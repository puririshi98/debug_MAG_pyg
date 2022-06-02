import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as opt
import torch_geometric
import torch_geometric.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import HeteroConv, MessagePassing, SAGEConv


def make_pyg_loader(graph, device):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / 2
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / 2
    num_work = int(num_work)
    return NeighborLoader(
        graph,
        num_neighbors=[50, 50],
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        input_nodes=("v0", None),
        num_workers=num_work,
        replace=True,
        transform=T.ToDevice(device),
    )


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    conv_type : MessagePassing
        GNN conv desired. Default: None. Uses SAGEConv as the default.
        Examples: torch_geometric.nn.conv.GraphConv, torch_geometric.nn.conv.GATConv
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        edge_types,
        *,
        activation=None,
        self_loop=False,
        dropout=0.0,
        conv_type=None,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.num_relations = len(self.edge_types)
        self.activation = activation
        self.self_loop = self_loop
        if conv_type is None:
            self.conv_type = SAGEConv
        else:
            self.conv_type = conv_type
        assert issubclass(
            self.conv_type, MessagePassing
        ), "Please only pass GNN convs that extend Message Passing. Actual: " + str(
            self.conv_type
        )
        if isinstance(in_feat, int):
            self.conv = HeteroConv(
                {rel: self.conv_type(in_feat, out_feat) for rel in self.edge_types}
            )
        else:
            self.conv = HeteroConv(
                {
                    rel: self.conv_type((in_feat[rel[0]], in_feat[rel[-1]]), out_feat)
                    for rel in self.edge_types
                }
            )

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, g, x):
        r"""
        Args:
            g: HeteroData object
            x: (Dict[str, Tensor]) –
                A dictionary holding node feature information for each individual node type.
        """
        edge_index_dict = g.collect("edge_index")
        h = self.conv(x, edge_index_dict)
        for node_type in h.keys():
            if self.dropout is not None:
                h[node_type] = self.dropout(h[node_type])
            if self.activation is not None:
                h[node_type] = self.activation(h[node_type])
        return h


class HeteroModule(nn.Module):
    # conv_type : MessagePassing
    #     GNN conv desired. Default: None. Uses SAGEConv as the default.
    #     Examples: torch_geometric.nn.conv.GraphConv, torch_geometric.nn.conv.GATConv
    def __init__(
        self,
        edge_types,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        activation=torch.nn.functional.relu,
        aggregator_type="mean",
        dropout=0.5,
        embedding=None,
        conv_type=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.edge_types = edge_types
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.node_dims = []
        self.layers.append(
            RelGraphConvLayer(
                self.dim_in,
                self.dim_hidden,
                self.edge_types,
                activation=self.activation,
                dropout=self.dropout,
                conv_type=conv_type,
            )
        )

        for i in range(1, self.n_layers - 1):
            self.layers.append(
                RelGraphConvLayer(
                    self.dim_hidden,
                    self.dim_hidden,
                    self.edge_types,
                    activation=self.activation,
                    dropout=self.dropout,
                    conv_type=conv_type,
                )
            )

        self.layers.append(
            RelGraphConvLayer(
                self.dim_hidden, self.dim_out, self.edge_types, conv_type=conv_type
            )
        )

    def forward(self, batch):
        h = batch.collect("x")
        for i, layer in enumerate(self.layers):
            h = layer(batch, h)
        return h


def run_fit(rank, trainer, child_conn):
    data = trainer.fit_process(rank)
    if rank == 0:
        child_conn.send(data)
        child_conn.close()


def reduce_tensor(tensor, num_gpus, average=False):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor).cuda()
    elif not tensor.is_cuda:
        tensor = tensor.cuda()

    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
    return rt


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def is_main_process():
    return get_rank() == 0


class Trainer:
    def __init__(
        self,
        data,
        model,
        optimizers,
        criterion,
        n_gpus=1,
        omp_num_threads=None,
        **kwargs,
    ):
        self.data = data
        self.model = model
        self.optimizers = optimizers
        self.criterion = criterion
        self.n_gpus = n_gpus
        self.omp_num_threads = omp_num_threads
        self.kwargs = kwargs

        self.epoch = 0
        self.train_steps = -1
        self.global_step = 0
        self.procs = []
        self.is_training = False

    def fit(self):
        """
        Runs self.epochs of training on the provided dataset
        """

        if self.omp_num_threads is None:
            if self.n_gpus == 0:
                omp_num_threads = str(mp.cpu_count() // 2)
            else:
                omp_num_threads = str(mp.cpu_count() // 2 // self.n_gpus)
            os.environ["OMP_NUM_THREADS"] = omp_num_threads
        else:
            os.environ["OMP_NUM_THREADS"] = self.omp_num_threads

        if self.n_gpus > 1:
            parent_conn, child_conn = mp.Pipe()
            pc = mp.spawn(
                run_fit,
                args=(
                    self,
                    child_conn,
                ),
                nprocs=self.n_gpus,
                join=False,
            )
            parent_conn.recv()
            parent_conn.close()
            pc.join()
        else:
            self.fit_process()

    def fit_process(self, rank=0):
        self.setup(rank)
        self.model.train()
        self.criterion.train()
        self.do_train_epoch()

    def setup(self, rank):
        use_ddp = self.n_gpus > 1

        self.rank = rank

        if self.n_gpus > 1:

            dist_init_method = "tcp://{master_ip}:{master_port}".format(
                master_ip="127.0.0.1", master_port="29500"
            )
            torch.cuda.set_device(rank)
            device = torch.device("cuda:" + str(rank))
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=dist_init_method,
                world_size=self.n_gpus,
                rank=rank,
            )
            torch.cuda.synchronize()
        elif self.n_gpus == 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.train_dataloader = make_pyg_loader(
            self.data,
            device,
        )
        self.initialize_module(self.model, device)
        self.model = self.model.to(device)

        if use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
            )

        self.device = device

    def do_train_epoch(self):
        train_step = 0
        for step, batch in enumerate(self.train_dataloader):
            train_step = step + 1
            self.step(batch, mode="train")
            if is_main_process():
                print("finished step", step)
            if step >= 5:
                break

        self.train_steps = train_step
        if self.n_gpus > 1:
            torch.distributed.barrier()

    def forward_pass(self, batch):
        output = self.model(batch)
        y = batch["v0"].y
        target = y[:1024]
        out = output["v0"][:1024]
        loss = self.criterion(out, target)

        return output, target, loss

    def step(self, batch, mode="train"):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        output, target, loss = self.forward_pass(batch)

        loss.backward()

        if self.n_gpus > 1:
            loss = reduce_tensor(loss, self.n_gpus, average=True)

    def initialize_module(self, module, device):
        for child in module.children():
            self.initialize_module(child, device)
        if hasattr(module, "init_device"):
            module.init_device(device)


DATA_DIR = "/workspace/data/"


def test_mag_workflow(n_gpus):
    conv_type = "SAGEConv"
    torch.cuda.empty_cache()
    conv_type = getattr(
        torch_geometric.nn.conv, conv_type
    )  # a normal user would just import the desired layer directly.
    from torch_geometric.datasets import FakeHeteroDataset

    torch_geometric.seed.seed_everything(42)
    data = FakeHeteroDataset(avg_num_nodes=20000).generate_data()
    print(data)
    model = HeteroModule(
        dim_in={
            node_type: data[node_type].x.shape[-1] for node_type in data.node_types
        },
        edge_types=data.edge_types,
        dim_hidden=400,
        dim_out=349,
        n_layers=1,
        conv_type=conv_type,
    )

    optimizers = opt.Adam(
        params=model.parameters(),
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=0.0,
        amsgrad=False,
    )

    trainer = Trainer(
        data=data,
        model=model,
        optimizers=[optimizers],
        criterion=nn.CrossEntropyLoss(),
        n_gpus=n_gpus,
    )

    trainer.fit()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_mag_workflow(1)
    print("1 GPU TEST PASSED!!!!")
    print()
    print()
    test_mag_workflow(2)
