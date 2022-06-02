import collections
import itertools
import os
import pathlib
import re
import time
from pathlib import Path

import dllogger
import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as opt
import torch_geometric
import torch_geometric.transforms as T
from dllogger import Logger, StdOutBackend
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import HeteroConv, MessagePassing, SAGEConv


def empty_step_format(step):
    return ""


def empty_prefix_format(timestamp):
    return ""


def get_framework_env_vars():
    return {
        "NVIDIA_PYTORCH_VERSION": os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION"),
        "CUBLAS_VERSION": os.environ.get("CUBLAS_VERSION"),
        "NCCL_VERSION": os.environ.get("NCCL_VERSION"),
        "CUDA_DRIVER_VERSION": os.environ.get("CUDA_DRIVER_VERSION"),
        "CUDNN_VERSION": os.environ.get("CUDNN_VERSION"),
        "CUDA_VERSION": os.environ.get("CUDA_VERSION"),
        "NVIDIA_PIPELINE_ID": os.environ.get("NVIDIA_PIPELINE_ID"),
        "NVIDIA_BUILD_ID": os.environ.get("NVIDIA_BUILD_ID"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def no_string_metric_format(metric, metadata, value):
    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    if metric == "String":
        return "{} {}".format(
            format.format(value) if value is not None else value, unit
        )
    return "{} : {} {}".format(
        metric, format.format(value) if value is not None else value, unit
    )


def setup_logger(rank, resume_training=False):
    if rank == 0:
        backends = [
            StdOutBackend(
                verbosity=dllogger.Verbosity.DEFAULT,
                step_format=empty_step_format,
                metric_format=no_string_metric_format,
                prefix_format=empty_prefix_format,
            ),
        ]

        logger = Logger(backends=backends)
    else:
        logger = Logger(backends=[])
    container_setup_info = get_framework_env_vars()
    logger.log(
        step="PARAMETER",
        data=container_setup_info,
        verbosity=dllogger.Verbosity.DEFAULT,
    )

    if not resume_training:
        logger.metadata("loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        logger.metadata("val_loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "VAL"})
    return logger


class Device:
    # assume nvml returns list of 64 bit ints
    _nvml_bit_affinity = 64

    _nvml_affinity_elements = (
        os.cpu_count() + _nvml_bit_affinity - 1
    ) // _nvml_bit_affinity

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self):
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self, scope):
        if scope == "socket":
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_SOCKET
        elif scope == "node":
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_NODE
        else:
            raise RuntimeError("Unknown scope")

        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinityWithinScope(
            self.handle, Device._nvml_affinity_elements, nvml_scope
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


def get_thread_siblings_list():
    """
    Returns a list of 2-element integer tuples representing pairs of
    hyperthreading cores.
    """
    path = "/sys/devices/system/cpu/cpu*/topology/thread_siblings_list"
    thread_siblings_list = []
    pattern = re.compile(r"(\d+)\D(\d+)")
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(sorted(map(int, res[0])))
                thread_siblings_list.append(pair)
    thread_siblings_list = list(set(thread_siblings_list))
    return thread_siblings_list


def build_thread_siblings_dict(siblings_list):
    siblings_dict = {}
    for siblings_tuple in siblings_list:
        for core in siblings_tuple:
            siblings_dict[core] = siblings_tuple

    return siblings_dict


def group_list_by_key(the_list, key):
    sorted_list = sorted(the_list, key=key)
    grouped = [tuple(group) for key, group in itertools.groupby(sorted_list, key=key)]
    return grouped


def group_by_siblings(affinities):
    siblings_list = get_thread_siblings_list()
    siblings_dict = build_thread_siblings_dict(siblings_list)

    def siblings_key(x):
        return siblings_dict.get(x, (x,))

    affinities = [
        tuple(group_list_by_key(affinity, key=siblings_key)) for affinity in affinities
    ]
    return affinities


def group_by_node(socket_affinities, node_affinities):
    socket_node_assigned_cores = collections.defaultdict(list)
    for socket, node_cores in zip(socket_affinities, node_affinities):
        socket_node_assigned_cores[socket].extend(node_cores)

    socket_node_assigned_cores = {
        key: tuple(sorted(set(value)))
        for key, value in socket_node_assigned_cores.items()
    }

    node_grouping = collections.defaultdict(list)

    for socket_cores, assigned_cores in socket_node_assigned_cores.items():
        unassigned_cores = sorted(list(set(socket_cores) - set(assigned_cores)))

        for assigned_core in assigned_cores:
            node_grouping[assigned_core].append(assigned_core)

        for assigned, unassigned in zip(
            itertools.cycle(assigned_cores), unassigned_cores
        ):
            node_grouping[assigned].append(unassigned)

    node_grouping = {key: tuple(value) for key, value in node_grouping.items()}

    grouped_affinities = [
        tuple(node_grouping[item] for item in node_affinity)
        for node_affinity in node_affinities
    ]
    return grouped_affinities


def ungroup_by_nodes(affinities, scope):
    if scope == "socket":
        affinities = [list(itertools.chain(*zip(*affinity))) for affinity in affinities]
    elif scope == "node":
        affinities = [[group[0] for group in affinity] for affinity in affinities]
    return affinities


def ungroup_by_siblings(affinities, cores):
    if cores == "all_logical":
        affinities = [list(itertools.chain(*affinity)) for affinity in affinities]
    elif cores == "single_logical":
        affinities = [[group[0] for group in affinity] for affinity in affinities]
    else:
        raise RuntimeError("Unknown cores mode")
    return affinities


def check_core_count(affinities, min_cores=1, max_cores=None):
    for gpu_id, affinity in enumerate(affinities):
        if len(affinity) < min_cores:
            raise RuntimeError(
                f"Number of available physical cores for GPU {gpu_id} is less "
                f"the predefinied minimum, min_cores={min_cores}, available "
                f"physical cores: {affinity} (count={len(affinity)})"
            )

    if max_cores is not None:
        affinities = [affinity[:max_cores] for affinity in affinities]

    return affinities


def ungroup_all_and_check_count(affinities, scope, cores, min_cores=1, max_cores=None):
    affinities = ungroup_by_nodes(affinities, scope)
    affinities = check_core_count(affinities, min_cores, max_cores)
    affinities = ungroup_by_siblings(affinities, cores)
    return affinities


def check_affinities(affinities):
    # sets of cores should be either identical or disjoint
    for i, j in itertools.product(affinities, affinities):
        if not set(i) == set(j) and not set(i).isdisjoint(set(j)):
            raise RuntimeError(
                f"Sets of cores should be either identical or disjoint, "
                f"but got {i} and {j}."
            )


def get_affinities(nproc_per_node, scope, exclude_unavailable_cores=True):
    devices = [Device(i) for i in range(nproc_per_node)]
    affinities = [dev.get_cpu_affinity(scope) for dev in devices]

    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        affinities = [
            sorted(list(set(affinity) & available_cores)) for affinity in affinities
        ]

    check_affinities(affinities)

    return affinities


def get_grouped_affinities(nproc_per_node, exclude_unavailable_cores=True):
    socket_affinities = get_affinities(
        nproc_per_node, "socket", exclude_unavailable_cores
    )
    node_affinities = get_affinities(nproc_per_node, "node", exclude_unavailable_cores)

    sibling_socket_affinities = group_by_siblings(socket_affinities)
    sibling_node_affinities = group_by_siblings(node_affinities)

    grouped_affinities = group_by_node(
        sibling_socket_affinities, sibling_node_affinities
    )

    return grouped_affinities


def get_unique(
    nproc_per_node,
    scope,
    cores,
    mode,
    min_cores,
    max_cores,
    balanced=True,
):
    """
    The process is assigned with a unique subset of available physical CPU
    cores from the list of all CPU cores recommended by pynvml for the GPU with
    a given id.

    Assignment automatically includes available hyperthreading siblings if
    cores='all_logical'.

    Args:
        nproc_per_node: number of processes per node
        scope: scope for retrieving affinity from pynvml, 'node' or 'socket'
        cores: 'all_logical' or 'single_logical'
        mode: 'unique_contiguous' or 'unique_interleaved'
        balanced: assign an equal number of physical cores to each process,
    """
    grouped_affinities = get_grouped_affinities(nproc_per_node)

    grouped_affinities_to_device_ids = collections.defaultdict(list)

    for idx, grouped_affinity in enumerate(grouped_affinities):
        grouped_affinities_to_device_ids[tuple(grouped_affinity)].append(idx)

    # compute minimal number of physical cores per GPU across all GPUs and
    # sockets, code assigns this number of cores per GPU if balanced == True
    min_physical_cores_per_gpu = min(
        [
            len(cores) // len(gpus)
            for cores, gpus in grouped_affinities_to_device_ids.items()
        ]
    )

    grouped_unique_affinities = [None] * nproc_per_node

    for (
        grouped_affinity,
        device_ids,
    ) in grouped_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            grouped_affinity = grouped_affinity[
                : devices_per_group * min_physical_cores_per_gpu
            ]
        else:
            cores_per_device = len(grouped_affinity) // devices_per_group

        for subgroup_id, device_id in enumerate(device_ids):
            # In theory there should be no difference in performance between
            # 'interleaved' and 'contiguous' pattern on Intel-based DGX-1,
            # but 'contiguous' should be better for DGX A100 because on AMD
            # Rome 4 consecutive cores are sharing L3 cache.
            # TODO: code doesn't attempt to automatically detect layout of
            # L3 cache, also external environment may already exclude some
            # cores, this code makes no attempt to detect it and to align
            # mapping to multiples of 4.

            if mode == "unique_interleaved":
                unique_grouped_affinity = list(
                    grouped_affinity[subgroup_id::devices_per_group]
                )
            elif mode == "unique_contiguous":
                unique_grouped_affinity = list(
                    grouped_affinity[
                        subgroup_id
                        * cores_per_device : (subgroup_id + 1)
                        * cores_per_device
                    ]
                )
            else:
                raise RuntimeError("Unknown set_unique mode")

            grouped_unique_affinities[device_id] = unique_grouped_affinity

    ungrouped_affinities = ungroup_all_and_check_count(
        grouped_unique_affinities, scope, cores, min_cores, max_cores
    )
    return ungrouped_affinities


def set_affinity(
    gpu_id,
    nproc_per_node,
    *,
    mode="unique_contiguous",
    scope="node",
    cores="all_logical",
    balanced=True,
    min_cores=1,
    max_cores=None,
):
    """
    The process is assigned with a proper CPU affinity that matches CPU-GPU
    hardware architecture on a given platform. Usually, setting proper affinity
    improves and stabilizes the performance of deep learning training workloads.

    This function assumes that the workload runs in multi-process single-device
    mode (there are multiple training processes, and each process is running on
    a single GPU). This is typical for multi-GPU data-parallel training
    workloads (e.g., using `torch.nn.parallel.DistributedDataParallel`).

    Available affinity modes:
    * 'all' - the process is assigned with all available physical CPU cores
    recommended by pynvml for the GPU with a given id.
    * 'single' - the process is assigned with the first available
    physical CPU core from the list of all physical CPU cores recommended by
    pynvml for the GPU with a given id (multiple GPUs could be assigned with
    the same CPU core).
    * 'single_unique' - the process is assigned with a single unique
    available physical CPU core from the list of all CPU cores recommended by
    pynvml for the GPU with a given id.
    * 'unique_interleaved' - the process is assigned with a unique subset of
    available physical CPU cores from the list of all physical CPU cores
    recommended by pynvml for the GPU with a given id, cores are assigned with
    interleaved indexing pattern
    * 'unique_contiguous' - (the default mode) the process is assigned with a
    unique subset of available physical CPU cores from the list of all physical
    CPU cores recommended by pynvml for the GPU with a given id, cores are
    assigned with contiguous indexing pattern

    Available "scope" modes:
    * 'node' - sets the scope for pynvml affinity queries to NUMA node
    * 'socket' - sets the scope for pynvml affinity queries to processor socket

    Available "cores" modes:
    * 'all_logical' - assigns the process with all logical cores associated with
    a given corresponding physical core (i.e., automatically includes all
    available hyperthreading siblings)
    * 'single_logical' - assigns the process with only one logical core
    associated with a given corresponding physical core (i.e., excludes
    hyperthreading siblings)

    'unique_contiguous' is the recommended mode for deep learning
    training workloads on NVIDIA DGX machines.

    Args:
        gpu_id: integer index of a GPU, value from 0 to 'nproc_per_node' - 1
        nproc_per_node: number of processes per node
        mode: affinity mode
        scope: scope for retrieving affinity from pynvml, 'node' or 'socket'
        cores: 'all_logical' or 'single_logical'
        balanced: assign an equal number of physical cores to each process,
            affects only 'unique_interleaved' and
            'unique_contiguous' affinity modes
        min_cores: (default=1) the intended minimum number of physical cores per
            process, code raises RuntimeError if the number of available cores
            is less than 'min_cores'
        max_cores: (default=None) the intended maxmimum number of physical cores
            per process, the list of assigned cores is trimmed to the first
            'max_cores' cores if max_cores is not None

    Returns a set of logical CPU cores on which the process is eligible to run.

    WARNING: On DGX A100, only half of the CPU cores have direct access to GPUs.
    set_affinity with scope='node' restricts execution only to the CPU cores
    directly connected to GPUs. On DGX A100, it will limit the code to half of
    the CPU cores and half of CPU memory bandwidth (which may be fine for many
    DL models). Use scope='socket' to use all available DGX A100 CPU cores.

    WARNING: Intel's OpenMP implementation resets affinity on the first call to
    an OpenMP function after a fork. It's recommended to run with env variable:
    `KMP_AFFINITY=disabled` if the affinity set by gpu_affinity should be
    preserved after a fork (e.g. in PyTorch DataLoader workers).

    Example:

    import argparse
    import os

    import gpu_affinity
    import torch


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--local_rank',
            type=int,
            default=os.getenv('LOCAL_RANK', 0),
        )
        args = parser.parse_args()

        nproc_per_node = torch.cuda.device_count()

        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
        print(f'{args.local_rank}: core affinity: {affinity}')


    if __name__ == "__main__":
        main()

    Launch the example with:
    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py
    """
    pynvml.nvmlInit()

    affinity = get_unique(
        nproc_per_node,
        scope,
        cores,
        mode,
        min_cores,
        max_cores,
        balanced,
    )

    os.sched_setaffinity(0, affinity[gpu_id])
    set_affinity = os.sched_getaffinity(0)
    return set_affinity


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


class NodeDataObject:
    def __init__(self, data):
        self.data = data

    def get_labels(self, batch):
        return batch["v0"].y

    def build_train_dataloader_post_dist(self, device, use_ddp):
        self.train_dataloader = make_pyg_loader(
            self.data,
            device,
        )


class ExampleMSE:
    def __call__(self, output, target):
        diff_sq = (output - target) ** 2
        return diff_sq.mean()


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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )


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


# dllogger doesn't accept list but json doesn't serialize tensor/ndarray
def metrics_list_to_num_dict(metrics):
    out = {}

    for k, v in metrics.items():
        if isinstance(v, list):
            for i, vv in enumerate(v):
                out["{}_{}".format(k, i)] = vv

        else:
            out[k] = v

    return out


def metrics_values_to_list(metrics):
    out = {}

    for k, v in metrics.items():
        if torch.is_tensor(v):
            out[k] = v.tolist()
        else:
            out[k] = v

    return out


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


def reduce_metrics(metrics, num_gpus, average=False):
    out = {}
    for k, v in metrics.items():
        reduced = reduce_tensor(v, num_gpus, average=average).tolist()
        out[k] = reduced

    return out


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


class GradientClipper:
    def __init__(self, parameters, max_norm, norm_type):
        self._parameters = parameters
        self._max_norm = max_norm
        self._norm_type = norm_type

    def __call__(self):
        torch.nn.utils.clip_grad_norm_(
            self._parameters, self._max_norm, self._norm_type
        )


class OptWrapper:
    def __init__(
        self,
        model,
        opt_partial=None,
        scheduler_partial=None,
        clip_gradient_norm_type=None,
        clip_gradient_max_norm=None,
    ):
        self.model = model
        self.opt_partial = opt_partial
        self.scheduler_partial = scheduler_partial
        self.clip_gradient_norm_type = clip_gradient_norm_type
        self.clip_gradient_max_norm = clip_gradient_max_norm

    def build(self):
        output = {}
        params = list(self.model.parameters())

        output["optimizer"] = self.opt_partial(params)
        if self.scheduler_partial is not None:
            output["scheduler"] = self.scheduler_partial(output["optimizer"])

        if (
            self.clip_gradient_norm_type is not None
            and self.clip_gradient_max_norm is not None
        ):
            output["gradient_clipper"] = GradientClipper(
                parameters=params,
                max_norm=self.clip_gradient_max_norm,
                norm_type=self.clip_gradient_norm_type,
            )

        return output


class Trainer:
    def __init__(
        self,
        data_object,
        model,
        optimizers,
        criterion,
        output_dir="./outputs/{}-{}-{}/{}-{}-{}".format(
            time.localtime().tm_year,
            time.localtime().tm_mon,
            time.localtime().tm_mday,
            time.localtime().tm_hour,
            time.localtime().tm_min,
            time.localtime().tm_sec,
        ),
        metrics=None,
        n_gpus=1,
        epochs=10,
        amp_enabled=False,
        log_frequency=100,
        user_callback_list=[],
        eval_interval=1,
        limit_batches=None,
        omp_num_threads=None,
        **kwargs,
    ):
        self.output_dir = output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.data_object = data_object
        self.model = model
        self.optimizers = optimizers
        self.criterion = criterion
        self.n_gpus = n_gpus
        self.epochs = epochs
        self.amp_enabled = amp_enabled
        self.log_frequency = log_frequency
        self.eval_interval = eval_interval
        self.limit_batches = limit_batches
        self.omp_num_threads = omp_num_threads
        self.kwargs = kwargs

        self.epoch = 0
        self.train_steps = -1
        self.valid_steps = -1
        self.test_steps = -1
        self.global_step = 0
        self.procs = []
        self.schedulers = []
        self.clip_gradients = []
        self.metrics = metrics or {}
        self.is_training = False

        self.logged_train_metrics = {}
        self.logged_valid_metrics = {}
        self.logged_test_metrics = {}

        self.check_parameters()

        return

    def check_parameters(self):
        if self.limit_batches is not None:
            assert self.limit_batches >= 1

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
            metrics = parent_conn.recv()
            parent_conn.close()
            pc.join()
        else:
            metrics = self.fit_process()

        return metrics

    def train_procedure(self, callback_metrics):
        self.do_train_epoch(callback_metrics)
        if self.rank == 0:
            print("TRAIN METRICS")
            print(self.logged_train_metrics)

    def fit_process(self, rank=0):
        self.setup(rank)
        callback_metrics = {}
        self.on_train_begin(callback_metrics)

        while self.epoch < self.epochs:
            self.on_epoch_begin(callback_metrics)
            self.train_procedure(callback_metrics)
            self.on_epoch_end(callback_metrics)

        self.on_train_end(callback_metrics)

        if self.n_gpus > 1:
            callback_metrics_ = reduce_metrics(callback_metrics, self.n_gpus)
            for k, v in callback_metrics_.items():
                callback_metrics[k] = v

    def setup(self, rank):
        use_ddp = self.n_gpus > 1

        self.rank = rank
        self.logger = setup_logger(self.rank)

        if self.n_gpus > 1:
            set_affinity(rank, self.n_gpus)

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

        self.data_object.build_train_dataloader_post_dist(device, use_ddp)
        self.initialize_module(self.model, device)
        self.model = self.model.to(device)

        for i in range(len(self.optimizers)):
            if isinstance(self.optimizers[i], OptWrapper):
                opt_out = self.optimizers[i].build()
                self.optimizers[i] = opt_out["optimizer"]
                if opt_out.get("scheduler", None) is not None:
                    self.schedulers.append(opt_out["scheduler"])
                if opt_out.get("gradient_clipper", None) is not None:
                    clipper = opt_out["gradient_clipper"]
                    self.clip_gradients.append(clipper)

        if use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
            )

        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def do_train_epoch(self, callback_metrics):
        self.on_train_epoch_begin(callback_metrics)
        data = {}
        train_step = 0
        for step, batch in enumerate(self.data_object.train_dataloader):
            train_step = step + 1
            self.on_batch_begin(step, callback_metrics)
            callback_metrics = self.step(batch, mode="train")
            for k, v in callback_metrics.items():
                data[k] = data.get(k, 0.0) + v
            self.on_batch_end(step, callback_metrics)
            print('finished step', step)
            if self.limit_batches is not None and step + 1 >= self.limit_batches:
                break

        self.train_steps = train_step

        for k, v in data.items():
            data[k] = v / len(self.data_object.train_dataloader)

        self.logged_train_metrics = data

        self.on_train_epoch_end(callback_metrics)

    def forward_pass(self, batch):
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            output = self.model(batch)
            y = batch['v0'].y
            target = y[:1024]
            out = output['v0'][:1024]
            loss = self.criterion(out, target)

            return output, target, loss

    def step(self, batch, mode="train"):
        metrics = {}
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        output, target, loss = self.forward_pass(batch)

        self.scaler.scale(loss).backward()
        loss = loss.detach()

        for gradient_clipper in self.clip_gradients:
            gradient_clipper()

        for optimizer in self.optimizers:
            self.scaler.step(optimizer)

        self.scaler.update()

        if self.n_gpus > 1:
            loss = reduce_tensor(loss, self.n_gpus, average=True)

        metrics["loss"] = loss.item()
        return metrics

    def on_train_begin(self, metrics):
        self.is_training = True
        return

    def on_train_end(self, metrics):
        self.logger.flush()

        if self.n_gpus > 1:
            torch.distributed.barrier()

        self.is_training = False

    def on_train_epoch_begin(self, metrics):
        self.model.train()
        self.criterion.train()
        return

    def on_train_epoch_end(self, metrics):
        self.logger.flush()

        if self.n_gpus > 1:
            torch.distributed.barrier()

    def on_epoch_begin(self, metrics):
        self.logger.log(
            step=self.global_step,
            data={"epoch": self.epoch},
            verbosity=dllogger.Verbosity.VERBOSE,
        )
        if self.n_gpus > 1:
            torch.distributed.barrier()

    def on_epoch_end(self, metrics):
        if self.n_gpus > 1:
            torch.distributed.barrier()
        self.epoch += 1
        for scheduler in self.schedulers:
            scheduler.step()
        return

    def on_batch_begin(self, step, metrics, synchronise=False):
        if synchronise:
            if self.n_gpus > 1:
                torch.distributed.barrier()

        return

    def on_batch_end(self, step, metrics, synchronise=False):
        if synchronise:
            if self.n_gpus > 1:
                torch.distributed.barrier()

        non_tensor_metrics = metrics_list_to_num_dict(metrics)
        self.logger.log(
            step=self.global_step,
            data=non_tensor_metrics,
            verbosity=dllogger.Verbosity.VERBOSE,
        )
        self.global_step += 1
        if self.global_step % self.log_frequency == 0:
            self.logger.flush()

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
    data_object = NodeDataObject(
        data=data,
    )
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
        data_object=data_object,
        model=model,
        optimizers=[optimizers],
        criterion=nn.CrossEntropyLoss(),
        n_gpus=n_gpus,
        epochs=1,
        metrics={"example_mse_1": ExampleMSE()},
        limit_batches=5,
    )

    trainer.fit()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_mag_workflow(1)
    print("1 GPU TEST PASSED!!!!")
    print()
    print()
    test_mag_workflow(2)