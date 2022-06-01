import collections
import enum
import itertools
import json
import os
import pathlib
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import dllogger
import numpy as np
import pandas as pd
import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as opt
import torch_geometric
from dllogger import Logger, StdOutBackend
from ogb.nodeproppred import NodePropPredDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import HeteroData
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


class StrEnum(str, enum.Enum):
    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, (str, enum.auto)):
                raise TypeError(
                    "Values of StrEnums must be strings: {} is a {}".format(
                        repr(arg), type(arg)
                    )
                )
        return super().__new__(cls, *args)

    def __str__(self):
        return self.value

    # def __repr__(self):
    #    return self.value

    def _generate_next_value_(name, *_):
        return name


class Meta(StrEnum):
    NODES = "nodes"
    NODE_TYPES = "node_types"
    EDGES = "edges"
    EDGE_TYPES = "edge_types"
    SPLIT = "split"
    SPLIT_NAME = "split_column_name"
    SRC_TYPE = "src_node_type"
    DST_TYPE = "dst_node_type"
    SRC_ID = "src_node_id"
    DST_ID = "dst_node_id"
    REVERSE = "generate_reverse_name"
    FEAT = "features"
    NAME = "name"
    LABEL = "label"
    FILES = "file_paths"
    FEAT_NAME = "name"
    EDGE_ID = "id"
    DTYPE = "dtype"
    SHAPE = "shape"
    NUM_NODES_DICT = "num_nodes_dict"
    # EDGE_SUBMATRIX_SHAPES_DICT = "edge_submatrix_shapes_dict"
    EDGE_FEATURE_DICT = "edge_feature_dict"
    NUM_EDGES_DICT = "num_edges_dict"
    EDGE_SPLIT_AT_DICT = "edge_split_at_dict"
    NODE_SPLIT_AT_DICT = "node_split_at_dict"
    # NODE_SUBMATRIX_SHAPES_DICT = "node_submatrix_shapes_dict"
    CATEGORICAL_WIDTH = "categorical_width"


MetadataKeys = Meta


def load_metadata(root_path):
    try:
        with open(os.path.join(root_path, "metadata.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def make_pyg_loader(graph, train_ids, metadata, device):
    src_node_types = set(
        [
            i[MetadataKeys.SRC_TYPE]
            for i in metadata[MetadataKeys.EDGES][MetadataKeys.EDGE_TYPES]
        ]
    )
    dst_node_types = set(
        [
            i[MetadataKeys.DST_TYPE]
            for i in metadata[MetadataKeys.EDGES][MetadataKeys.EDGE_TYPES]
        ]
    )
    unupdated_nodes = list(set(src_node_types - dst_node_types))
    T_list = [T.ToDevice(device)]
    if len(unupdated_nodes) > 0:
        # If a node type does not get filled with message passing this will cause errors
        # Solve this by adding reverse edges:
        T_list += [T.ToUndirected(merge=False)]
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
        input_nodes=(graph.labeled_node_type, train_ids[graph.labeled_node_type]),
        num_workers=num_work,
        replace=True,
        transform=T.Compose(T_list),
    )


def get_split_nids(graph, metadata, split):
    split_values = {}
    for node in metadata[Meta.NODES][Meta.NODE_TYPES]:
        compare = None
        if node[Meta.NAME] in graph[Meta.NODE_SPLIT_AT_DICT].keys():
            compare = graph[Meta.NODE_SPLIT_AT_DICT][node[Meta.NAME]]
        if compare is not None:
            split_values[node[Meta.NAME]] = (
                (compare == split).nonzero()[:, 0].type(torch.int64)
            )
    return split_values


def extract_block_one_hot_labels(
    batch, label_name, label_size, node_type, device, logits
):
    batch_size = batch[batch.labeled_node_type].batch_size
    return (
        logits[:batch_size],
        torch.nn.functional.one_hot(
            batch[node_type].y[:batch_size], num_classes=label_size
        )
        .float()
        .to(device),
    )


def get_logit_labels_types_block(batch, logits, node_label_map, device):
    all_logits = []
    all_labels = []
    for node_type, logits in logits.items():
        label_names = node_label_map[node_type]
        for label_name, label_size in label_names:
            # - XXX: assumes labels are present for all nodes in batch
            #        will require a mask for filtering
            logits, labels = extract_block_one_hot_labels(
                batch, label_name, label_size, node_type, device, logits
            )
            all_logits.append(logits)
            all_labels.append(labels)
    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    return logits, labels


def read_pandas_feats(root_path, dicty):
    return pd.concat(
        [pd.read_parquet(os.path.join(root_path, path)) for path in dicty[Meta.FILES]]
    )


def make_split(num, g, key):
    train, val = int(0.8 * num), int(0.1 * num)
    test = num - train - val
    split = torch.cat((torch.zeros(train), torch.ones(val), 2 * torch.ones(test)))[
        torch.randperm(num)
    ].to("cpu")
    dictkey = (
        Meta.NODE_SPLIT_AT_DICT
        if not isinstance(key, tuple)
        else Meta.EDGE_SPLIT_AT_DICT
    )
    g[dictkey][key] = split
    return g


def load_graph(root_path, metadata, use_reverse_edges_features=False):
    # Initialize PyG Graph object
    g = HeteroData()
    # print("Raw Metadata:", metadata)
    relation_types = []
    g[Meta.NUM_EDGES_DICT] = {}
    biggest_node_id_dict = {}
    # Parse Edge Indexes
    for edge in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
        feats = read_pandas_feats(root_path, edge)
        relation = (
            edge[Meta.SRC_TYPE],
            edge[Meta.NAME],
            edge[Meta.DST_TYPE],
        )
        relation_types.append(relation)
        src = (
            torch.tensor(feats[Meta.SRC_ID].astype("int64").values)
            .to("cpu")
            .reshape(1, -1)
        )
        dst = (
            torch.tensor(feats[Meta.DST_ID].astype("int64").values)
            .to("cpu")
            .reshape(1, -1)
        )
        if edge[Meta.SRC_TYPE] not in biggest_node_id_dict.keys():
            biggest_node_id_dict[edge[Meta.SRC_TYPE]] = int(torch.max(src))
        else:
            biggest_node_id_dict[edge[Meta.SRC_TYPE]] = max(
                biggest_node_id_dict[edge[Meta.SRC_TYPE]], int(torch.max(src))
            )
        if edge[Meta.DST_TYPE] not in biggest_node_id_dict.keys():
            biggest_node_id_dict[edge[Meta.DST_TYPE]] = int(torch.max(dst))
        else:
            biggest_node_id_dict[edge[Meta.DST_TYPE]] = max(
                biggest_node_id_dict[edge[Meta.DST_TYPE]], int(torch.max(dst))
            )
        g[relation].edge_index = torch.cat((src, dst), axis=0)
        g[Meta.NUM_EDGES_DICT][relation] = g[relation].edge_index.size()[-1]
        if edge.get(Meta.REVERSE, None):
            relation = (
                edge[Meta.DST_TYPE],
                edge[Meta.REVERSE],
                edge[Meta.SRC_TYPE],
            )
            g[relation].edge_index = torch.cat((dst, src), axis=0)
            g[Meta.NUM_EDGES_DICT][relation] = g[relation].edge_index.size()[-1]
    node_types = []
    filled_node_types = []
    # PyG Hetero Loaders don't like additional information in the node stores
    # Make seperate dicts to hold info
    g[Meta.NODE_SPLIT_AT_DICT] = {}
    # Parse Node Features
    g.labeled_node_type = ""
    for node in metadata[Meta.NODES][Meta.NODE_TYPES]:
        node_types.append(node[Meta.NAME])
        if node.get(Meta.FILES):
            filled_node_types.append(node)
            feats = read_pandas_feats(root_path, node)
            list_of_submtrx_to_cat = []
            # submtrx_shapes = []
            for feature in node[Meta.FEAT]:
                # Store Node Labels
                if feature[Meta.NAME] == Meta.LABEL or Meta.LABEL in feature.keys():
                    g[node[Meta.NAME]].y = (
                        torch.tensor(feats[feature[Meta.NAME]])
                        .reshape(-1)
                        .to(torch.int64)
                    )
                    g.labeled_node_type = node[Meta.NAME]
                elif isinstance(feature[Meta.NAME], list):
                    for name in feature[Meta.NAME]:
                        submatrix = torch.Tensor(feats[name]).to("cpu")
                        list_of_submtrx_to_cat.append(
                            submatrix
                            if len(submatrix.size()) == 2
                            else submatrix.reshape(1, -1)
                        )
                        # For look up table
                        # submtrx_shapes.append(
                        #     (name, list_of_submtrx_to_cat[-1].size()[0])
                        # )
                else:
                    submatrix = torch.Tensor(feats[feature[Meta.NAME]]).to("cpu")
                    list_of_submtrx_to_cat.append(
                        submatrix
                        if len(submatrix.size()) == 2
                        else submatrix.reshape(1, -1)
                    )
                    # submtrx_shapes.append(
                    #     (
                    #         feature[Meta.NAME],
                    #         list_of_submtrx_to_cat[-1].size()[0],
                    #     )
                    # )
            # Store node features
            g[node[Meta.NAME]].x = torch.cat(list_of_submtrx_to_cat).T
            # Store individual feature shapes for future retrieval
            # g[node[Meta.NAME]].submtrx_shapes = dict(submtrx_shapes)
            g[node[Meta.NAME]].num_nodes = g[node[Meta.NAME]].x.size()[0]
            # Store given split or choose at random
            g = make_split(g[node[Meta.NAME]].num_nodes, g, node[Meta.NAME])
        else:
            # Need to atleast store num nodes. DGL does this automatically
            # PyG does not
            g[node[Meta.NAME]].num_nodes = biggest_node_id_dict[node[Meta.NAME]] + 1
        g[node[Meta.NAME]].n_id = torch.arange(g[node[Meta.NAME]].num_nodes)

    if Meta.NUM_NODES_DICT in metadata.keys():
        g.num_nodes = sum(list(metadata[Meta.NUM_NODES_DICT].values()))
    else:
        g.num_nodes = sum([g[node[Meta.NAME]].num_nodes for node in filled_node_types])

    # PyG Hetero Loaders don't like additional information in the edge stores
    # Make seperate dicts to hold info
    # g[Meta.EDGE_SUBMATRIX_SHAPES_DICT] = {}
    g[Meta.EDGE_FEATURE_DICT] = {}
    g[Meta.EDGE_SPLIT_AT_DICT] = {}
    g.predictable_edge_type = ""
    for edge in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
        relation = (
            edge[Meta.SRC_TYPE],
            edge[Meta.NAME],
            edge[Meta.DST_TYPE],
        )
        if edge.get(Meta.FILES):
            feats = read_pandas_feats(root_path, edge)
            list_of_submtrx_to_cat = []
            # submtrx_shapes = []
            for feature in edge[Meta.FEAT]:
                if feature[Meta.NAME] == Meta.LABEL or Meta.LABEL in feature.keys():
                    g[relation].y = (
                        torch.tensor(feats[feature[Meta.NAME]].values)
                        .reshape(-1)
                        .to(torch.int64)
                    )
                    g.predictable_edge_type = tuple(relation)
                elif isinstance(feature[Meta.NAME], list):
                    for name in feature[Meta.NAME]:
                        submatrix = torch.Tensor(feats[name]).to("cpu")
                        list_of_submtrx_to_cat.append(
                            submatrix
                            if len(submatrix.size()) == 2
                            else submatrix.reshape(1, -1)
                        )
                        # submtrx_shapes.append(
                        #     (name, list_of_submtrx_to_cat[-1].size()[0])
                        # )
                else:
                    submatrix = torch.Tensor(feats[feature[Meta.NAME]].values).to("cpu")
                    list_of_submtrx_to_cat.append(
                        submatrix
                        if len(submatrix.size()) == 2
                        else submatrix.reshape(1, -1)
                    )
                    # submtrx_shapes.append(
                    #     (
                    #         str(feature[Meta.NAME]),
                    #         list_of_submtrx_to_cat[-1].size()[0],
                    #     )
                    # )
            if list_of_submtrx_to_cat:
                g[Meta.EDGE_FEATURE_DICT][relation] = torch.cat(
                    list_of_submtrx_to_cat
                ).T
                if edge.get(Meta.REVERSE, None) and use_reverse_edges_features:
                    rev_relation = (
                        edge[Meta.DST_TYPE],
                        edge[Meta.REVERSE],
                        edge[Meta.SRC_TYPE],
                    )
                    g[Meta.EDGE_FEATURE_DICT][rev_relation] = g[Meta.EDGE_FEATURE_DICT][
                        relation
                    ].clone()
            # g[Meta.EDGE_SUBMATRIX_SHAPES_DICT][relation] = dict(submtrx_shapes)
            g = make_split(g[Meta.NUM_EDGES_DICT][relation], g, relation)
    g.edge_types = relation_types
    g.node_types = node_types
    return g


class DataObject:
    def __init__(
        self,
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        data_path: Optional[str] = None,
        target_extraction_policy=None,
        use_reverse_edges_features=False,
    ):
        self.data_path = data_path
        self.use_reverse_edges_features = use_reverse_edges_features
        self.construct_cache = {}
        self.metadata = load_metadata(data_path)
        # import pdb; pdb.set_trace()
        self.load_data(data_path)
        self.construct_cache["train_dataloader"] = train_dataloader
        self.construct_cache["valid_dataloader"] = valid_dataloader
        self.construct_cache["test_dataloader"] = test_dataloader
        self.build_train_dataloader_pre_dist(train_dataloader)
        self.build_valid_dataloader_pre_dist(valid_dataloader)
        self.build_test_dataloader_pre_dist(test_dataloader)
        self.target_extraction_policy = self.get_default_target_extraction_policy()

    def get_default_target_extraction_policy(self):
        raise NotImplementedError(
            f"{self} needs to implement `get_default_target_extraction_policy` "
            f"or you should pass `target_extraction_policy` explicitly"
        )

    def extract_output_target(self, batch, output, device):
        return self.target_extraction_policy(self, batch, output, device)

    def load_data(self, data_path):
        self.construct_cache["graph"] = load_graph(
            data_path,
            self.metadata,
            use_reverse_edges_features=self.use_reverse_edges_features,
        )

    def init_post_dist(self, device="cuda", use_ddp=False):
        self.device = device
        self.use_ddp = use_ddp
        self.build_train_dataloader_post_dist(device, use_ddp)
        self.build_valid_dataloader_post_dist(device, use_ddp)
        self.build_test_dataloader_post_dist(device, use_ddp)

    # For task specific dataloading, users should implement an inheritor class of DataObject
    def build_train_dataloader_pre_dist(self, dataloader_spec):
        self.train_dataloader = (dataloader_spec, self.construct_cache["graph"])

    def build_valid_dataloader_pre_dist(self, dataloader_spec):
        self.valid_dataloader = (dataloader_spec, self.construct_cache["graph"])

    def build_test_dataloader_pre_dist(self, dataloader_spec):
        self.test_dataloader = (dataloader_spec, self.construct_cache["graph"])

    def build_train_dataloader_post_dist(self, device, use_ddp):
        return

    def build_valid_dataloader_post_dist(self, device, use_ddp):
        return

    def build_test_dataloader_post_dist(self, device, use_ddp):
        return

    @property
    def train(self):
        return self.train_dataloader

    @property
    def valid(self):
        return self.valid_dataloader

    @property
    def test(self):
        return self.test_dataloader

    def get_labels(self, batch):
        raise NotImplementedError

    @property
    def get_metadata(self):
        return self.metadata

    @property
    def graph(self):
        return self.construct_cache.get("graph")



class DefaultNodeTargetExtractionPolicy:
    def __call__(self, data_object, batch, output, device):
        output, target = get_logit_labels_types_block(
            batch, output, data_object.node_label_map, device
        )
        return output, target

class NodeDataObject(DataObject):
    def __init__(
        self,
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        data_path: Optional["str"] = None,
        target_extraction_policy=None,
    ):
        super().__init__(
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            data_path,
            target_extraction_policy,
        )
        self.node_label_map = defaultdict(list)
        for node in self.metadata[MetadataKeys.NODES][MetadataKeys.NODE_TYPES]:
            node_features = node[MetadataKeys.FEAT]
            for feature in node_features:
                if feature.get(MetadataKeys.LABEL, False):
                    y = self.construct_cache["graph"][node[MetadataKeys.NAME]].y
                    # TODO: THIS ASSUMES THAT ONLY A SINGLE NODE TYPE HAS A LABEL
                    # (If there are multiple chooses the last by default)
                    self.label_sizes = y.max().item() + 1
                    self.label_tag = [
                        node[MetadataKeys.NAME],
                        feature[MetadataKeys.NAME],
                    ]

                    # Tracks all types
                    self.node_label_map[node[MetadataKeys.NAME]].append(
                        (feature[MetadataKeys.NAME], self.label_sizes)
                    )

    def get_labels(self, batch):
        return extract_block_one_hot_labels(
            batch, self.label_tag[1], self.label_sizes, self.label_tag[0]
        )

    def get_default_target_extraction_policy(self):
        return DefaultNodeTargetExtractionPolicy()

    def build_train_dataloader_pre_dist(self, dataloader_spec):
        train_ids = get_split_nids(
            self.construct_cache["graph"], self.metadata, split=0
        )
        self.construct_cache["train_ids"] = train_ids

    def build_train_dataloader_post_dist(self, device, use_ddp):
        train_ids = self.construct_cache["train_ids"]
        self.train_dataloader = make_pyg_loader(
            self.construct_cache["graph"],
            train_ids,
            self.metadata,
            device,
        )

    def build_valid_dataloader_pre_dist(self, dataloader_spec):
        valid_ids = get_split_nids(
            self.construct_cache["graph"], self.metadata, split=1
        )
        self.construct_cache["valid_ids"] = valid_ids

    def build_valid_dataloader_post_dist(self, device, use_ddp):
        # Build block sampler
        valid_ids = self.construct_cache["valid_ids"]
        self.valid_dataloader = make_pyg_loader(
            self.construct_cache["graph"],
            valid_ids,
            self.metadata,
            device,
        )

    def build_test_dataloader_pre_dist(self, dataloader_spec):
        test_ids = get_split_nids(self.construct_cache["graph"], self.metadata, split=2)
        self.construct_cache["test_ids"] = test_ids
        # Build block sampler

    def build_test_dataloader_post_dist(self, device, use_ddp):
        test_ids = self.construct_cache["test_ids"]
        self.test_dataloader = make_pyg_loader(
            self.construct_cache["graph"],
            test_ids,
            self.metadata,
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
        rel_names,
        *,
        activation=None,
        self_loop=False,
        dropout=0.0,
        conv_type=None,
    ):
        super().__init__()
        self.rel_names = rel_names
        self.num_relations = len(rel_names)
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
                {rel: self.conv_type(in_feat, out_feat) for rel in self.rel_names}
            )
        else:
            self.conv = HeteroConv(
                {
                    rel: self.conv_type((in_feat[rel[0]], in_feat[rel[-1]]), out_feat)
                    for rel in self.rel_names
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


def get_self_loop_dim(metadata, embedding=None):
    node_dims = {}
    for node_type in metadata[MetadataKeys.NODES][MetadataKeys.NODE_TYPES]:

        node_dim = 0
        for feature in node_type[MetadataKeys.FEAT]:
            if feature.get(MetadataKeys.LABEL, False):  # - skip label
                continue
            # - assumes last shape dim is feature dimension
            # - TODO: store cont/cat/discrete/mixed/ explicitely in metadata
            if feature[MetadataKeys.DTYPE] == "torch.FloatTensor":
                if type(feature[MetadataKeys.NAME]) == list:
                    node_dim += len(feature[MetadataKeys.NAME])
                else:
                    node_dim += feature[MetadataKeys.SHAPE][
                        -1
                    ]  # - treat last shape element as dim
            elif feature[MetadataKeys.DTYPE] == "torch.LongTensor":
                # - TODO: for now simply treat same as above
                if type(feature[MetadataKeys.NAME]) == list:
                    node_dim += len(feature[MetadataKeys.NAME])
                else:
                    node_dim += feature[MetadataKeys.SHAPE][
                        -1
                    ]  # - treat last shape element as dim
                continue
                # - TODO: add 'cat_dim' and 'cat_size' to metadata
                if hasattr(feature, "cat_dim"):
                    node_dim += feature["cat_dim"]
                else:
                    # - TODO: the default should never be hit,
                    # and this info should be stored in metadata
                    pass
            #      node_dim += cat_embedding_heuristic(feature.get('cat_size', 5))

            # - TODO: add support for mixed type/discrete
            # must first decide how we will store the feature type in metadata.

        node_dims[node_type[MetadataKeys.NAME]] = node_dim
        if embedding is not None:
            node_dims[node_type[MetadataKeys.NAME]] += embedding.get_embedding_size(
                node_type[MetadataKeys.NAME]
            )
    return node_dims


class HeteroModule(nn.Module):
    # conv_type : MessagePassing
    #     GNN conv desired. Default: None. Uses SAGEConv as the default.
    #     Examples: torch_geometric.nn.conv.GraphConv, torch_geometric.nn.conv.GATConv
    def __init__(
        self,
        metadata,
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
        self.metadata = metadata
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.aggregator_type = aggregator_type
        self.dropout = dropout
        self.rel_names = []

        for i in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
            self.rel_names.append((i[Meta.SRC_TYPE], i[Meta.NAME], i[Meta.DST_TYPE]))
            if i.get(Meta.REVERSE):
                self.rel_names.append(
                    (i[Meta.DST_TYPE], i[Meta.REVERSE], i[Meta.SRC_TYPE])
                )
        self.rel_names.sort()
        self.src_node_types = set([key[0] for key in self.rel_names])
        self.dst_node_types = set([key[-1] for key in self.rel_names])
        self.unupdated_nodes = list(set(self.src_node_types - self.dst_node_types))
        if len(self.unupdated_nodes) > 0:
            # If a node type does not get filled with message passing this will cause errors
            # Solve this by adding reverse edges:
            for i in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
                if not i.get(Meta.REVERSE):
                    self.rel_names.append(
                        (
                            i[Meta.DST_TYPE],
                            str("rev_") + str(i[Meta.NAME]),
                            i[Meta.SRC_TYPE],
                        )
                    )
        self.node_dims = get_self_loop_dim(metadata, embedding=embedding)
        self.layers.append(
            RelGraphConvLayer(
                self.node_dims,
                self.dim_hidden,
                self.rel_names,
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
                    self.rel_names,
                    activation=self.activation,
                    dropout=self.dropout,
                    conv_type=conv_type,
                )
            )

        self.layers.append(
            RelGraphConvLayer(
                self.dim_hidden, self.dim_out, self.rel_names, conv_type=conv_type
            )
        )

    def forward(self, batch):
        h = batch.collect("x")
        for i, layer in enumerate(self.layers):
            h = layer(batch, h)
        return h


class OGBN_MAG:
    """
    The OGBN_MAG class includes the transformation
    operation for a subset of the Microsoft Academic Graph (MAG).
    It's a heterogeneous network that contains four types of entities—papers
    (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes),
    and fields of study (59,965 nodes)—as well as four types of directed relations
    connecting two types of entities—an author is “affiliated with” an institution,
    an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic
    of” a field of study. For more information, please check
    https://ogb.stanford.edu/docs/nodeprop/

    Example usage::
    # Create an instance and call transform

    from gp.preprocessing.trans_dataset import OGBN_Products
    o = OGBN_MAG(source_path, destination_path)
    o.transform()

    Parameters
    ----------
    source_path: str
        source path for downloading the original data.
    dest_path: str
        destination path for putting the transformed data.
    """

    def __init__(self, source_path: str, dest_path: str):
        self.source_path = source_path
        self.dest_path = dest_path
        self.labels = None
        self.node_data = None
        self.edge_data = None
        self.metadata = None
        self.node_types = []
        self.edge_types = []

    def transform(self):
        """
        Transforms the OGBN Mag dataset to GP's data format.
        :return: None
        """
        import cudf

        # Download and prepare the OGBN MAG dataset using the
        # OGBN's  NodePropPredDatasetfunction.
        dataset = NodePropPredDataset(name="ogbn-mag", root=self.source_path)[0]
        feat_key = "paper"
        data = dataset[0]
        labels = torch.tensor(dataset[1][feat_key])
        # All the edge types have features. So, we get each edge type one by one.
        self.edge_data = {}
        for e, edges in data["edge_index_dict"].items():
            # Get the given edge type in the order of source to destination.
            # So, first column will have the source and the second one will have dest.
            # Third column has the ids of the edges.
            edata = torch.tensor(data["edge_reltype"][e])
            src_nodes = np.array(edges[0, :])
            dest_nodes = np.array(edges[1, :])
            edge_ids = np.array(torch.arange(edges.shape[1]))
            self.edge_data[e[1]] = cudf.DataFrame(
                {
                    MetadataKeys.EDGE_ID: edge_ids,
                    MetadataKeys.SRC_ID: src_nodes,
                    MetadataKeys.DST_ID: dest_nodes,
                }
            )
            self.edge_data[e[1]]["feat"] = cudf.DataFrame(np.array(edata))

            feature = {
                MetadataKeys.NAME: "feat",
                MetadataKeys.DTYPE: str(edata.type()),
                MetadataKeys.SHAPE: edata.size(),
            }
            # e[0] = source node type, e[1] = edge type, e[2] = destination node type
            edge_type = {
                MetadataKeys.NAME: e[1],
                MetadataKeys.FILES: ["./edge_data/edge_type_" + e[1] + ".parquet"],
                MetadataKeys.SRC_TYPE: e[0],
                MetadataKeys.DST_TYPE: e[2],
                MetadataKeys.FEAT: [feature],
            }

            self.edge_types.append(edge_type)

        # Only the node type 'paper' has features in this dataset.
        feat_val = torch.tensor(data["node_feat_dict"][feat_key])

        self.node_data = dict()
        # Get the 'paper' node type data and convert it to cudf
        self.node_data["paper"] = cudf.DataFrame(np.array(feat_val)).astype("float32")

        # Set a string column name so that parquet doesn't complain about it
        new_col_names = {i: "feat_" + str(i) for i in self.node_data["paper"].columns}
        feat_col_names = [val for val in new_col_names.values()]
        self.node_data["paper"] = self.node_data["paper"].rename(columns=new_col_names)

        # Another node level data in the graph is the 'year' info for the samples.
        # Putting the 'year info into the graph data as a feature.
        year_data = torch.tensor(data["node_year"][feat_key])
        self.node_data["paper"]["year"] = cudf.DataFrame(np.array(year_data)).astype(
            "int32"
        )
        # venue is the node labels for this dataset.
        self.node_data["paper"]["venue"] = cudf.DataFrame(np.array(labels)).astype(
            "int32"
        )

        # Split the data based on the year like what's described in OGBN's website.
        self.node_data["paper"]["split"] = cudf.Series(
            np.zeros(self.node_data["paper"]["venue"].size), dtype=np.int8
        )
        self.node_data["paper"].loc[
            self.node_data["paper"]["year"] == 2018, "split"
        ] = 1
        self.node_data["paper"].loc[self.node_data["paper"]["year"] > 2018, "split"] = 2
        self.node_data["paper"].drop(columns=["year"], inplace=True)

        # Calculate author, institution features.
        self.node_data["paper"]["paper_id"] = list(
            range(0, self.node_data["paper"].shape[0])
        )

        author_feat = (
            self.edge_data["writes"]
            .merge(
                self.node_data["paper"],
                left_on="dst_node_id",
                right_on="paper_id",
                how="left",
            )
            .groupby("src_node_id", sort=True)
            .mean()
        )

        new_feat_col_names = [val for val in new_col_names.values()]
        self.node_data["author"] = author_feat[new_feat_col_names].astype("float32")
        self.node_data["author"]["author_id"] = list(
            range(0, self.node_data["author"].shape[0])
        )
        self.node_data["author"]["split"] = cudf.Series(
            np.zeros(self.node_data["author"].shape[0]), dtype=np.int8
        )

        institution_feat = (
            self.edge_data["affiliated_with"]
            .merge(
                self.node_data["author"], left_on="src_node_id", right_on="author_id"
            )
            .groupby("dst_node_id", sort=True)
            .mean()
        )
        self.node_data["institution"] = institution_feat[new_feat_col_names].astype(
            "float32"
        )
        self.node_data["institution"]["split"] = cudf.Series(
            np.zeros(self.node_data["institution"].shape[0]), dtype=np.int8
        )

        field_of_study = (
            self.edge_data["has_topic"]
            .merge(self.node_data["paper"], left_on="src_node_id", right_on="paper_id")
            .groupby("dst_node_id", sort=True)
            .mean()
        )
        self.node_data["field_of_study"] = field_of_study[new_feat_col_names].astype(
            "float32"
        )
        self.node_data["field_of_study"]["split"] = cudf.Series(
            np.zeros(self.node_data["field_of_study"].shape[0]), dtype=np.int8
        )

        # Get the required info for the metadata file
        self.node_data["paper"].drop(columns=["paper_id"], inplace=True)
        features = dict()
        features["paper"] = list()
        features["paper"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: feat_val.size(),
            }
        )
        features["paper"].append(
            {
                MetadataKeys.NAME: "venue",
                MetadataKeys.DTYPE: str(labels.type()),
                MetadataKeys.SHAPE: labels.size(),
                MetadataKeys.LABEL: True,
            }
        )

        self.node_data["author"].drop(columns=["author_id"], inplace=True)
        features["author"] = list()
        features["author"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["author"].shape[0],
                    self.node_data["author"].shape[1] - 1,
                ],
            }
        )

        features["institution"] = list()
        features["institution"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["institution"].shape[0],
                    self.node_data["institution"].shape[1] - 1,
                ],
            }
        )

        features["field_of_study"] = list()
        features["field_of_study"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["field_of_study"].shape[0],
                    self.node_data["field_of_study"].shape[1] - 1,
                ],
            }
        )
        ntypes = data["num_nodes_dict"].keys()
        # There are multiple node types in this dataset.
        for i, ntype in enumerate(ntypes):
            # Only one node type has features. Once that is found, features can be added
            # into the metadata.
            if ntype in features:
                node_type = {
                    MetadataKeys.NAME: ntype,
                    MetadataKeys.FILES: ["./node_data/node_type_" + ntype + ".parquet"],
                    MetadataKeys.FEAT: features[ntype],
                }
            else:
                # Since these node types don't have any feature, nothing is added.
                node_type = {MetadataKeys.NAME: ntype, MetadataKeys.FEAT: []}

            self.node_types.append(node_type)

        # We create the metadata in the end.
        self.metadata = {
            MetadataKeys.NODES: {
                MetadataKeys.NODE_TYPES: self.node_types,
                MetadataKeys.SPLIT_NAME: "split",
            },
            MetadataKeys.EDGES: {MetadataKeys.EDGE_TYPES: self.edge_types},
        }

        self._write_to_files()

    def _write_to_files(self, exist_ok=True):
        os.makedirs(os.path.join(self.dest_path, "node_data/"), exist_ok=exist_ok)
        os.makedirs(os.path.join(self.dest_path, "edge_data/"), exist_ok=exist_ok)

        for i in range(len(self.node_types)):
            if self.node_types[i][MetadataKeys.NAME] in self.node_data.keys():
                self.node_data[self.node_types[i][MetadataKeys.NAME]].to_parquet(
                    os.path.join(
                        self.dest_path, self.node_types[i][MetadataKeys.FILES][0]
                    )
                )
        for i in range(len(self.edge_types)):
            if self.edge_types[i][MetadataKeys.NAME] in self.edge_data.keys():
                self.edge_data[self.edge_types[i][MetadataKeys.NAME]].to_parquet(
                    os.path.join(
                        self.dest_path, self.edge_types[i][MetadataKeys.FILES][0]
                    )
                )

        with open(os.path.join(self.dest_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)


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


def run_valid(rank, trainer, child_conn):
    data = trainer.valid_process(rank)
    if rank == 0:
        child_conn.send(data)
        child_conn.close()


def run_test(rank, trainer, child_conn):
    data = trainer.test_process(rank)
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


def merge_fit_process_metrics(
    train_metrics, valid_metrics, test_metrics, callback_metrics
):
    final_metrics = {}

    for k, v in train_metrics.items():
        final_metrics["train_{}".format(k)] = v
    for k, v in valid_metrics.items():
        final_metrics["valid_{}".format(k)] = v
    for k, v in test_metrics.items():
        final_metrics["test_{}".format(k)] = v
    for k, v in callback_metrics.items():
        final_metrics[k] = v

    return final_metrics


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

    def valid(self):
        """
        Runs an evaluation on the validation dataset, returns the metrics
        """
        if self.n_gpus > 1:
            parent_conn, child_conn = mp.Pipe()
            pc = mp.spawn(
                run_valid,
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
            return metrics
        else:
            return self.valid_process()

    def test(self):
        """
        Runs an evaluation on the test dataset, returns the metrics
        """
        if self.n_gpus > 1:
            parent_conn, child_conn = mp.Pipe()
            pc = mp.spawn(
                run_test,
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
            return metrics
        else:
            return self.test_process(0)

    def train_procedure(self, callback_metrics):
        self.do_train_epoch(callback_metrics)
        if self.rank == 0:
            print("TRAIN METRICS")
            print(self.logged_train_metrics)

    def valid_procedure(self, callback_metrics):
        self.do_valid_epoch(callback_metrics)
        non_tensor_metrics = metrics_values_to_list(self.logged_valid_metrics)
        if self.rank == 0:
            print("VALID METRICS")
            print(non_tensor_metrics)
        self.logger.log(
            step=self.global_step,
            data=non_tensor_metrics,
            verbosity=dllogger.Verbosity.VERBOSE,
        )

    def test_procedure(self, callback_metrics):
        self.do_test_epoch(callback_metrics)
        non_tensor_metrics = metrics_values_to_list(self.logged_test_metrics)
        if self.rank == 0:
            print("TEST METRICS")
            print(non_tensor_metrics)
        self.logger.log(
            step=self.global_step,
            data=non_tensor_metrics,
            verbosity=dllogger.Verbosity.VERBOSE,
        )

    def fit_process(self, rank=0):
        self.setup(rank)
        callback_metrics = {}
        self.on_train_begin(callback_metrics)

        while self.epoch < self.epochs:
            self.on_epoch_begin(callback_metrics)
            self.train_procedure(callback_metrics)

            if (self.epoch + 1) % self.eval_interval == 0:
                self.valid_procedure(callback_metrics)

            self.on_epoch_end(callback_metrics)

        self.on_train_end(callback_metrics)
        self.test_procedure(callback_metrics)

        if self.n_gpus > 1:
            callback_metrics_ = reduce_metrics(callback_metrics, self.n_gpus)
            for k, v in callback_metrics_.items():
                callback_metrics[k] = v

        merged_metrics = merge_fit_process_metrics(
            self.logged_train_metrics,
            self.logged_valid_metrics,
            self.logged_test_metrics,
            callback_metrics,
        )

        return metrics_values_to_list(merged_metrics)

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
        self.data_object.build_valid_dataloader_post_dist(device, use_ddp)
        self.data_object.build_test_dataloader_post_dist(device, use_ddp)
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

            output, target = self.data_object.extract_output_target(
                batch, output, self.device
            )

            loss = self.criterion(output, target)

            return output, target, loss

    def step(self, batch, mode="train"):
        metrics = {}
        if mode == "train":
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

        else:
            with torch.no_grad():
                output, target, loss = self.forward_pass(batch)

                for k, v in self.metrics.items():
                    metrics[k] = v(output, target)

            metrics["loss"] = loss

        return metrics

    def valid_process(self, rank=0):
        self.setup(rank)
        callback_metrics = {}
        self.do_valid_epoch(callback_metrics)
        metrics = self.logged_valid_metrics
        for k, v in callback_metrics.items():
            metrics[k] = v

        return metrics_values_to_list(metrics)

    def test_process(self, rank=0):
        self.setup(rank)
        callback_metrics = {}
        self.do_test_epoch(callback_metrics)
        metrics = self.logged_test_metrics
        for k, v in callback_metrics.items():
            metrics[k] = v

        return metrics_values_to_list(metrics)

    def do_valid_epoch(self, callback_metrics):
        self.on_valid_begin(callback_metrics)
        data = {}
        valid_step = 0
        for step, batch in enumerate(self.data_object.valid_dataloader):
            valid_step = step + 1
            metrics = self.step(batch, mode="valid")
            for k, v in metrics.items():
                data[k] = data.get(k, 0.0) + v

            if self.limit_batches is not None and step + 1 >= self.limit_batches:
                break

        self.valid_steps = valid_step

        for k, v in data.items():
            data[k] = v / len(self.data_object.valid_dataloader)

        if self.n_gpus > 1:
            data = reduce_metrics(data, self.n_gpus, average=True)
            callback_metrics_ = reduce_metrics(callback_metrics, self.n_gpus)
            for k, v in callback_metrics_.items():
                callback_metrics[k] = v

        data = {k: v for k, v in data.items()}
        self.logged_valid_metrics = data

        self.on_valid_end(callback_metrics)

    def do_test_epoch(self, callback_metrics):
        self.on_test_begin(callback_metrics)
        metrics = {}
        test_step = 0
        for step, batch in enumerate(self.data_object.test_dataloader):
            test_step = step + 1
            step_metrics = self.step(batch, mode="valid")

            for k, v in step_metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v

            if self.limit_batches is not None and step + 1 >= self.limit_batches:
                break

        self.test_steps = test_step

        for k, v in metrics.items():
            metrics[k] = v / len(self.data_object.test_dataloader)

        if self.n_gpus > 1:
            metrics = reduce_metrics(metrics, self.n_gpus, average=True)
            callback_metrics_ = reduce_metrics(callback_metrics, self.n_gpus)
            for k, v in callback_metrics_.items():
                callback_metrics[k] = v

        data = {k: v for k, v in metrics.items()}
        self.logged_test_metrics = data

        self.on_test_end(callback_metrics)

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

    def on_valid_begin(self, metrics):
        self.model.eval()
        self.criterion.eval()

        if self.n_gpus > 1:
            torch.distributed.barrier()


    def on_valid_end(self, metrics):
        if self.n_gpus > 1:
            torch.distributed.barrier()


    def on_test_begin(self, metrics):
        self.model.eval()
        self.criterion.eval()

        if self.n_gpus > 1:
            torch.distributed.barrier()


    def on_test_end(self, metrics):
        if self.n_gpus > 1:
            torch.distributed.barrier()


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
    source_path = os.path.join(DATA_DIR, "ogbn/mag/")
    destination_path = os.path.join(DATA_DIR, "ogbn/mag/GP_Transformed/")

    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    prep = OGBN_MAG(source_path, destination_path)
    prep.transform()

    data_object = NodeDataObject(
        data_path=destination_path,
    )
    metadata = data_object.metadata
    model = HeteroModule(
        metadata=metadata, dim_hidden=400, dim_out=349, n_layers=1, conv_type=conv_type
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
    shutil.rmtree(destination_path)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_mag_workflow(1)
    print("1 GPU TEST PASSED!!!!")
    print()
    print()
    test_mag_workflow(2)
