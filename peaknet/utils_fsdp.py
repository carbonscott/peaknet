# -- Imports for basic PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nccl as nccl
import torch.distributed as dist

# -- Imports for printing debug messages
import colorama
colorama.init(autoreset=True)

# -- Imports for FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# -- Imports for saving sharded state dict
# Use old APIs
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

from torch.distributed.fsdp.api import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)

# -- Imports for understanding package versions
from pkg_resources import packaging
from packaging import version

# -- Imports for dataclasses
from dataclasses import dataclass, asdict
from typing import Optional, Dict

# -- Rest
import pickle
import os
from datetime import datetime

# -- Patch PyTorch
from .patches.build_metadata import patch_build_metadata

# -- Logging
import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  MEMORY TOOL
# ----------------------------------------------------------------------- #
# This code is adapted from https://github.com/carbonscott/pytorch-fsdp-transformers/blob/main/performance/gpu_memory.py
#
# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# Summary:
# the utility class Memory_Maximizer tracks reserved per epoch or per minibatch reserved GPU memory, in GB and as % of GPU VRAM,
# and most importantly programmatically confirms if any cudaMalloc retries took place.

# cudaMalloc retries can significantly lower performance (likely due to resetting the cache), but are otherwise
# not normally visible as an actual 'error' the way OOM is.

# usage - create instance,
# start() to reset internal stats, and begin,
# update() at end of epoch or minibatch,
# stop() to stop and print details.

# adjust batch size until you no longer see any cudaMalloc retries for best performance/memory maximization.

"""
example usage:

from peaknet.utils_performance import Memory_Maximizer

if rank == 0:
        memmax = Memory_Maximizer()

# memory and timing tracking
    if local_rank == 0:
        memmax.start()  # start will reset all tracking points

# in training loop - at minibatch or epoch end point:
    # update durations and memory tracking
    if local_rank == 0:
        memmax.update()

# at end of training - stop and print stats
    # memory summary
    if local_rank == 0:
        memmax.stop()  # stop and display info  
"""

gigabyte_size = 1073741824
megabyte_size = 1048576


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


class MemoryMaximizer:
    def __init__(
        self,
    ):

        current_free, full_gpu_mem = torch.cuda.mem_get_info()

        self.m_total_gpu_memory = format_to_gb(full_gpu_mem)

        print(f"--> total memory per gpu (GB) = {self.m_total_gpu_memory}")

        self.m_reserved_memory_list = []
        self.m_reserved_memory_pct = []
        self.m_allocated_memory_list = []
        self.m_allocated_memory_pct = []
        self.m_active_memory_list = []
        self.m_active_memory_pct = []

        self.m_total_ooms = 0
        self.m_num_retries = 0
        self.m_max_reserved = 0
        self.m_max_allocated = 0
        self.m_max_active = 0

    def _convert_to_gpu_pct(self, value):
        return round(100 * (value / self.m_total_gpu_memory), 2)

    def start(self):
        """start memory tracking, reset any current info"""

        torch.cuda.reset_peak_memory_stats()
        self.m_reserved_memory_list = []
        self.m_reserved_memory_pct = []
        self.m_allocated_memory_list = []
        self.m_allocated_memory_pct = []
        self.m_active_memory_list = []
        self.m_active_memory_pct = []

        self.m_total_ooms = 0
        self.m_num_retries = 0
        self.m_max_reserved = 0
        self.m_max_allocated = 0
        self.m_max_active = 0

        print(f"memory stats reset, ready to track")

    def update(
        self,
    ):
        """update reserved memory for this epoch"""
        updated_reserved = format_to_gb(torch.cuda.memory_reserved())
        updated_allocated = format_to_gb(torch.cuda.memory_allocated())

        self.m_reserved_memory_list.append(updated_reserved)
        self.m_reserved_memory_pct.append(self._convert_to_gpu_pct(updated_reserved))

        self.m_allocated_memory_list.append(updated_allocated)
        self.m_allocated_memory_pct.append(self._convert_to_gpu_pct(updated_allocated))

    def stop(
        self,
        verbose=False,
    ):
        """end of training...get various stats and display"""

        if verbose:
            print(f"\nreserved memory = {self.m_reserved_memory_list}")
            print(f"memory % = {self.m_reserved_memory_pct}\n")
            print(f"allocated memory = {self.m_allocated_memory_list}")
            print(f"allocated memory % = {self.m_allocated_memory_pct}")

        cuda_max_reserved = format_to_gb(torch.cuda.max_memory_reserved())
        print(f"\n--> cuda max reserved memory = {cuda_max_reserved}")
        res_percentage = self._convert_to_gpu_pct(cuda_max_reserved)

        print(f"--> max reserved percentage = {round(res_percentage,4)} %\n")

        cuda_max_allocated = format_to_gb(torch.cuda.max_memory_allocated())
        print(f"--> cuda max memory allocated = {cuda_max_allocated}")
        alloc_percentage = self._convert_to_gpu_pct(cuda_max_allocated)
        print(f"--> max allocated percentage = {alloc_percentage} %\n")

        cuda_info = torch.cuda.memory_stats()

        active_peak = cuda_info.get("active_bytes.all.peak", 0)
        active_peak_memory_gb = format_to_gb(active_peak)

        self.m_num_retries = cuda_info.get("num_alloc_retries", 0)
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)

        print(f"--> peak active memory = {active_peak_memory_gb}")
        print(
            f"--> peak active memory {self._convert_to_gpu_pct(active_peak_memory_gb)} %\n"
        )

        print(f"cudaMalloc retries = {self.m_num_retries}")
        print(f"cuda OOM = {self.m_cuda_ooms}\n")
        if self.m_num_retries > 0:
            print(
                f"--> Recommend decreasing batch size...cuda retries can greatly degrade perf!"
            )

    def summary(
        self,
    ):
        pass


# ----------------------------------------------------------------------- #
#  BFLOAT16 SUPPORT
# ----------------------------------------------------------------------- #
# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

verify_bfloat_support = (
    torch.cuda.is_available()
    and torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)


# ----------------------------------------------------------------------- #
#  BROADCAST DICTIONARY
# ----------------------------------------------------------------------- #
def broadcast_dict(obj, src=0, device = 'cpu'):
    rank = dist.get_rank()
    if rank == src:
        # Serialize the dictionary...
        buffer = pickle.dumps(obj)
        ## tensor = torch.ByteTensor(list(buffer), device = device)
        tensor = torch.tensor(list(buffer), dtype=torch.uint8, device=device)

        # Communicate about the size of the underlying data...
        tensor_size = torch.tensor([len(buffer)], dtype=torch.long, device = device)
    else:
        # Prepare to receive the size of the underlying data...
        tensor_size = torch.tensor([0], dtype=torch.long, device = device)

    # Broadcast the size of the tensor to all processes...
    dist.broadcast(tensor_size, src)

    # Prepare to receive data...
    if rank != src:
        tensor = torch.empty((tensor_size.item(),), dtype=torch.uint8, device = device)

    # Broadcast the data...
    dist.broadcast(tensor, src)

    if rank != src:
        # Deserialize the tensor back into a dictionary...
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)

    return obj


# ----------------------------------------------------------------------- #
#  CHECKPOINT
# ----------------------------------------------------------------------- #
# -- 1. FULL STATE DICT
class FullStateDictCheckpoint:
    MODEL_STATE_DICT_FILE = 'model_full_state_dict.pt'
    OPTIM_STATE_DICT_FILE = 'optim_full_state_dict.pt'
    LR_STATE_DICT_FILE    = 'lr_full_state_dict.pt'
    ITER_STATE_DICT_FILE  = 'iter_full_state_dict.pt'

    def __init__(self):
        self.state_dict_config = FullStateDictConfig(
            offload_to_cpu = True,
            rank0_only     = True,
        )
        self.optim_dict_config = FullOptimStateDictConfig(
            offload_to_cpu = True,
            rank0_only     = True,
        )

    def save_model_checkpoint(self, rank, model, path_checkpoint_model):
        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        # Pull full state dict from the sharded model...
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        ):
            model_state_dict = model.state_dict()

            if rank == 0:
                torch.save(model_state_dict, path_checkpoint_model)

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        """
        Must run before FSDP wrapper.
        """
        dist.barrier()

        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        model_state_dict = torch.load(path_checkpoint_model)

        # [NOTE] Context manager will throw errors
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        )
        model.load_state_dict(model_state_dict)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist.barrier()

        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        full_optim_state_dict = None

        if rank == 0 or not optim_dict_config.rank0_only:
            full_optim_state_dict = torch.load(path_checkpoint_optim)

        sharded_optim_state_dict = FSDP.scatter_full_optim_state_dict(
            full_optim_state_dict = full_optim_state_dict,
            model = model,
        )
        optimizer.load_state_dict(sharded_optim_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        torch_version = torch.__version__
        torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
        if version.parse(torch_version) <= version.parse("2.0.1"):
            optim_state_dict = FSDP.full_optim_state_dict(model, optimizer)
        else:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config       = state_dict_config,
                optim_state_dict_config = optim_dict_config,
            ):
                optim_state_dict = FSDP.optim_state_dict(model, optimizer)

        if rank == 0:
            torch.save(optim_state_dict, path_checkpoint_optim)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist.barrier()

        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        torch_version = torch.__version__
        torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
        if version.parse(torch_version) <= version.parse("2.0.1"):
            full_optim_state_dict = None

            if rank == 0 or not optim_dict_config.rank0_only:
                full_optim_state_dict = torch.load(path_checkpoint_optim)

            sharded_optim_state_dict = FSDP.scatter_full_optim_state_dict(
                full_optim_state_dict = full_optim_state_dict,
                model = model,
            )
            optimizer.load_state_dict(sharded_optim_state_dict)
        else:
            # [NOTE] Context manager will throw errors
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config       = state_dict_config,
                optim_state_dict_config = optim_dict_config,
            )
            full_optim_state_dict = None

            if rank == 0 or not optim_dict_config.rank0_only:
                full_optim_state_dict = torch.load(path_checkpoint_optim)

            flattened_optim_state_dict = FSDP.optim_state_dict_to_load(
                model = model,
                optim = optimizer,
                optim_state_dict = full_optim_state_dict,
            )
            optimizer.load_state_dict(flattened_optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            lr_state_dict = lr_scheduler.state_dict()
            torch.save(lr_state_dict, path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            lr_state_dict = torch.load(path_checkpoint_lr, map_location = 'cpu')
            object_list = [lr_state_dict, ]

        dist.broadcast_object_list(object_list, src = 0)
        lr_state_dict = object_list[0]
        lr_scheduler.load_state_dict(lr_state_dict)

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            iter_state_saved = torch.load(path_checkpoint_iter_state, map_location = 'cpu')
            object_list = [iter_state_saved, ]

        dist.broadcast_object_list(object_list, src = 0)
        iter_state_saved = object_list[0]
        iter_state = iter_state_saved

    def save(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        os.makedirs(path_checkpoint, exist_ok = True)
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE)
        path_checkpoint_lr         = os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE)
        path_checkpoint_iter_state = os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE)

        if model is not None:
            self.save_model_checkpoint(rank, model, path_checkpoint_model)

        if optimizer is not None:
            self.save_optimizer_checkpoint(rank, model, optimizer, path_checkpoint_optim)

        if lr_scheduler is not None:
            self.save_lr_checkpoint(rank, lr_scheduler, path_checkpoint_lr)

        if iter_state is not None:
            self.save_iter_state_checkpoint(rank, iter_state, path_checkpoint_iter_state)

    def pre_fsdp_load(self, rank, model, path_checkpoint):
        """
        Only the model needs to be loaded pre FSDP wrapper.
        """
        path_checkpoint_model = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        self.load_model_checkpoint(rank, model, path_checkpoint_model)

        dist.barrier()

    def post_fsdp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        """
        Users have to pass in the current model, optimizer, lr_scheduler and
        training state so that the checkpointer has the best knowledge of the
        FSDP stages.
        """
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE)
        path_checkpoint_lr         = os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE)
        path_checkpoint_iter_state = os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE)

        if optimizer is not None:
            self.load_optimizer_checkpoint(rank, model, optimizer, path_checkpoint_optim)

        if lr_scheduler is not None:
            self.load_lr_checkpoint(rank, lr_scheduler, path_checkpoint_lr)

        if iter_state is not None:
            self.load_iter_state_checkpoint(rank, iter_state, path_checkpoint_iter_state)

        dist.barrier()

    def load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE)
        path_checkpoint_lr         = os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE)
        path_checkpoint_iter_state = os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE)

        if model is not None:
            self.load_model_checkpoint(rank, model, path_checkpoint_model)

        if optimizer is not None:
            self.load_optimizer_checkpoint(rank, model, optimizer, path_checkpoint_optim)

        if lr_scheduler is not None:
            self.load_lr_checkpoint(rank, lr_scheduler, path_checkpoint_lr)

        if iter_state is not None:
            self.load_iter_state_checkpoint(rank, iter_state, path_checkpoint_iter_state)


# -- 2. SHARDED STATE DICT
class ShardedStateDictCheckpoint:
    MODEL_STATE_DICT_FILE = 'model_full_state_dict.pt'
    OPTIM_STATE_DICT_FILE = 'optim_full_state_dict.pt'
    LR_STATE_DICT_FILE    = 'lr_full_state_dict.pt'
    ITER_STATE_DICT_FILE  = 'iter_full_state_dict.pt'

    def __init__(self):
        self.state_dict_config = ShardedStateDictConfig(
            offload_to_cpu = True,
        )
        self.optim_dict_config = ShardedOptimStateDictConfig(
            offload_to_cpu = True,
        )

    def save_model_checkpoint(self, rank, model, path_checkpoint_model):
        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        dist_writer = FileSystemWriter(path_checkpoint_model)
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        )
        model_state_dict = model.state_dict()
        state_dict_to_save = {"model": model_state_dict}  # FSDP writer requires it.

        save_state_dict(
            state_dict     = state_dict_to_save,
            storage_writer = dist_writer,
            planner        = DefaultSavePlanner(),
        )

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        """
        Must run before FSDP wrapper.
        """
        dist.barrier()

        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        dist_reader = FileSystemReader(path_checkpoint_model)
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        )
        model_state_dict = model.state_dict()
        state_dict_to_load = {"model": model_state_dict}  # FSDP reader requires it.

        load_state_dict(
            state_dict     = state_dict_to_load,
            storage_reader = dist_reader,
            planner        = DefaultLoadPlanner(),
        )
        model_state_dict = state_dict_to_load.get("model")
        model.load_state_dict(model_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        dist_writer = FileSystemWriter(path_checkpoint_optim)
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        )
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        state_dict_to_save = {"optim": optim_state_dict}  # FSDP writer requires it.

        save_state_dict(
            state_dict     = state_dict_to_save,
            storage_writer = dist_writer,
            planner        = DefaultSavePlanner(),
        )


    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist.barrier()

        state_dict_config = self.state_dict_config
        optim_dict_config = self.optim_dict_config

        dist_reader = FileSystemReader(path_checkpoint_optim)
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config       = state_dict_config,
            optim_state_dict_config = optim_dict_config,
        )
        model_state_dict = model.state_dict()
        state_dict_to_load = load_sharded_optimizer_state_dict(
            model_state_dict = model_state_dict,
            optimizer_key    = 'optim',
            storage_reader   = dist_reader,
        )
        optim_state_dict = state_dict_to_load.get("optim")

        flattened_optim_state_dict = FSDP.optim_state_dict_to_load(
            model = model,
            optim = optimizer,
            optim_state_dict = optim_state_dict,
        )
        optimizer.load_state_dict(flattened_optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            lr_state_dict = lr_scheduler.state_dict()
            torch.save(lr_state_dict, path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            lr_state_dict = torch.load(path_checkpoint_lr, map_location = 'cpu')
            object_list = [lr_state_dict, ]

        dist.broadcast_object_list(object_list, src = 0)
        lr_state_dict = object_list[0]
        lr_scheduler.load_state_dict(lr_state_dict)

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            iter_state_saved = torch.load(path_checkpoint_iter_state, map_location = 'cpu')
            object_list = [iter_state_saved, ]

        dist.broadcast_object_list(object_list, src = 0)
        iter_state_saved = object_list[0]
        iter_state = iter_state_saved

    def save(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        os.makedirs(path_checkpoint, exist_ok = True)
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE)
        path_checkpoint_lr         = os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE)
        path_checkpoint_iter_state = os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE)

        if model is not None:
            self.save_model_checkpoint(rank, model, path_checkpoint_model)

        if optimizer is not None:
            self.save_optimizer_checkpoint(rank, model, optimizer, path_checkpoint_optim)

        if lr_scheduler is not None:
            self.save_lr_checkpoint(rank, lr_scheduler, path_checkpoint_lr)

        if iter_state is not None:
            self.save_iter_state_checkpoint(rank, iter_state, path_checkpoint_iter_state)

    def load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE)
        path_checkpoint_lr         = os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE)
        path_checkpoint_iter_state = os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE)

        if model is not None:
            self.load_model_checkpoint(rank, model, path_checkpoint_model)

        if optimizer is not None:
            self.load_optimizer_checkpoint(rank, model, optimizer, path_checkpoint_optim)

        if lr_scheduler is not None:
            self.load_lr_checkpoint(rank, lr_scheduler, path_checkpoint_lr)

        if iter_state is not None:
            self.load_iter_state_checkpoint(rank, iter_state, path_checkpoint_iter_state)

    def pre_fsdp_load(self, rank, model, path_checkpoint):
        pass

    def post_fsdp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        self.load(rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint)

# ----------------------------------------------------------------------- #
#  Logger
# ----------------------------------------------------------------------- #
def init_logger(uses_dist, dist_rank, device, fl_prefix = None, drc_log = "logs", level = 'info'):
    timestamp = None

    if dist_rank == 0:
        # Create a timestamp to name the log file...
        now = datetime.now()
        timestamp = now.strftime("%Y_%m%d_%H%M_%S")

    if uses_dist:
        timestamp = broadcast_dict(dict(timestamp=timestamp), src = 0, device = device).get('timestamp')

    # Set up the log file...
    # ...base directory
    base_log = f"{timestamp}"
    if fl_prefix is not None: base_log = f"{fl_prefix}.{base_log}"
    path_log = os.path.join(drc_log, base_log)

    # ...path
    os.makedirs(path_log, exist_ok = True)
    fl_log = f"rank{dist_rank}.log"
    path_log = os.path.join(path_log, fl_log)

    # Config logging behaviors
    log_level_spec = {
        "info"  : logging.INFO,
        "debug" : logging.DEBUG,
    }
    log_level = log_level_spec.get(level, logging.INFO)
    logging.basicConfig( filename = path_log,
                         filemode = 'w',
                         format="%(asctime)s %(levelname)s %(name)s\n%(message)s",
                         datefmt="%m/%d/%Y %H:%M:%S",
                         level=log_level, )
    logger = logging.getLogger(__name__)

    return timestamp
