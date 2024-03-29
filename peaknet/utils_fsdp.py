import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nccl as nccl
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    BackwardPrefetch,
)

from pkg_resources import packaging

from dataclasses import dataclass, asdict
from typing import Optional

#####################
# Memory tool
#####################
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


#####################
# bfloat16 support
#####################

# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)


#####################
# Add dict broadcast
#####################
def broadcast_dict(obj, src=0, device = 'cpu'):
    rank = dist.get_rank()
    if rank == src:
        # Serialize the dictionary...
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer), device = device)

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


#####################
# checkpoint
#####################
# Refer to `$DATA_NOTES/pytorch/fsdp.checkpoint.md`

@dataclass
class TrainingStateDictConfig:
    current_epoch      : int
    current_mini_batch : int
    current_micro_batch: int
    loss_min           : float

@dataclass
class FullStateDictCheckpointConfig:
    model          : Optional[nn.Module]    # A FSDP wrapped model
    optimizer      : Optional[torch.optim.Optimizer]
    lr_scheduler   : Optional[torch.optim.lr_scheduler._LRScheduler]
    training_state : TrainingStateDictConfig
    rank           : int
    device         : str
    path_checkpoint: Optional[str]

class FullStateDictCheckpoint:
    def __init__(self, config):
        self.config = config
        self.full_state_dict = None


    def prepare_model_full_state_dict(self):
        model = self.config.model

        state_dict = None

        # Configure full state dict saving...
        full_state_saving_policy = FullStateDictConfig(
            offload_to_cpu = True,
            rank0_only     = True,
        )

        # Pull full state dict from the sharded model...
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            full_state_saving_policy
        ):
            state_dict = model.state_dict()

        return state_dict


    def prepare_optim_full_state_dict(self):
        print(f"Preparing optim full state dict...")
        model     = self.config.model
        optimizer = self.config.optimizer

        state_dict = None
        if optimizer is not None:
            state_dict = FSDP.full_optim_state_dict(model, optimizer)

        return state_dict


    def prepare_lr_scheduler_full_state_dict_by_rank0(self):
        rank = self.config.rank
        lr_scheduler_state_dict = None

        if rank == 0:
            lr_scheduler = self.config.lr_scheduler
            lr_scheduler_state_dict = lr_scheduler.state_dict()

        return lr_scheduler_state_dict


    def prepare_training_state_dict_by_rank0(self):
        rank = self.config.rank
        training_state = None
        if rank == 0:
            training_state = self.config.training_state
            training_state = asdict(training_state)

        return training_state


    def save_full_state_dict(self):
        dist.barrier()

        rank = self.config.rank

        model_full_state_dict   = self.prepare_model_full_state_dict()
        optim_full_state_dict   = self.prepare_optim_full_state_dict()
        lr_scheduler_state_dict = self.prepare_lr_scheduler_full_state_dict_by_rank0()
        training_state_dict     = self.prepare_training_state_dict_by_rank0()

        if rank == 0:
            path_checkpoint = self.config.path_checkpoint
            full_state_dict = {
                'model_state_dict'     : model_full_state_dict,
                'optimizer_state_dict' : optim_full_state_dict,
                'scheduler_state_dict' : lr_scheduler_state_dict,
                'training_state_dict'  : training_state_dict,
            }
            torch.save(full_state_dict, path_checkpoint)

        dist.barrier()


    def load_model_full_state_dict(self):
        """
        Must run before FSDP wrapper.
        """
        rank            = self.config.rank
        path_checkpoint = self.config.path_checkpoint
        model           = self.config.model

        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            model_full_state_dict = self.full_state_dict.get('model_state_dict')
            model.load_state_dict(model_full_state_dict)


    def load_optim_full_state_dict(self):
        """
        Must run after FSDP wrapper.
        """
        rank      = self.config.rank
        optimizer = self.config.optimizer
        model     = self.model

        optim_full_state_dict = None
        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            optim_full_state_dict = self.full_state_dict.get('optimizer_state_dict')

        # Scatter the optimizer state to all ranks...
        sharded_optim_state_dict = FSDP.scatter_full_optim_state_dict(optim_full_state_dict, model)
        optimizer.load_state_dict(sharded_optim_state_dict)


    def load_training_state_dict(self):
        rank           = self.config.rank
        device         = self.config.device
        training_state = self.config.training_state

        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            training_state = self.full_state_dict.get('training_state_dict')

        # Scatter the training state to all ranks...
        training_state = broadcast_dict(training_state, src = 0, device = device)

        self.config.training_state = training_state
