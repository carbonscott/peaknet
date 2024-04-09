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

# -- Imports for saving full state dict
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# -- Imports for saving sharded state dict
try:
    # Use new API
    from torch.distributed.checkpoint import (
        FileSystemReader,
        FileSystemWriter,
        load,
        save,
        DefaultLoadPlanner,
        DefaultSavePlanner,
        load_sharded_optimizer_state_dict,
    )
except ImportError:
    # Fall back to the old API
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

    import warnings
    warnings.warn("Using older version of the FSDP checkpoint API.", DeprecationWarning)


# -- Imports for understanding package versions
from pkg_resources import packaging

# -- Imports for dataclasses
from dataclasses import dataclass, asdict
from typing import Optional

# -- Rest
import pickle
import os

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
@dataclass
class TrainingStateDictConfig:
    epoch      : int
    start_idx  : int
    seg_size   : int
    loss_min   : float


# -- 1. FULL STATE DICT
@dataclass
class FullStateDictCheckpointConfig:
    model          : Optional[nn.Module]    # A FSDP wrapped model on all ranks
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


    @staticmethod
    def contains_fsdp(module):
        return hasattr(module, 'module')


    def _prepare_model_full_state_dict(self):
        model = self.config.model    # A FSDP wrapped model on all ranks

        # Sanity check if the model is wrapped with FSDP
        if not FullStateDictCheckpoint.contains_fsdp(model):
            raise ValueError(f"RANK {self.config.rank} - The model subject to "  \
            "checkpointing must be wrapped with an FSDP wrapper before saving a "\
            "full state dict.")

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


    def _prepare_optim_full_state_dict(self):
        model     = self.config.model
        optimizer = self.config.optimizer

        # Sanity check if the model is wrapped with FSDP
        if not FullStateDictCheckpoint.contains_fsdp(model):
            raise ValueError(f"RANK {self.config.rank} - The model subject to "  \
            "checkpointing must be wrapped with an FSDP wrapper before saving a "\
            "full state dict.")

        state_dict = None
        if optimizer is not None:
            state_dict = FSDP.full_optim_state_dict(model, optimizer)

        return state_dict


    def _prepare_lr_scheduler_state_dict_by_rank0(self):
        rank = self.config.rank
        lr_scheduler_state_dict = None

        if rank == 0:
            lr_scheduler = self.config.lr_scheduler
            lr_scheduler_state_dict = lr_scheduler.state_dict()

        return lr_scheduler_state_dict


    def _prepare_training_state_dict_by_rank0(self):
        rank = self.config.rank
        training_state = None
        if rank == 0:
            training_state = self.config.training_state
            training_state = asdict(training_state)

        return training_state


    def _load_model_full_state_dict(self):
        """
        Must run before FSDP wrapper.
        """
        rank            = self.config.rank
        path_checkpoint = self.config.path_checkpoint
        model           = self.config.model

        # Sanity check if the model is wrapped with FSDP
        if FullStateDictCheckpoint.contains_fsdp(model):
            raise ValueError(f"RANK {self.config.rank} - The model subject to "  \
            "checkpointing with full state dict must be resumed before the FSDP "\
            "wrapper.")

        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            model_full_state_dict = self.full_state_dict.get('model_state_dict')
            model.load_state_dict(model_full_state_dict)


    def _load_optim_full_state_dict(self):
        """
        Must run after FSDP wrapper.
        """
        rank      = self.config.rank
        optimizer = self.config.optimizer
        model     = self.config.model

        optim_full_state_dict = None
        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            optim_full_state_dict = self.full_state_dict.get('optimizer_state_dict')

        # Scatter the optimizer state to all ranks...
        sharded_optim_state_dict = FSDP.scatter_full_optim_state_dict(optim_full_state_dict, model)
        optimizer.load_state_dict(sharded_optim_state_dict)


    def _load_training_state_dict(self):
        rank            = self.config.rank
        device          = self.config.device
        training_state  = self.config.training_state
        path_checkpoint = self.config.path_checkpoint

        training_state = None
        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            training_state = self.full_state_dict.get('training_state_dict')

        # Scatter the training state to all ranks...
        training_state = broadcast_dict(training_state, src = 0, device = device)

        self.config.training_state = TrainingStateDictConfig(**training_state)


    def _load_lr_scheduler_state_dict(self):
        rank         = self.config.rank
        device       = self.config.device
        lr_scheduler = self.config.lr_scheduler
        path_checkpoint = self.config.path_checkpoint

        lr_scheduler_state_dict = None
        if rank == 0:
            if self.full_state_dict is None:
                self.full_state_dict = torch.load(path_checkpoint, map_location = 'cpu')
            lr_scheduler_state_dict = self.full_state_dict.get('scheduler_state_dict')

        # Scatter the training state to all ranks...
        lr_scheduler_state_dict = broadcast_dict(lr_scheduler_state_dict, src = 0, device = device)

        self.config.lr_scheduler.load_state_dict(lr_scheduler_state_dict)


    def save(self, model, optimizer, lr_scheduler, training_state, path_checkpoint):
        dist.barrier()    # Make sure all shards are at the same timepoint in training.

        self.update_config(model, optimizer, lr_scheduler, training_state, path_checkpoint)

        rank = self.config.rank

        model_full_state_dict   = self._prepare_model_full_state_dict()
        optim_full_state_dict   = self._prepare_optim_full_state_dict()
        lr_scheduler_state_dict = self._prepare_lr_scheduler_state_dict_by_rank0()
        training_state_dict     = self._prepare_training_state_dict_by_rank0()

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


    def update_config(self, model = None, optimizer = None, lr_scheduler = None, training_state = None, path_checkpoint = None):
        if model is not None:
            self.config.model = model
            print(f"RANK {self.config.rank} - Model loaded.")

        if optimizer is not None:
            self.config.optimizer = optimizer
            print(f"RANK {self.config.rank} - Optimizer loaded.")

        if lr_scheduler is not None:
            self.config.lr_scheduler = lr_scheduler
            print(f"RANK {self.config.rank} - Scheduler loaded.")

        if training_state is not None:
            self.config.training_state = training_state
            print(f"RANK {self.config.rank} - Training state loaded.")

        if path_checkpoint is not None:
            self.config.path_checkpoint = path_checkpoint
            print(f"RANK {self.config.rank} - Checkpoint path loaded.")


    def pre_fsdp_load(self):
        """
        Only the model needs to be loaded pre FSDP wrapper.
        """
        self._load_model_full_state_dict()
        dist.barrier()    # Make sure all shards are at the same timepoint in training.


    def post_fsdp_load(self, model, optimizer, lr_scheduler, training_state):
        """
        Users have to pass in the current model, optimizer, lr_scheduler and
        training state so that the checkpointer has the best knowledge of the
        FSDP stages.
        """

        self.update_config(model, optimizer, lr_scheduler, training_state)

        self._load_optim_full_state_dict()
        self._load_training_state_dict()
        self._load_lr_scheduler_state_dict()
        dist.barrier()    # Make sure all shards are at the same timepoint in training.


# -- 2. SHARDED STATE DICT
@dataclass
class ShardedStateDictCheckpointConfig:
    model          : Optional[nn.Module]    # A FSDP wrapped model on all ranks
    optimizer      : Optional[torch.optim.Optimizer]
    lr_scheduler   : Optional[torch.optim.lr_scheduler._LRScheduler]
    training_state : TrainingStateDictConfig
    rank           : int
    device         : str
    path_checkpoint: Optional[str]    # A directory

class ShardedStateDictCheckpoint:
    def __init__(self, config):
        self.config = config


    @staticmethod
    def contains_fsdp(module):
        return hasattr(module, 'module')


    def _prepare_model_state_dict(self):
        model = self.config.model

        # Sanity check if the model is wrapped with FSDP
        if not ShardedStateDictCheckpoint.contains_fsdp(model):
            raise ValueError(f"RANK {self.config.rank} - The model subject to "  \
            "checkpointing must be wrapped with an FSDP wrapper before saving a "\
            "full state dict.")

        # Pulling sharded state dict
        model_state_dict = None
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state_dict = model.state_dict()

        return model_state_dict


    def _prepare_optim_state_dict(self):
        optimizer = self.config.optimizer
        model     = self.config.model

        # Pulling sharded state dict
        optim_state_dict = None
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            optim_state_dict = FSDP.sharded_optim_state_dict(model, optimizer)

        return optim_state_dict


    def _save_model_and_optimizer(self):
        # Prepare the state_dict
        model_state_dict = self._prepare_model_state_dict()
        optim_state_dict = self._prepare_optim_state_dict()

        state_dict = dict(
            model_state_dict     = model_state_dict,
            optimizer_state_dict = optim_state_dict,
        )

        # Create a directory for saving checkpoints
        device = self.config.device
        path_checkpoint = self.config.path_checkpoint
        path_checkpoint = broadcast_dict(dict(path_checkpoint=path_checkpoint), src = 0, device = device).get('path_checkpoint')
        os.makedirs(path_checkpoint, exist_ok=True)

        # Write the state_dict to a checkpoint directory
        model = self.config.model
        writer = FileSystemWriter(path_checkpoint, single_file_per_rank=True)
        if state_dict is not None:
            save_state_dict(
                state_dict     = state_dict,
                storage_writer = writer,
                process_group  = model.process_group,
                planner        = DefaultSavePlanner(),
            )


    def _load_model_and_optimizer(self):
        model           = self.config.model
        optimizer       = self.config.optimizer
        path_checkpoint = self.config.path_checkpoint

        # -- Model
        # Sanity check if the model is wrapped with FSDP
        if not ShardedStateDictCheckpoint.contains_fsdp(model):
            raise ValueError(f"RANK {self.config.rank} - The model subject to "  \
            "checkpointing must be wrapped with an FSDP wrapper before saving a "\
            "full state dict.")

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = dict(model_state_dict = model.state_dict())
            load_state_dict(
                state_dict     = state_dict,
                storage_reader = FileSystemReader(path_checkpoint),
                planner        = DefaultLoadPlanner(),
            )
            model.load_state_dict(state_dict.get('model_state_dict'))
            ## model.to(self.config.rank)

        # -- Optimizer
        if optimizer is None:
            raise ValueError("Optimizer has not been properly initialized")

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict = model.state_dict(),
                optimizer_key    = 'optimizer_state_dict',
                storage_reader   = FileSystemReader(path_checkpoint),
            )
        flattened_optim_state_dict = FSDP.optim_state_dict_to_load(
            model = model, optim = optimizer, optim_state_dict = optim_state.get('optimizer_state_dict')
        )
        optimizer.load_state_dict(flattened_optim_state_dict)


    def _prepare_lr_scheduler_state_dict_by_rank0(self):
        rank = self.config.rank
        lr_scheduler_state_dict = None

        if rank == 0:
            lr_scheduler = self.config.lr_scheduler
            lr_scheduler_state_dict = lr_scheduler.state_dict()

        return lr_scheduler_state_dict


    def _prepare_training_state_dict_by_rank0(self):
        rank = self.config.rank
        training_state = None
        if rank == 0:
            training_state = self.config.training_state
            training_state = asdict(training_state)

        return training_state


    def _load_training_state_dict(self):
        rank            = self.config.rank
        device          = self.config.device
        training_state  = self.config.training_state
        path_checkpoint = self.config.path_checkpoint

        training_state = None
        if rank == 0:
            path_scheduler_and_training_checkpoint = os.path.join(path_checkpoint, 'scheduler_and_training.chkpt')
            full_state_dict = torch.load(path_scheduler_and_training_checkpoint, map_location = 'cpu')
            training_state = full_state_dict.get('training_state_dict')

        # Scatter the training state to all ranks...
        training_state = broadcast_dict(training_state, src = 0, device = device)

        self.config.training_state = TrainingStateDictConfig(**training_state)


    def _load_lr_scheduler_state_dict(self):
        rank         = self.config.rank
        device       = self.config.device
        lr_scheduler = self.config.lr_scheduler
        path_checkpoint = self.config.path_checkpoint

        lr_scheduler_state_dict = None
        if rank == 0:
            path_scheduler_and_training_checkpoint = os.path.join(path_checkpoint, 'scheduler_and_training.chkpt')
            full_state_dict = torch.load(path_scheduler_and_training_checkpoint, map_location = 'cpu')
            lr_scheduler_state_dict = full_state_dict.get('scheduler_state_dict')

        # Scatter the training state to all ranks...
        lr_scheduler_state_dict = broadcast_dict(lr_scheduler_state_dict, src = 0, device = device)

        self.config.lr_scheduler.load_state_dict(lr_scheduler_state_dict)


    def update_config(self, model = None, optimizer = None, lr_scheduler = None, training_state = None, path_checkpoint = None):
        if model is not None:
            self.config.model = model
            print(f"RANK {self.config.rank} - Model loaded.")

        if optimizer is not None:
            self.config.optimizer = optimizer
            print(f"RANK {self.config.rank} - Optimizer loaded.")

        if lr_scheduler is not None:
            self.config.lr_scheduler = lr_scheduler
            print(f"RANK {self.config.rank} - Scheduler loaded.")

        if training_state is not None:
            self.config.training_state = training_state
            print(f"RANK {self.config.rank} - Training state loaded.")

        if path_checkpoint is not None:
            self.config.path_checkpoint = path_checkpoint
            print(f"RANK {self.config.rank} - Checkpoint path loaded.")

        dist.barrier()


    def save(self, model, optimizer, lr_scheduler, training_state, path_checkpoint):
        dist.barrier()

        self.update_config(model, optimizer, lr_scheduler, training_state, path_checkpoint)

        # -- Model and Optimizer
        self._save_model_and_optimizer()

        # -- Scheduler and training state (rank0 only)
        lr_scheduler_state_dict = self._prepare_lr_scheduler_state_dict_by_rank0()
        training_state_dict     = self._prepare_training_state_dict_by_rank0()

        rank = self.config.rank
        if rank == 0:
            path_checkpoint = self.config.path_checkpoint
            path_scheduler_and_training_checkpoint = os.path.join(path_checkpoint, 'scheduler_and_training.chkpt')
            full_state_dict = {
                'scheduler_state_dict' : lr_scheduler_state_dict,
                'training_state_dict'  : training_state_dict,
            }
            torch.save(full_state_dict, path_scheduler_and_training_checkpoint)

        dist.barrier()


    def load(self):
        # -- Model and Optimizer
        self._load_model_and_optimizer()

        # -- Scheduler and training state (rank0 only)
        self._load_training_state_dict()
        self._load_lr_scheduler_state_dict()

        dist.barrier()
