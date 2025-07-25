import os

import torch
import torch.distributed as dist

# -- Imports for FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# -- Imports for saving sharded state dict
# Use new APIs (migrated from torch.distributed._shard.checkpoint)
from torch.distributed.checkpoint import (
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
from packaging import version

import logging

logger = logging.getLogger(__name__)

class Checkpoint:
    MODEL_STATE_DICT_FILE = 'model_state_dict.pt'
    OPTIM_STATE_DICT_FILE = 'optim_state_dict.pt'
    LR_STATE_DICT_FILE    = 'lr_state_dict.pt'
    ITER_STATE_DICT_FILE  = 'iter_state_dict.pt'

    def __init__(self):
        pass

    def save_model_checkpoint(self, rank, model, path_checkpoint_model):
        if rank == 0:
            model_state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
            torch.save(model_state_dict, path_checkpoint_model)

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        """
        Must run before FSDP wrapper.
        """
        if dist.is_initialized():
            dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            model_state_dict = torch.load(path_checkpoint_model, map_location='cpu')
            object_list = [model_state_dict,]

        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)

        model_state_dict = object_list[0]
        model.load_state_dict(model_state_dict)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        if dist.is_initialized():
            dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            full_optim_state_dict = torch.load(path_checkpoint_optim, map_location='cpu')
            object_list = [full_optim_state_dict, ]

        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)
        full_optim_state_dict = object_list[0]
        optimizer.load_state_dict(full_optim_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        if rank == 0:
            optim_state_dict = optimizer.state_dict()
            torch.save(optim_state_dict, path_checkpoint_optim)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            lr_state_dict = lr_scheduler.state_dict()
            torch.save(lr_state_dict, path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if dist.is_initialized():
            dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            lr_state_dict = torch.load(path_checkpoint_lr, map_location = 'cpu')
            object_list = [lr_state_dict, ]

        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)
        lr_state_dict = object_list[0]
        lr_scheduler.load_state_dict(lr_state_dict)

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if dist.is_initialized():
            dist.barrier()

        object_list = [None, ]  # For the use of dist.broadcast_object_list
        if rank == 0:
            iter_state_saved = torch.load(path_checkpoint_iter_state, map_location = 'cpu')
            object_list = [iter_state_saved, ]

        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)
        iter_state_saved = object_list[0]
        iter_state.clear()
        iter_state.update(iter_state_saved)

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

    def pre_dp_load(self, rank, model, path_checkpoint):
        """
        Only the model needs to be loaded pre FSDP wrapper.
        """
        path_checkpoint_model = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        self.load_model_checkpoint(rank, model, path_checkpoint_model)

    def post_dp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
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

        if dist.is_initialized():
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


# -- 1. FULL STATE DICT
class FullStateDictCheckpoint:
    MODEL_STATE_DICT_FILE = 'model_state_dict.pt'
    OPTIM_STATE_DICT_FILE = 'optim_state_dict.pt'
    LR_STATE_DICT_FILE    = 'lr_state_dict.pt'
    ITER_STATE_DICT_FILE  = 'iter_state_dict.pt'

    def __init__(self, offload_to_cpu=True, rank0_only=True, **kwargs):
        self.state_dict_config = FullStateDictConfig(
            offload_to_cpu = offload_to_cpu,
            rank0_only     = rank0_only,
        )
        self.optim_dict_config = FullOptimStateDictConfig(
            offload_to_cpu = offload_to_cpu,
            rank0_only     = rank0_only,
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
        iter_state.clear()
        iter_state.update(iter_state_saved)

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

    def pre_dp_load(self, rank, model, path_checkpoint):
        """
        Only the model needs to be loaded pre FSDP wrapper.
        """
        path_checkpoint_model = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        self.load_model_checkpoint(rank, model, path_checkpoint_model)

        dist.barrier()

    def post_dp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
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
    MODEL_STATE_DICT_DIR = 'model_state_dict.pt'
    OPTIM_STATE_DICT_DIR = 'optim_state_dict.pt'
    LR_STATE_DICT_FILE   = 'lr_state_dict.pt'
    ITER_STATE_DICT_FILE = 'iter_state_dict.pt'

    def __init__(self, offload_to_cpu=True, **kwargs):
        self.state_dict_config = ShardedStateDictConfig(
            offload_to_cpu = offload_to_cpu,
        )
        self.optim_dict_config = ShardedOptimStateDictConfig(
            offload_to_cpu = offload_to_cpu,
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
        iter_state.clear()
        iter_state.update(iter_state_saved)

    def save(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        os.makedirs(path_checkpoint, exist_ok = True)
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_DIR)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_DIR)
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
        path_checkpoint_model      = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_DIR)
        path_checkpoint_optim      = os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_DIR)
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

    def pre_dp_load(self, rank, model, path_checkpoint):
        pass

    def post_dp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        self.load(rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint)

def init_checkpointer(state_dict_type, uses_fsdp, offload_to_cpu=True, rank0_only=True):
    checkpoint_func = {
        "full"    : FullStateDictCheckpoint,
        "sharded" : ShardedStateDictCheckpoint,
    }[state_dict_type] if uses_fsdp else Checkpoint

    if uses_fsdp:
        if state_dict_type == "full":
            checkpointer = checkpoint_func(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        else:  # sharded
            checkpointer = checkpoint_func(offload_to_cpu=offload_to_cpu)
    else:
        checkpointer = checkpoint_func()

    return checkpointer
