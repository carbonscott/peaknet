import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._shard.checkpoint import (
    FileSystemReader, FileSystemWriter, load_state_dict, save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner, DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp.api import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
    ShardedStateDictConfig, ShardedOptimStateDictConfig,
)
from packaging import version
import logging

logger = logging.getLogger(__name__)

class Checkpoint:
    """Basic checkpoint class for non-FSDP models."""
    
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
        if dist.is_initialized():
            dist.barrier()
        object_list = [None, ]
        if rank == 0:
            model_state_dict = torch.load(path_checkpoint_model, map_location='cpu')
            object_list = [model_state_dict,]
        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)
        model_state_dict = object_list[0]
        model.load_state_dict(model_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        if rank == 0:
            optim_state_dict = optimizer.state_dict()
            torch.save(optim_state_dict, path_checkpoint_optim)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        if dist.is_initialized():
            dist.barrier()
        object_list = [None, ]
        if rank == 0:
            full_optim_state_dict = torch.load(path_checkpoint_optim, map_location='cpu')
            object_list = [full_optim_state_dict, ]
        if dist.is_initialized():
            dist.broadcast_object_list(object_list, src = 0)
        full_optim_state_dict = object_list[0]
        optimizer.load_state_dict(full_optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            lr_state_dict = lr_scheduler.state_dict()
            torch.save(lr_state_dict, path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if dist.is_initialized():
            dist.barrier()
        object_list = [None, ]
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
        object_list = [None, ]
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

    def pre_fsdp_load(self, rank, model, path_checkpoint):
        path_checkpoint_model = os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE)
        self.load_model_checkpoint(rank, model, path_checkpoint_model)

    def post_fsdp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
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


class FullStateDictCheckpoint:
    """FSDP-compatible full state dict checkpoint."""
    
    MODEL_STATE_DICT_FILE = 'model_state_dict.pt'
    OPTIM_STATE_DICT_FILE = 'optim_state_dict.pt'
    LR_STATE_DICT_FILE    = 'lr_state_dict.pt'
    ITER_STATE_DICT_FILE  = 'iter_state_dict.pt'

    def __init__(self, offload_to_cpu=True, rank0_only=True, **kwargs):
        self.state_dict_config = FullStateDictConfig(
            offload_to_cpu = offload_to_cpu, rank0_only = rank0_only,
        )
        self.optim_dict_config = FullOptimStateDictConfig(
            offload_to_cpu = offload_to_cpu, rank0_only = rank0_only,
        )

    def save_model_checkpoint(self, rank, model, path_checkpoint_model):
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            state_dict_config=self.state_dict_config,
            optim_state_dict_config=self.optim_dict_config,
        ):
            model_state_dict = model.state_dict()
            if rank == 0:
                torch.save(model_state_dict, path_checkpoint_model)

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        dist.barrier()
        model_state_dict = torch.load(path_checkpoint_model)
        FSDP.set_state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            state_dict_config=self.state_dict_config,
            optim_state_dict_config=self.optim_dict_config,
        )
        model.load_state_dict(model_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        torch_version = torch.__version__
        torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
        if version.parse(torch_version) <= version.parse("2.0.1"):
            optim_state_dict = FSDP.full_optim_state_dict(model, optimizer)
        else:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT,
                state_dict_config=self.state_dict_config,
                optim_state_dict_config=self.optim_dict_config,
            ):
                optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        if rank == 0:
            torch.save(optim_state_dict, path_checkpoint_optim)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist.barrier()
        torch_version = torch.__version__
        torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
        if version.parse(torch_version) <= version.parse("2.0.1"):
            full_optim_state_dict = None
            if rank == 0 or not self.optim_dict_config.rank0_only:
                full_optim_state_dict = torch.load(path_checkpoint_optim)
            sharded_optim_state_dict = FSDP.scatter_full_optim_state_dict(
                full_optim_state_dict = full_optim_state_dict, model = model,
            )
            optimizer.load_state_dict(sharded_optim_state_dict)
        else:
            FSDP.set_state_dict_type(
                model, StateDictType.FULL_STATE_DICT,
                state_dict_config=self.state_dict_config,
                optim_state_dict_config=self.optim_dict_config,
            )
            full_optim_state_dict = None
            if rank == 0 or not self.optim_dict_config.rank0_only:
                full_optim_state_dict = torch.load(path_checkpoint_optim)
            flattened_optim_state_dict = FSDP.optim_state_dict_to_load(
                model=model, optim=optimizer, optim_state_dict=full_optim_state_dict,
            )
            optimizer.load_state_dict(flattened_optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            torch.save(lr_scheduler.state_dict(), path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        dist.barrier()
        object_list = [None, ]
        if rank == 0:
            object_list = [torch.load(path_checkpoint_lr, map_location='cpu'), ]
        dist.broadcast_object_list(object_list, src = 0)
        lr_scheduler.load_state_dict(object_list[0])

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        dist.barrier()
        object_list = [None, ]
        if rank == 0:
            object_list = [torch.load(path_checkpoint_iter_state, map_location='cpu'), ]
        dist.broadcast_object_list(object_list, src = 0)
        iter_state.clear()
        iter_state.update(object_list[0])

    def save(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        os.makedirs(path_checkpoint, exist_ok = True)
        if model is not None:
            self.save_model_checkpoint(rank, model, os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE))
        if optimizer is not None:
            self.save_optimizer_checkpoint(rank, model, optimizer, os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE))
        if lr_scheduler is not None:
            self.save_lr_checkpoint(rank, lr_scheduler, os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE))
        if iter_state is not None:
            self.save_iter_state_checkpoint(rank, iter_state, os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE))

    def pre_fsdp_load(self, rank, model, path_checkpoint):
        self.load_model_checkpoint(rank, model, os.path.join(path_checkpoint, self.MODEL_STATE_DICT_FILE))
        dist.barrier()

    def post_fsdp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        if optimizer is not None:
            self.load_optimizer_checkpoint(rank, model, optimizer, os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_FILE))
        if lr_scheduler is not None:
            self.load_lr_checkpoint(rank, lr_scheduler, os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE))
        if iter_state is not None:
            self.load_iter_state_checkpoint(rank, iter_state, os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE))
        dist.barrier()


class ShardedStateDictCheckpoint:
    """FSDP-compatible sharded state dict checkpoint."""
    
    MODEL_STATE_DICT_DIR = 'model_state_dict.pt'
    OPTIM_STATE_DICT_DIR = 'optim_state_dict.pt'
    LR_STATE_DICT_FILE   = 'lr_state_dict.pt'
    ITER_STATE_DICT_FILE = 'iter_state_dict.pt'

    def __init__(self, offload_to_cpu=True, **kwargs):
        self.state_dict_config = ShardedStateDictConfig(offload_to_cpu=offload_to_cpu)
        self.optim_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=offload_to_cpu)

    def save_model_checkpoint(self, rank, model, path_checkpoint_model):
        dist_writer = FileSystemWriter(path_checkpoint_model)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                state_dict_config=self.state_dict_config,
                                optim_state_dict_config=self.optim_dict_config)
        model_state_dict = model.state_dict()
        save_state_dict({"model": model_state_dict}, dist_writer, DefaultSavePlanner())

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        dist.barrier()
        dist_reader = FileSystemReader(path_checkpoint_model)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                state_dict_config=self.state_dict_config,
                                optim_state_dict_config=self.optim_dict_config)
        model_state_dict = model.state_dict()
        state_dict_to_load = {"model": model_state_dict}
        load_state_dict(state_dict_to_load, dist_reader, DefaultLoadPlanner())
        model.load_state_dict(state_dict_to_load["model"])

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist_writer = FileSystemWriter(path_checkpoint_optim)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                state_dict_config=self.state_dict_config,
                                optim_state_dict_config=self.optim_dict_config)
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        save_state_dict({"optim": optim_state_dict}, dist_writer, DefaultSavePlanner())

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        dist.barrier()
        dist_reader = FileSystemReader(path_checkpoint_optim)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                state_dict_config=self.state_dict_config,
                                optim_state_dict_config=self.optim_dict_config)
        model_state_dict = model.state_dict()
        state_dict_to_load = load_sharded_optimizer_state_dict(
            model_state_dict=model_state_dict, optimizer_key='optim', storage_reader=dist_reader,
        )
        flattened_optim_state_dict = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=state_dict_to_load["optim"],
        )
        optimizer.load_state_dict(flattened_optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            torch.save(lr_scheduler.state_dict(), path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        dist.barrier()
        object_list = [None, ]
        if rank == 0:
            object_list = [torch.load(path_checkpoint_lr, map_location='cpu'), ]
        dist.broadcast_object_list(object_list, src = 0)
        lr_scheduler.load_state_dict(object_list[0])

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        dist.barrier()
        object_list = [None, ]
        if rank == 0:
            object_list = [torch.load(path_checkpoint_iter_state, map_location='cpu'), ]
        dist.broadcast_object_list(object_list, src = 0)
        iter_state.clear()
        iter_state.update(object_list[0])

    def save(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        os.makedirs(path_checkpoint, exist_ok = True)
        if model is not None:
            self.save_model_checkpoint(rank, model, os.path.join(path_checkpoint, self.MODEL_STATE_DICT_DIR))
        if optimizer is not None:
            self.save_optimizer_checkpoint(rank, model, optimizer, os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_DIR))
        if lr_scheduler is not None:
            self.save_lr_checkpoint(rank, lr_scheduler, os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE))
        if iter_state is not None:
            self.save_iter_state_checkpoint(rank, iter_state, os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE))

    def pre_fsdp_load(self, rank, model, path_checkpoint):
        pass

    def post_fsdp_load(self, rank, model, optimizer, lr_scheduler, iter_state, path_checkpoint):
        if model is not None:
            self.load_model_checkpoint(rank, model, os.path.join(path_checkpoint, self.MODEL_STATE_DICT_DIR))
        if optimizer is not None:
            self.load_optimizer_checkpoint(rank, model, optimizer, os.path.join(path_checkpoint, self.OPTIM_STATE_DICT_DIR))
        if lr_scheduler is not None:
            self.load_lr_checkpoint(rank, lr_scheduler, os.path.join(path_checkpoint, self.LR_STATE_DICT_FILE))
        if iter_state is not None:
            self.load_iter_state_checkpoint(rank, iter_state, os.path.join(path_checkpoint, self.ITER_STATE_DICT_FILE))


def init_checkpointer(state_dict_type, uses_fsdp, offload_to_cpu=True, rank0_only=True):
    """Factory function to create appropriate checkpointer."""
    if uses_fsdp:
        checkpoint_func = {"full": FullStateDictCheckpoint, "sharded": ShardedStateDictCheckpoint}[state_dict_type]
        if state_dict_type == "full":
            return checkpoint_func(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        else:
            return checkpoint_func(offload_to_cpu=offload_to_cpu)
    else:
        return Checkpoint()
