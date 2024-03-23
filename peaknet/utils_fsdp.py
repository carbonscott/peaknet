import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint import FileSystemWriter, save_state_dict, FileSystemReader, load_state_dict


def save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt, process_group=dist.group.WORLD, coordinator_rank=0):
    """
    Save a checkpoint for models in distributed training using PyTorch Distributed Checkpoint (DCP).

    Parameters:
    - model           : The model, which can be a regular model or wrapped with FSDP for sharded training.
    - optimizer       : The optimizer used in training.
    - scheduler       : Learning rate scheduler.
    - epoch           : Current epoch or iteration number.
    - loss_min        : Minimum or best loss achieved so far.
    - path_chkpt      : Path to save the checkpoint (e.g. f"{timestamp}.epoch_{epoch}.rank_{fsdp_rank}.chkpt").
    - process_group   : The process group used for distributed training.
    - coordinator_rank: The rank of the process coordinating the checkpoint.
    """
    storage_writer = FileSystemWriter(path_chkpt)

    state_dict = {
        'model'     : FSDP.full_optim_state_dict(model) if isinstance(model, FSDP) else model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch'     : epoch,
        'loss_min'  : loss_min,
    }

    # Use DCP to save the state_dict
    save_state_dict(
        state_dict       = state_dict,
        storage_writer   = storage_writer,
        process_group    = process_group,
        coordinator_rank = coordinator_rank,
    )


def load_checkpoint(model, optimizer, scheduler, path_chkpt, process_group=dist.group.WORLD):
    """
    Load a checkpoint for models, along with optimizer and scheduler states, in distributed training using PyTorch Distributed Checkpoint (DCP).

    Parameters:
    - model: The FSDP-wrapped model.
    - optimizer: The optimizer used in training.
    - scheduler: The learning rate scheduler used in training.
    - path_chkpt: Path to the checkpoint file.
    - process_group: The process group used for distributed training.
    """
    storage_reader = FileSystemReader(path_chkpt)

    # Prepare the empty state_dict for loading
    state_dict = {}

    # Use DCP to load the state_dict
    load_state_dict(
        state_dict     = state_dict,
        storage_reader = storage_reader,
        process_group  = process_group
    )

    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    if 'optimizer' in state_dict and optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    if 'scheduler' in state_dict and scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])

    return state_dict['epoch'], state_dict['loss_min']
