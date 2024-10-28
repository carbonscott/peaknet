from dataclasses import dataclass, asdict
from typing import Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, path_checkpoint_model)

    def load_model_checkpoint(self, rank, model, path_checkpoint_model):
        """
        Must run before FSDP wrapper.
        """
        model_state_dict = torch.load(path_checkpoint_model)
        model.load_state_dict(model_state_dict)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        full_optim_state_dict = torch.load(path_checkpoint_optim)
        optimizer.load_state_dict(full_optim_state_dict)

    def save_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        if rank == 0:
            optim_state_dict = optimizer.state_dict()
            torch.save(optim_state_dict, path_checkpoint_optim)

    def load_optimizer_checkpoint(self, rank, model, optimizer, path_checkpoint_optim):
        optim_state_dict = torch.load(path_checkpoint_optim)
        optimizer.load_state_dict(optim_state_dict)

    def save_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        if rank == 0:
            lr_state_dict = lr_scheduler.state_dict()
            torch.save(lr_state_dict, path_checkpoint_lr)

    def load_lr_checkpoint(self, rank, lr_scheduler, path_checkpoint_lr):
        lr_state_dict = torch.load(path_checkpoint_lr, map_location = 'cpu')
        lr_scheduler.load_state_dict(lr_state_dict)

    def save_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        if rank == 0:
            torch.save(iter_state, path_checkpoint_iter_state)

    def load_iter_state_checkpoint(self, rank, iter_state, path_checkpoint_iter_state):
        iter_state_saved = torch.load(path_checkpoint_iter_state, map_location = 'cpu')
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
