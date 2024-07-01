from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingStateDictConfig:
    epoch      : int
    seg        : int
    start_idx  : int
    end_idx    : int
    loss_min   : float

@dataclass
class CheckpointConfig:
    model          : Optional[nn.Module]    # A FSDP wrapped model on all ranks
    optimizer      : Optional[torch.optim.Optimizer]
    lr_scheduler   : Optional[torch.optim.lr_scheduler._LRScheduler]
    training_state : Optional[TrainingStateDictConfig]
    rank           : int
    device         : str
    path_checkpoint: Optional[str]

class Checkpoint:
    def __init__(self, config):
        self.config = config
        self.full_state_dict = None


    def _prepare_model_full_state_dict(self):
        return self.config.model.state_dict()


    def _prepare_optim_full_state_dict(self):
        return self.config.optimizer.state_dict()


    def _prepare_lr_scheduler_state_dict_by_rank0(self):
        return self.config.lr_scheduler.state_dict()


    def _prepare_training_state_dict_by_rank0(self):
        return asdict(self.config.training_state)


    def _load_model_full_state_dict(self):
        if self.full_state_dict is None:
            self.full_state_dict = torch.load(self.config.path_checkpoint, map_location = 'cpu')
        model_full_state_dict = self.full_state_dict.get('model_state_dict')
        self.config.model.load_state_dict(model_full_state_dict)


    def _load_optim_full_state_dict(self):
        if self.full_state_dict is None:
            self.full_state_dict = torch.load(self.config.path_checkpoint, map_location = 'cpu')
        optim_full_state_dict = self.full_state_dict.get('optim_state_dict')
        self.config.optimizer.load_state_dict(optim_full_state_dict)


    def _load_training_state_dict(self):
        if self.full_state_dict is None:
            self.full_state_dict = torch.load(self.config.path_checkpoint, map_location = 'cpu')
        training_state = self.full_state_dict.get('training_state_dict')
        self.config.training_state = TrainingStateDictConfig(**training_state)


    def _load_lr_scheduler_state_dict(self):
        if self.full_state_dict is None:
            self.full_state_dict = torch.load(self.config.path_checkpoint, map_location = 'cpu')
        lr_scheduler_state_dict = self.full_state_dict.get('scheduler_state_dict')
        self.config.lr_scheduler.load_state_dict(lr_scheduler_state_dict)


    def save(self, model, optimizer, lr_scheduler, training_state, path_checkpoint):
        self.update_config(model, optimizer, lr_scheduler, training_state, path_checkpoint)

        model_full_state_dict   = self._prepare_model_full_state_dict()
        optim_full_state_dict   = self._prepare_optim_full_state_dict()
        lr_scheduler_state_dict = self._prepare_lr_scheduler_state_dict_by_rank0()
        training_state_dict     = self._prepare_training_state_dict_by_rank0()

        path_checkpoint = self.config.path_checkpoint
        full_state_dict = {
            'model_state_dict'     : model_full_state_dict,
            'optim_state_dict'     : optim_full_state_dict,
            'scheduler_state_dict' : lr_scheduler_state_dict,
            'training_state_dict'  : training_state_dict,
        }
        torch.save(full_state_dict, path_checkpoint)


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
        self._load_model_full_state_dict()


    def post_fsdp_load(self, model, optimizer, lr_scheduler, training_state):
        self.update_config(model, optimizer, lr_scheduler, training_state)

        self._load_optim_full_state_dict()
        self._load_training_state_dict()
        self._load_lr_scheduler_state_dict()
