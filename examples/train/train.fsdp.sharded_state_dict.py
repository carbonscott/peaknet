#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import socket
import tqdm
import signal
import numpy as np
import h5py
import argparse
import logging

from functools import partial

from peaknet.datasets.safetensors_dataset  import PeakNetDataset
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet, SegHeadConfig
from peaknet.modeling.convnextv2_encoder   import ConvNextV2BackboneConfig
from peaknet.modeling.bifpn_config         import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from peaknet.criterion                     import CategoricalFocalLoss
from peaknet.utils                         import init_logger, split_dataset, set_seed, init_weights
from peaknet.lr_scheduler                  import CosineLRScheduler
from peaknet.perf                          import Timer
from peaknet.tensor_transforms             import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift
from peaknet.utils_fsdp                    import (
    MemoryMaximizer,
    verify_bfloat_support,
    TrainingStateDictConfig,
    FullStateDictCheckpointConfig,
    FullStateDictCheckpoint,
    ShardedStateDictCheckpointConfig,
    ShardedStateDictCheckpoint,
)

import torch
import torch.nn as nn
import torch.optim as optim

# Libraryies used for Fully Sharded Data Parallel (FSDP)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,

    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed as dist

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap

from contextlib import contextmanager

torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16
                                            # Reference: https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4

import colorama
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)

# [[[ ARG ]]]
parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")

args = parser.parse_args()


# [[[ HYPER-PARAMERTERS ]]]
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# ...Checkpoint
chkpt_config        = config.get("checkpoint")
dir_root_chkpt      = chkpt_config.get("directory")
path_pretrain_chkpt = chkpt_config.get("pretrain")
fl_chkpt_prefix     = chkpt_config.get("filename_prefix")
dir_chkpt_prefix    = chkpt_config.get("dir_chkpt_prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")
epoch_unstable_end  = chkpt_config.get("epoch_unstable_end")

# ...Dataset
dataset_config    = config.get("dataset")
path_train_csv    = dataset_config.get("path_train")
path_validate_csv = dataset_config.get("path_validate")
size_batch        = dataset_config.get("batch_size")
num_workers       = dataset_config.get("num_workers")
transforms_config = dataset_config.get("transforms")
num_patch         = transforms_config.get("num_patch")
size_patch        = transforms_config.get("size_patch")
frac_shift_max    = transforms_config.get("frac_shift_max")
angle_max         = transforms_config.get("angle_max")
var_size_patch    = transforms_config.get("var_size_patch")
downscale_factors = transforms_config.get("downscale_factors")
H_pad             = transforms_config.get("H_pad")
W_pad             = transforms_config.get("W_pad")

# ...Model
model_params              = config.get("model")
uses_random_weights       = model_params.get("uses_random_weights")
freezes_backbone          = model_params.get("freezes_backbone")
backbone_params           = model_params.get("backbone")
bifpn_params              = model_params.get("bifpn")
bifpn_block_params        = bifpn_params.get("block")
bifpn_block_bn_params     = bifpn_block_params.get("bn")
bifpn_block_fusion_params = bifpn_block_params.get("fusion")
seghead_params            = model_params.get("seg_head")
seghead_num_classes       = seghead_params.get("num_classes")

# ...Loss
loss_config  = config.get("loss")
focal_config = loss_config.get("focal")
focal_alpha  = focal_config.get("alpha")
focal_gamma  = focal_config.get("gamma")

# ...Optimizer
optim_config = config.get("optim")
lr           = float(optim_config.get("lr"))
weight_decay = float(optim_config.get("weight_decay"))
grad_clip    = float(optim_config.get("grad_clip"))

# ...Scheduler
lr_scheduler_config = config.get("lr_scheduler")
patience            = lr_scheduler_config.get("patience")
warmup_epochs       = lr_scheduler_config.get("warmup_epochs")
total_epochs        = lr_scheduler_config.get("total_epochs")
uses_prev_scheduler = lr_scheduler_config.get("uses_prev")
min_lr              = float(lr_scheduler_config.get("min_lr"))


# ...FSDP
fsdp_config            = config.get("fsdp")
fsdp_backend           = fsdp_config.get("backend")
uses_unique_world_seed = fsdp_config.get("uses_unique_world_seed")

# ...Logging
logging_config = config.get("logging")
drc_log       = logging_config.get("directory")
fl_log_prefix = logging_config.get("filename_prefix")

# ...Misc
misc_config = config.get("misc")
uses_mixed_precision = misc_config.get("uses_mixed_precision")
max_epochs           = misc_config.get("max_epochs")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")
cache_clear_after_n_batch = 5


# [[[ ERROR HANDLING ]]]
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# [[[ FSDP INIT ]]]
# Initialize distributed environment
uses_fsdp = int(os.environ.get("RANK", -1)) != -1
if uses_fsdp:
    fsdp_rank       = int(os.environ["RANK"      ])
    fsdp_local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = fsdp_backend,
                            rank        = fsdp_rank,
                            world_size  = fsdp_world_size,
                            init_method = "env://",)
    print(f"RANK:{fsdp_rank},LOCAL_RANK:{fsdp_local_rank},WORLD_SIZE:{fsdp_world_size}")
else:
    fsdp_rank       = 0
    fsdp_local_rank = 0
    fsdp_world_size = 1
    print(f"NO FSDP is used.  RANK:{fsdp_rank},LOCAL_RANK:{fsdp_local_rank},WORLD_SIZE:{fsdp_world_size}")

# Set up GPU device
device = f'cuda:{fsdp_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = fsdp_rank if uses_unique_world_seed else 0

# [[[ PERFORMANCE UTILITY ]]]
memmax = MemoryMaximizer if fsdp_local_rank == 0 else None

# [[[ FSDP POLICY ]]]
# === Mixed precision ===
dtype           = 'float32'
mixed_precision = None
if uses_mixed_precision:
    if verify_bfloat_support:
        dtype = 'bfloat16'
        mixed_precision = MixedPrecision(
            param_dtype  = torch.bfloat16,
            reduce_dtype = torch.bfloat16,
            buffer_dtype = torch.bfloat16,
        )
    else:
        dtype = 'float16'
        mixed_precision = MixedPrecision(
            param_dtype  = torch.float16,
            reduce_dtype = torch.float16,
            buffer_dtype = torch.float16,
        )

# === Sharding strategy ===
sharding_strategy = ShardingStrategy.FULL_SHARD

# === Wrapping strategy ===
min_num_params   = 500_000
auto_wrap_policy = partial(size_based_auto_wrap_policy,
                           min_num_params = min_num_params,)

# [[[ USE YAML CONFIG TO INITIALIZE HYPERPARAMETERS ]]]
# Set Seed
base_seed   = 0
world_seed  = base_seed + seed_offset

if fsdp_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)


# [[[ DATASET ]]]
# Set global seed...
set_seed(world_seed)

# Set up transformation...
trans_list = (
    Pad(H_pad, W_pad),
    DownscaleLocalMean(factors = downscale_factors),
    RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    RandomRotate(angle_max),
    RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
)

# Load raw data...
set_seed(base_seed)    # ...Make sure data split is consistent across devices
cache_size       = 10
dataset_train    = PeakNetDataset(path_csv = path_train_csv   , trans_list = trans_list, cache_size = cache_size, perfs_runtime = False)
dataset_validate = PeakNetDataset(path_csv = path_validate_csv, trans_list = trans_list, cache_size = cache_size, perfs_runtime = False)

# Define the training set
sampler_train    = torch.utils.data.DistributedSampler(dataset_train) if uses_fsdp else None
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                sampler         = sampler_train,
                                                shuffle         = False,
                                                pin_memory      = True,
                                                batch_size      = size_batch,
                                                num_workers     = num_workers,
                                                prefetch_factor = 2)

# Define validation set...
sampler_validate    = torch.utils.data.DistributedSampler(dataset_validate, shuffle=False) if uses_fsdp else None
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   sampler         = sampler_validate,
                                                   shuffle         = False,
                                                   pin_memory      = True,
                                                   batch_size      = size_batch,
                                                   num_workers     = num_workers,
                                                   prefetch_factor = 2)


# [[[ MODEL ]]]
# Use the model architecture -- regnet + bifpn...
# ...Backbone
backbone_config = ConvNextV2BackboneConfig(**backbone_params)

# ...BiFPN
bifpn_block_params["bn"]     = BNConfig        (**bifpn_block_bn_params)
bifpn_block_params["fusion"] = FusionConfig    (**bifpn_block_fusion_params)
bifpn_params["block"]        = BiFPNBlockConfig(**bifpn_block_params)
bifpn_config                 = BiFPNConfig     (**bifpn_params)

# ...Seg head
seghead_config = SegHeadConfig(**seghead_params)

# ...PeakNet
peaknet_config = PeakNetConfig(
    backbone = backbone_config,
    bifpn    = bifpn_config,
    seg_head = seghead_config,
)
model = PeakNet(config = peaknet_config)
## model.to(device)

if fsdp_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

## # Initialized by the main rank and weights will be broadcast by FSDP wrapper
## if fsdp_rank == 0:
##     if uses_random_weights:
##         # Use random weights...
##         model.apply(init_weights)
##     ## else:
##     ##     # Use pretrained weights...
##     ##     pretrain_chkpt = torch.load(path_pretrain_chkpt)
##     ##     model.backbone.encoder.load_state_dict(pretrain_chkpt, strict = False)

# Freeze the backbone???
if freezes_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False

## model.to(dtype = torch.bfloat16)


# [[[ CRITERION ]]]
criterion = CategoricalFocalLoss(alpha       = focal_alpha,
                                 gamma       = focal_gamma,
                                 num_classes = seghead_num_classes,)

# [[[ OTHER SETTINGS ]]]
# Initialize a GradScaler. If enabled=False scaler is a no-op
if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# Wrapping the model in FSDP...
if uses_fsdp:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP...
    model = FSDP(model,
                 auto_wrap_policy  = auto_wrap_policy,
                 mixed_precision   = mixed_precision,
                 sharding_strategy = sharding_strategy,
                 device_id         = device)

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {fsdp_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

if fsdp_rank == 0:
    print(f"Current timestamp: {timestamp}")

# [[[ OPTIMIZER ]]]
param_iter = model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = CosineLRScheduler(optimizer     = optimizer,
                              warmup_epochs = warmup_epochs,
                              total_epochs  = total_epochs,
                              min_lr        = min_lr)
## scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
##                                          factor         = 2e-1,
##                                          patience       = patience,
##                                          threshold      = 1e-4,
##                                          threshold_mode ='rel',
##                                          verbose        = True)

# [[[ CHECKPOINT ]]]
# From a prev training???
## epoch_min = 0
## mini_batch = 0
## micro_batch = 0
## loss_min  = float('inf')
training_state_dict_config = TrainingStateDictConfig(
    epoch       = 0,
    mini_batch  = 0,
    micro_batch = 0,
    loss_min    = float('inf'),
)
chkpt_config = ShardedStateDictCheckpointConfig(
    model           = model,
    optimizer       = optimizer,
    lr_scheduler    = scheduler,
    training_state  = training_state_dict_config,
    rank            = fsdp_rank,
    device          = device,
    path_checkpoint = path_chkpt_prev,
)
checkpointer = ShardedStateDictCheckpoint(config = chkpt_config)

# Resume training state...
if path_chkpt_prev is not None:
    # ...from a full state dict checkpoint
    if isinstance(checkpointer, ShardedStateDictCheckpoint):
        checkpointer.load()

        training_state = checkpointer.config.training_state
        epoch_min      = training_state.epoch
        mini_batch     = training_state.mini_batch
        micro_batch    = training_state.micro_batch
        loss_min       = training_state.loss_min

        print(colorama.Fore.RED + f"RANK {fsdp_rank} - epoch_min   = {epoch_min  }")
        print(colorama.Fore.RED + f"RANK {fsdp_rank} - mini_batch  = {mini_batch }")
        print(colorama.Fore.RED + f"RANK {fsdp_rank} - micro_batch = {micro_batch}")
        print(colorama.Fore.RED + f"RANK {fsdp_rank} - loss_min    = {loss_min   }")

        ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
        epoch_min += 1    # Next epoch
        mini_batch  += 1
        micro_batch += 1
        logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

# Let's simulate saving a full state dict chkpt...
training_state = TrainingStateDictConfig(**dict(
    epoch       = 4,
    micro_batch = 10,
    mini_batch  = 10,
    loss_min    = 0.2,
))
if path_chkpt_prev is None:
    path_chkpt = None
    if fsdp_rank == 0:
        epoch = training_state.epoch
        micro_batch = training_state.micro_batch
        dir_chkpt = f"{timestamp}.epoch_{epoch}.microbatch_{micro_batch}"
        if dir_chkpt_prefix is not None: dir_chkpt = f"{dir_chkpt_prefix}.{dir_chkpt}"
        path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
    checkpointer.save(model, optimizer, scheduler, training_state, path_chkpt)

dist.barrier()
if dist.is_initialized():
    dist.destroy_process_group()

## # [[[ TRAIN LOOP ]]]
## try:
##     for epoch in tqdm.tqdm(range(max_epochs)):
##         epoch += epoch_min
## 
##         if uses_fsdp:
##             # Shuffle the training examples...
##             sampler_train.set_epoch(epoch)
## 
##         # ___/ TRAIN \___
##         # Turn on training related components in the model...
##         model.train()
## 
##         # Fetch batches...
##         batch_train  = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
##         train_loss   = torch.zeros(len(batch_train)).to(device).float()
##         train_sample = torch.zeros(len(batch_train)).to(device).float()
##         for batch_idx, batch_entry in batch_train:
##             with Timer(tag = "Moving data to gpu", is_on = False):
##                 # Unpack the batch entry and move them to device...
##                 batch_input, batch_target = batch_entry
##                 batch_input  = batch_input.to(device, non_blocking=True)
##                 batch_target = batch_target.to(device, non_blocking=True)
## 
##             # Forward, backward and update...
##             if uses_mixed_precision:
##                 with Timer(tag = "Forward", is_on = False):
##                     with torch.cuda.amp.autocast(dtype = torch.bfloat16):
##                         # Forward pass...
##                         batch_output = model(batch_input)
## 
##                         # Calculate the loss...
##                         loss = criterion(batch_output, batch_target)
##                         loss = loss.mean()
## 
##                 with Timer(tag = "Backward", is_on = False):
##                     # Backward pass and optimization...
##                     optimizer.zero_grad()
##                     scaler.scale(loss).backward()
##                     if grad_clip != 0.0:
##                         scaler.unscale_(optimizer)
##                         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
##                     scaler.step(optimizer)
##                     scaler.update()
##             else:
##                 # Forward pass...
##                 batch_output = model(batch_input)
## 
##                 # Calculate the loss...
##                 loss = criterion(batch_output, batch_target)
##                 loss = loss.mean()
## 
##                 # Backward pass and optimization...
##                 optimizer.zero_grad()
##                 loss.backward()
##                 if grad_clip != 0.0:
##                     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
##                 optimizer.step()
## 
##             # Reporting...
##             train_loss  [batch_idx] = loss
##             train_sample[batch_idx] = len(batch_input)
## 
##         # Calculate the wegihted mean...
##         train_loss_sum   = torch.dot(train_loss, train_sample)
##         train_sample_sum = train_sample.sum()
## 
##         if uses_fsdp:
##             # Gather training metrics
##             world_train_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(fsdp_world_size) ]
##             world_train_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(fsdp_world_size) ]
##             dist.all_gather(world_train_loss_sum  , train_loss_sum)
##             dist.all_gather(world_train_sample_sum, train_sample_sum)
## 
##             world_train_loss_mean = torch.tensor(world_train_loss_sum).sum() / torch.tensor(world_train_sample_sum).sum()
##         else:
##             world_train_loss_mean = train_loss_sum / train_sample_sum
## 
##         if fsdp_rank == 0:
##             logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {world_train_loss_mean:.8f}")
## 
## 
##         # ___/ VALIDATE \___
##         model.eval()
## 
##         # Fetch batches...
##         batch_validate  = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
##         validate_loss   = torch.zeros(len(batch_validate)).to(device).float()
##         validate_sample = torch.zeros(len(batch_validate)).to(device).float()
##         for batch_idx, batch_entry in batch_validate:
##             # Unpack the batch entry and move them to device...
##             batch_input, batch_target = batch_entry
##             batch_input  = batch_input.to(device, non_blocking=True)
##             batch_target = batch_target.to(device, non_blocking=True)
## 
##             # Forward only...
##             with torch.no_grad():
##                 if uses_mixed_precision:
##                     with torch.cuda.amp.autocast(dtype = torch.bfloat16):
##                         # Forward pass...
##                         batch_output = model(batch_input)
## 
##                         # Calculate the loss...
##                         loss = criterion(batch_output, batch_target)
##                         loss = loss.mean()
##                 else:
##                     # Forward pass...
##                     batch_output = model(batch_input)
## 
##                     # Calculate the loss...
##                     loss = criterion(batch_output, batch_target)
##                     loss = loss.mean()
## 
##             # Reporting...
##             validate_loss  [batch_idx] = loss
##             validate_sample[batch_idx] = len(batch_input)
## 
##         # Calculate the wegihted mean...
##         validate_loss_sum   = torch.dot(validate_loss, validate_sample)
##         validate_sample_sum = validate_sample.sum()
## 
##         if uses_fsdp:
##             # Gather training metrics
##             world_validate_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(fsdp_world_size) ]
##             world_validate_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(fsdp_world_size) ]
##             dist.all_gather(world_validate_loss_sum  , validate_loss_sum)
##             dist.all_gather(world_validate_sample_sum, validate_sample_sum)
## 
##             world_validate_loss_mean = torch.tensor(world_validate_loss_sum).sum() / torch.tensor(world_validate_sample_sum).sum()
##         else:
##             world_validate_loss_mean = validate_loss_sum / validate_sample_sum
## 
##         if fsdp_rank == 0:
##             logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {world_validate_loss_mean:.8f}")
## 
##             # Report the learning rate used in the last optimization...
##             lr_used = optimizer.param_groups[0]['lr']
##             logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")
## 
##         # Update learning rate in the scheduler...
##         scheduler.step()
## 
## 
##         # ___/ SAVE CHECKPOINT??? \___
##         # Conditionally save checkpoint...
##         if world_validate_loss_mean < loss_min:
##             loss_min = world_validate_loss_mean
## 
##             if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
##                 dir_chkpt_timestamp = f"{timestamp}"
##                 if dir_chkpt_prefix is not None: dir_chkpt_timestamp = f"{dir_chkpt_prefix}.{dir_chkpt_timestamp}"
##                 os.makedirs(dir_chkpt_timestamp, exist_ok = True)
## 
##                 dir_chkpt_epoch = f"epoch_{epoch}"
##                 dir_chkpt = os.path.join(dir_chkpt_timestamp, dir_chkpt_epoch)
##                 os.makedirs(dir_chkpt_epoch, exist_ok = True)
## 
##                 fl_chkpt = f"rank_{fsdp_rank}.chkpt"
##                 path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt, fl_chkpt)
##                 save_checkpoint(model,
##                                 optimizer,
##                                 scheduler,
##                                 epoch,
##                                 loss_min,
##                                 path_chkpt,
##                                 process_group    = dist.group.WORLD,
##                                 coordinator_rank = 0)    # ...rank 0 is the coordinator
##                 logger.info(f"MSG (device:{device}) - save {path_chkpt}")
## 
##         if uses_fsdp: dist.barrier()
## 
## except KeyboardInterrupt:
##     print(f"FSDP RANK {fsdp_rank}: Training was interrupted!")
## except Exception as e:
##     print(f"FSDP RANK {fsdp_rank}: Error occurred: {e}")
## finally:
##     # Ensure that the process group is always destroyed
##     if dist.is_initialized():
##        dist.destroy_process_group()
