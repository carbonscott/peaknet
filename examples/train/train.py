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

import torch
import torch.nn as nn
import torch.optim as optim
## from torch.optim.lr_scheduler import ReduceLROnPlateau

# Libraries used for Distributed Data Parallel (DDP)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from peaknet.datasets.peaknet_dataset      import PeakNetDatasetLoader
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet, SegHeadConfig
from peaknet.modeling.convnextv2_encoder   import ConvNextV2BackboneConfig
from peaknet.modeling.bifpn_config         import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from peaknet.criterion                     import CategoricalFocalLoss
from peaknet.utils                         import init_logger, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights
from peaknet.lr_scheduler                  import CosineLRScheduler

from peaknet.trans import RandomShift,  \
                          RandomRotate, \
                          RandomPatch,  \
                          PadResize

torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16
                                            # Reference: https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4

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
drc_chkpt           = chkpt_config.get("directory")
path_pretrain_chkpt = chkpt_config.get("pretrain")
fl_chkpt_prefix     = chkpt_config.get("filename_prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")
epoch_unstable_end  = chkpt_config.get("epoch_unstable_end")

# ...Dataset
dataset_config    = config.get("dataset")
path_train_csv    = dataset_config.get("path_train")
path_validate_csv = dataset_config.get("path_validate")
## cxi_key           = dataset_config.get("cxi_key")
## size_sample       = dataset_config.get("sample_size")
## frac_train        = dataset_config.get("frac_train")
## frac_validate     = dataset_config.get("frac_validate")
size_batch        = dataset_config.get("batch_size")
num_workers       = dataset_config.get("num_workers")

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


# ...DDP
ddp_config             = config.get("ddp")
ddp_backend            = ddp_config.get("backend")
uses_unique_world_seed = ddp_config.get("uses_unique_world_seed")

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


# [[[ ERROR HANDLING ]]]
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# [[[ DDP INIT ]]]
# Initialize distributed environment
uses_ddp = int(os.environ.get("RANK", -1)) != -1
if uses_ddp:
    ddp_rank       = int(os.environ["RANK"      ])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=ddp_backend,
                            rank = ddp_rank,
                            world_size = ddp_world_size,
                            init_method = "env://",)
    print(f"RANK:{ddp_rank},LOCAL_RANK:{ddp_local_rank},WORLD_SIZE:{ddp_world_size}")
else:
    ddp_rank       = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    print(f"NO DDP is used.  RANK:{ddp_rank},LOCAL_RANK:{ddp_local_rank},WORLD_SIZE:{ddp_world_size}")

# Set up GPU device
device = f'cuda:{ddp_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = ddp_rank if uses_unique_world_seed else 0


# [[[ USE YAML CONFIG TO INITIALIZE HYPERPARAMETERS ]]]
# Set Seed
base_seed   = 0
world_seed  = base_seed + seed_offset

if ddp_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)


# [[[ DATASET ]]]
# Set global seed...
set_seed(world_seed)

# Set up transformation rules
num_patch        = 250
size_patch       = 20
frac_shift_max   = 0.1
angle_max        = 360
H_input, W_input = 1024, 1024

trans_list = (
    PadResize(H_input, W_input),
    RandomRotate(angle_max = angle_max, order = 0),
    RandomShift (frac_shift_max, frac_shift_max),
    RandomPatch (num_patch = num_patch, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
)

# Load raw data...
set_seed(base_seed)    # ...Make sure data split is consistent across devices
dataset_train    = PeakNetDatasetLoader(path_csv = path_train_csv   , trans_list = trans_list)
dataset_validate = PeakNetDatasetLoader(path_csv = path_validate_csv, trans_list = trans_list)

# Define the training set
sampler_train    = torch.utils.data.DistributedSampler(dataset_train) if uses_ddp else None
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                sampler     = sampler_train,
                                                shuffle     = False,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

# Define validation set...
sampler_validate    = torch.utils.data.DistributedSampler(dataset_validate, shuffle=False) if uses_ddp else None
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   sampler     = sampler_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )


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
model.to(device)
if ddp_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# Initialized by the main rank and weights will be broadcast by DDP wrapper
if ddp_rank == 0:
    if uses_random_weights:
        # Use random weights...
        model.apply(init_weights)
    ## else:
    ##     # Use pretrained weights...
    ##     pretrain_chkpt = torch.load(path_pretrain_chkpt)
    ##     model.backbone.encoder.load_state_dict(pretrain_chkpt, strict = False)

# Freeze the backbone???
if freezes_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False

model.float()


# [[[ CRITERION ]]]
criterion = CategoricalFocalLoss(alpha       = focal_alpha,
                                 gamma       = focal_gamma,
                                 num_classes = seghead_num_classes,)

# [[[ OPTIMIZER ]]]
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
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


# [[[ TRAIN LOOP ]]]
# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model,
                                          optimizer,
                                          scheduler if uses_prev_scheduler else None,
                                          path_chkpt_prev,
                                          device)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

# Uses mixed precision???
if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

# Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# Wrapping the model in DDP...
if uses_ddp:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using DDP...
    model = DDP(model, device_ids = [ddp_local_rank], find_unused_parameters=True)

if ddp_rank == 0:
    print(f"Current timestamp: {timestamp}")

try:
    for epoch in tqdm.tqdm(range(max_epochs)):
        epoch += epoch_min

        if uses_ddp:
            # Shuffle the training examples...
            sampler_train.set_epoch(epoch)

        # ___/ TRAIN \___
        # Turn on training related components in the model...
        model.train()

        # Fetch batches...
        batch_train  = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
        train_loss   = torch.zeros(len(batch_train)).to(device).float()
        train_sample = torch.zeros(len(batch_train)).to(device).float()
        for batch_idx, batch_entry in batch_train:
            # Unpack the batch entry and move them to device...
            batch_input, batch_target = batch_entry
            batch_input  = batch_input.to(device, dtype = torch.float)
            batch_target = batch_target.to(device, dtype = torch.float)

            # Forward, backward and update...
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    # Forward pass...
                    batch_output = model(batch_input)

                    ## # Crop the target mask as u-net might change the output dimension...
                    ## size_y, size_x = batch_output.shape[-2:]
                    ## batch_target_crop = center_crop(batch_target, size_y, size_x)

                    # Calculate the loss...
                    ## loss = criterion(batch_output, batch_target_crop)
                    loss = criterion(batch_output, batch_target)
                    loss = loss.mean()

                # Backward pass and optimization...
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass...
                batch_output = model(batch_input)

                ## # Crop the target mask even if the output dimension size is changed...
                ## size_y, size_x = batch_output.shape[-2:]
                ## batch_target_crop = center_crop(batch_target, size_y, size_x)

                # Calculate the loss...
                ## loss = criterion(batch_output, batch_target_crop)
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()

                # Backward pass and optimization...
                optimizer.zero_grad()
                loss.backward()
                if grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Reporting...
            train_loss  [batch_idx] = loss
            train_sample[batch_idx] = len(batch_input)

        # Calculate the wegihted mean...
        train_loss_sum   = torch.dot(train_loss, train_sample)
        train_sample_sum = train_sample.sum()

        if uses_ddp:
            # Gather training metrics
            world_train_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
            world_train_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
            dist.all_gather(world_train_loss_sum  , train_loss_sum)
            dist.all_gather(world_train_sample_sum, train_sample_sum)

            world_train_loss_mean = torch.tensor(world_train_loss_sum).sum() / torch.tensor(world_train_sample_sum).sum()
        else:
            world_train_loss_mean = train_loss_sum / train_sample_sum

        if ddp_rank == 0:
            logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {world_train_loss_mean:.8f}")


        # ___/ VALIDATE \___
        model.eval()

        # Fetch batches...
        batch_validate  = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
        validate_loss   = torch.zeros(len(batch_validate)).to(device).float()
        validate_sample = torch.zeros(len(batch_validate)).to(device).float()
        for batch_idx, batch_entry in batch_validate:
            # Unpack the batch entry and move them to device...
            batch_input, batch_target = batch_entry
            batch_input  = batch_input.to(device, dtype = torch.float)
            batch_target = batch_target.to(device, dtype = torch.float)

            # Forward only...
            with torch.no_grad():
                if uses_mixed_precision:
                    with torch.cuda.amp.autocast(dtype = torch.float16):
                        # Forward pass...
                        batch_output = model(batch_input)

                        ## # Crop the target mask as u-net might change the output dimension...
                        ## size_y, size_x = batch_output.shape[-2:]
                        ## batch_target_crop = center_crop(batch_target, size_y, size_x)

                        # Calculate the loss...
                        ## loss = criterion(batch_output, batch_target_crop)
                        loss = criterion(batch_output, batch_target)
                        loss = loss.mean()
                else:
                    # Forward pass...
                    batch_output = model(batch_input)

                    ## # Crop the target mask as u-net might change the output dimension...
                    ## size_y, size_x = batch_output.shape[-2:]
                    ## batch_target_crop = center_crop(batch_target, size_y, size_x)

                    # Calculate the loss...
                    ## loss = criterion(batch_output, batch_target_crop)
                    loss = criterion(batch_output, batch_target)
                    loss = loss.mean()

            # Reporting...
            validate_loss  [batch_idx] = loss
            validate_sample[batch_idx] = len(batch_input)

        # Calculate the wegihted mean...
        validate_loss_sum   = torch.dot(validate_loss, validate_sample)
        validate_sample_sum = validate_sample.sum()

        if uses_ddp:
            # Gather training metrics
            world_validate_loss_sum   = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
            world_validate_sample_sum = [ torch.tensor(0.0).to(device).float() for _ in range(ddp_world_size) ]
            dist.all_gather(world_validate_loss_sum  , validate_loss_sum)
            dist.all_gather(world_validate_sample_sum, validate_sample_sum)

            world_validate_loss_mean = torch.tensor(world_validate_loss_sum).sum() / torch.tensor(world_validate_sample_sum).sum()
        else:
            world_validate_loss_mean = validate_loss_sum / validate_sample_sum

        if ddp_rank == 0:
            logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {world_validate_loss_mean:.8f}")

            # Report the learning rate used in the last optimization...
            lr_used = optimizer.param_groups[0]['lr']
            logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")

        # Update learning rate in the scheduler...
        scheduler.step()


        # ___/ SAVE CHECKPOINT??? \___
        if ddp_rank == 0:
            if world_validate_loss_mean < loss_min:
                loss_min = world_validate_loss_mean

                if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
                    fl_chkpt = f"{timestamp}.epoch_{epoch}.chkpt"
                    if fl_chkpt_prefix is not None: fl_chkpt = f"{fl_chkpt_prefix}.{fl_chkpt}"
                    path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
                    save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
                    logger.info(f"MSG (device:{device}) - save {path_chkpt}")

        if uses_ddp: dist.barrier()

except KeyboardInterrupt:
    print(f"DDP RANK {ddp_rank}: Training was interrupted!")
except Exception as e:
    print(f"DDP RANK {ddp_rank}: Error occurred: {e}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
       dist.destroy_process_group()