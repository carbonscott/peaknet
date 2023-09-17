#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import socket
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from peaknet.datasets.SFX           import SFXMulticlassDataset
from peaknet.modeling.reg_bifpn_net import PeakNet
from peaknet.criterion              import CategoricalFocalLoss
from peaknet.utils                  import init_logger, MetaLog, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights

from peaknet.config import CONFIG

from peaknet.trans import RandomShift,  \
                          RandomRotate, \
                          RandomPatch,  \
                          center_crop

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

# [[[ CONFIG ]]]
with CONFIG.enable_auto_create():
    CONFIG.BACKBONE.FREEZE_ALL = True

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0505_1249_26"
epoch          = None # 21

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

# Set up parameters for an experiment...
drc_dataset  = 'datasets'
fl_dataset   = 'mfx13016_0028_N68+mfxp22820_0013_N37+mfx13016_0028_N63_low_photon.68v37v30.data.label_corrected.npy'       # size_sample = 3000
path_dataset = os.path.join(drc_dataset, fl_dataset)

size_sample   = 3000
frac_train    = 0.8
frac_validate = 1.0
dataset_usage = 'train'

uses_mixed_precision = True

base_channels = 8
focal_alpha   = 1.0 * 10**(0)
focal_gamma   = 2 * 10**(0)

lr           = 3e-4
weight_decay = 1e-4

num_gpu     = 1
size_batch  = 5
num_workers = 5    # mutiple of size_sample // size_batch
seed        = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            File (Dataset)         : {path_dataset}
            Fraction    (train)    : {frac_train}
            Dataset size           : {size_sample}
            Batch  size            : {size_batch}
            Number of GPUs         : {num_gpu}
            lr                     : {lr}
            weight_decay           : {weight_decay}
            base_channels          : {base_channels}
            focal_alpha            : {focal_alpha}
            focal_gamma            : {focal_gamma}
            uses_mixed_precision   : {uses_mixed_precision}
            num_workers            : {num_workers}
            continued training???  : from {fl_chkpt_prev}

            """

timestamp = init_logger(returns_timestamp = True)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Load raw data...
dataset_list = np.load(path_dataset, allow_pickle = True)
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = seed)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = seed)

# Set global seed...
set_seed(seed)

# Set up transformation rules
num_patch      = 50
size_patch     = 100
frac_shift_max = 0.2
angle_max      = 360

trans_list = (
    RandomRotate(angle_max = angle_max, order = 0),
    RandomShift(frac_shift_max, frac_shift_max),
    RandomPatch(num_patch = num_patch, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
)

# Define the training set
dataset_train = SFXMulticlassDataset( data_list          = data_train,
                                      size_sample        = size_sample,
                                      trans_list         = trans_list,
                                      normalizes_data    = True,
                                      prints_cache_state = True,
                                      mpi_comm           = None, )
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                shuffle     = False,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

# Define validation set...
dataset_validate = SFXMulticlassDataset( data_list          = data_validate,
                                         size_sample        = size_sample//2,
                                         trans_list         = trans_list,
                                         normalizes_data    = True,
                                         prints_cache_state = True,
                                         mpi_comm           = None, )
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )


# [[[ MODEL ]]]
# Use the model architecture -- attention u-net...
model = PeakNet(num_blocks = 1, num_features = 64)
print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# Use pretrained weights...
path_chkpt = os.path.join("pretrained_chkpts", "transfered_weights.resnet50.chkpt")
chkpt      = torch.load(path_chkpt)
model.backbone.encoder.load_state_dict(chkpt, strict = False)

# Set device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to gpu(s)...
# [IMPROVE] Replace the following with ddp
## if torch.cuda.device_count() > 1:
##    model = nn.DataParallel(model)
model.to(device)

# Freeze the backbone???
if CONFIG.BACKBONE.FREEZE_ALL:
    for param in model.backbone.parameters():
        param.requires_grad = False


# [[[ CRITERION ]]]
criterion = CategoricalFocalLoss(alpha = focal_alpha, gamma = focal_gamma)

# [[[ OPTIMIZER ]]]
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
                                         factor         = 2e-1,
                                         patience       = 10,
                                         threshold      = 1e-4,
                                         threshold_mode ='rel',
                                         verbose        = True)


# [[[ TRAIN LOOP ]]]
max_epochs = 3000

# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

print(f"Current timestamp: {timestamp}")

chkpt_saving_period = 1
epoch_unstable_end  = -1
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
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

                # Crop the target mask as u-net might change the output dimension...
                size_y, size_x = batch_output.shape[-2:]
                batch_target_crop = center_crop(batch_target, size_y, size_x)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target_crop)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass...
            batch_output = model(batch_input)

            # Crop the target mask as u-net might change the output dimension...
            size_y, size_x = batch_output.shape[-2:]
            batch_target_crop = center_crop(batch_target, size_y, size_x)

            # Calculate the loss...
            loss = criterion(batch_output, batch_target_crop)
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting...
        train_loss  [batch_idx] = loss
        train_sample[batch_idx] = len(batch_input)

    # Calculate the wegihted mean...
    train_loss_sum   = torch.dot(train_loss, train_sample)
    train_sample_sum = train_sample.sum()
    train_loss_mean  = train_loss_sum / train_sample_sum

    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
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

                    # Crop the target mask as u-net might change the output dimension...
                    size_y, size_x = batch_output.shape[-2:]
                    batch_target_crop = center_crop(batch_target, size_y, size_x)

                    # Calculate the loss...
                    loss = criterion(batch_output, batch_target_crop)
                    loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus
            else:
                # Forward pass...
                batch_output = model(batch_input)

                # Crop the target mask as u-net might change the output dimension...
                size_y, size_x = batch_output.shape[-2:]
                batch_target_crop = center_crop(batch_target, size_y, size_x)

                # Calculate the loss...
                loss = criterion(batch_output, batch_target_crop)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

        # Reporting...
        validate_loss  [batch_idx] = loss
        validate_sample[batch_idx] = len(batch_input)

    # Calculate the wegihted mean...
    validate_loss_sum   = torch.dot(validate_loss, validate_sample)
    validate_sample_sum = validate_sample.sum()
    validate_loss_mean  = validate_loss_sum / validate_sample_sum

    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {validate_loss_mean:.8f}")

    # Report the learning rate used in the last optimization...
    lr_used = optimizer.param_groups[0]['lr']
    logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")

    # Update learning rate in the scheduler...
    scheduler.step(validate_loss_mean)


    # ___/ SAVE CHECKPOINT??? \___
    if validate_loss_mean < loss_min:
        loss_min = validate_loss_mean

        if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
            fl_chkpt   = f"{timestamp}.epoch_{epoch}.chkpt"
            path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
            save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
            logger.info(f"MSG (device:{device}) - save {path_chkpt}")
