#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import socket
import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from peaknet.datasets.CXI      import CXIManager, CXIDataset
from peaknet.modeling.att_unet import PeakNet
from peaknet.criterion         import CategoricalFocalLoss
from peaknet.utils             import init_logger, MetaLog, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights

from peaknet.trans import RandomPatch,  \
                          center_crop

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0825_2111_46"
epoch          = None # 57

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

# Set up parameters for an experiment...
path_yaml   = 'train.cxic00318_run0123.autolabel.yaml'
frac_train  = 0.8
size_sample_train = 3000
size_sample_validate = 1500

frac_train    = 0.8
frac_validate = 1.0
dataset_usage = 'train'

uses_skip_connection = True    # Default: True
uses_mixed_precision = True    # Default: True
num_classes          = 3

base_channels = 8
focal_alpha   = 1.0 * 10**(0)
focal_gamma   = 2 * 10**(0)

lr           = 10**(-3.0)
weight_decay = 1e-4
grad_clip    = 1.0

num_gpu     = 1
size_batch  = 10 * num_gpu
num_workers = 5  * num_gpu    # mutiple of size_sample // size_batch
seed        = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            File (Dataset)         : {path_yaml}
            Fraction    (train)    : {frac_train}
            Dataset size           : {size_sample_train}(train), {size_sample_validate}(validate)
            Batch  size            : {size_batch}
            Number of GPUs         : {num_gpu}
            lr                     : {lr}
            weight_decay           : {weight_decay}
            base_channels          : {base_channels}
            focal_alpha            : {focal_alpha}
            focal_gamma            : {focal_gamma}
            uses_skip_connection   : {uses_skip_connection}
            uses_mixed_precision   : {uses_mixed_precision}
            num_workers            : {num_workers}
            continued training???  : from {fl_chkpt_prev}

            """

timestamp = init_logger(returns_timestamp = True)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Load the CXI manager...
cxi_manager = CXIManager(path_yaml)
num_seed_img = len(cxi_manager)

seed_data_idx_list = list(range(num_seed_img))
seed_data_train_idx_list, seed_data_validate_idx_list = split_dataset(seed_data_idx_list, frac_train)

data_train_idx_list = random.choices(seed_data_train_idx_list, k = size_sample_train)
data_validate_idx_list = random.choices(seed_data_validate_idx_list, k = size_sample_validate)

# Set global seed...
set_seed(seed)

# Set up transformation rules
num_patch      = 50
size_patch     = 100
frac_shift_max = 0.2
angle_max      = 360

trans_list = (
    ## RandomRotate(angle_max = angle_max, order = 0),
    ## RandomShift(frac_shift_max, frac_shift_max),
    RandomPatch(num_patch = num_patch, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
)

# Define the training set
dataset_train = CXIDataset(cxi_manager     = cxi_manager,
                           idx_list        = data_train_idx_list,
                           trans_list      = None,
                           normalizes_data = True,
                           reverses_bg     = False)
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                shuffle     = False,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

# Define validation set...
dataset_validate = CXIDataset(cxi_manager     = cxi_manager,
                              idx_list        = data_validate_idx_list,
                              trans_list      = None,
                              normalizes_data = True,
                              reverses_bg     = False)
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers, )


# [[[ MODEL ]]]
# Config the channels in the network...
config_channels = {
    "input_layer"    : (1, 8),
    "encoder_layers" : (
        (8,  16),
        (16, 32),
        (32, 64),
        (64, 128),
    ),
    "fusion_layers" : (
        (128, 64),
        (64,  32),
        (32,  16),
        (16,   8),
    ),
    "head_segmask_layer": (8, 3),
}

# Use the model architecture -- attention u-net...
model = PeakNet( config_channels = config_channels,
                 uses_skip_connection = uses_skip_connection, )

# Initialize weights...
model.apply(init_weights)

# Set device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to gpu(s)...
# [IMPROVE] Replace the following with ddp
## if torch.cuda.device_count() > 1:
##    model = nn.DataParallel(model)
model.to(device)


# [[[ CRITERION ]]]
criterion = CategoricalFocalLoss(alpha                = focal_alpha,
                                 gamma                = focal_gamma,
                                 num_classes          = num_classes,
                                 uses_mixed_precision = uses_mixed_precision)

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

chkpt_saving_period = 5
epoch_unstable_end  = 40
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    train_loss_list = []
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
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

            # Backward pass, optional gradient clipping and optimization...
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

            # Crop the target mask as u-net might change the output dimension...
            size_y, size_x = batch_output.shape[-2:]
            batch_target_crop = center_crop(batch_target, size_y, size_x)

            # Calculate the loss...
            loss = criterion(batch_output, batch_target_crop)
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass, optional gradient clipping and optimization...
            optimizer.zero_grad()
            loss.backward()
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Reporting...
        train_loss_list.append(loss.item())

    train_loss_mean = np.mean(train_loss_list)
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    validate_loss_list = []
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
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
        validate_loss_list.append(loss.item())

    validate_loss_mean = np.mean(validate_loss_list)
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
