#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn            as nn
import torch.nn.functional as F

import numpy as np
import math
import yaml
import h5py
import time

from peaknet.datasets.SFX_Inference import SFXInferenceDataset

# Suppose we are processing event_num events...
event_num  = 100
event_list = torch.randint(0, event_num + 1, (event_num,)).tolist()

# ___/ Psana \___
exp           = 'mfxx49820'
run           = 107
img_load_mode = 'calib'
access_mode   = 'idx'
detector_name = 'MfxEndstation.0:Epix10ka2M.0'
## sfx_inference_dataset = SFXInferenceDataset(exp, run, access_mode, detector_name, img_load_mode)
sfx_inference_dataset = SFXInferenceDataset(exp, run, access_mode, detector_name, img_load_mode, event_list = event_list)

# ___/ Data loader \___
size_batch  = 10
num_workers = 5
dataloader = torch.utils.data.DataLoader( sfx_inference_dataset,
                                          shuffle     = False,
                                          pin_memory  = True,
                                          batch_size  = size_batch,
                                          num_workers = num_workers, )


# ___/ Peak finder \___
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

## # Load CONFIG from YAML...
## path_yaml = "experiments/yaml/opt_cxi.00.yaml"
## pf = PeakFinder(path_chkpt = "experiments/chkpts/opt_cxi.00.2023_1108_2307_12.epoch_96.chkpt", path_yaml_config = path_yaml)

peak_list = []
batch_times = []
dataloader_iter = iter(dataloader)
idx = 0
while True:
    try:
        t_s = time.monotonic()  # Start timing before retrieving the next batch
        batch_data, batch_metadata = next(dataloader_iter)
        t_e = time.monotonic()  # Stop timing after the batch is retrieved

        # Calculate and store the loading time for this batch
        loading_time_ms = (t_e - t_s) * 1e3  # Convert to milliseconds

        print(f"Batch {idx}: {loading_time_ms} ms per {len(batch_data)} img...")
        idx += 1

        batch_times.append(loading_time_ms)

    except StopIteration:
        break

# After the loop, you can process batch_times list as needed
# For example, print average loading time
if batch_times:
    average_load_time = sum(batch_times) / len(batch_times)
    print(f"Average load time per batch: {average_load_time:.2f} ms")
else:
    print("No batches were processed.")
