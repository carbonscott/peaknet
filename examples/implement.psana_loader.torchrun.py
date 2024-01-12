#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
psana to cheetah tile is required in the cxi file.
"""

import os
import torch
import random
import numpy as np
import h5py
import time
import pickle
import yaml
import signal
import argparse

from peaknet.datasets.SFX_Inference import SFXInferenceDataset
from peaknet.plugins                import PsanaImg
from peaknet.utils                  import split_list_into_chunk, set_seed

# Libraries used for Distributed Data Parallel (DDP)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# [[[ ERROR HANDLING ]]]
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# [[[ DDP INIT ]]]
# Initialize distributed environment
## ddp_backend = 'gloo'
ddp_backend = 'nccl'
uses_unique_world_seed = True
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

# Set global seed...
set_seed(world_seed)

## # Set up MPI
## from mpi4py import MPI
## mpi_comm = MPI.COMM_WORLD
## mpi_rank = mpi_comm.Get_rank()
## mpi_size = mpi_comm.Get_size()
## mpi_data_tag = 11


# [[[ ARG PARSE ]]]
parser = argparse.ArgumentParser(description='Process a yaml file.')
parser.add_argument('yaml', help='The input yaml file.')
args = parser.parse_args()

# [[[ Configure ]]]
fl_yaml = args.yaml
basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

# Load the YAML file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# Access the values
# ___/ PeakNet model \___
path_chkpt           = config['path_chkpt']
path_yaml            = config['path_yaml' ]

# ___/ Experimental data \___
# Psana...
exp                  = config['exp'          ]
run                  = config['run'          ]
img_load_mode        = config['img_load_mode']
access_mode          = config['access_mode'  ]
detector_name        = config['detector_name']
photon_energy        = config['photon_energy']
encoder_value        = config['encoder_value']

# Data range...
event_min            = config['event_min']
event_max            = config['event_max']

# ___/ Peak finding \___
uses_int_coords        = config['uses_int_coords']
mask_custom_npy        = config['mask_custom_npy']
max_num_peak           = config['max_num_peak']
num_model_on_gpu       = config['num_model_on_gpu']
returns_prediction_map = config['returns_prediction_map']
dataloader_batch_size  = config["dataloader_batch_size"]
dataloader_num_workers = config["dataloader_num_workers"]

## if mpi_size < num_model_on_gpu: num_model_on_gpu = mpi_size
gpu_worker_ranks = list(range(num_model_on_gpu))    # Manager rank (0) is on gpu by default


# ___/ Output \___
dir_results           = config["dir_results"]
pf_tag                = config["pf_tag"]
ddp_batch_size        = config["mpi_batch_size"]

basename_out  = f"{basename_yaml}"
basename_out += pf_tag

# ___/ Misc \___
path_cheetah_geom    = config["path_cheetah_geom"]


# MPI GPU
## device = 'cpu'
mask_custom = None

try:
    # [[[ Psana ]]]
    # Set up experiments...
    psana_img = PsanaImg(exp, run, access_mode, detector_name)

    if event_min is None: event_min = 0
    if event_max is None: event_max = len(psana_img.timestamps)

    def find_peak_per_rank(events_per_rank, ddp_batch_idx, ddp_rank):
        ''' Assume all variables are global.  This function is only to wrap a
            process to save lines of codes.
        '''
        # Establish dataloader...
        sfx_inference_dataset = SFXInferenceDataset(exp, run, access_mode, detector_name, img_load_mode, events_per_rank)

        sampler    = torch.utils.data.DistributedSampler(sfx_inference_dataset)
        dataloader = torch.utils.data.DataLoader( sfx_inference_dataset,
                                                  sampler     = sampler,
                                                  shuffle     = False,
                                                  pin_memory  = True,
                                                  batch_size  = dataloader_batch_size,
                                                  num_workers = dataloader_num_workers, )

        event_filtered_list = []
        for batch_enum_idx, (batch_data, batch_metadata) in enumerate(dataloader):
            t_s = time.monotonic()

            # Apply custom mask...
            if mask_custom is not None: batch_data = apply_mask(batch_data, mask_custom, mask_value = 0)

            # Package the image into tensor...
            batch_data = batch_data.type(dtype=torch.float).to(device)


            batch_data     = batch_data.flatten(start_dim = 0, end_dim = 1)[:, None]
            batch_metadata = batch_metadata.flatten(start_dim = 0, end_dim = 1)
            batch_data     = batch_data.to(device)
            batch_metadata = batch_metadata.to(device)

            t_e = time.monotonic()
            t_d = t_e - t_s
            freq = dataloader_batch_size / t_d
            ## print(f"MPI batch {ddp_batch_idx:02d}, mini batch {batch_enum_idx:06d} (ddp_rank {ddp_rank:02d}), time: {freq:.2f} Hz...")
            print(f"DDP batch {ddp_batch_idx:02d}, mini batch {batch_enum_idx:06d} (ddp_rank {ddp_rank:02d}), events: {len(batch_metadata)}, time: {freq:.2f} Hz...")

        return event_filtered_list

    # Split all events into DDP Chunks...
    events = range(event_min, event_max)
    batch_events = split_list_into_chunk(events, max_num_chunk = ddp_batch_size)

    for batch_idx, events in enumerate(batch_events):
        events_in_chunk = split_list_into_chunk(events, max_num_chunk = num_model_on_gpu)

        # DDP manager
        # - Find peaks
        # - Collect peaks from other workers
        if ddp_rank == 0:
            if batch_idx == 0: event_filtered_list = []

            # Find peaks
            events_per_rank = events_in_chunk[ddp_rank]
            event_filtered_list_per_rank = find_peak_per_rank(events_per_rank, batch_idx, ddp_rank)

            event_filtered_list_per_rank = torch.tensor(event_filtered_list_per_rank, dtype = torch.float32, requires_grad = False).to(device)
            event_filtered_list.extend(event_filtered_list_per_rank)

            # Collect peaks from other workers
            for i in range(1, num_model_on_gpu, 1):
                ## data_received = mpi_comm.recv(source = i, tag = mpi_data_tag)

                # Receive the size of the incoming data...
                size_data = torch.tensor([0], dtype = torch.int64, requires_grad = False).to(device)
                dist.recv(size_data, src = i)

                # Receive the actual data...
                data_recv = torch.empty(size_data.item(), dtype = torch.float32, requires_grad = False)
                dist.recv(data_recv, src = i)

                event_filtered_list.extend(data_recv)

        if ddp_rank != 0:
            # Find peaks
            events_per_rank = events_in_chunk[ddp_rank] if len(events_in_chunk) > ddp_rank else []
            event_filtered_list_per_rank = find_peak_per_rank(events_per_rank, batch_idx, ddp_rank)

            # Send peaks to the manager
            data_to_send = torch.tensor(event_filtered_list_per_rank, dtype = torch.float32, requires_grad = False).to(device)
            ## mpi_comm.send(data_to_send, dest = 0, tag = mpi_data_tag)
            size_data = torch.tensor([data_to_send.numel()], dtype = torch.int64, requires_grad = False).to(device)
            dist.send(size_data, dst = 0)

            dist.send(data_to_send, dst = 0)


    # GPU nodes are done with peak finding...
    dist.barrier()


except KeyboardInterrupt:
    print(f"DDP RANK {ddp_rank}: Training was interrupted!")
except Exception as e:
    print(f"DDP RANK {ddp_rank}: Error occurred: {e}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.destroy_process_group()
