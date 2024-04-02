import io
import csv
import requests
import numpy as np

import torch
import torch.distributed as dist

from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List

from ..utils_fsdp import broadcast_dict
from ..perf import Timer

@dataclass
class RemoteDistributedSegmentedDatasetConfig:
    """Configuration for the Remote Distributed Segmented Dataset.

    Attributes:
        full_dataset (List): The complete dataset details to be segmented and distributed.
        micro_batch_size_per_rank (int): The size of each micro-batch to be processed by each rank.
        world_size (int): Total number of distributed processes (ranks) in use.
        transforms (List): A list of transformations to apply to each data item.
        is_perf (bool): Flag to enable performance timing for transformations. Default is False.
        url (str): URL of the server to fetch data from. Defaults to 'http://localhost:5001'.
    """
    full_dataset             : List
    micro_batch_size_per_rank: int
    world_size               : int
    transforms               : List
    is_perf                  : bool = False
    url                      : str = 'http://localhost:5001'

class RemoteDistributedSegmentedDataset(Dataset):
    """A dataset class designed for fetching and distributing segments of data
    in a distributed training environment.

    This class allows for efficient data loading and processing across multiple
    distributed processes.
    """
    def __init__(self, config: RemoteDistributedSegmentedDatasetConfig):
        self.full_dataset              = config.full_dataset
        self.micro_batch_size_per_rank = config.micro_batch_size_per_rank
        self.world_size                = config.world_size
        self.url                       = config.url
        self.transforms                = config.transforms
        self.is_perf                   = config.is_perf
        self.total_size                = len(config.full_dataset)

        self.start_idx  = 0
        self.end_idx    = self.calculate_end_idx()

    def calculate_end_idx(self):
        # Calculate and return the end index for the current dataset segment.
        return min(self.start_idx + self.micro_batch_size_per_rank * self.world_size, self.total_size)

    def set_start_idx(self, start_idx):
        self.start_idx = start_idx
        self.end_idx = self.calculate_end_idx()

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # Ensure idx is within the bounds of the current segment
        if idx >= (self.end_idx - self.start_idx):
            raise IndexError("Index out of range for the current segment")

        # Map the local index to the correct global index within the segment
        global_idx = self.start_idx + idx

        # Obtain dataset handle
        exp, run, access_mode, detector_name, event = self.full_dataset[global_idx]

        # Fetch event
        image = self.fetch_event(self.url, exp, run, access_mode, detector_name, event)    # psana image: (H, W)

        # Apply transforms
        image_tensor = None
        if image is not None and self.transforms is not None:
            image_tensor = torch.from_numpy(image[None, None])    # (B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = None, is_on = self.is_perf):
                    image_tensor = trans(image_tensor)

        return image_tensor[0]    # Dataloader only wants data with shape of (C, H, W)

    def save_checkpoint(self, checkpoint_path, rank):
        if rank == 0:
            checkpoint = {
                'end_idx'       : self.end_idx,
                'micro_batch_size_per_rank': self.micro_batch_size_per_rank
            }
            torch.save(checkpoint, checkpoint_path)
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint_and_broadcast(self, checkpoint_path, rank, device):
        checkpoint = None
        if rank == 0:
            checkpoint = torch.load(checkpoint_path)
        checkpoint = broadcast_dict(checkpoint, src=0, device=device)

        if checkpoint:
            self.set_start_idx(checkpoint.get('end_idx', 0))
            if 'micro_batch_size_per_rank' in checkpoint and checkpoint['micro_batch_size_per_rank'] != self.micro_batch_size_per_rank:
                warnings.warn(f"micro_batch_size_per_rank has been changed from {checkpoint['micro_batch_size_per_rank']} to {self.micro_batch_size_per_rank}. Resetting to {checkpoint['micro_batch_size_per_rank']}.")
                self.micro_batch_size_per_rank = checkpoint['micro_batch_size_per_rank']

        if dist.is_initialized():
            dist.barrier()

    def fetch_event(self, url, exp, run, access_mode, detector_name, event):
        payload = {
            'exp'          : exp,
            'run'          : run,
            'access_mode'  : access_mode,
            'detector_name': detector_name,
            'event'        : event
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            with io.BytesIO(response.content) as buffer:
                data_array = np.load(buffer)
            return data_array
        else:
            print(f"Failed to fetch data for event {event}: {response.status_code}")
            return None
