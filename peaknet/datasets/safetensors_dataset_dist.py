import csv

from safetensors.torch import load_file
from collections import deque

import torch
import torch.distributed as dist

from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List

from ..utils_fsdp import broadcast_dict
from ..perf import Timer

@dataclass
class DistributedSegmentedDatasetConfig:
    full_dataset    : List
    micro_batch_size: int
    world_size      : int

class DistributedSegmentedDataset(Dataset):
    def __init__(self, config: DistributedSegmentedDatasetConfig):
        self.full_dataset          = config.full_dataset
        self.micro_batch_size      = config.micro_batch_size
        self.world_size            = config.world_size
        self.total_size            = len(config.full_dataset)
        self.segment_size          = self.micro_batch_size * self.world_size
        self.total_segments        = (self.total_size + self.segment_size - 1) // self.segment_size

        self.start_idx             = 0
        self.end_idx               = 0
        self.current_segment_index = 0
        self.update_segment_indices()

    def update_segment_indices(self):
        # Compute the global start and end indices of the current segment
        segment_global_start = self.current_segment_index * self.segment_size
        self.start_idx       = segment_global_start
        self.end_idx         = min(segment_global_start + self.segment_size, self.total_size)

    def set_segment(self, segment_idx):
        if segment_idx < 0 or segment_idx >= self.total_segments:
            warnings.warn(f"Segment index {segment_idx} is out of bounds. Resetting to segment 0.")
            segment_idx = 0
        self.current_segment_index = segment_idx
        self.update_segment_indices()

    def __len__(self):
        return min(self.segment_size, self.total_size - self.start_idx)

    def __getitem__(self, idx):
        # Map the local index to the correct global index within the segment
        global_idx = self.start_idx + idx
        if global_idx >= self.total_size:
            raise IndexError("Index out of range for the current segment")
        return self.full_dataset[global_idx]

    def save_checkpoint(self, checkpoint_path, rank):
        if rank == 0:
            checkpoint = {
                'current_segment_index': self.current_segment_index,
                'last_world_size'      : self.world_size,
                'last_micro_batch_size': self.micro_batch_size
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
            old_world_size = checkpoint['last_world_size']
            current_world_size = self.world_size

            # Temporarily set the world size to the old one for correct segment calculation
            self.world_size = old_world_size

            if old_world_size != current_world_size:
                warnings.warn(f"World size at checkpoint ({old_world_size}) does not match current world size ({current_world_size}).")

            if 'last_micro_batch_size' in checkpoint and checkpoint['last_micro_batch_size'] != self.micro_batch_size:
                warnings.warn(f"micro_batch_size has been changed from {checkpoint['last_micro_batch_size']} to {self.micro_batch_size}. Resetting to {checkpoint['last_micro_batch_size']}.")
                self.micro_batch_size = checkpoint['last_micro_batch_size']

            self.segment_size = self.micro_batch_size * old_world_size

            # Use the old world size for setting the segment
            self.set_segment(checkpoint['current_segment_index'])

            # Restore the current world size and adjust dataset properties accordingly
            self.world_size = current_world_size
            self.segment_size = self.micro_batch_size * current_world_size

    if dist.is_initialized():
        dist.barrier()


class PeakNetDataset(Dataset):
    """
    This class only loads events sequentially saved in a list of safetensors files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, path_csv, trans_list = None, cache_size = 10, perfs_runtime = False):
        super().__init__()

        self.trans_list    = trans_list
        self.cache_size    = cache_size
        self.perfs_runtime = perfs_runtime

        # Importing data from csv...
        self.file_paths     = []
        self.file_index_map = []
        with open(path_csv, 'r') as fh:
            lines = list(csv.reader(fh))
        self.file_paths = [ line[0] for line in lines ]

        with Timer(tag = f"Build index map", is_on = self.perfs_runtime):
            # Open all safetensors files and track their status...
            for file_idx, file_path in enumerate(self.file_paths):
                data = load_file(file_path, device = 'cpu')
                num_items = data["image"].shape[0]
                for item_idx in range(num_items):
                    self.file_index_map.append((file_idx, item_idx))

        self.cache = {}
        self.cache_order = deque([], maxlen=cache_size)

        return None


    def get_img(self, idx):
        file_idx, item_idx = self.file_index_map[idx]
        file_path = self.file_paths[file_idx]

        # Load file into cache if not already present
        if file_path not in self.cache:
            self._load_file_to_cache(file_path)

        # Retrieve specific image and label tensors
        data = self.cache[file_path]
        image = data["image"][item_idx]
        label = data["label"][item_idx]

        return image, label


    def __getitem__(self, idx):
        img, label = self.get_img(idx)    # (C, H, W)

        # Apply transformation to image and label at the same time...
        if self.trans_list is not None:
            data = torch.cat([img[None,], label[None,]], dim = 0)    # (2*B=1, C, H, W)
            for enum_idx, trans in enumerate(self.trans_list):
                with Timer(tag = f"Transform method {enum_idx:d}", is_on = self.perfs_runtime):
                    data = trans(data)

            img   = data[0]    # (1, C, H, W)
            label = data[1]    # (1, C, H, W)

        # Binary the label...
        label = label > 0

        return img, label


    def _load_file_to_cache(self, file_path):
        # Remove the oldest entry from the cache if it exceeds the specified cache size
        if file_path not in self.cache and len(self.cache_order) == self.cache_order.maxlen:
            oldest_file = self.cache_order.popleft()
            del self.cache[oldest_file]

        if file_path not in self.cache:
            self.cache[file_path] = load_file(file_path, device='cpu')
            self.cache_order.append(file_path)


    def __len__(self):
        return len(self.file_index_map)
