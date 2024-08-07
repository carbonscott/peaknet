import csv
import zarr

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List, Tuple

from math import ceil

from collections import OrderedDict

from ..perf import Timer
from ..tensor_transforms import InstanceNorm, NoTransform

from itertools import islice

import logging
logger = logging.getLogger(__name__)

@dataclass
class SegmentedPeakNetDatasetConfig:
    """
    Attributes:
        path_csv (str)      : CSV file to configure the dataset list.
        seg_size (int)      : The segment size by each rank.
        dist_world_size (int)    : Total number of distributed process (ranks) in use.
        transforms (List)   : A list of transformations to apply to each data item.
        buffer_size (str)   : The number of data buffered in memory.
        dtype (torch.dtype) : The torch.dtype for representing underlying data.
        perfs_runtime (bool): Whether to measure performance.
    """
    path_csv       : str
    seg_size       : int
    transforms     : List
    buffer_size    : int
    dist_rank      : int
    dist_world_size: int
    device         : str
    dtype          : Optional[torch.dtype] = None
    uses_norm      : bool = True
    perfs_runtime  : bool = False


class SegmentedPeakNetDataset(Dataset):
    """
    This class only loads events sequentially saved in a list of safetensors files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, config: SegmentedPeakNetDatasetConfig):
        super().__init__()

        self.path_csv        = config.path_csv
        self.seg_size        = config.seg_size
        self.dtype           = config.dtype
        self.transforms      = config.transforms
        self.uses_norm       = config.uses_norm
        self.perfs_runtime   = config.perfs_runtime
        self.buffer_size     = config.buffer_size
        self.device          = config.device
        self.dist_rank       = config.dist_rank
        self.dist_world_size = config.dist_world_size

        self.file_paths = self._init_file_paths()
        self.total_size = self._get_total_size()

        self.data_buffer = OrderedDict()
        self.start_idx   = 0
        self.end_idx     = 0

        self.item_generator  = None
        self.current_dataset = None

        self.norm = InstanceNorm() if self.uses_norm else NoTransform

        return None


    def _init_file_paths(self):
        object_list = [None,]  # For communication
        file_paths = []
        if self.dist_rank == 0:
            with open(self.path_csv, 'r') as fh:
                lines = list(csv.reader(fh))
            file_paths = [ line[0] for line in lines ]
            object_list = [file_paths,]

        if self.dist_world_size > 1:
            logger.debug(f"[RANK {self.dist_rank}] Syncing file paths.")
            dist.broadcast_object_list(object_list, src = 0)
            file_paths = object_list[0]

        return file_paths


    def _item_generator(self):
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                data = zarr.open(file_path, mode = 'r')
                num_items = data.get('images').shape[0]
                for item_idx in range(num_items):
                    yield file_idx, item_idx

            except Exception as e:
                # Log the error and skip to the next file
                logger.error(f"Skipping file {file_path} due to error: {e}")
                continue


    def _get_total_size(self):
        total_size = 0
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                data = zarr.open(file_path, mode = 'r')
                num_items = data.get('images').shape[0]
                total_size += num_items

            except Exception as e:
                # Log the error and skip to the next file
                logger.error(f"Skipping file {file_path} due to error: {e}")
                continue

        return total_size


    def reset(self):
        self.start_idx       = 0
        self.end_idx         = 0
        self.item_generator  = None
        self.current_dataset = None
        logger.debug(f"[RANK {self.dist_rank}] Resetting Dataset.")


    def update_dataset_segment(self):
        logger.debug(f"[RANK {self.dist_rank}] Updating segment to {self.start_idx}-{self.end_idx}.")
        return list(islice(self.item_generator, 0, self.end_idx - self.start_idx))


    def calculate_end_idx(self):
        """
        end_idx is not inclusive (up to, but not including end_idx)
        """
        # Calculate and return the end index for the current dataset segment.
        return min(self.start_idx + self.seg_size * self.dist_world_size, self.total_size)


    @property
    def num_seg(self):
        return ceil(self.total_size / (self.seg_size * self.dist_world_size))


    def set_start_idx(self, start_idx):
        requires_reset = False

        logger.debug(f"[RANK {self.dist_rank}] Setting start idx to {start_idx}.")

        self.start_idx = start_idx
        self.end_idx   = self.calculate_end_idx()

        # Optionally reset and/or advance the generator
        if self.dist_rank == 0 and self.item_generator is None:
            # Initialize the generator for a resumption or rewind
            item_generator = self._item_generator()
            self.item_generator = islice(item_generator, self.start_idx, None)

        # Update dataset segment and sync across ranks
        object_list = [None,]  # For communication
        if self.dist_rank == 0:
            self.current_dataset = self.update_dataset_segment()
            object_list = [self.current_dataset,]

        if self.dist_world_size > 1:
            logger.debug(f"[RANK {self.dist_rank}] Syncing current dataset.")
            dist.broadcast_object_list(object_list, src = 0)
            self.current_dataset = object_list[0]

        # Reset if reached the end of the item generator???
        if len(self.current_dataset) == 0:
            requires_reset = True
            self.reset()

        return requires_reset


    def __getitem__(self, idx):
        file_idx, item_idx = self.current_dataset[idx]
        file_path = self.file_paths[file_idx]

        # Load data into buffer if not already present
        if file_path not in self.data_buffer:
            if len(self.data_buffer) == self.buffer_size:
                file_path_oldest, data_oldest = self.data_buffer.popitem(last=False)
                ## print(f"{file_path_oldest} is cleared.")
            self.data_buffer[file_path] = zarr.open(file_path, mode = 'r')

        # Retrieve specific image and label tensors
        data = self.data_buffer[file_path]
        image = torch.from_numpy(data.get("images")[item_idx][None,None,])
        label = torch.from_numpy(data.get("labels")[item_idx][None,None,])

        # Apply transformation to image and label at the same time...
        if self.transforms is not None:
            data = torch.cat([image, label], dim = 0)    # (2*B=1, C, H, W)
            if self.dtype is not None: data = data.to(self.dtype)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = f"Transform method {enum_idx:d}", is_on = self.perfs_runtime):
                    data = trans(data)

            image = data[0]    # (1, C, H, W)
            label = data[1]    # (1, C, H, W)

            # Binarize the label...
            label = label > 0.5

        image = self.norm(image[None,])[0]

        return image, label    # (C, H, W)


    def __len__(self):
        return self.end_idx - self.start_idx
