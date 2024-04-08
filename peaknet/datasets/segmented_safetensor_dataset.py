import csv

from safetensors.torch import load_file
from collections import deque

import torch

from torch.utils.data import Dataset

import warnings

from dataclasses import dataclass
from typing import Optional, List

from collections import OrderedDict

from ..perf import Timer

class SegmentedPeakNetDataset(Dataset):
    """
    This class only loads events sequentially saved in a list of safetensors files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, path_csv, seg_size, dtype = None, transforms = None, buffer_size = 1, perfs_runtime = False):
        super().__init__()


        # -- Preprocessing
        # --- Create an item list based on the input csv file
        file_paths = []
        item_list  = []
        with open(path_csv, 'r') as fh:
            lines = list(csv.reader(fh))
        file_paths = [ line[0] for line in lines ]

        with Timer(tag = f"Build index map", is_on = perfs_runtime):
            # Open all safetensors files and track their status...
            for file_idx, file_path in enumerate(file_paths):
                data = load_file(file_path, device = 'cpu')
                num_items = data["image"].shape[0]
                for item_idx in range(num_items):
                    item_list.append((file_idx, item_idx))

        # -- Create attributes
        self.path_csv      = path_csv
        self.seg_size      = len(item_list) if seg_size is None else seg_size
        self.dtype         = dtype
        self.transforms    = transforms
        self.perfs_runtime = perfs_runtime
        self.file_paths    = file_paths
        self.item_list     = item_list
        self.buffer_size   = buffer_size
        self.data_buffer   = OrderedDict()
        self.start_idx     = 0
        self.end_idx       = self.seg_size

        return None


    def set_next_seg(self, start_idx = None):
        if start_idx is None:
            start_idx = self.end_idx if self.end_idx < len(self.item_list) else 0
        end_idx = min(start_idx + self.seg_size, len(self.item_list))

        self.start_idx = start_idx
        self.end_idx   = end_idx


    def __getitem__(self, idx):
        global_idx = self.start_idx + idx

        file_idx, item_idx = self.item_list[global_idx]
        file_path = self.file_paths[file_idx]

        # Load data into buffer if not already present
        if file_path not in self.data_buffer:
            if len(self.data_buffer) == self.buffer_size:
                file_path_oldest, data_oldest = self.data_buffer.popitem(last=False)
                print(f"{file_path_oldest} is cleared.")
            self.data_buffer[file_path] = load_file(file_path, device = 'cpu')
            print(f"{file_path} is loaded.")

        print(f'Global index: {global_idx}; Local index: {idx}; Start index: {self.start_idx};')

        # Retrieve specific image and label tensors
        data = self.data_buffer[file_path]
        image = data["image"][item_idx]
        label = data["label"][item_idx]

        # Apply transformation to image and label at the same time...
        if self.transforms is not None:
            data = torch.cat([image[None,], label[None,]], dim = 0)    # (2*B=1, C, H, W)
            if self.dtype is not None: data = data.to(self.dtype)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = f"Transform method {enum_idx:d}", is_on = self.perfs_runtime):
                    data = trans(data)

            image = data[0]    # (1, C, H, W)
            label = data[1]    # (1, C, H, W)

        # Binarize the label...
        label = label > 0

        return image, label    # (C, H, W)


    def __len__(self):
        return self.end_idx - self.start_idx
