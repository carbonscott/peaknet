import csv

from safetensors.torch import load_file
from collections import deque

import torch

from torch.utils.data import Dataset

from ..perf import Timer


class PeakNetDataset(Dataset):
    """
    This class only loads events sequentially saved in a list of safetensors files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, path_csv, transforms = None, cache_size = 10, perfs_runtime = False):
        super().__init__()

        self.transforms    = transforms
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

        # Create an item index for sub sampling
        self.reset_sample_idx_map()

        self.cache = {}
        self.cache_order = deque([], maxlen=cache_size)

        return None


    def get_img(self, idx):
        sample_idx = self.sample_idx_map[idx]
        file_idx, item_idx = self.file_index_map[sample_idx]
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
        if self.transforms is not None:
            data = torch.cat([img[None,], label[None,]], dim = 0)    # (2*B=1, C, H, W)
            for enum_idx, trans in enumerate(self.transforms):
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
        return len(self.sample_idx_map)


    def reset_sample_idx_map(self):
        self.sample_idx_map = range(len(self.file_index_map))


    def sample_subset(self, subset_start_idx, subset_size):
        subset_start_idx = max(subset_start_idx, 0)
        subset_end_idx   = min(subset_start_idx + subset_size, len(self.file_index_map))

        self.sample_idx_map = range(subset_start_idx, subset_end_idx)
