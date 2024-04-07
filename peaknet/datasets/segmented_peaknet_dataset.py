import csv
import h5py
import numpy as np

from ..perf import Timer

from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Optional, List

from collections import OrderedDict

class SegmentedPeakNetDataset(Dataset):
    """
    This class only loads events sequentially saved in a list of hdf5 files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, path_csv, seg_size = None, transforms = None, applies_norm = True, perfs_runtime = False):
        super().__init__()

        # -- Preprocessing
        # --- Create an item list based on the input csv file
        item_list = []
        with open(path_csv, 'r') as fh:
            lines = csv.reader(fh)
            for line in lines:
                path_hdf5, dataset_mean, dataset_std = line
                with h5py.File(path_hdf5, 'r') as f:
                    groups = f.get('data')
                    for group_idx in range(len(groups)):
                        item_list.append((path_hdf5, float(dataset_mean), float(dataset_std), group_idx))

        # -- Create attributes
        self.path_csv      = path_csv
        self.seg_size      = len(item_list) if seg_size is None else seg_size
        self.transforms    = transforms
        self.applies_norm  = applies_norm
        self.perfs_runtime = perfs_runtime
        self.item_list     = item_list
        self.start_idx     = 0
        self.end_idx       = self.seg_size
        self.file_buffer   = OrderedDict()

        return None


    def set_next_seg(self, start_idx = None):
        if start_idx is None:
            start_idx = self.end_idx if self.end_idx < len(self.item_list) else 0
        end_idx = min(start_idx + self.seg_size, len(self.item_list))

        self.start_idx = start_idx
        self.end_idx   = end_idx


    def get_img(self, idx):
        global_idx = self.start_idx + idx
        path_hdf5, dataset_mean, dataset_std, idx_in_hdf5 = self.item_list[global_idx]

        with Timer(tag = "Read data from hdf5", is_on = self.perfs_runtime):
            # Add a file buffer
            if not path_hdf5 in self.file_buffer:
                # Remove any existing buffer
                if self.file_buffer:
                    path_hdf5_prev, f_prev = self.file_buffer.popitem(last=False)
                    f_prev.close()
                    print(f"{path_hdf5_prev} is closed.")
                self.file_buffer[path_hdf5] = h5py.File(path_hdf5, 'r')

            print(f'Global index: {global_idx}; Local index: {idx}; Start index: {self.start_idx};')

            # Retrieve file buffer
            f = self.file_buffer.get(path_hdf5)

            # Obtain image
            k = f"data/data_{idx_in_hdf5:04d}/image"
            image = f.get(k)[()]

            # Obtain label
            k = f"data/data_{idx_in_hdf5:04d}/label"
            label = f.get(k)[()]

            # Obtain pixel map
            k = f"data/data_{idx_in_hdf5:04d}/metadata/pixel_map"
            pixel_map_x, pixel_map_y, pixel_map_z = f.get(k)[()]

        with Timer(tag = "Assemble panels into an image", is_on = self.perfs_runtime):
            pixel_map_x = np.round(pixel_map_x).astype(int)
            pixel_map_y = np.round(pixel_map_y).astype(int)
            pixel_map_z = np.round(pixel_map_z).astype(int)

            detector_image = np.zeros((pixel_map_x.max() - pixel_map_x.min() + 1,
                                       pixel_map_y.max() - pixel_map_y.min() + 1,
                                       pixel_map_z.max() - pixel_map_z.min() + 1), dtype = float)
            detector_label = np.zeros((pixel_map_x.max() - pixel_map_x.min() + 1,
                                       pixel_map_y.max() - pixel_map_y.min() + 1,
                                       pixel_map_z.max() - pixel_map_z.min() + 1), dtype = int)

            detector_image[pixel_map_x, pixel_map_y, pixel_map_z] = image
            detector_label[pixel_map_x, pixel_map_y, pixel_map_z] = label

        with Timer(tag = "Transpose data (some internal ops)", is_on = self.perfs_runtime):
            detector_image = detector_image.transpose((2, 0, 1))    # (H, W, C=1) -> (C, H, W)
            detector_label = detector_label.transpose((2, 0, 1))    # (H, W, C=1) -> (C, H, W)

        with Timer(tag = "Normalization", is_on = self.perfs_runtime):
            if self.applies_norm:
                detector_image = (detector_image - dataset_mean) / dataset_std

        # Apply transformation to image and label at the same time...
        if self.transforms is not None:
            data = np.concatenate([detector_image, detector_label], axis = 0)    # (2, H, W)
            for enum_idx, trans in enumerate(self.transforms):
                with Timer(tag = f"Transform method {enum_idx:d}", is_on = self.perfs_runtime):
                    data = trans(data)

            detector_image = data[0:1]    # (1, H, W)
            detector_label = data[1: ]    # (1, H, W)

        return detector_image, detector_label


    def __getitem__(self, idx):
        img, label = self.get_img(idx)

        return img, label


    def __len__(self):
        return self.end_idx - self.start_idx

