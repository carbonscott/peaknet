import os
import csv
import h5py
import random
import numpy as np
import logging

from ..utils   import split_dataset
from ..plugins import apply_mask

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CXIManager:
    """
    Main tasks of this class:
    - Offers `get_img` function that returns a data point tensor with the shape
      (2, H, W).
    - Offers an interface that allows users to modify the label tensor with the
      shape (1, H, W).  The label tensor only supports integer type.

    YAML
    - CXI 0
      - EVENT 0
      - EVENT 1
    - CXI 1
      - EVENT 0
      - EVENT 1
    """
    def __init__(self, path_csv = None):
        super().__init__()

        # Imported variables...
        self.path_csv    = path_csv

        # Define the keys used below...
        CXI_KEY = {
            "data"      : "/entry_1/data_1/data",
            "segmask"   : "/entry_1/data_1/segmask",
        }

        # Importing data from csv...
        cxi_dict = {}
        with open(path_csv, 'r') as fh:
            lines = csv.reader(fh)

            # Skip the header
            next(lines)

            # Open all cxi files and track their status...
            for line in lines:
                # Fetch metadata of a dataset
                path_cxi, sample_weight = line

                # Open a new file???
                if path_cxi not in cxi_dict:
                    print(f"Opening {path_cxi}...")
                    cxi_dict[path_cxi] = {
                        "file_handle"   : h5py.File(path_cxi, 'r'),
                        "is_open"       : True,
                        "sample_weight" : float(sample_weight),
                    }

        # Internal variables...
        self.cxi_dict      = cxi_dict
        self.CXI_KEY       = CXI_KEY

        # Init dataset...
        self.dataset = []

        return None


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def close(self):
        for path_cxi, cxi in self.cxi_dict.items():
            is_open = cxi.get("is_open")
            if is_open:
                cxi.get("file_handle").close()
                cxi["is_open"] = False
                print(f"{path_cxi} is closed.")


    @staticmethod
    def count_sample_by_category(num_samples, weights):
        """
        Provide a list detailing the count of samples from each category.

        Arguments:

            num_samples : Int
                Total number of samples across all categories.

            weights : List
                Weight for each category.

        Returns:

            sample_counts_int : List
                A list of sample counts for each category.

        """
        # Normalizing weights...
        normalized_weights = weights / weights.sum()

        # Allocate samples by weights...
        sample_counts = num_samples * normalized_weights

        # Split into integer and fractional parts...
        sample_counts_int  = np.floor(sample_counts).astype(int)
        sample_counts_frac = sample_counts - sample_counts_int

        # Determine number of left over samples...
        num_remaining_samples = num_samples - sample_counts_int.sum()
        num_remaining_samples = int(num_remaining_samples)

        # Find the indices of the largest fractional parts...
        indices_sorted_by_remainder = np.argsort(sample_counts_frac)[::-1]    # ...Descending order
        indices_to_accept_remanider = indices_sorted_by_remainder[:num_remaining_samples]

        # Assign a singular remaining sample to these indices...
        sample_counts_int[indices_to_accept_remanider] += 1

        return sample_counts_int


    def create_dataset(self, num_samples):
        cxi_dict = self.cxi_dict

        # Get the sample weights...
        cxi_list           = []
        sample_weight_list = []
        for path_cxi, metadata in cxi_dict.items():
            sample_weight = metadata['sample_weight']
            sample_weight_list.append(sample_weight)
            cxi_list.append(path_cxi)

        # Determine sample counts for each file...
        sample_weight_list = np.asarray(sample_weight_list)
        sample_counts = CXIManager.count_sample_by_category(num_samples, sample_weight_list)

        # Sample with replacement from each file...
        key_data = self.CXI_KEY['data']
        dataset  = []
        for path_cxi, sample_count in zip(cxi_list, sample_counts):
            # Fetch the total number of samples in the category...
            cxi_metadata = cxi_dict[path_cxi]
            fh = cxi_metadata['file_handle']
            num_data = fh.get(key_data).shape[0]

            # Choose...
            sample_idx_list = random.choices(range(num_data), k = sample_count)

            dataset.extend((path_cxi, sample_idx) for sample_idx in sample_idx_list)

        return dataset


    def initialize_dataset(self, num_samples):
        self.dataset = self.create_dataset(num_samples)


    def get_img(self, idx):
        path_cxi, idx_in_cxi = self.dataset[idx]

        # Obtain the file handle...
        fh = self.cxi_dict[path_cxi]['file_handle']

        # Obtain the image...
        k   = self.CXI_KEY["data"]
        img = fh.get(k)[idx_in_cxi]

        # Obtain the segmask...
        k       = self.CXI_KEY["segmask"]
        segmask = fh.get(k)[idx_in_cxi]

        return img[None,], segmask[None,]


    def __getitem__(self, idx):
        return self.get_img(idx)


    def __len__(self):
        return len(self.dataset)




class CXIDataset(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, cxi_manager, idx_list, trans_list = None, normalizes_img = True):
        self.cxi_manager    = cxi_manager
        self.idx_list       = idx_list
        self.normalizes_img = normalizes_img
        self.trans_list     = trans_list

        return None


    def __len__(self):
        return len(self.idx_list)


    def __getitem__(self, idx):
        cxi_idx = self.idx_list[idx]

        img, label = self.cxi_manager[cxi_idx]

        # Apply transformation to image and label at the same time...
        if self.trans_list is not None:
            data = np.concatenate([img, label], axis = 0)
            for trans in self.trans_list:
                data = trans(data)

            img   = data[0:1]
            label = data[1: ]

        if self.normalizes_img:
            # Normalize input image...
            img_mean = np.nanmean(img)
            img_std  = np.nanstd(img)
            img      = img - img_mean

            if img_std == 0:
                img[:]   = 0
                label[:] = 0

        return img, label
