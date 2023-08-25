import os
import yaml
import h5py
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

    def __init__(self, path_yaml = None):
        super().__init__()

        # Imported variables...
        self.path_yaml = path_yaml

        # Load the YAML file
        with open(self.path_yaml, 'r') as fh:
            config = yaml.safe_load(fh)
        path_cxi_list = config['cxi']

        # Define the keys used below...
        CXI_KEY = {
            "num_peaks" : "/entry_1/result_1/nPeaks",
            "peak_y"    : "/entry_1/result_1/peakYPosRaw",
            "peak_x"    : "/entry_1/result_1/peakXPosRaw",
            "data"      : "/entry_1/data_1/data",
            "mask"      : "/entry_1/data_1/mask",
            "segmask"   : "/entry_1/data_1/segmask",
        }

        # Open all cxi files and track their status...
        cxi_dict = {}
        for path_cxi in path_cxi_list:
            # Open a new file???
            if path_cxi not in cxi_dict:
                cxi_dict[path_cxi] = {
                    "file_handle" : h5py.File(path_cxi, 'r'),
                    "is_open"     : True,
                }

        # Build an entire idx list...
        idx_list = []
        for path_cxi, cxi in cxi_dict.items():
            fh = cxi["file_handle"]
            k  = CXI_KEY["num_peaks"]
            num_event = fh.get(k)[()]
            for event_idx in range(len(num_event)):
                idx_list.append((path_cxi, event_idx, fh))


        # Internal variables...
        self.cxi_dict      = cxi_dict
        self.CXI_KEY       = CXI_KEY
        self.path_cxi_list = path_cxi_list
        self.idx_list      = idx_list

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


    def get_img(self, idx):
        path_cxi, event_idx, fh = self.idx_list[idx]

        # Obtain the image...
        k   = self.CXI_KEY["data"]
        img = fh.get(k)[event_idx]

        # Obtain the bad pixel mask...
        k    = self.CXI_KEY['mask']
        mask = fh.get(k)
        mask = mask[event_idx] if mask.ndim == 3 else mask[()]

        # Apply mask...
        img = apply_mask(img, 1 - mask, mask_value = 0)

        # Obtain the segmask...
        k       = self.CXI_KEY["segmask"]
        segmask = fh.get(k)[event_idx]

        return img[None,], segmask[None,]


    def __getitem__(self, idx):
        return self.get_img(idx)


    def __len__(self):
        return len(self.idx_list)




class CXIDataset(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, cxi_manager, idx_list, trans_list      = None,
                                              normalizes_data = False,
                                              reverses_bg     = False,):
        self.cxi_manager     = cxi_manager
        self.idx_list        = idx_list
        self.trans_list      = trans_list
        self.normalizes_data = normalizes_data
        self.reverses_bg     = reverses_bg

        return None


    def __len__(self):
        return len(self.idx_list)


    def __getitem__(self, idx):
        sample_idx   = self.idx_list[idx]
        img, segmask = self.cxi_manager[sample_idx]    # ...Each has shape (1, H, W)

        batch_data = np.concatenate([img, segmask], axis = 0)
        if self.trans_list is not None:
            for trans in self.trans_list:
                batch_data = trans(batch_data)

        data  = batch_data[0:1]
        label = batch_data[1: ]

        if self.reverses_bg:
            label[-1] = 1 - label[-1]

        if self.normalizes_data:
            # Normalize input image...
            data_mean = np.nanmean(data)
            data_std  = np.nanstd(data)
            data      = data - data_mean

            if data_std == 0:
                data_std = 1.0
                label[:] = 0

            data /= data_std

        return data, label
