import os
import h5py
import random
import numpy as np
import logging

from ..plugins import PsanaImg, apply_mask

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SFXInferenceDataset(Dataset):
    """
    This class provides batched images for high through model inference.
    """

    def __init__(self, exp, run, mode, detector_name, img_mode):
        super().__init__()

        self.exp           = exp
        self.run           = run
        self.mode          = mode
        self.detector_name = detector_name
        self.img_mode      = img_mode

        # Define the Psana image handle...
        self.psana_img = PsanaImg(exp, run, mode, detector_name)
        self.bad_pixel_mask = self.psana_img.create_bad_pixel_mask()


    def __len__(self):
        return len(self.psana_img)


    def __getitem__(self, event):
        # Fetch pixel data using psana...
        data = self.psana_img.get(event, None, self.img_mode)    # (B, H, W) or (H, W)

        # Mask out bad pixels...
        data = apply_mask(data, self.bad_pixel_mask, mask_value = 0)

        # Unify the data dimension...
        if data.ndim == 2: data = data[None,]    # (H, W) -> (1, H, W)

        # Build metadata...
        metadata = np.array([ (event, idx) for idx, _ in enumerate(data) ], dtype = int)

        return data, metadata
