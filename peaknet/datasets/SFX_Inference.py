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

    def __init__(self, exp, run, mode, detector_name, img_mode, event_list = None):
        super().__init__()

        self.exp           = exp
        self.run           = run
        self.mode          = mode
        self.detector_name = detector_name
        self.img_mode      = img_mode
        self.event_list    = event_list

        self.psana_img      = None
        self.bad_pixel_mask = None


    def _initialize_psana(self):
        exp           = self.exp
        run           = self.run
        mode          = self.mode
        detector_name = self.detector_name

        self.psana_img      = PsanaImg(exp, run, mode, detector_name)
        self.bad_pixel_mask = self.psana_img.create_bad_pixel_mask()


    def __len__(self):
        if self.psana_img is None:
            self._initialize_psana()

        return len(self.psana_img) if self.event_list is None else len(self.event_list)


    def __getitem__(self, idx):
        if self.psana_img is None:
            self._initialize_psana()

        # Fetch the event based on idx...
        event = idx if self.event_list is None else self.event_list[idx]

        # Fetch pixel data using psana...
        data = self.psana_img.get(event, None, self.img_mode)    # (B, H, W) or (H, W)

        if data is None:
            data = np.zeros_like(self.bad_pixel_mask, dtype = np.float32)

        # Mask out bad pixels...
        data = apply_mask(data, self.bad_pixel_mask, mask_value = 0)

        # Unify the data dimension...
        if data.ndim == 2: data = data[None,]    # (H, W) -> (1, H, W)

        # Build metadata...
        metadata = np.array([ (idx, event, panel_idx_in_batch) for panel_idx_in_batch, _ in enumerate(data) ], dtype = np.int32)

        return data, metadata
