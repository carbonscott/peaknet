#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import cupy  as cp

import torch
import torch.nn.functional as F

from math import isnan
from cupyx.scipy import ndimage

from .modeling.reg_bifpn_net import PeakNet

## from .trans import coord_crop_to_img, center_crop

from .configurator import Configurator


class PeakFinder:

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG.MODEL.NUM_BLOCKS   = 3
            CONFIG.MODEL.NUM_FEATURES = 64
            CONFIG.MODEL.AUX_CONFIG   = PeakNet.get_default_config()

        return CONFIG


    def __init__(self, path_chkpt = None, path_cheetah_geom = None, config = None):
        self.config = PeakFinder.get_default_config() if config is None else config

        # Set up default path to load default files...
        default_path = os.path.dirname(__file__)

        # [[[ MODEL ]]]
        # Create model...
        self.model, self.device = self.config_model(config = self.config.MODEL)

        # Load weights...
        if path_chkpt is not None:
            chkpt = torch.load(path_chkpt)
            self.model.module.load_state_dict(chkpt['model_state_dict']) if hasattr(self.model, 'module') else \
                   self.model.load_state_dict(chkpt['model_state_dict'])

        # [[[ CHEETAH GEOM ]]]
        # Load the cheetah geometry...
        self.path_cheetah_geom = os.path.join(default_path, 'data/cheetah_geom.pickle') if path_cheetah_geom is None else \
                                 path_cheetah_geom

        with open(self.path_cheetah_geom, 'rb') as handle:
            cheetah_geom_dict = pickle.load(handle)
        self.cheetah_geom_list = list(cheetah_geom_dict.values())[::2]

        # Set up structure to find connected component in 2D only...
        self.structure = cp.array([[1,1,1],
                                   [1,1,1],
                                   [1,1,1]])

        self.model.eval()



    def config_model(self, config = None):
        # Cofnig the model architecture...
        model = PeakNet( num_blocks   = config.NUM_BLOCKS,
                         num_features = config.NUM_FEATURES,
                         config       = config.AUX_CONFIG)

        # Set device...
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        return model, device


    def calc_batch_center_of_mass(self, img_stack, batch_mask):
        structure = self.structure

        imgs  = cp.asarray(img_stack[:, 0])
        masks = cp.asarray(batch_mask)

        # Obtain the tensor dimension...
        B, H, W = batch_mask.shape

        batch_center_of_mass = [ [] for _ in range(B) ]
        for i in range(B):
            img  = imgs[i]
            mask = masks[i]

            # Fetch labels...
            label, num_feature = ndimage.label(mask, structure = structure)

            # Calculate batch center of mass...
            center_of_mass = ndimage.center_of_mass(img, label, cp.asarray(range(1, num_feature+1)))
            batch_center_of_mass[i] = center_of_mass

        batch_center_of_mass = [ (i, y.get(), x.get()) for i, center_of_mass in enumerate(batch_center_of_mass) for y, x in center_of_mass ]

        return batch_center_of_mass


    def calc_batch_mean_position(self, batch_mask):
        batch_mask = cp.asarray(batch_mask)

        # Set up structure to find connected component in 2D only...
        structure = cp.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        structure[1] = cp.array([[1,1,1],
                                 [1,1,1],
                                 [1,1,1]])

        # Fetch labels...
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

        # Calculate batch center of mass...
        batch_mean_position = [ cp.argwhere(batch_label == i).mean(axis = 0) for i in range(1, batch_num_feature + 1) ]

        return batch_mean_position


    def calc_fmap(self, img_stack, uses_mixed_precision = True):
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)

        # Get activation feature map given the image stack...
        ## self.model.eval()
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    fmap_stack = self.model.forward(img_stack)
            else:
                fmap_stack = self.model.forward(img_stack)

        return fmap_stack


    def find_peak_w_softmax(self, img_stack, min_num_peaks = 0, uses_geom = True, returns_prediction_map = False, uses_mixed_precision = True):
        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)

        # Get activation feature map given the image stack...
        ## self.model.eval()
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    fmap_stack = self.model.forward(img_stack)
            else:
                fmap_stack = self.model.forward(img_stack)

        # Convert to probability with the softmax function...
        mask_stack_predicted = fmap_stack.softmax(dim = 1)

        B, C, H, W = mask_stack_predicted.shape
        mask_stack_predicted = mask_stack_predicted.argmax(dim = 1, keepdims = True)
        mask_stack_predicted = F.one_hot(mask_stack_predicted.reshape(B, -1), num_classes = C).permute(0, 2, 1).reshape(B, -1, H, W)
        label_predicted = mask_stack_predicted[:, 1]
        label_predicted = label_predicted.to(torch.int)

        # Find center of mass for each image in the stack...
        num_stack, size_y, size_x = label_predicted.shape
        batch_peak_in_panels = self.calc_batch_center_of_mass(img_stack, label_predicted.view(num_stack, size_y, size_x))

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(batch_peak_in_panels) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in batch_peak_in_panels:
                idx_panel, y, x = peak_pos
                idx_panel = int(idx_panel)

                if uses_geom:
                    x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                    x += x_min
                    y += y_min

                peak_list.append((idx_panel, y, x))

        ret = peak_list, None
        if returns_prediction_map: ret = peak_list, mask_stack_predicted.cpu().numpy()

        return ret
