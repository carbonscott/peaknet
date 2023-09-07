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

from .modeling.att_unet import PeakNet
from .trans             import coord_crop_to_img, center_crop


## class PeakFinderRefactor:
class PeakFinder:

    def __init__(self, path_chkpt = None, path_cheetah_geom = None):
        # Set up default path to load default files...
        default_path = os.path.dirname(__file__)

        # [[[ CHECKPOINT ]]]
        self.path_chkpt = os.path.join(default_path, 'data/rayonix.2023_0506_0308_15.chkpt') if path_chkpt is None else \
                          path_chkpt
        chkpt = torch.load(self.path_chkpt)

        # [[[ MODEL ]]]
        # Create model...
        self.model, self.device = self.init_default_model()

        # Load weights...
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



    def init_default_model(self, path_chkpt = None):
        # Config the channels in the network...
        config_channels = { 
            "input_layer"    : (1, 8), 
            "encoder_layers" : ( 
                (8,  16),
                (16, 32),
                (32, 64),
                (64, 128),
            ),  
            "fusion_layers" : ( 
                (128, 64),
                (64,  32),
                (32,  16),
                (16,   8), 
            ),  
            "head_segmask_layer": (8, 3), 
        }

        # Use the model architecture -- attention u-net...
        uses_skip_connection = True
        model = PeakNet( config_channels = config_channels,
                         uses_skip_connection = uses_skip_connection, )

        # Set device...
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## if torch.cuda.device_count() > 1:
        ##     model  = torch.nn.DataParallel(model)
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


    ## def calc_batch_center_of_mass(self, img_stack, batch_mask):
    ##     img_stack  = cp.asarray(img_stack[:, 0])    # B, C, H, W and C = 1
    ##     batch_mask = cp.asarray(batch_mask)

    ##     # Set up structure to find connected component in 2D only...
    ##     structure = self.structure

    ##     # Fetch labels...
    ##     batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)

    ##     # Calculate batch center of mass...
    ##     batch_center_of_mass = ndimage.center_of_mass(img_stack, batch_label, cp.asarray(range(1, batch_num_feature+1)))

    ##     return batch_center_of_mass


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


    def find_peak(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)

        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)

        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()

        # Thresholding the probability...
        ## is_background = mask_stack_predicted < threshold_prob
        ## mask_stack_predicted[ is_background ] = 0
        ## mask_stack_predicted[~is_background ] = 1
        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(img_stack, mask_stack_predicted.view(num_stack, size_y, size_x))

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(peak_pos_predicted_stack) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in peak_pos_predicted_stack:
                idx_panel, y, x = peak_pos.get()

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

                x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                x += x_min
                y += y_min

                peak_list.append((y, x))

        return peak_list


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

        # Guarantee image and prediction mask have the same size...
        size_y, size_x = img_stack.shape[-2:]
        mask_stack_predicted = center_crop(mask_stack_predicted, size_y, size_x, returns_offset_tuple = False)

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

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], label_predicted.shape[-2:])

                if uses_geom:
                    x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                    x += x_min
                    y += y_min

                peak_list.append((idx_panel, y, x))

        ret = peak_list, None
        if returns_prediction_map: ret = peak_list, mask_stack_predicted.cpu().numpy()

        return ret


    def find_peak_w_softmax_and_perf(self, img_stack, min_num_peaks = 15, uses_geom = True, returns_prediction_map = False, uses_mixed_precision = True):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalize'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # Get activation feature map given the image stack...
        self.model.eval()
        time_start = time.time()
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    fmap_stack = self.model.forward(img_stack)
            else:
                fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the softmax function...
        mask_stack_predicted = fmap_stack.softmax(dim = 1)

        # Guarantee image and prediction mask have the same size...
        size_y, size_x = img_stack.shape[-2:]
        mask_stack_predicted = center_crop(mask_stack_predicted, size_y, size_x, returns_offset_tuple = False)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Crop'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        B, C, H, W = mask_stack_predicted.shape
        mask_stack_predicted = mask_stack_predicted.argmax(dim = 1, keepdims = True)
        mask_stack_predicted = F.one_hot(mask_stack_predicted.reshape(B, -1), num_classes = C).permute(0, 2, 1).reshape(B, -1, H, W)
        label_predicted = mask_stack_predicted[:, 1]
        label_predicted = label_predicted.to(torch.int)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Softmax'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Find center of mass for each image in the stack...
        num_stack, size_y, size_x = label_predicted.shape
        batch_peak_in_panels = self.calc_batch_center_of_mass(img_stack, label_predicted.view(num_stack, size_y, size_x))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:map to peak'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(batch_peak_in_panels) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in batch_peak_in_panels:
                ## idx_panel, y, x = peak_pos.get()
                idx_panel, y, x = peak_pos

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], label_predicted.shape[-2:])

                if uses_geom:
                    x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                    x += x_min
                    y += y_min

                peak_list.append((idx_panel, y, x))

        ret = peak_list, None
        ## if returns_prediction_map: ret = peak_list, label_predicted.cpu().numpy()
        if returns_prediction_map: ret = peak_list, mask_stack_predicted.cpu().numpy()

        return ret


    def save_peak_and_fmap(self, img_stack, threshold_prob = 1 - 1e-4):
        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)

        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)

        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()

        # Thresholding the probability...
        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        batch_peak_in_panels = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))

        return batch_peak_in_panels, mask_stack_predicted