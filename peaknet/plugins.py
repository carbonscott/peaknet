#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import cupy  as cp

import torch
import torch.nn.functional as F

from math import isnan
from cupyx.scipy import ndimage

from .trans import coord_crop_to_img


class CheetahPeakFinder:

    def __init__(self, model = None, path_cheetah_geom = None):
        self.model             = model
        self.path_cheetah_geom = path_cheetah_geom

        # Load the cheetah geometry...
        with open(self.path_cheetah_geom, 'rb') as handle:
            cheetah_geom_dict = pickle.load(handle)
        self.cheetah_geom_list = list(cheetah_geom_dict.values())[::2]

        # Set up structure to find connected component in 2D only...
        self.structure = cp.zeros((3, 3, 3))
        #                     ^  ^^^^
        # batch_______________|   |
        #                         |
        # 2D image________________|

        # Define structure in 2D image at the middle layer
        self.structure[1] = cp.array([[1,1,1],
                                      [1,1,1],
                                      [1,1,1]])



    def calc_batch_center_of_mass(self, batch_mask):
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
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, cp.asarray(range(1, batch_num_feature+1)))

        return batch_center_of_mass


    def calc_batch_center_of_mass_perf(self, batch_mask):
        import time

        batch_mask = cp.asarray(batch_mask)


        # Set up structure to find connected component in 2D only...
        structure = self.structure
        ## structure = cp.zeros((3, 3, 3))
        ## #                     ^  ^^^^
        ## # batch_______________|   |
        ## #                         |
        ## # 2D image________________|

        ## # Define structure in 2D image at the middle layer
        ## structure[1] = cp.array([[1,1,1],
        ##                          [1,1,1],
        ##                          [1,1,1]])

        # Fetch labels...
        time_start = time.time()
        batch_label, batch_num_feature = ndimage.label(batch_mask, structure = structure)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(L)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # Calculate batch center of mass...
        time_start = time.time()
        batch_center_of_mass = ndimage.center_of_mass(batch_mask, batch_label, cp.asarray(range(1, batch_num_feature+1)))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(C)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return batch_center_of_mass


    def calc_batch_center_of_mass_2d_perf(self, batch_mask):
        import time

        batch_mask = cp.asarray(batch_mask)

        # Define structure in 2D image at the middle layer
        structure = cp.array([[1,1,1],
                              [1,1,1],
                              [1,1,1]])

        # Fetch labels...
        time_start = time.time()
        batch_label = []
        batch_num_feature = []
        for batch_idx in range(len(batch_mask)):
            label, num_feature = ndimage.label(batch_mask[batch_idx], structure = structure)

            batch_label.append(label)
            batch_num_feature.append(num_feature)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(L)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # Calculate batch center of mass...
        time_start = time.time()
        batch_center_of_mass = []
        for batch_idx in range(len(batch_mask)):
            mask = batch_mask[batch_idx]
            label = batch_label[batch_idx]
            num_feature = batch_num_feature[batch_idx]
            center_of_mass = ndimage.center_of_mass(mask, label, cp.asarray(range(1, num_feature+1)))
            batch_center_of_mass.append([batch_idx, *center_of_mass])
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass(C)'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

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
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))

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


    def find_peak_w_softmax(self, img_stack, min_num_peaks = 15):
        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)

        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
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
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(label_predicted.view(num_stack, size_y, size_x))

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(peak_pos_predicted_stack) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in peak_pos_predicted_stack:
                idx_panel, y, x = peak_pos.get()

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], label_predicted.shape[-2:])

                x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                x += x_min
                y += y_min

                peak_list.append((y, x))

        return peak_list


    def find_peak_w_softmax_and_perf(self, img_stack, min_num_peaks = 15):
        import time

        peak_list = []

        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (-1, -2), keepdim = True)) / img_stack.std(axis = (-1, -2), keepdim = True)

        # Get activation feature map given the image stack...
        self.model.eval()

        time_start = time.time()

        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)

        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the softmax function...
        mask_stack_predicted = fmap_stack.softmax(dim = 1)

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
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(label_predicted.view(num_stack, size_y, size_x))

        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:map to peak'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        if len(peak_pos_predicted_stack) >= min_num_peaks:
            # Convert to cheetah coordinates...
            for peak_pos in peak_pos_predicted_stack:
                idx_panel, y, x = peak_pos.get()

                if isnan(y) or isnan(x): continue

                idx_panel = int(idx_panel)

                y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], label_predicted.shape[-2:])

                x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

                x += x_min
                y += y_min

                peak_list.append((y, x))

        return peak_list


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
        peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))

        return peak_pos_predicted_stack, mask_stack_predicted


    def find_peak_and_perf(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalization'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Sigmoid'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Thresholding the probability...
        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Thresholding'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Find center of mass for each image in the stack...
        num_stack, _, size_y, size_x = mask_stack_predicted.shape
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))
        peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass_2d_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## peak_pos_predicted_stack = self.calc_batch_mean_position(mask_stack_predicted.view(num_stack, size_y, size_x))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        ## if len(peak_pos_predicted_stack) >= min_num_peaks:
        ##     time_start = time.time()
        ##     # Convert to cheetah coordinates...
        ##     for peak_pos in peak_pos_predicted_stack:
        ##         idx_panel, y, x = peak_pos.get()

        ##         if isnan(y) or isnan(x): continue

        ##         idx_panel = int(idx_panel)

        ##         y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

        ##         x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

        ##         x += x_min
        ##         y += y_min

        ##         peak_list.append((y, x))

        ##     time_end = time.time()
        ##     time_delta = time_end - time_start
        ##     time_delta_name = 'pf:Convert coords'
        ##     print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return peak_list


    def find_peak_and_perf_downsized(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalization'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Sigmoid'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Thresholding the probability...
        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)
        size_y, size_x = mask_stack_predicted.shape[-2:]
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Thresholding'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        mask_stack_predicted_downsized = torch.nn.functional.avg_pool2d(mask_stack_predicted.type(torch.float), kernel_size = 3, stride = 2, padding=1)
        ## mask_stack_predicted_downsized = torch.nn.functional.avg_pool2d(mask_stack_predicted_downsized.type(torch.float), kernel_size = 3, stride = 2, padding=1)
        mask_stack_predicted_downsized = (mask_stack_predicted_downsized > 0).type(torch.int32)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Downsize'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Find center of mass for each image in the stack...
        num_stack, _, size_y_downsized, size_x_downsized = mask_stack_predicted_downsized.shape
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted_downsized.view(num_stack, size_y_downsized, size_x_downsized))
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Center of mass'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## time_start = time.time()
        ## # Find center of mass for each image in the stack...
        ## peak_pos_predicted_stack[:,1] *= size_y / size_y_downsized
        ## peak_pos_predicted_stack[:,2] *= size_x / size_x_downsized
        ## time_end = time.time()
        ## time_delta = time_end - time_start
        ## time_delta_name = 'pf:Rescale coordinates'
        ## print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        ## if len(peak_pos_predicted_stack) >= min_num_peaks:
        ##     time_start = time.time()
        ##     # Convert to cheetah coordinates...
        ##     for peak_pos in peak_pos_predicted_stack:
        ##         idx_panel, y, x = peak_pos.get()

        ##         if isnan(y) or isnan(x): continue

        ##         idx_panel = int(idx_panel)

        ##         y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

        ##         x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

        ##         x += x_min
        ##         y += y_min

        ##         peak_list.append((y, x))

        ##     time_end = time.time()
        ##     time_delta = time_end - time_start
        ##     time_delta_name = 'pf:Convert coords'
        ##     print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return peak_list


    def count_hl_pixel_and_perf(self, img_stack, threshold_prob = 1 - 1e-4, min_num_peaks = 15):
        import time

        peak_list = []

        time_start = time.time()
        # Normalize the image stack...
        img_stack = (img_stack - img_stack.mean(axis = (2, 3), keepdim = True)) / img_stack.std(axis = (2, 3), keepdim = True)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Normalization'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Get activation feature map given the image stack...
        self.model.eval()
        with torch.no_grad():
            fmap_stack = self.model.forward(img_stack)
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Inference'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Convert to probability with the sigmoid function...
        mask_stack_predicted = fmap_stack.sigmoid()
        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Sigmoid'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        time_start = time.time()
        # Thresholding the probability...
        ## is_background = mask_stack_predicted < threshold_prob
        ## mask_stack_predicted[ is_background ] = 0
        ## mask_stack_predicted[~is_background ] = 1

        mask_stack_predicted = (mask_stack_predicted >= threshold_prob).type(torch.int32)

        time_end = time.time()
        time_delta = time_end - time_start
        time_delta_name = 'pf:Thresholding'
        print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## time_start = time.time()
        ## # Find center of mass for each image in the stack...
        ## num_stack, _, size_y, size_x = mask_stack_predicted.shape
        ## ## peak_pos_predicted_stack = self.calc_batch_center_of_mass(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## peak_pos_predicted_stack = self.calc_batch_center_of_mass_perf(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## ## peak_pos_predicted_stack = self.calc_batch_mean_position(mask_stack_predicted.view(num_stack, size_y, size_x))
        ## time_end = time.time()
        ## time_delta = time_end - time_start
        ## time_delta_name = 'pf:Center of mass'
        ## print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        ## # A workaround to avoid copying gpu memory to cpu when num of peaks is small...
        ## if len(peak_pos_predicted_stack) >= min_num_peaks:
        ##     time_start = time.time()
        ##     # Convert to cheetah coordinates...
        ##     for peak_pos in peak_pos_predicted_stack:
        ##         idx_panel, y, x = peak_pos.get()

        ##         if isnan(y) or isnan(x): continue

        ##         idx_panel = int(idx_panel)

        ##         y, x = coord_crop_to_img((y, x), img_stack.shape[-2:], mask_stack_predicted.shape[-2:])

        ##         x_min, y_min, x_max, y_max = self.cheetah_geom_list[idx_panel]

        ##         x += x_min
        ##         y += y_min

        ##         peak_list.append((y, x))

        ##     time_end = time.time()
        ##     time_delta = time_end - time_start
        ##     time_delta_name = 'pf:Convert coords'
        ##     print(f"Time delta ({time_delta_name:20s}): {time_delta * 1e3:.4f} ms.")

        return []




class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):
        import psana

        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }


    def __len__(self):
        return len(self.timestamps)


    def get(self, event_num, id_panel = None, mode = "calib"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[event_num]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
            f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        # Fetch image data based on timestamp from detector...
        data = self.read[mode](event)
        img  = data[int(id_panel)] if id_panel is not None else data

        return img


    def assemble(self, multipanel = None, mode = "image", fake_event_num = 0):
        # Set up a fake event_num...
        event_num = fake_event_num

        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector...
        img = self.read[mode](event, multipanel)

        return img


    def create_bad_pixel_mask(self):
        return self.read["mask"](self.run_current, calib       = True,
                                                   status      = True,
                                                   edges       = True,
                                                   central     = True,
                                                   unbond      = True,
                                                   unbondnbrs  = True,
                                                   unbondnbrs8 = False).astype(np.uint16)





def remove_outliers(data, percentile = 5):
    """Removes outliers from a numpy array using the IQR method."""
    q1, q3 = np.percentile(data, [percentile, 100 - percentile])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = np.logical_and(data >= lower_bound, data <= upper_bound)
    return data * mask
