#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import csv
import os
import h5py
import pickle
import logging

from scipy            import ndimage
from torch.utils.data import Dataset

from peaknet.utils                  import set_seed, split_dataset
from peaknet.datasets.stream_parser import StreamParser, GeomInterpreter

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class SFXPanelDataset(Dataset):
    """
    SFX images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  

    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, config):
        self.fl_csv         = getattr(config, 'fl_csv'        , None)
        self.drc_project    = getattr(config, 'drc_project'   , None)
        self.size_sample    = getattr(config, 'size_sample'   , None)
        self.frac_train     = getattr(config, 'frac_train'    , None)    # Proportion/Fraction of training examples
        self.frac_validate  = getattr(config, 'frac_validate' , None)    # Proportion/Fraction of validation examples
        self.dataset_usage  = getattr(config, 'dataset_usage' , None)    # train, validate, test
        self.seed           = getattr(config, 'seed'          , None)
        self.dist           = getattr(config, 'dist'          , 5)       # Max distance to consider as an indexed found peak.
        self.trans          = getattr(config, 'trans'         , None)
        self.mpi_comm       = getattr(config, 'mpi_comm'      , None)
        self.add_channel_ok = getattr(config, 'add_channel_ok', True)
        self.mask_radius    = getattr(config, 'mask_radius'   , 3)
        self.is_batch_mask  = getattr(config, 'is_batch_mask' , False)
        self.user_mask      = getattr(config, 'user_mask'     , None)
        self.snr_threshold  = getattr(config, 'snr_threshold' , 0.3)
        self.uses_indexed   = getattr(config, 'uses_indexed'  , False)
        self.adu_threshold  = getattr(config, 'adu_threshold' , 1000)

        # Variables that capture raw information from the input (stream files)
        self.fl_stream_list     = []
        self.metadata_orig_list = [] # ...A list of (fl_cxi, event_crystfel) that have labeled peaks
        self.peak_orig_list     = [] # ...A list of labeled peaks
        self.stream_cache_dict  = {} # ...A dictionary of file to stream mapping
        self.is_cache           = False

        # Variables that capture information in data spliting
        self.metadata_list = []
        self.peak_list     = []

        # Variables for caching...
        self.peak_cache_dict      = {} # ...A list of image data and their corresponding labels
        self.raw_img_cache_dict   = {}
        self.raw_label_cache_dict = {}

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

        # Collect stream files from the csv...
        with open(self.fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            next(lines)

            for line in lines:
                fl_stream = line[0]
                self.fl_stream_list.append(fl_stream)

        # Obtain all indexed SFX event from stream files...
        # [COMMENT] I tried not to have repeating code when dealing with MPI on/off
        # but I will keep it like this for now.  Room to improve.
        for fl_stream in self.fl_stream_list:
            # Create a list of data entry from a stream file...
            # With MPI
            if self.mpi_comm is not None:
                # Sync stream cache...
                if self.mpi_rank == 0:
                    # Get the stream object and save it to a global dictionary...
                    if not fl_stream in self.stream_cache_dict:
                        self.stream_cache_dict[fl_stream] = self.parse_stream(fl_stream)

                    # Sync the cache across workers...
                    for i in range(1, self.mpi_size, 1):
                        data_to_send = self.stream_cache_dict[fl_stream]
                        self.mpi_comm.send(data_to_send, dest = i, tag = self.mpi_data_tag)

                if self.mpi_rank !=0:
                    data_received = self.mpi_comm.recv(source = 0, tag = self.mpi_data_tag)
                    self.stream_cache_dict[fl_stream] = data_received

                # Get metadata and label...
                metadata_per_stream, peak_list_per_stream = \
                    self.mpi_extract_metadata_and_labeled_peak_from_streamfile(fl_stream)

                # Sync metadata and label across workers for completeness of the class object
                if self.mpi_rank == 0:
                    for i in range(1, self.mpi_size, 1):
                        data_to_send = (metadata_per_stream, peak_list_per_stream)
                        self.mpi_comm.send(data_to_send, dest = i, tag = self.mpi_data_tag)

                if self.mpi_rank != 0:
                    data_received = self.mpi_comm.recv(source = 0, tag = self.mpi_data_tag)
                    metadata_per_stream, peak_list_per_stream = data_received

            # Without MPI
            else:
                # Get the stream object and save it to a global dictionary...
                if not fl_stream in self.stream_cache_dict:
                    self.stream_cache_dict[fl_stream] = self.parse_stream(fl_stream)

                metadata_per_stream, peak_list_per_stream = \
                    self.extract_metadata_and_labeled_peak_from_streamfile(fl_stream)

            # Accumulate metadata...
            self.metadata_orig_list.extend(metadata_per_stream)    # (fl_stream, fl_cxi, event_crystfel)

            # Accumulate all peaks...
            self.peak_orig_list.extend(peak_list_per_stream)

        # Split original dataset sequence into training sequence and holdout sequence...
        seq_orig_list = list(range(len(self.metadata_orig_list)))
        seq_train_list, seq_holdout_list = split_dataset(seq_orig_list, self.frac_train)

        # Calculate the percentage of validation in the whole holdout set...
        frac_holdout = 1.0 - self.frac_train
        frac_validate_in_holdout = self.frac_validate / frac_holdout if self.frac_validate is not None else 0.5

        # Split holdout dataset into validation and test...
        seq_valid_list, seq_test_list = split_dataset(seq_holdout_list, frac_validate_in_holdout)

        # Choose which dataset is going to be used, defaults to original set...
        dataset_by_usage_dict = {
            'train'    : seq_train_list,
            'validate' : seq_valid_list,
            'test'     : seq_test_list,
        }
        seq_random_list = seq_orig_list
        if self.dataset_usage in dataset_by_usage_dict:
            seq_random_list = dataset_by_usage_dict[self.dataset_usage]

        # Create data list based on the sequence...
        self.metadata_list = [ self.metadata_orig_list[i] for i in seq_random_list ]
        self.peak_list     = [ self.peak_orig_list[i]     for i in seq_random_list ]

        return None


    def parse_stream(self, fl_stream):
        # Initialize the object to return...
        stream_dict = None

        # Find the basename of the stream file...
        basename_stream = os.path.basename(fl_stream)
        basename_stream = basename_stream[:basename_stream.rfind('.')]

        # Check if a corresponding pickle file exists...
        fl_pickle         = f"{basename_stream}.pickle"
        prefix_pickle     = 'pickles'
        prefixpath_pickle = os.path.join(self.drc_project, prefix_pickle)
        if not os.path.exists(prefixpath_pickle): os.makedirs(prefixpath_pickle)
        path_pickle = os.path.join(prefixpath_pickle, fl_pickle)

        # Obtain key information from stream by loading the pickle file if it exists...
        if os.path.exists(path_pickle):
            with open(path_pickle, 'rb') as fh:
                stream_dict = pickle.load(fh)

        # Otherwise, parse the stream file...
        else:
            # Employ stream parser to extract key info from stream...
            stream_parser = StreamParser(fl_stream)
            stream_parser.parse()
            stream_dict = stream_parser.stream_dict

            # Save the stream result in a pickle file...
            with open(path_pickle, 'wb') as fh:
                pickle.dump(stream_dict, fh, protocol = pickle.HIGHEST_PROTOCOL)

        return stream_dict


    def get_raw_peak(self, idx):
        '''
        Return raw found peaks and indexed peaks in a tuple.
        '''
        # Read image...
        # Ignore which stream file this information is extracted from
        fl_stream, fl_cxi, event_crystfel, panel = self.metadata_list[idx]

        # Fetch stream either from scratch or from a cached dictionary...
        stream_dict = self.parse_stream(fl_stream) if not fl_stream in self.stream_cache_dict \
                                                   else self.stream_cache_dict[fl_stream]

        # Get the right chunk
        panel_dict = stream_dict['chunk'][fl_cxi][event_crystfel]

        # Find all peaks...
        peak_saved_dict        = panel_dict[panel]
        found_per_panel_dict   = peak_saved_dict['found']
        indexed_per_panel_dict = peak_saved_dict['indexed']

        return found_per_panel_dict, indexed_per_panel_dict


    def extract_metadata_and_labeled_peak_from_streamfile(self, fl_stream):
        # Fetch stream either from scratch or from a cached dictionary...
        stream_dict = self.parse_stream(fl_stream) if not fl_stream in self.stream_cache_dict \
                                                   else self.stream_cache_dict[fl_stream]

        # Get the chunk...
        chunk_dict = stream_dict['chunk']

        # Work on each cxi file...
        metadata_list = []
        peak_list     = []
        for fl_cxi, event_crystfel_list in chunk_dict.items():
            # Only keep those both found and indexed peaks...
            for event_crystfel in event_crystfel_list:
                # Get panels in an event...
                panel_dict = chunk_dict[fl_cxi][event_crystfel]

                # Find all peaks...
                for panel, peak_per_panel_dict in panel_dict.items():
                    # Accumulate metadata of a label...
                    panel_descriptor = (fl_stream, fl_cxi, event_crystfel, panel)
                    metadata_list.append(panel_descriptor)

                    # Accumulate data for making a label...
                    peak_list.append(peak_per_panel_dict)

        return metadata_list, peak_list


    def mpi_extract_metadata_and_labeled_peak_from_streamfile(self, fl_stream):
        '''
        Extract metadata and labeled peak from chunks with MPI.
        '''
        # Import chunking method...
        from peaknet.utils import split_dict_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # Fetch stream either from scratch or from a cached dictionary...
        stream_dict = self.parse_stream(fl_stream) if not fl_stream in self.stream_cache_dict \
                                                   else self.stream_cache_dict[fl_stream]

        # Get the chunk...
        chunk_dict = stream_dict['chunk']

        for fl_cxi, event_crystfel_list in chunk_dict.items():
            # Split the workload...
            event_crystfel_list_in_chunk = split_dict_into_chunk(event_crystfel_list, max_num_chunk = mpi_size)

            # Process chunk by each worker...
            if mpi_rank != 0:
                event_crystfel_list_per_worker = event_crystfel_list_in_chunk[mpi_rank]
                metadata_list, peak_list = self.extract_metadata_and_labeled_peak_from_event(fl_stream, chunk_dict, fl_cxi, event_crystfel_list_per_worker)

                data_to_send = (metadata_list, peak_list)
                mpi_comm.send(data_to_send, dest = 0, tag = mpi_data_tag)

            if mpi_rank == 0:
                event_crystfel_list_per_worker = event_crystfel_list_in_chunk[mpi_rank]
                metadata_list, peak_list = self.extract_metadata_and_labeled_peak_from_event(fl_stream, chunk_dict, fl_cxi, event_crystfel_list_per_worker)

                for i in range(1, mpi_size, 1):
                    data_received = mpi_comm.recv(source = i, tag = mpi_data_tag)
                    metadata_list_per_worker, peak_list_per_worker = data_received

                    metadata_list.extend(metadata_list_per_worker)
                    peak_list.extend(peak_list_per_worker)

        return metadata_list, peak_list


    def extract_metadata_and_labeled_peak_from_event(self, fl_stream, chunk_dict, fl_cxi, event_crystfel_list):
        metadata_list = []
        peak_list     = []
        # Only keep those both found and indexed peaks...
        for event_crystfel in event_crystfel_list:
            # Get panels in an event...
            panel_dict = chunk_dict[fl_cxi][event_crystfel]

            # Find all peaks...
            for panel, peak_per_panel_dict in panel_dict.items():
                # Accumulate metadata of a label...
                panel_descriptor = (fl_stream, fl_cxi, event_crystfel, panel)
                metadata_list.append(panel_descriptor)

                # Accumulate data for making a label...
                peak_list.append(peak_per_panel_dict)

        return metadata_list, peak_list


    def calc_peak_SNR(self, img_peak, radius_inner_box = 2):
        size_y, size_x = img_peak.shape[-2:]

        cy = size_y // 2
        cx = size_x // 2

        x_min_inner_box = max(cx - radius_inner_box-1, 0)
        x_max_inner_box = min(cx + radius_inner_box, size_x)
        y_min_inner_box = max(cy - radius_inner_box-1, 0)
        y_max_inner_box = min(cy + radius_inner_box, size_y)

        ## y_min_inner_box = 0      + inner_box_radius_offset
        ## x_min_inner_box = 0      + inner_box_radius_offset
        ## y_max_inner_box = size_y - inner_box_radius_offset
        ## x_max_inner_box = size_x - inner_box_radius_offset

        select_matrix = np.zeros((size_y, size_x), dtype = bool)

        # Create a selection matrix for extract signal peaks...
        select_matrix[y_min_inner_box : y_max_inner_box,
                      x_min_inner_box : x_max_inner_box] = True

        # Choose signal pixel and bg pixel...
        img_signal = img_peak[..., select_matrix]
        img_bg     = img_peak[...,~select_matrix]

        # Calculate the mean bg...
        mean_img_bg = np.mean(img_bg)

        # Background subtraction...
        img_signal -= mean_img_bg
        mean_img_signal = np.mean(img_signal)

        # Calculate signal noise ratio...
        snr = mean_img_signal / mean_img_bg

        return snr


    def calc_peak_SNR_with_mask(self, img_peak, mask_peak):
        # Choose signal pixel and bg pixel...
        img_sg = img_peak[ mask_peak]
        img_bg = img_peak[~mask_peak]

        ## img_sg *= img_sg
        ## img_bg *= img_bg

        ## img_sg = img_sg.sum() / img_sg.size
        ## img_bg = img_bg.sum() / img_bg.size

        ## snr = 10 * np.log10(img_sg / img_bg)

        ## # Calculate the mean bg...
        ## mean_img_bg = np.mean(img_bg)

        ## # Background subtraction...
        ## img_sg -= mean_img_bg
        ## mean_img_sg = np.mean(img_sg)

        ## # Calculate signal noise ratio...
        ## snr = mean_img_sg / mean_img_bg

        snr = img_sg.mean() / img_bg.mean()

        return snr


    def __len__(self):
        return len(self.seq_random_list)


    def cache_img(self, idx_list = []):
        ''' Cache image in the seq_random_list unless a subset is specified.
        '''
        # If subset is not give, then go through the whole set...
        if not len(idx_list): idx_list = range(len(self.metadata_list))

        for idx in idx_list:
            # Skip those have been recorded...
            if idx in self.peak_cache_dict: continue

            # Otherwise, record it...
            img, label = self.get_img_and_label(idx, verbose = True)
            self.peak_cache_dict[idx] = (img, label)

        return None


    def mpi_cache_img(self):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from peaknet.utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        idx_list = range(len(self.metadata_list))

        # Split the workload...
        idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

        # Process chunk by each worker...
        # No need to sync the peak_cache_dict across workers
        if mpi_rank != 0:
            if mpi_rank < len(idx_list_in_chunk):
                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                self.peak_cache_dict = self._mpi_cache_img_per_rank(idx_list_per_worker)

            mpi_comm.send(self.peak_cache_dict, dest = 0, tag = mpi_data_tag)

        if mpi_rank == 0:
            idx_list_per_worker = idx_list_in_chunk[mpi_rank]
            self.peak_cache_dict = self._mpi_cache_img_per_rank(idx_list_per_worker)

            for i in range(1, mpi_size, 1):
                peak_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)

                self.peak_cache_dict.update(peak_cache_dict)

        return None


    def _mpi_cache_img_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        peak_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in peak_cache_dict: continue

            # Otherwise, record it...
            img, label = self.get_img_and_label(idx, verbose = True)
            peak_cache_dict[idx] = (img, label)

        return peak_cache_dict


    def get_img_and_label(self, idx, verbose = False):
        '''
        Get both the image and the label (a mask).  

        Caveat: The model requires an extra dimension in returned image.
        '''
        # Unpack the descriptor to locate a panel...
        fl_stream, fl_cxi, event_crystfel, panel = self.metadata_list[idx]

        # Get geom information...
        stream_dict       = self.stream_cache_dict[fl_stream]
        geom_dict         = stream_dict['geom']
        ## cheetah_geom_dict = GeomInterpreter(geom_dict).to_cheetah()
        python_geom_dict = GeomInterpreter(geom_dict).to_python()

        # Load an image...
        # From scratch and cache it
        cache_key = (fl_stream, fl_cxi, event_crystfel)
        if not cache_key in self.raw_img_cache_dict:
            with h5py.File(fl_cxi, 'r') as fh:
                # Fetch raw image and mask...
                ## raw_img = fh["/entry_1/instrument_1/detector_1/data"][event_crystfel]
                raw_img = fh["/entry_1/data_1/data"][event_crystfel]
                mask    = fh["/entry_1/data_1/mask"][event_crystfel if self.is_batch_mask else ()]

                # Subtract user mask from the global mask...
                # mask in psana definition:
                # - 1 or True means bad pixel
                # - 0 or False means good pixel
                if self.user_mask is not None:
                    mask[mask == self.user_mask] = 0

                # Apply mask to image...
                raw_img *= np.where(mask > 0, 0, 1)

                if self.add_channel_ok: raw_img = raw_img[None,]

                self.raw_img_cache_dict[cache_key] = raw_img

        # Otherwise, load it from cache
        raw_img = self.raw_img_cache_dict[cache_key]

        # Select an area as a panel image and a masked panel...
        ## x_min, y_min, x_max, y_max = cheetah_geom_dict[panel]
        x_min, y_min, x_max, y_max = python_geom_dict[panel]
        panel_img = raw_img[..., y_min : y_max, x_min : x_max]

        # Create a mask that works as the label...
        panel_label         = np.zeros_like(panel_img, dtype = int)    # ...A mask
        mask_radius         = self.mask_radius
        snr_threshold       = self.snr_threshold
        size_y, size_x      = panel_img.shape[-2:]
        peak_per_panel_list = self.peak_list[idx]

        # Cutoff threshold using min ADU...
        adu_threshold = self.adu_threshold

        # Set up structure to find connected component in 2D only...
        structure = np.ones((3, 3, 3))

        for peak_type, peak_list in peak_per_panel_list.items():
            if peak_type == 'indexed' and (not self.uses_indexed): continue

            for peak in peak_list:
                # Unack coordiante...
                x, y = peak

                # Round an 'indexed' peak to the nearest integer coordinate...
                if peak_type == 'indexed':
                    x += 0.5
                    y += 0.5

                x = int(x)
                y = int(y)

                # Get local coordinate for the panel...
                x = x - x_min
                y = y - y_min

                # Find a patch around a peak with global image coordinate...
                patch_x_min = max(x - mask_radius-1, 0)
                patch_x_max = min(x + mask_radius, size_x)
                patch_y_min = max(y - mask_radius-1, 0)
                patch_y_max = min(y + mask_radius, size_y)
                patch_img = panel_img[..., patch_y_min : patch_y_max, patch_x_min : patch_x_max]

                ## # Calculate the SNR of the patch...
                ## snr = self.calc_peak_SNR(patch_img)
                ## if snr < snr_threshold: continue

                ## # Figure out the local threshold...
                std_level = 1
                patch_mean = np.mean(patch_img)
                patch_std  = np.std (patch_img)
                threshold  = patch_mean + std_level * patch_std

                # [OVER-ENGINEERING WARNING] Find the most interesting area...
                mask_peak = patch_img >= threshold
                label_peak, num_peak = ndimage.label(mask_peak, structure = structure)

                # Modify label if necessary...
                if num_peak > 1:
                    # [DETAIL] i+1 is the label
                    label_human_engineered, _ = sorted([ (i+1, (label_peak == i+1).sum()) for i in range(num_peak) ], key = lambda x:x[1], reverse = True)[0]
                    label_peak[label_peak != label_human_engineered] = 0

                mask_peak = label_peak > 0

                # Calculate the SNR of the patch...
                snr = self.calc_peak_SNR_with_mask(patch_img, mask_peak)
                if snr < snr_threshold: continue

                # Mask the patch area out...
                ## panel_label[..., patch_y_min : patch_y_max, patch_x_min : patch_x_max][~(patch_img < adu_threshold)] += 1
                panel_label[..., patch_y_min : patch_y_max, patch_x_min : patch_x_max][mask_peak] += 1

        # Assure that it is a binary mask...
        panel_label[panel_label != 0] = 1

        if verbose: logger.info(f'DATA LOADING - {fl_cxi} {event_crystfel} {panel}.')

        return panel_img, panel_label


    def __getitem__(self, idx):
        img, label = self.peak_cache_dict[idx] if   idx in self.peak_cache_dict \
                                               else self.get_img_and_label(idx)

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        ## # Normalize input image...
        ## img_mean = np.mean(img)
        ## img_std  = np.std(img)
        ## img_norm = (img - img_mean) / img_std

        img_norm = img

        return img_norm, label




class SFXPanelDatasetMini(SFXPanelDataset):
    def __init__(self, config):
        super().__init__(config)

        logger.info("___/ Dataset size after splitting \___")
        logger.info(f"KV - size : {len(self.metadata_list)}")

        if self.size_sample is not None: self.form_miniset()


    def __len__(self):
        return len(self.metadata_list)


    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)

        return img, label


    def form_miniset(self):
        size_sample = self.size_sample

        # Draw a miniset (subset)...
        idx_metadata_list = range(len(self.metadata_list))
        idx_metadata_miniset = random.sample(idx_metadata_list, k = size_sample)

        # Reconstruct metadata_list and peak_list...
        self.metadata_list = [ self.metadata_list[i] for i in idx_metadata_miniset ]
        self.peak_list     = [ self.peak_list[i]     for i in idx_metadata_miniset ]
