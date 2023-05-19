#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import h5py
import time
import pickle
import yaml
import argparse

from cupyx.scipy import ndimage
import cupy as cp

from peaknet.app     import PeakFinder
from peaknet.plugins import PsanaImg, apply_mask
from peaknet.trans   import center_crop
from peaknet.utils   import split_list_into_chunk


# Set up MPI
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_data_tag = 11


# [[[ ARG PARSE ]]]
parser = argparse.ArgumentParser(description='Process a yaml file.')
parser.add_argument('yaml', help='The input yaml file.')
args = parser.parse_args()

# [[[ Configure ]]]
fl_yaml = args.yaml
basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

# Load the YAML file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# Access the values
# ___/ PeakNet model \___
timestamp            = config['timestamp'           ]
epoch                = config['epoch'               ]
tag                  = config['tag'                 ]
uses_skip_connection = config['uses_skip_connection']
uses_DataParallel    = config['uses_DataParallel'   ]

# ___/ Experimental data \___
# Psana...
exp                  = config['exp'          ]
run                  = config['run'          ]
img_load_mode        = config['img_load_mode']
access_mode          = config['access_mode'  ]
detector_name        = config['detector_name']
photon_energy        = config['photon_energy']
encoder_value        = config['encoder_value']

# Data range...
event_min            = config['event_min']
event_max            = config['event_max']

# ___/ Peak finding \___
uses_int_coords      = config['uses_int_coords']
mask_custom_npy      = config['mask_custom_npy']
max_num_peak         = config['max_num_peak']
num_model_on_gpu     = config['num_model_on_gpu']

if mpi_size < num_model_on_gpu: num_model_on_gpu = mpi_size
gpu_worker_ranks = list(range(num_model_on_gpu))    # Manager rank (0) is on gpu by default


# ___/ Output \___
dir_results          = config["dir_results"]
pf_tag               = config["pf_tag"]
batch_size           = config["batch_size"]

basename_out  = f"{basename_yaml}.{timestamp}.epoch_{epoch}"
basename_out += pf_tag

# ___/ Misc \___
path_cheetah_geom    = config["path_cheetah_geom"]


# MPI GPU
if mpi_rank in gpu_worker_ranks:
    # [[[ Psana ]]]
    # Set up experiments...
    psana_img = PsanaImg(exp, run, access_mode, detector_name)

    # [[[ Peakfinder ]]]
    # Load trained model...
    fl_chkpt = None if timestamp is None else f"{timestamp}.epoch_{epoch}{tag}.chkpt"
    path_chkpt = None if fl_chkpt is None else os.path.join("chkpts", fl_chkpt)

    # Load the peak finder...
    pf = PeakFinder(path_chkpt = path_chkpt, path_cheetah_geom = path_cheetah_geom)
    device = pf.device

    # [[[ Peakfinding Metadata ]]]
    min_num_peaks = 15
    if event_min is None: event_min = 0
    if event_max is None: event_max = len(psana_img.timestamps)

    mask_bad_pixel = psana_img.create_bad_pixel_mask()
    mask_custom    = np.load(mask_custom_npy) if mask_custom_npy is not None else None

    def find_peak_per_rank(events_per_rank):
        ''' Assume all variables are global.  This function is only to wrap a
            process to save lines of codes.
        '''
        event_filtered_list = []
        for enum_idx, event in enumerate(events_per_rank):
            print(f"___/ Event {event:06d} (from rank {mpi_rank}) \___")

            img = psana_img.get(event, None, img_load_mode)

            if img is None: continue

            # Apply psana mask...
            img = apply_mask(img, mask_bad_pixel, mask_value = 0)

            # Apply custom mask...
            if mask_custom is not None: img = apply_mask(img, mask_custom, mask_value = 0)

            # Package the image into tensor...
            img = torch.tensor(img).type(dtype=torch.float)[:,None].to(device)

            # Normalization is done in find_peak below

            time_start = time.time()
            peak_list, prediction_map = pf.find_peak_w_softmax(img, min_num_peaks = min_num_peaks, uses_geom = False, returns_prediction_map = True)
            time_end = time.time()
            time_delta = time_end - time_start
            print(f"Time delta: {time_delta * 1e3:.4f} millisecond.")

            # [NEW] Maybe do some safeguarding here for num_peaks > max_num_peak
            if len(peak_list) < min_num_peaks: continue

            # Save coordinates...
            if uses_int_coords:
                # Make coordinates compatible with Cheetah before saving...
                peak_list = [ (_, round(y), round(x)) for _, y, x in peak_list ]

            # Save masks...
            mask_peaknet = prediction_map[:,2]    # 0 : bg
                                                  # 1 : peaks
                                                  # 2 : nozzle ring scattering
                                                  # B, C, H, W

            mask = mask_peaknet.astype(int)
            if mask_bad_pixel is not None: mask += (1-mask_bad_pixel[None,].astype(int))
            if mask_custom    is not None: mask += (1-mask_custom.astype(int))

            # Cheetah wants the masked area to be True...
            mask_to_save = mask.astype(bool)

            # Saving...
            event_filtered_list.append((event, peak_list, mask_to_save))

        return event_filtered_list

    # Split all events into MPI Chunks...
    events = range(event_min, event_max)
    batch_events = split_list_into_chunk(events, max_num_chunk = batch_size)

    for batch_idx, events in enumerate(batch_events):
        events_in_chunk = split_list_into_chunk(events, max_num_chunk = num_model_on_gpu)

        # MPI manager
        # - Find peaks
        # - Collect peaks from other workers
        if mpi_rank == 0:
            if batch_idx == 0: event_filtered_list = []

            # Find peaks
            events_per_rank = events_in_chunk[mpi_rank]
            event_filtered_list_per_rank = find_peak_per_rank(events_per_rank)

            event_filtered_list.extend(event_filtered_list_per_rank)

            # Collect peaks from other workers
            for i in range(1, num_model_on_gpu, 1):
                data_received = mpi_comm.recv(source = i, tag = mpi_data_tag)
                event_filtered_list.extend(data_received)

        if mpi_rank != 0:
            # Find peaks
            events_per_rank = events_in_chunk[mpi_rank] if len(events_in_chunk) > mpi_rank else []
            event_filtered_list_per_rank = find_peak_per_rank(events_per_rank)

            # Send peaks to the manager
            data_to_send = event_filtered_list_per_rank
            mpi_comm.send(data_to_send, dest = 0, tag = mpi_data_tag)


# GPU nodes are done with peak finding...
mpi_comm.Barrier()

# [[[ Export ]]]
def export(batch_idx, chunk_idx, event_filtered_list, psana_img):
    # Metadata...
    # !!!CAVEAT: it looks like cheetah thinks image and mask should be in (H, W).
    # !!!However, (B, H, W) might be required for multi-panel detectors.
    num_event      = len(event_filtered_list)
    size_y, size_x = event_filtered_list[0][-1].shape[-2:] if num_event > 0 else (0, 0)

    # Create h5...
    fl_h5 = f"{basename_out}.batch_{batch_idx:02d}.chunk_{chunk_idx:02d}.cxi"
    path_h5 = os.path.join(dir_results, fl_h5)
    print(f"Writing {path_h5} from rank {chunk_idx}.", flush = True)
    with h5py.File(path_h5, 'w') as f:
        # Initialize h5...
        # ...Img
        f.create_dataset('/entry_1/data_1/data'         , (num_event, size_y, size_x),
                         dtype = 'float32'   , )

        # ...Mask
        f.create_dataset('/entry_1/data_1/mask'         , (num_event, size_y, size_x),
                         dtype            = 'int'       ,
                         compression_opts = 6           ,
                         compression      = 'gzip'      , )

        # ...Pos X
        f.create_dataset('/entry_1/result_1/peakXPosRaw', (num_event, max_num_peak),
                         dtype  = 'float32'             ,
                         chunks = (1, max_num_peak)     , )

        # ...Pos Y
        f.create_dataset('/entry_1/result_1/peakYPosRaw', (num_event, max_num_peak),
                         dtype  = 'float32'             ,
                         chunks = (1, max_num_peak)     , )

        # ...Event
        f.create_dataset('/entry_1/result_1/peakEvent'  , (num_event, ),
                         dtype  = 'int'                 , )

        # ...nPeaks
        f.create_dataset('/entry_1/result_1/nPeaks'     , (num_event, ),
                         dtype  = 'int'                 , )

        # Write data per event...
        for event_enum_idx, (event, peaks_per_event, mask) in enumerate(event_filtered_list):
            time_start = time.monotonic()

            # ...Data
            img = psana_img.get(event, None, img_load_mode)
            f['/entry_1/data_1/data'][event_enum_idx] = img     # !!!CAVEAT (H, W)

            # ...Mask
            f['/entry_1/data_1/mask'][event_enum_idx] = mask[0]    # !!!CAVEAT (B, H, W), B = 1

            # ...Pos X, Pos Y
            for peak_enum_idx, peak in enumerate(peaks_per_event):
                if not (peak_enum_idx < max_num_peak): break

                seg, cheetahRow, cheetahCol = peak
                f['/entry_1/result_1/peakYPosRaw'][event_enum_idx, peak_enum_idx] = cheetahRow
                f['/entry_1/result_1/peakXPosRaw'][event_enum_idx, peak_enum_idx] = cheetahCol

            # ...Event
            f['/entry_1/result_1/peakEvent'][event_enum_idx] = event

            # ...nPeaks
            f['/entry_1/result_1/nPeaks'][event_enum_idx] = len(peaks_per_event)

            time_end = time.monotonic()
            time_delta = time_end - time_start
            print(f"Time delta: {time_delta * 1e3:.4f} millisecond. (enum {event_enum_idx}, event {event}, batch = {batch_idx}, rank = {chunk_idx})", flush = True)

# MPI manager (rank = 0) disseminates data to all MPI workers...
if mpi_rank == 0:
    batch_event_filtered_list = split_list_into_chunk(event_filtered_list, max_num_chunk = batch_size)

    # Inform all workers the number of chunks to work on...
    for i in range(1, mpi_size, 1):
        num_chunks   = len(batch_event_filtered_list)
        mpi_comm.send(num_chunks, dest = i, tag = mpi_data_tag)

    for batch_idx, batch_events in enumerate(batch_event_filtered_list):
        events_in_chunk = split_list_into_chunk(batch_events, max_num_chunk = mpi_size)
        for i in range(1, mpi_size, 1):
            data_to_send = events_in_chunk[i] if len(events_in_chunk) > i else []
            mpi_comm.send(data_to_send, dest = i, tag = mpi_data_tag)

        events_per_rank = events_in_chunk[0]

        # All ranks work on exporting the file...
        psana_img = PsanaImg(exp, run, access_mode, detector_name)
        chunk_idx = mpi_rank
        export(batch_idx, chunk_idx, events_per_rank, psana_img)

if mpi_rank != 0:
    num_chunks = mpi_comm.recv(source = 0, tag = mpi_data_tag)
    for batch_idx in range(num_chunks):
        events_per_rank = mpi_comm.recv(source = 0, tag = mpi_data_tag)

        if len(events_per_rank) == 0: continue

        # All ranks work on exporting the file...
        psana_img = PsanaImg(exp, run, access_mode, detector_name)

        chunk_idx = mpi_rank
        export(batch_idx, chunk_idx, events_per_rank, psana_img)

print(f"Done (rank {mpi_rank:02d})")

if mpi_rank == 0:
    MPI.Finalize()
    print(f"Main rank is working on merging.")

    # [[[ Merge ]]]
    # Metadata...
    num_event = len(event_filtered_list)
    size_y, size_x = event_filtered_list[0][-1].shape[-2:] if num_event > 0 else (0, 0)

    # Create a layout for the virtual dataset...
    layout = {
        '/entry_1/data_1/data'          : h5py.VirtualLayout(shape = (num_event, size_y, size_x), dtype = 'float32'),
        '/entry_1/data_1/mask'          : h5py.VirtualLayout(shape = (num_event, size_y, size_x), dtype = 'int'    ),
        '/entry_1/result_1/peakXPosRaw' : h5py.VirtualLayout(shape = (num_event, max_num_peak  ), dtype = 'float32'),
        '/entry_1/result_1/peakYPosRaw' : h5py.VirtualLayout(shape = (num_event, max_num_peak  ), dtype = 'float32'),
        '/entry_1/result_1/peakEvent'   : h5py.VirtualLayout(shape = (num_event,               ), dtype = 'int'    ),
        '/entry_1/result_1/nPeaks'      : h5py.VirtualLayout(shape = (num_event,               ), dtype = 'int'    ),
    }


    # Write data per event...
    num_events_processed = 0
    for batch_idx, batch_events in enumerate(batch_event_filtered_list):
        events_in_chunk = split_list_into_chunk(batch_events, max_num_chunk = mpi_size)

        for chunk_idx, events_per_chunk in enumerate(events_in_chunk):
            # Metadata
            ## h5_link = f"{basename_out}.mpi_{chunk_idx:02d}.batch_{batch_idx:02d}.cxi"
            h5_link = f"{basename_out}.batch_{batch_idx:02d}.chunk_{chunk_idx:02d}.cxi"
            num_event_per_chunk = len(events_per_chunk)

            # ...Data
            vsource = h5py.VirtualSource(h5_link, '/entry_1/data_1/data', shape = (num_event_per_chunk, size_y, size_x))
            layout['/entry_1/data_1/data'][num_events_processed : num_events_processed + num_event_per_chunk, :, :] = vsource

            # ...Mask
            vsource = h5py.VirtualSource(h5_link, '/entry_1/data_1/mask', shape = (num_event_per_chunk, size_y, size_x))
            layout['/entry_1/data_1/mask'][num_events_processed : num_events_processed + num_event_per_chunk, :, :] = vsource

            # ...Pos X
            vsource = h5py.VirtualSource(h5_link, '/entry_1/result_1/peakXPosRaw', shape = (num_event_per_chunk, max_num_peak))
            layout['/entry_1/result_1/peakXPosRaw'][num_events_processed : num_events_processed + num_event_per_chunk, :] = vsource

            # ...Pos Y
            vsource = h5py.VirtualSource(h5_link, '/entry_1/result_1/peakYPosRaw', shape = (num_event_per_chunk, max_num_peak))
            layout['/entry_1/result_1/peakYPosRaw'][num_events_processed : num_events_processed + num_event_per_chunk, :] = vsource

            # ...Event
            vsource = h5py.VirtualSource(h5_link, '/entry_1/result_1/peakEvent', shape = (num_event_per_chunk, ))
            layout['/entry_1/result_1/peakEvent'][num_events_processed : num_events_processed + num_event_per_chunk] = vsource

            # ...nPeaks
            vsource = h5py.VirtualSource(h5_link, '/entry_1/result_1/nPeaks', shape = (num_event_per_chunk, ))
            layout['/entry_1/result_1/nPeaks'][num_events_processed : num_events_processed + num_event_per_chunk] = vsource

            num_events_processed += num_event_per_chunk

    # Create h5...
    fl_h5 = f"{basename_out}.cxi"
    path_h5 = os.path.join(dir_results, fl_h5)

    with h5py.File(path_h5, 'w') as f:
        # Write virtual data...
        for k, v in layout.items():
            f.create_virtual_dataset(k, v, fillvalue = 0)

        # Write global information...
        num_event_all = event_max - event_min
        # ...nPeaksAll
        f.create_dataset('/entry_1/result_1/nPeaksAll',
                         data  = -np.ones(num_event_all,),
                         dtype = 'int')

        # ...Pos X All
        f.create_dataset('/entry_1/result_1/peakXPosRawAll'    , (num_event_all, max_num_peak),
                         dtype = 'float32',
                         chunks = (1, max_num_peak))

        # ...Pos Y All
        f.create_dataset('/entry_1/result_1/peakYPosRawAll'    , (num_event_all, max_num_peak),
                         dtype = 'float32',
                         chunks = (1, max_num_peak))

        # ...Intensity
        f.create_dataset('/entry_1/result_1/peakTotalIntensity', (num_event_all, max_num_peak),
                         dtype = 'float32',
                         chunks = (1, max_num_peak))

        # ...LCLS
        f.create_dataset('/LCLS/detector_1/EncoderValue', (1, ), dtype = 'float32', )
        f.create_dataset('/LCLS/photon_energy_eV'       , (1, ), dtype = 'float32', )

        for event_enum_idx, (event, peaks_per_event, _) in enumerate(event_filtered_list):
            f['/entry_1/result_1/nPeaksAll'][event] = len(peaks_per_event)

            for peak_enum_idx, peak in enumerate(peaks_per_event):
                if not (i < max_num_peak): break

                seg, cheetahRow, cheetahCol = peak
                f['/entry_1/result_1/peakXPosRawAll'][event, peak_enum_idx] = cheetahCol
                f['/entry_1/result_1/peakYPosRawAll'][event, peak_enum_idx] = cheetahRow

        f['/LCLS/detector_1/EncoderValue'       ][0] = encoder_value
        f['/LCLS/photon_energy_eV'              ][0] = photon_energy
        ## f['/entry_1/result_1/peakTotalIntensity'][0] = photon_energy


    print(f"Main rank is done with merging. [END]")

    print(f"Main rank is generating lst file.")
    fl_lst = f"{basename_out}.lst"
    path_lst = os.path.join(dir_results, fl_lst)
    with open(path_lst,'w') as fh:
        for i, (event, peaks_per_event, _) in enumerate(event_filtered_list): 
            # Index this event???
            nPeaks = len(peaks_per_event)
            if nPeaks > max_num_peak: continue

            fh.write(f"{path_h5} //{i}")
            fh.write("\n")
    print(f"Main rank is done with generating lst file. [END]")
