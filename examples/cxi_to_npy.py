#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
import pickle
import yaml
import argparse

from configurator import Configurator

parser = argparse.ArgumentParser(description = "Yaml file." ,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--yaml_file",
                    required = True,
                    type     = str,
                    help     = """Yaml file""",)
args = parser.parse_args()

# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config_dict = yaml.safe_load(fh)
CONFIG = Configurator.from_dict(config_dict)

path_cheetah_geom_dict = CONFIG.PATH_CHEETAH_GEOM_DICT
cxi_key                = CONFIG.CXI_KEY
path_cxi               = CONFIG.PATH_CXI
panel_selected         = CONFIG.PANEL_SELECTED
path_output            = CONFIG.PATH_OUTPUT
bad_pixel_label        = CONFIG.BAD_PIXEL_LABEL

# Obtain cheetah geom...
with open(path_cheetah_geom_dict, 'rb') as f:
    cheetah_geom_dict = pickle.load(f)
name_to_enum    = { k : i for i, k in enumerate(cheetah_geom_dict.keys()) }
panel_enum_list = [ name_to_enum[i] for i in panel_selected ]

# Obtain the panel size...
meta_panel = next(iter(cheetah_geom_dict.keys()))
x_min, y_min, x_max, y_max = cheetah_geom_dict[meta_panel]
H          = y_max - y_min + 1
W          = x_max - x_min + 1

key_data    = cxi_key['data']
key_mask    = cxi_key['mask']
key_segmask = cxi_key['segmask']
with h5py.File(path_cxi, 'r') as fh:
    # ___/ MASK \___
    # Obtain a batch of masks...
    batch_mask = fh.get(key_mask)[:]

    # Fetch the cheetah tile size...
    H_cheetah, W_cheetah = batch_mask.shape[-2:]

    NH_cheetah = H_cheetah // H
    NW_cheetah = W_cheetah // W

    # Reshape a cheetah tile to a group of panels...
    batch_mask = batch_mask.reshape(-1, NH_cheetah, H, NW_cheetah, W)
    batch_mask = batch_mask.transpose((0, 1, 3, 2, 4))
    batch_mask = batch_mask.reshape(-1, NH_cheetah * NW_cheetah, H, W)

    # Only select these specified...
    batch_mask = batch_mask[:, panel_enum_list] # (B, N, H, W)
    batch_mask = batch_mask.reshape(-1, H, W)   # (B*N, H, W)

    # ___/ IMAGE \___
    # Obtain a batch of images...
    batch_img = fh.get(key_data)[:]

    # Reshape a cheetah tile to a group of panels...
    batch_img  = batch_img.reshape(-1, NH_cheetah, H, NW_cheetah, W)
    batch_img  = batch_img.transpose((0, 1, 3, 2, 4))
    batch_img  = batch_img.reshape(-1, NH_cheetah * NW_cheetah, H, W)

    # Only select these specified...
    batch_img = batch_img[:, panel_enum_list] # (B, N, H, W)

    # Apply mask...
    batch_img *= 1-batch_mask[None,]

    # Regroup...
    batch_img = batch_img.reshape(-1, H, W)   # (B*N, H, W)

    # ___/ SEGMASK \___
    # Obtain a batch of segmasks...
    batch_segmask = fh.get(key_segmask)[:]

    # Fetch the cheetah tile size...
    H_cheetah, W_cheetah = batch_segmask.shape[-2:]

    # Reshape a cheetah tile to a group of panels...
    batch_segmask = batch_segmask.reshape(-1, NH_cheetah, H, NW_cheetah, W)
    batch_segmask = batch_segmask.transpose((0, 1, 3, 2, 4))
    batch_segmask = batch_segmask.reshape(-1, NH_cheetah * NW_cheetah, H, W)

    # Only select these specified...
    batch_segmask = batch_segmask[:, panel_enum_list] # (B, N, H, W)
    batch_segmask = batch_segmask.reshape(-1, H, W)   # (B*N, H, W)

    batch_img[batch_segmask == bad_pixel_label] = 0.0

    ## # Compiple image and segmask...
    ## data = np.concatenate((batch_img, batch_segmask), axis = 1)    # (B*N, 2, 1, H, W)

# Save...
B, H, W = batch_img.shape
with h5py.File(path_output, 'w') as f:
    # ...Img
    f.create_dataset(key_data, (B, H, W),
                     dtype = 'float32', )

    # ...Mask
    f.create_dataset(key_segmask, (B, H, W),
                     dtype            = 'int',
                     compression_opts = 6,
                     compression      = 'gzip', )

    f[key_data][:]    = batch_img
    f[key_segmask][:] = batch_segmask
