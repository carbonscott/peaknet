import torch
import torch.nn            as nn
import torch.nn.functional as F

import numpy as np
import os
import h5py

from PIL import Image

from peaknet.app     import PeakFinder
from peaknet.plugins import CheetahConverter, PsanaImg, apply_mask

path_chkpt = "experiments/chkpts/opt_cxi.00.2023_1108_2307_12.epoch_96.chkpt"
path_yaml = "experiments/yaml/opt_cxi.00.yaml"
pf = PeakFinder(path_chkpt = path_chkpt, path_cheetah_geom = None, path_yaml_config = path_yaml)

# path_geom = "cheetah_geom.jungfrau.pickle"
# cheetah_converter = CheetahConverter(path_geom)

drc_data  = 'label_studio'

exp           = 'cxic00318'
run           = 123
img_load_mode = 'calib'
access_mode   = 'idx'
detector_name = 'jungfrau4M'

psana_img = PsanaImg(exp, run, access_mode, detector_name)
mask_bad_pixel = psana_img.create_bad_pixel_mask()
basename = f"{exp}_{run:04d}"

def save_float32_image(img, sigma_cut = 0.2):
    img_copy = img[:]
    vmin = img_copy.mean()
    vmax = img_copy.mean() + sigma_cut * img_copy.std()
    img_copy[img_copy > vmax] = vmax
    img_copy[img_copy < vmin] = vmin

    # Normalize the image data to 0-255
    img_norm = (img_copy - np.min(img_copy)) / (np.max(img_copy) - np.min(img_copy))
    img_scale = (255 * img_norm).astype(np.uint8)
    img_pil = Image.fromarray(img_scale)

    return img_pil, img_copy

device = pf.device
events = [5100, 5177, 16987, 18789, 25243, 29400, 29741, 66893, 69304, 241841]
sigma_cuts = [0.2, 4, 6]
for event in events:
    basename_event = f"{basename}_{event:06d}"

    # peakfinding
    img = psana_img.get(event, None, 'calib')
    img = apply_mask(img, mask_bad_pixel, mask_value = 0)

    img_tensor = torch.tensor(img).type(dtype=torch.float)[:,None,].to(device)

    # H, W = img.shape[-2:]
    # base_size = 2**5
    # dummy_offset = 0
    # H_pad, W_pad = math.ceil(H / base_size + dummy_offset) * base_size, math.ceil(W / base_size + dummy_offset) * base_size

    peaks, mask_predicted = pf.find_peak_w_softmax(img_tensor, returns_prediction_map = True, uses_geom = False, uses_mixed_precision = True, uses_batch_norm = True)

    label = mask_predicted[:, 1]

    # Assemble the image for output...
    img_output   = psana_img.assemble(multipanel = img)
    label_output = psana_img.assemble(multipanel = label)

    # Export label to hdf5 (when using requests is not allowed in the network)...    
    output_label_file = f"{basename_event}.npy"
    output_label_path = os.path.join(drc_data, output_label_file)
    np.save(output_label_path, label_output)

    # Save the image as JPEG
    for sig_idx, sigma_cut in enumerate(sigma_cuts):
        img_pil, img_copy = save_float32_image(img_output, sigma_cut = sigma_cut)

        # Create an image object and save it
        output_file = f"{basename_event}.sig{sig_idx:01d}.jpeg"
        output_jpeg_path = os.path.join(drc_data, output_file)
        img_pil.save(output_jpeg_path)
