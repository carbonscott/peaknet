import numpy as np
import os
import h5py
import torch
import torch.nn            as nn
import torch.nn.functional as F
from peaknet.app     import PeakFinder
from peaknet.plugins import CheetahConverter, PsanaImg, apply_mask
import cupy as cp
from skimage import measure
from PIL import Image
import json
import matplotlib.pyplot as plt

path_cxi = "inference_results/peaknet.cxic00318_0123.cxi"

try:
    fh.close()
except:
    pass

fh = h5py.File(path_cxi, "r")


path_chkpt = "experiments/chkpts/opt_cxi.00.2023_1108_2307_12.epoch_96.chkpt"
path_yaml = "experiments/yaml/opt_cxi.00.yaml"
pf = PeakFinder(path_chkpt = path_chkpt, path_cheetah_geom = None, path_yaml_config = path_yaml)

device = pf.device


idx = 0
img = fh.get('entry_1/data_1/data')[idx:idx+4]

img_tensor = torch.tensor(img[:, None,]).to(device)
batch_ndimage_labels = pf.generate_ndimage_labels(img_tensor)


drc_root = 'data_root/pf/'
coco_dict = {
    "images" : [],
    "annotations" : [],
    "categories" : [
        {
            "id": 1,
            "name": "peak"
        },
        {
            "id": 2,
            "name": "artifact scattering"
        },
        {
            "id": 3,
            "name": "bad pixel"
        },
    ],
}
for enum_idx, (ndimage_label, num_peaks) in enumerate(batch_ndimage_labels):
    contours = measure.find_contours(cp.asnumpy(ndimage_label > 0))

    # Convert contours to COCO format
    coco_polygons = []
    for contour in contours:
        # Flatten the contour array and round off the values
        contour = np.around(contour).astype(int).flatten().tolist()
        coco_polygons.append(contour)

    # Create a COCO annotation
    coco_annotation = {
        "segmentation": coco_polygons,
        "area": 0, # calculate the area of the polygon if needed
        "iscrowd": 0, # 0 or 1 depending on your data
        "image_id": enum_idx, # ID of the image this annotation belongs to
        "category_id": 1, # ID of the category this annotation belongs to
        "id": enum_idx # ID of this annotation
    }

    file_img = f"demo_{enum_idx:06d}.png"
    rel_path_img = os.path.join("label_studio", file_img)
    path_img = os.path.join(drc_root, rel_path_img)

    H, W = ndimage_label.shape
    coco_dict["images"].append(
            {
                "file_name": path_img,
                "id": enum_idx,
                "width": W,
                "height": H,
            },
    )
    coco_dict["annotations"].append(
        coco_annotation
    )

    # Export to png...
    data = img[enum_idx]

    # Apply a colormap using Matplotlib
    cmap = plt.cm.viridis  # Replace with any other colormap you like

    # Calculate the dynamic range clipping
    sigma_cut = 1
    vmin = np.mean(data) - 0 * data.std()
    vmax = np.mean(data) + sigma_cut * data.std()

    # Clip the values and normalize between 0 and 1
    data = np.clip(data, vmin, vmax)

    # Normalize the image data
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    data_normalized = norm(data)

    # Apply the colormap
    colored_data = cmap(data_normalized)

    # Convert to an image
    colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)  # Discard the alpha channel and convert to 8-bit array
    img_pil = Image.fromarray(colored_data)

    # Save the image with PIL
    img_pil.save(rel_path_img)