import numpy as np
import pickle
import regex

import torch
import torch.nn.functional as F

from math import isnan

from .trans import coord_crop_to_img


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




class CheetahConverter:

    def __init__(self, path_cheetah_geom):
        self.path_cheetah_geom = path_cheetah_geom

        with open(self.path_cheetah_geom, 'rb') as handle:
            cheetah_geom_dict = pickle.load(handle)

        cheetah_geom_regex = regex.compile(r"""
            (?x)
            [a-z](?<PANEL>[0-9]+)[a-z](?<ASIC>[0-9]+)
        """)
        cheetah2psana_geom_dict = {}
        for panel_str, (x_min, y_min, x_max, y_max) in cheetah_geom_dict.items():
            match = cheetah_geom_regex.match(panel_str)
            if match is None:
                continue

            capture_dict = match.capturesdict()

            panel_id = capture_dict['PANEL'][0]
            asic_id  = capture_dict['ASIC' ][0]

            x_max += 1
            y_max += 1

            panel_id = int(panel_id)
            if panel_id not in cheetah2psana_geom_dict: cheetah2psana_geom_dict[panel_id] = [x_min, y_min, x_max, y_max]
            panel_x_min, panel_y_min, panel_x_max, panel_y_max = cheetah2psana_geom_dict[panel_id]
            panel_x_min = min(panel_x_min, x_min)
            panel_y_min = min(panel_y_min, y_min)
            panel_x_max = max(panel_x_max, x_max)
            panel_y_max = max(panel_y_max, y_max)
            cheetah2psana_geom_dict[panel_id] = panel_x_min, panel_y_min, panel_x_max, panel_y_max

        self.cheetah_geom_dict       = cheetah_geom_dict
        self.cheetah2psana_geom_dict = cheetah2psana_geom_dict


    def convert_to_cheetah_img(self, img):
        W_cheetah, H_cheetah = list(self.cheetah2psana_geom_dict.values())[-1][-2:]
        cheetah_img = np.zeros((H_cheetah, W_cheetah), dtype = np.float32)

        for panel_id, (x_min, y_min, x_max, y_max) in self.cheetah2psana_geom_dict.items():
            H = y_max - y_min
            W = x_max - x_min
            cheetah_img[y_min:y_max, x_min:x_max] = img[panel_id, 0:H, 0:W]

        return cheetah_img


    def convert_to_psana_img(self, cheetah_img):
        # Figure out channel dimension...
        C = len(self.cheetah2psana_geom_dict)

        # Figure out spatial dimension...
        x_min, y_min, x_max, y_max = self.cheetah2psana_geom_dict[0]
        H = y_max - y_min
        W = x_max - x_min

        # Initialize a zero value image...
        img = np.zeros((C, H, W), dtype = np.float32)

        for panel_id, (x_min, y_min, x_max, y_max) in self.cheetah2psana_geom_dict.items():
            img[panel_id] = cheetah_img[y_min:y_max, x_min:x_max]

        return img


    def convert_to_cheetah_coords(self, peaks_psana_list):
        peaks_cheetah_list = [
            self.convert_to_cheetah_coord(idx_panel, y, x)
            for idx_panel, y, x in peaks_psana_list
        ]

        return peaks_cheetah_list


    def convert_to_cheetah_coord(self, idx_panel, y, x):
        x_min, y_min, x_max, y_max = self.cheetah2psana_geom_dict[idx_panel]

        x += x_min
        y += y_min

        return idx_panel, y, x




def remove_outliers(data, percentile = 5):
    """Removes outliers from a numpy array using the IQR method."""
    q1, q3 = np.percentile(data, [percentile, 100 - percentile])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = np.logical_and(data >= lower_bound, data <= upper_bound)
    return data * mask




def apply_mask(data, mask, mask_value = np.nan):
    """ 
    Return masked data.

    Args:
        data: numpy.ndarray with the shape of (B, H, W).·
              - B: batch of images.
              - H: height of an image.
              - W: width of an image.

        mask: numpy.ndarray with the shape of (B, H, W).·

    Returns:
        data_masked: numpy.ndarray.
    """ 
    # Mask unwanted pixels with np.nan...
    data_masked = np.where(mask, data, mask_value)

    return data_masked
