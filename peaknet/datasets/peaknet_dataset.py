import csv
import h5py
import numpy as np

from torch.utils.data import Dataset

class PeakNetDatasetLoader:
    """
    This class only loads events sequentially saved in a list of hdf5 files
    The data distribution should be handled externally (like how you construct
    the csv).
    """
    def __init__(self, path_csv, trans_list = None):
        super().__init__()

        self.trans_list = trans_list

        # Importing data from csv...
        self.data_list = []
        with open(path_csv, 'r') as fh:
            lines = csv.reader(fh)

            # Open all hdf5 files and track their status...
            for line in lines:
                path_hdf5 = line[0]
                with h5py.File(path_hdf5, 'r') as f:
                    groups = f.get('data')
                    for group_idx in range(len(groups)):
                        self.data_list.append((path_hdf5, group_idx))

        return None


    def get_img(self, idx):
        path_hdf5, idx_in_hdf5 = self.data_list[idx]

        with h5py.File(path_hdf5, 'r') as f:
            # Obtain the image...
            k = f"data/data_{idx_in_hdf5:04d}/image"
            image = f.get(k)[()]

            # Obtain the label...
            k = f"data/data_{idx_in_hdf5:04d}/label"
            label = f.get(k)[()]

            # Obtain pixel map...
            k = f"data/data_{idx_in_hdf5:04d}/metadata/pixel_map"
            pixel_map_x, pixel_map_y, pixel_map_z = f.get(k)[()]

        pixel_map_x = np.round(pixel_map_x).astype(int)
        pixel_map_y = np.round(pixel_map_y).astype(int)
        pixel_map_z = np.round(pixel_map_z).astype(int)

        detector_image = np.zeros((pixel_map_x.max() - pixel_map_x.min() + 1,
                                   pixel_map_y.max() - pixel_map_y.min() + 1,
                                   pixel_map_z.max() - pixel_map_z.min() + 1), dtype = float)
        detector_label = np.zeros((pixel_map_x.max() - pixel_map_x.min() + 1,
                                   pixel_map_y.max() - pixel_map_y.min() + 1,
                                   pixel_map_z.max() - pixel_map_z.min() + 1), dtype = int)

        detector_image[pixel_map_x, pixel_map_y, pixel_map_z] = image
        detector_label[pixel_map_x, pixel_map_y, pixel_map_z] = label

        detector_image = detector_image.transpose((2, 0, 1))    # (H, W, C=1) -> (C, H, W)
        detector_label = detector_label.transpose((2, 0, 1))    # (H, W, C=1) -> (C, H, W)

        return detector_image, detector_label


    def __getitem__(self, idx):
        img, label = self.get_img(idx)

        # Apply transformation to image and label at the same time...
        if self.trans_list is not None:
            data = np.concatenate([img, label], axis = 0)    # (2, H, W)
            for trans in self.trans_list:
                data = trans(data)

            img   = data[0:1]    # (1, H, W)
            label = data[1: ]    # (1, H, W)

        return img, label


    def __len__(self):
        return len(self.data_list)
