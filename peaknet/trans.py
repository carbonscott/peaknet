import torch
import numpy as np
import random

import torch.nn.functional as F

from skimage.transform import resize
from scipy.ndimage     import rotate


'''
Batch augmentation.

Image shape: (B, H, W)
'''


class PadBottomRight:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x


    def __call__(self, img):
        size_y = self.size_y
        size_x = self.size_x

        size_img_y, size_img_x = img.shape[-2:]

        dy_padded = max(size_y - size_img_y, 0)
        dx_padded = max(size_x - size_img_x, 0)
        pad_width = (
            (0, 0),
            (0, dy_padded),
            (0, dx_padded),
        )

        img_padded = np.pad(img, pad_width = pad_width, mode = 'constant', constant_values = 0)

        return img_padded




class Pad:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x


    def __call__(self, img):
        size_y = self.size_y
        size_x = self.size_x

        size_img_y, size_img_x = img.shape[-2:]

        dy_padded = max(size_y - size_img_y, 0)
        dx_padded = max(size_x - size_img_x, 0)
        pad_width = (
            (0, 0),
            (dy_padded // 2, dy_padded - dy_padded // 2),
            (dx_padded // 2, dx_padded - dx_padded // 2),
        )

        img_padded = np.pad(img, pad_width = pad_width, mode = 'constant', constant_values = 0)

        return img_padded




class Crop:
    def __init__(self, crop_center, crop_window_size):
        self.crop_center      = crop_center
        self.crop_window_size = crop_window_size


    def __call__(self, img):
        crop_center      = self.crop_center
        crop_window_size = self.crop_window_size

        # ___/ IMG \___
        # Calcualte the crop window range...
        y_min = crop_center[0] - crop_window_size[0] // 2
        x_min = crop_center[1] - crop_window_size[1] // 2
        y_max = crop_center[0] + crop_window_size[0] // 2
        x_max = crop_center[1] + crop_window_size[1] // 2

        # Resolve over the bound issue...
        size_img_y, size_img_x = img.shape[-2:]
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        y_max = min(y_max, size_img_y)
        x_max = min(x_max, size_img_x)

        # Crop...
        img_crop = img[:, y_min:y_max, x_min:x_max]

        return img_crop




class Resize:
    def __init__(self, size_y, size_x, anti_aliasing = True):
        self.size_y        = size_y
        self.size_x        = size_x
        self.anti_aliasing = anti_aliasing


    def __call__(self, img):
        size_y        = self.size_y
        size_x        = self.size_x
        anti_aliasing = self.anti_aliasing

        B = img.shape[0]

        img_resize = resize(img, (B, size_y, size_x), anti_aliasing = anti_aliasing)

        return img_resize




## class Downsample:
##     def __init__(self, block_size = 2, func = np.max, cval = 0, func_kwargs = None):
##         self.block_size  = block_size
##         self.func        = func
##         self.cval        = cval
##         self.func_kwargs = func_kwargs
## 
## 
##     def __call__(self, img):
##         block_size  = self.block_size
##         func        = self.func
##         cval        = self.cval
##         func_kwargs = self.func_kwargs
## 
##         img_downsampled = block_reduce(img, block_size = block_size, func = func, cval = cval, func_kwargs = func_kwargs)
## 
##         return img_downsampled




class MaxPool2D:
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.ceil_mode      = ceil_mode
        self.return_indices = return_indices

        return None


    def __call__(self, img):
        kernel_size    = self.kernel_size
        padding        = self.padding
        stride         = self.stride
        dilation       = self.dilation
        ceil_mode      = self.ceil_mode
        return_indices = self.return_indices

        B, H, W = img.shape
        img_torch = torch.tensor(img).view(B, 1, H, W)
        img_downsampled = F.max_pool2d(img_torch, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

        B, _, H_down, W_down = img_downsampled.shape

        return img_downsampled.view(B, H_down, W_down).numpy()




class RandomCenterCropZoom:
    def __init__(self, trim_factor_max = 0.2):
        self.trim_factor_max = trim_factor_max

        return None


    def __call__(self, img):
        trim_factor_max = self.trim_factor_max

        size_img_y, size_img_x = img.shape[-2:]

        # ___/ Trim \___
        trim_factor = np.random.uniform(low = 0, high = trim_factor_max)

        img_cy = size_img_y // 2
        img_cx = size_img_x // 2
        crop_window_size = (int(size_img_y * (1 - trim_factor)),
                            int(size_img_x * (1 - trim_factor)))
        cropper = Crop( crop_center = (img_cy, img_cx), crop_window_size = crop_window_size )

        img_crop = cropper(img)

        # ___/ Zoom \___
        resizer = Resize(size_img_y, size_img_x)

        img_resize = resizer(img_crop)

        return img_resize




class RandomCrop:
    def __init__(self, center_shift_max = (0, 0), crop_window_size = (200, 200)):
        self.center_shift_max = center_shift_max
        self.crop_window_size = crop_window_size

        return None


    def __call__(self, img):
        cy_shift_max, cx_shift_max = self.center_shift_max
        crop_window_size           = self.crop_window_size

        size_img_y, size_img_x = img.shape
        cy, cx = center

        cy_shift = np.random.randint(low = cy - cy_shift_max, high = cy + cy_shift_max)
        cx_shift = np.random.randint(low = cx - cx_shift_max, high = cx + cx_shift_max)

        crop = Crop( crop_center = (cy_shift, cx_shift), crop_window_size = crop_window_size )

        img_crop = crop(img, center = center)

        return img_crop




class RandomShift:
    def __init__(self, frac_y_shift_max = 0.01, frac_x_shift_max = 0.01):
        self.frac_y_shift_max = frac_y_shift_max
        self.frac_x_shift_max = frac_x_shift_max

    def __call__(self, img, verbose = False):
        frac_y_shift_max = self.frac_y_shift_max
        frac_x_shift_max = self.frac_x_shift_max

        # Get the size of the image...
        batch_size, size_img_y, size_img_x = img.shape

        # Draw a random value for shifting along x and y, respectively...
        y_shift_abs_max = size_img_y * frac_y_shift_max
        y_shift_abs_max = y_shift_abs_max
        y_shift = random.uniform(-y_shift_abs_max, y_shift_abs_max)
        y_shift = int(y_shift)

        x_shift_abs_max = size_img_x * frac_x_shift_max
        x_shift_abs_max = int(x_shift_abs_max)
        x_shift = random.uniform(-x_shift_abs_max, x_shift_abs_max)
        x_shift = int(x_shift)

        # Determine the size of the super image...
        size_super_y = size_img_y + 2 * abs(y_shift)
        size_super_x = size_img_x + 2 * abs(x_shift)

        # Construct a super image by padding (with zero) the absolute y and x shift...
        super = np.zeros((batch_size, size_super_y, size_super_x))

        # Move the image to the target area...
        target_y_min = abs(y_shift) + y_shift
        target_x_min = abs(x_shift) + x_shift
        target_y_max = size_img_y + target_y_min
        target_x_max = size_img_x + target_x_min
        super[:, target_y_min:target_y_max, target_x_min:target_x_max] = img[:, :]

        # Crop super...
        crop_y_min = abs(y_shift)
        crop_x_min = abs(x_shift)
        crop_y_max = size_img_y + crop_y_min
        crop_x_max = size_img_x + crop_x_min
        crop = super[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if verbose: print( f"y-shift = {y_shift}, x-shift = {x_shift}" )

        return crop




class RandomRotate:
    def __init__(self, angle_max = 360, order = 1):
        self.angle_max = angle_max
        self.order     = order

        return None

    def __call__(self, img):
        angle_max = self.angle_max
        order     = self.order

        angle = np.random.uniform(low = 0, high = angle_max)
        img_rot = rotate(img, angle = angle, order = order, axes = (-2, -1), reshape = False)    # Scikit image wants (x, y) instead of (y, x)

        return img_rot




class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, num_patch, size_patch_y,    size_patch_x, 
                                  var_patch_y = 0, var_patch_x = 0, 
                                  returns_mask = False):
        self.num_patch    = num_patch                   # ...Number of patches
        self.size_patch_y = size_patch_y                # ...Size of the patch in y dimension
        self.size_patch_x = size_patch_x                # ...Size of the patch in x dimension
        self.var_patch_y  = max(0, min(var_patch_y, 1)) # ...Percent variation with respect to the patch size in x dimension
        self.var_patch_x  = max(0, min(var_patch_x, 1)) # ...Percent variation with respect to the patch size in y dimension
        self.returns_mask = returns_mask                # ...Is it allowed to return a mask

        return None


    def __call__(self, img):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape[-2:]

        # Construct a mask of ones with the same size of the image...
        mask = np.ones_like(img)

        # Generate a number of random position...
        pos_y = np.random.randint(low = 0, high = size_img_y, size = self.num_patch)
        pos_x = np.random.randint(low = 0, high = size_img_x, size = self.num_patch)

        # Stack two column vectors to form an array of (x, y) indices...
        pos_y = pos_y.reshape(-1,1)
        pos_x = pos_x.reshape(-1,1)
        pos   = np.hstack((pos_y, pos_x))

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = self.size_patch_y
            size_patch_x = self.size_patch_x

            # Apply random variance...
            # Find the absolute max pixel to vary
            varsize_patch_y = int(size_patch_y * self.var_patch_y)
            varsize_patch_x = int(size_patch_x * self.var_patch_x)

            # Sample an integer from the min-max pixel to vary
            # Also allow user to set var_patch = 0 when variance is not desired
            delta_patch_y = np.random.randint(low = -varsize_patch_y, high = varsize_patch_y if varsize_patch_y else 1)
            delta_patch_x = np.random.randint(low = -varsize_patch_x, high = varsize_patch_x if varsize_patch_x else 1)

            # Apply the change
            size_patch_y += delta_patch_y
            size_patch_x += delta_patch_x

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[:, y : y_end, x : x_end] = 0

        # Appy the mask...
        img_masked = mask * img

        # Construct the return value...
        # Parentheses are necessary
        output = img_masked if not self.returns_mask else (img_masked, mask)

        return output




def center_crop(img, size_y_crop, size_x_crop, returns_offset_tuple = False):
    '''
    Return the cropped area and associated offset for coordinate transformation
    purposes.  
    '''
    # Get metadata (device)...
    device = img.device

    # Get the size the original image...
    # It might have other dimensions
    dim = img.shape
    size_y_img, size_x_img = dim[-2:]
    dim_storage = dim[:-2]

    # Initialize the super image that covers both the image and the crop...
    size_y_super = max(size_y_img, size_y_crop)
    size_x_super = max(size_x_img, size_x_crop)
    img_super = torch.zeros((*dim_storage, size_y_super, size_x_super), device = device)

    # Paint the original image onto the super image...
    # Area
    y_min_img = (size_y_super - size_y_img) // 2
    x_min_img = (size_x_super - size_x_img) // 2
    img_super[...,
              y_min_img : y_min_img + size_y_img,
              x_min_img : x_min_img + size_x_img] = img

    # Find crop range...
    # Min
    y_min_crop = (size_y_super - size_y_crop) // 2
    x_min_crop = (size_x_super - size_x_crop) // 2

    # Max
    y_max_crop = y_min_crop + size_y_crop
    x_max_crop = x_min_crop + size_x_crop

    # Crop the image area...
    img_crop =  img_super[...,
                          y_min_crop : y_max_crop,
                          x_min_crop : x_max_crop]

    # Pack things to return in a tuple
    ret_tuple = img_crop
    if returns_offset_tuple:
        # Min float for finding the offset
        y_min_crop_float = (size_y_super - size_y_crop) / 2
        x_min_crop_float = (size_x_super - size_x_crop) / 2

        # Offset introduced due to the integer division
        offset_tuple = (y_min_crop_float - y_min_crop, x_min_crop_float - x_min_crop)

        ret_tuple = (img_crop, offset_tuple)

    return ret_tuple




## def center_crop(img, size_y_crop, size_x_crop, returns_offset_tuple=False):
##     ''' 
##     Return the cropped area and associated offset for coordinate transformation
##     purposes.
##     ''' 
##     # Get the size of the original image
##     size_y_img, size_x_img = img.shape[-2:]
## 
##     # Calculate crop indices
##     y_min_crop = (size_y_img - size_y_crop) // 2
##     x_min_crop = (size_x_img - size_x_crop) // 2
## 
##     # Crop the image using torch.narrow
##     img_crop = img.narrow(-2, y_min_crop, size_y_crop).narrow(-1, x_min_crop, size_x_crop)
## 
##     if returns_offset_tuple:
##         # Calculate the offset introduced due to the integer division
##         y_min_crop_float = (size_y_img - size_y_crop) / 2 
##         x_min_crop_float = (size_x_img - size_x_crop) / 2 
##         offset_tuple = (y_min_crop_float - y_min_crop, x_min_crop_float - x_min_crop)
## 
##         return img_crop, offset_tuple
## 
##     return img_crop




def coord_img_to_crop(coord_tuple, size_img_tuple, size_crop_tuple, offset_tuple = ()):
    '''
    Need some unit test.
    '''
    # Unpack all inputs...
    y          , x           = coord_tuple
    size_y_img ,  size_x_img = size_img_tuple
    size_y_crop, size_x_crop = size_crop_tuple

    # Transform...
    y_crop = (size_y_crop - size_y_img) / 2 + y
    x_crop = (size_x_crop - size_x_img) / 2 + x

    if len(offset_tuple) == 2:
        y_offset, x_offset = offset_tuple
        y_crop += y_offset
        x_crop += x_offset

    return y_crop, x_crop




def coord_crop_to_img(coord_tuple, size_img_tuple, size_crop_tuple, offset_tuple = ()):
    '''
    Need some unit test.
    '''
    # Unpack all inputs...
    y_crop     , x_crop      = coord_tuple
    size_y_img ,  size_x_img = size_img_tuple
    size_y_crop, size_x_crop = size_crop_tuple

    # Transform...
    y_img = y_crop - (size_y_crop - size_y_img) / 2
    x_img = x_crop - (size_x_crop - size_x_img) / 2

    if len(offset_tuple) == 2:
        y_offset, x_offset = offset_tuple
        y_img += y_offset
        x_img += x_offset

    return y_img, x_img
