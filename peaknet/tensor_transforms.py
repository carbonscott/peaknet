import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms.functional import rotate, normalize

import random

from math import ceil

'''
Batch augmentation.

Image shape: (B, C, H, W)
'''


class Pad:
    def __init__(self, H, W):
        self.H = H
        self.W = W


    def calc_pad_width(self, img):
        H = self.H
        W = self.W
        _, _, H_img, W_img = img.shape

        dH_padded = max(H - H_img, 0)
        dW_padded = max(W - W_img, 0)

        pad_width = (
            dW_padded // 2, dW_padded - dW_padded // 2,    # -1 dimension (left, right)
            dH_padded // 2, dH_padded - dH_padded // 2,    # -2 dimension (top, bottom)
        )

        return pad_width


    def __call__(self, img, **kwargs):
        pad_width  = self.calc_pad_width(img)
        img_padded = F.pad(img, pad_width, 'constant', 0)

        return img_padded


class DownscaleLocalMean:
    def __init__(self, factors = (2, 2)):
        self.factors = factors


    def __call__(self, img, **kwargs):
        kernel_size = self.factors
        stride      = self.factors

        img_downsized = F.avg_pool2d(img, kernel_size = kernel_size, stride = stride)

        return img_downsized


class RandomPatch:
    def __init__(self, num_patch,
                       H_patch,
                       W_patch,
                       var_H_patch  = 0,
                       var_W_patch  = 0,
                       returns_mask = False):
        self.num_patch    = num_patch
        self.H_patch      = H_patch
        self.W_patch      = W_patch
        self.var_H_patch  = max(0, min(var_H_patch, 1))
        self.var_W_patch  = max(0, min(var_W_patch, 1))
        self.returns_mask = returns_mask


    def __call__(self, img, **kwargs):
        num_patch    = self.num_patch
        H_patch      = self.H_patch
        W_patch      = self.W_patch
        var_H_patch  = self.var_H_patch
        var_W_patch  = self.var_W_patch
        returns_mask = self.returns_mask

        H_img, W_img = img.shape[-2:]

        mask = torch.ones_like(img)

        # Generate random positions
        pos_y = torch.randint(low=0, high=H_img, size=(num_patch,))
        pos_x = torch.randint(low=0, high=W_img, size=(num_patch,))

        for i in range(num_patch):
            max_delta_H_patch = int(H_patch * var_H_patch)
            max_detla_W_patch = int(W_patch * var_W_patch)

            delta_patch_H = torch.randint(low=-max_delta_H_patch, high=max_delta_H_patch+1, size=(1,))
            delta_patch_W = torch.randint(low=-max_detla_W_patch, high=max_detla_W_patch+1, size=(1,))

            H_this_patch = H_patch + delta_patch_H.item()
            W_this_patch = W_patch + delta_patch_W.item()

            y_start = pos_y[i]
            x_start = pos_x[i]
            y_end   = min(y_start + H_this_patch, H_img)
            x_end   = min(x_start + W_this_patch, W_img)

            mask[:, :, y_start:y_end, x_start:x_end] = 0    # (B, C, H, W)

        img_masked = mask * img

        output = img_masked if not returns_mask else (img_masked, mask)

        return output



class RandomRotate:
    def __init__(self, angle_max=360):
        self.angle_max = angle_max

    def __call__(self, img, **kwargs):
        angle = random.uniform(0, self.angle_max)

        original_dtype = img.dtype
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        img_rot = rotate(img,
                         angle         = angle,
                         interpolation = torchvision.transforms.InterpolationMode.BILINEAR)

        img_rot = img_rot.to(original_dtype)

        return img_rot


class RandomShift:
    def __init__(self, frac_y_shift_max=0.01, frac_x_shift_max=0.01):
        self.frac_y_shift_max = frac_y_shift_max
        self.frac_x_shift_max = frac_x_shift_max

    def __call__(self, img, verbose=False, **kwargs):
        frac_y_shift_max = self.frac_y_shift_max
        frac_x_shift_max = self.frac_x_shift_max

        B, C, H, W = img.size()

        # Draw a random value for shifting along x and y, respectively...
        y_shift_abs_max = H * frac_y_shift_max
        y_shift = random.uniform(-y_shift_abs_max, y_shift_abs_max)
        y_shift = int(y_shift)

        x_shift_abs_max = W * frac_x_shift_max
        x_shift = random.uniform(-x_shift_abs_max, x_shift_abs_max)
        x_shift = int(x_shift)

        # Construct a super tensor by padding (with zero) the absolute y and x shift...
        size_super_y = H + 2 * abs(y_shift)
        size_super_x = W + 2 * abs(x_shift)
        super_tensor = torch.zeros(B, C, size_super_y, size_super_x, device=img.device, dtype=img.dtype)

        # Move the image to the target area...
        target_y_min = abs(y_shift) + y_shift
        target_x_min = abs(x_shift) + x_shift
        target_y_max = H + target_y_min
        target_x_max = W + target_x_min
        super_tensor[:, :, target_y_min:target_y_max, target_x_min:target_x_max] = img

        # Crop super tensor...
        crop_y_min = abs(y_shift)
        crop_x_min = abs(x_shift)
        crop_y_max = H + crop_y_min
        crop_x_max = W + crop_x_min
        crop = super_tensor[:, :, crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if verbose:
            print(f"y-shift = {y_shift}, x-shift = {x_shift}")

        return crop


class Patchify:
    """
    Examples:
        >>> patch_size = 244
        >>> stride = 244
        >>> patchifier = Patchify(patch_size, stride)
        >>> patches = patchifier(imgs)
        >>> print(patches.shape)
    """
    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride     = stride

    def __call__(self, batch_img, **kwargs):
        """
        Arguments:
            img: (B, C, H, W)
        """
        patch_size = self.patch_size
        stride     = self.stride

        B, C, H, W = batch_img.shape

        H_padded, W_padded = patch_size * ceil(H / patch_size), patch_size * ceil(W / patch_size)

        padder = Pad(H_padded, W_padded)
        batch_img_padded = padder(batch_img)

        # (B, C, H, W) -> (B, C * patch_size * patch_size, num_patches)
        batch_patches = F.unfold(
            batch_img_padded,
            kernel_size = (patch_size, patch_size),
            stride=stride
        )

        # (B, C * patch_size * patch_size, num_patches) - > (B, C, patch_size, patch_size, num_patches)
        batch_patches = batch_patches.view(
            B, C, patch_size, patch_size, -1
        )

        # (B, C, patch_size, patch_size, num_patches) -> (B, num_patches, C, patch_size, patch_size)
        batch_patches = batch_patches.permute(0, 4, 1, 2, 3).contiguous()

        return batch_patches

class Norm:
    def __init__(self, detector_norm_params):
        self.detector_norm_params = detector_norm_params

    def __call__(self, img, detector_name, **kwargs):
        mean, std = self.detector_norm_params[detector_name]["mean"], self.detector_norm_params[detector_name]["std"]
        C = img.shape[-3]
        return normalize(img, [mean]*C, [std]*C)
