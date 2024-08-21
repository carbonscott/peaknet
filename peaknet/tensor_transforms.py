import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms.functional import rotate, normalize

import random

import math

'''
Batch augmentation.

Image shape: (B, C, H, W)
'''

import logging
logger = logging.getLogger(__name__)

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

        H_padded, W_padded = patch_size * math.ceil(H / patch_size), patch_size * math.ceil(W / patch_size)

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


class InstanceNorm:
    def __init__(self, eps = 1e-6, checks_nan = True):
        self.eps = eps
        self.checks_nan = checks_nan

    def __call__(self, img, **kwargs):
        if self.checks_nan and torch.isnan(img).any():
            logger.debug(f"NaN found in the input.")

        B, C, H, W = img.size()
        img_view = img.view(B, C, H*W)
        mean = img_view.mean(dim=-1, keepdim=True)
        var  = img_view.var (dim=-1, keepdim=True, correction=0)

        img_norm = (img_view - mean) / torch.sqrt(var + self.eps)

        return img_norm.view(B, C, H, W)


class PolarCenterCrop:
    def __init__(self, Hv, Wv, sigma = 0.33, num_crop = 1):
        """
        Performs random center cropping on batches of images using polar coordinates.

        This class implements a batched multi-crop operation where crop centers are
        determined using polar coordinates. It supports processing multiple images
        simultaneously and generating multiple crops per image.

        Key Features:
        - Batch processing: Can process multiple images in a single call.
        - Multi-crop: Generates multiple crops per image.
        - Polar sampling: Uses polar coordinates for random center selection,
          allowing for more controlled distribution of crop centers.
        - Efficient implementation: Utilizes PyTorch's advanced indexing for fast cropping.

        Args:
            Hv (int): Height of the crop.
            Wv (int): Width of the crop.
            sigma (float, optional): Standard deviation for the radial distribution of crop centers.
                                     Defaults to 0.33, following the 68-95-99.7 rule.
            num_crop (int, optional): Number of crops to generate per image. Defaults to 1.

        Example:
            >>> cropper = PolarCenterCrop(224, 224, sigma=0.5, num_crop=5)
            >>> batch = torch.rand(32, 1, 1920, 1920)  # 32 single-channel images of size 256x256
            >>> crops = cropper(batch)  # Returns tensor of shape (32, 5, 1, 1920, 1920)

        Note:
            The input images should be PyTorch tensors with shape (B, C, H, W),
            where B is the batch size, C is the number of channels, and H and W
            are the height and width of the images respectively.
        """
        self.Hv       = Hv
        self.Wv       = Wv
        self.sigma    = sigma
        self.num_crop = num_crop


    def __call__(self, img, **kwargs):
        """
        Perform the multi-crop operation on a batch of images.

        Args:
            img (torch.Tensor): Input images tensor of shape (B, C, H, W),
                                where B is the batch size, C is the number of channels,
                                and H and W are the height and width of the images.

        Returns:
            torch.Tensor: Cropped images tensor of shape (B, N, C, Hv, Wv),
                          where N is the number of crops per image, and Hv and Wv
                          are the height and width of each crop.
        """
        Hv    = self.Hv
        Wv    = self.Wv
        sigma = self.sigma
        N     = self.num_crop

        # -- Valid H and W
        B, C, H, W = img.shape
        Hv = min(Hv, H)
        Wv = min(Wv, W)
        H_valid = H - Hv
        W_valid = W - Wv

        # -- Polar random sampling for determining the crop center
        # Sample from uniform angle and Gaussian radius
        device = img.device
        theta  = torch.rand (B, N, device=device) * math.pi * 2  # (B, N)
        radius = torch.randn(B, N, device=device).abs() * sigma  # (B, N)

        # Convert them to Cartesian
        cy = radius * torch.cos(theta)  # image y = Cartesian x
        cx = radius * torch.sin(theta)  # image x = Cartesian y

        # Clamp to avoid overflow
        cy.clamp_(min = -1, max = 1)
        cx.clamp_(min = -1, max = 1)

        # Rescale it to half image length
        cy *= H_valid/2
        cx *= W_valid/2

        # Change the origin to the center of the input image
        cy += H/2
        cx += W/2

        # Calculate the top-left(tl) coordinates used for fancy indexing
        tly = cy - Hv//2  # (B, N)
        tlx = cx - Wv//2  # (B, N)
        tly = tly.int()
        tlx = tlx.int()

        # -- Multi-crop
        # Create a view of multiple images
        img_expanded = img.unsqueeze(1).expand(-1,N,-1,-1,-1) # (B,C,H,W) -> (B,N,C,H,W)

        # Create indexing tensor along each dimension (B, N, C, H, W)
        idx_tensor_B = torch.arange(B , dtype=torch.int, device=device)
        idx_tensor_N = torch.arange(N , dtype=torch.int, device=device)
        idx_tensor_C = torch.arange(C , dtype=torch.int, device=device)
        idx_tensor_H = torch.arange(Hv, dtype=torch.int, device=device)
        idx_tensor_W = torch.arange(Wv, dtype=torch.int, device=device)

        # Create meshgrid for advanced indexing
        # Note: We only use the first three outputs, others are discarded
        mesh_B, mesh_N, mesh_C, _, _ = torch.meshgrid(
            idx_tensor_B,
            idx_tensor_N,
            idx_tensor_C,
            idx_tensor_H,
            idx_tensor_W,
            indexing = 'ij'
        )

        # Generate indexing tensors for height and width
        idx_tensor_W = tlx[:,:,None] + idx_tensor_W[None,None,:]  # tlx:(B,N,1) + idx_tensor_W:(1,1,Wv) -> (B,N,Wv)
        idx_tensor_H = tly[:,:,None] + idx_tensor_H[None,None,:]  # tly:(B,N,1) + idx_tensor_H:(1,1,Hv) -> (B,N,Hv)

        # Expand indexing tensors to match desired dimensions
        idx_tensor_W = idx_tensor_W[:,:,None,:].expand(-1,-1,Hv,-1)  # (B,N,1,Wv) -> (B,N,Hv,Wv)
        idx_tensor_H = idx_tensor_H[:,:,:,None].expand(-1,-1,-1,Wv)  # (B,N,Hv,1) -> (B,N,Hv,Wv)

        # Create final advanced/fancy indexing tensors
        mesh_H = idx_tensor_H[:,:,None,:,:].expand(B,-1,C,-1,-1)  # (B,N,1,Hv,Wv) -> (B,N,C,Hv,Wv)
        mesh_W = idx_tensor_W[:,:,None,:,:].expand(B,-1,C,-1,-1)  # (B,N,1,Hv,Wv) -> (B,N,C,Hv,Wv)

        # Apply indexing to get the final crops
        return img_expanded[mesh_B, mesh_N, mesh_C, mesh_H, mesh_W]


class MergeBatchPatchDims:
    def __call__(self, x, **kwargs):
        B, N, C, H, W = x.size()
        return x.view(B*N, C, H, W)


class BatchSampler:
    def __init__(self, sampling_fraction = None, dim = 0):
        if sampling_fraction is not None and (sampling_fraction <= 0.0 or sampling_fraction > 1.0):
            raise ValueError("sampling_fraction must be None or a number between 0 and 1.")
        self.sampling_fraction = sampling_fraction
        self.dim               = dim

    def __call__(self, image_tensor, **kwargs):
        if self.sampling_fraction is not None:
            dim_size       = image_tensor.size(self.dim)
            sample_size    = max(int(dim_size * self.sampling_fraction), 1)
            sample_indices = torch.randperm(dim_size)[:sample_size]
            image_tensor   = image_tensor.transpose(self.dim, 0)[sample_indices].transpose(0, self.dim)

        return image_tensor


class NDArrayToTensor:
    def __call__(self, x, **kwargs):
        return torch.from_numpy(x)


class NoTransform:
    def __call__(self, x, **kwargs):
        return x
