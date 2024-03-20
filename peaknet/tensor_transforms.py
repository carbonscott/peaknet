import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def __call__(self, img):
        pad_width  = self.calc_pad_width(img)
        img_padded = F.pad(img, pad_width, 'constant', 0)

        return img_padded


class DownscaleLocalMean:
    def __init__(self, factors = (2, 2)):
        self.factors = factors


    def __call__(self, img):
        kernel_size = self.factors
        stride      = self.factors

        img_downsized = F.avg_pool2d(img, kernel_size = kernel_size, stride = stride)

        return img_downsized


class RandomPatch:
    """ Randomly place num_patch patches with the size of H * W onto an image.
    """
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


    def __call__(self, img):
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
