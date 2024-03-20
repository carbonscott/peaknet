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
