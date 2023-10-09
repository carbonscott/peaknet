"""
Reference: https://github.com/facebookresearch/pycls/blob/main/pycls/models/blocks.py
"""

import torch
import torch.nn            as nn
import torch.nn.functional as F


def conv2d(in_channels, out_channels, kernel_size, *, stride = 1, groups = 1, bias = False):    # ...`*` forces the rest arguments to be keyword arguments
    """Helper for building a conv2d layer."""
    assert kernel_size % 2 == 1, "Only odd size kernels supported to avoid padding issues."

    padding = (kernel_size - 1)//2

    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size = kernel_size,
                     stride      = stride,
                     padding     = padding,
                     groups      = groups,
                     bias        = bias)




def pool2d(kernel_size, *, stride = 2):    # ...`*` forces the rest arguments to be keyword arguments
    """Helper for building a pool2d layer."""
    assert kernel_size % 2 == 1, "Only odd size kernels supported to avoid padding issues."

    padding = (kernel_size - 1)//2

    return nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding)
