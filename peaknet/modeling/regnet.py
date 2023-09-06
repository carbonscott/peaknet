"""
RegNet

Building blocks
- STEM
- BODY
  - STAGES (1..4)
    - BLOCKS (1..N)
- HEAD

width_per_stage: specifies the width (i.e., number of channels) for each stage.

Reference:
https://github.com/facebookresearch/pycls/blob/main/pycls/models/blocks.py
"""

import torch
import torch.nn            as nn
import torch.nn.functional as F

from ..config import CONFIG

from .blocks import conv2d, pool2d


class ResStem(nn.Module):
    """
    This class implments the first layer (STEM in RegNet's nomenclature) of
    ResNet.

    Structure:
    - Conv2d kernel (7, 7)
    - BatchNorm2d
    - Activation
    - MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        print(CONFIG.RESSTEM.BN.EPS)
        self.conv = conv2d(in_channels,
                           out_channels,
                           kernel_size = 7,
                           stride      = 2,)
        self.bn   = nn.BatchNorm2d(num_features = out_channels,
                                   eps          = CONFIG.RESSTEM.BN.EPS,
                                   momentum     = CONFIG.RESSTEM.BN.MOMENTUM,)

        self.af   = nn.ReLU(inplace = CONFIG.RESSTEM.RELU_INPLACE)
        self.pool = pool2d(kernel_size = 3, stride = 2)


    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x




class ResStage(nn.Module):
    """
    This class implments the four stages in the RegNet architecture.
    """

    def __init__(self):
        pass




class ResBlock(nn.Module):
    """
    Create X blocks for the RegNet architecture.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            conv2d(in_channels,
                   out_channels,
                   kernel_size = 1,
                   stride      = 1,),
            nn.BatchNorm2d(num_features = out_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),
        )
