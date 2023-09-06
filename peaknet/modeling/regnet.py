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




class ResBlock(nn.Module):
    """
    Create X blocks for the RegNet architecture.

    in_channels, bnk_channels (bottleneck channels), out_channels
    """

    def __init__(self, in_channels, out_channels, bnk_channels, bnk_stride = 1, bnk_groups = 1):
        super().__init__()

        self.in_conv = nn.Sequential(
            conv2d(in_channels,
                   bnk_channels,
                   kernel_size = 1,
                   stride      = 1,),
            nn.BatchNorm2d(num_features = out_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),
        )

        self.bnk_conv = nn.Sequential(
            conv2d(bnk_channels,
                   bnk_channels,
                   kernel_size = 3,
                   stride      = bnk_stride,
                   groups      = bnk_groups),
            nn.BatchNorm2d(num_features = out_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),
        )

        self.out_conv = nn.Sequential(
            conv2d(bnk_channels,
                   out_channels,
                   kernel_size = 1,
                   stride      = 1,),
            nn.BatchNorm2d(num_features = in_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
        )

        self.res_conv = None
        if (in_channels != out_channels) or (bnk_stride != 1):
            self.res_conv = nn.Sequential(
                conv2d(in_channels,
                       out_channels,
                       kernel_size = 1,
                       stride      = 2,),
                nn.BatchNorm2d(num_features = in_channels,
                               eps          = CONFIG.RESBLOCK.BN.EPS,
                               momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            )

        self.afunc = nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),


    def forward(self, x):
        y = self.in_conv(x)
        y = self.bnk_conv(y)
        y = self.out_conv(y)

        if self.res_conv is not None:
            y = y + self.res_conv(x)

        return y




class ResStage(nn.Module):
    """
    This class implments the four stages in the RegNet architecture.

    Block means a res block.
    """

    def __init__(self, in_channels, out_channels, num_blocks, bnk_channels, bnk_stride, bnk_groups):
        super().__init__()

        self.blocks = nn.ModuleList([
            ResBlock(in_channels  = in_channels if block_idx == 0 else out_channels,
                     out_channels = out_channels,
                     bnk_channels = bnk_channels,
                     bnk_stride   = bnk_stride  if block_idx == 0 else 1,
                     bnk_groups   = bnk_groups)
            for block_idx in range(num_blocks)
        ])


    def forward(self, x):
        for block in self.blocks:
            x = blocks(x)

        return x
