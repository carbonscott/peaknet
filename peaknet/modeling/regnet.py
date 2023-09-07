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

    Spatial dimension change: (H, W) -> (H//4, W//4)
    """

    def __init__(self, stem_in_channels, stem_out_channels):
        super().__init__()

        self.conv = conv2d(stem_in_channels,
                           stem_out_channels,
                           kernel_size = 7,
                           stride      = 2,)
        self.bn   = nn.BatchNorm2d(num_features = stem_out_channels,
                                   eps          = CONFIG.RESSTEM.BN.EPS,
                                   momentum     = CONFIG.RESSTEM.BN.MOMENTUM,)

        self.relu = nn.ReLU(inplace = CONFIG.RESSTEM.RELU_INPLACE)
        self.pool = pool2d(kernel_size = 3, stride = 2)


    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x




class ResBlock(nn.Module):
    """
    Create X blocks for the RegNet architecture.

    in_channels, mid_channels (bottleneck channels), out_channels
    """

    def __init__(self, block_in_channels, block_out_channels, mid_conv_channels, mid_conv_groups = 1, in_conv_stride = 1):
        super().__init__()

        self.in_conv = nn.Sequential(
            conv2d(block_in_channels,
                   mid_conv_channels,
                   kernel_size = 1,
                   stride      = in_conv_stride,),
            nn.BatchNorm2d(num_features = mid_conv_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),
        )

        self.mid_conv = nn.Sequential(
            conv2d(mid_conv_channels,
                   mid_conv_channels,
                   kernel_size = 3,
                   stride      = 1,
                   groups      = mid_conv_groups),
            nn.BatchNorm2d(num_features = mid_conv_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE),
        )

        self.out_conv = nn.Sequential(
            conv2d(mid_conv_channels,
                   block_out_channels,
                   kernel_size = 1,
                   stride      = 1,),
            nn.BatchNorm2d(num_features = block_out_channels,
                           eps          = CONFIG.RESBLOCK.BN.EPS,
                           momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
        )

        self.res_conv = None
        if (block_in_channels != block_out_channels) or (in_conv_stride != 1):
            self.res_conv = nn.Sequential(
                conv2d(block_in_channels,
                       block_out_channels,
                       kernel_size = 1,
                       stride      = in_conv_stride,),
                nn.BatchNorm2d(num_features = block_out_channels,
                               eps          = CONFIG.RESBLOCK.BN.EPS,
                               momentum     = CONFIG.RESBLOCK.BN.MOMENTUM,),
            )

        self.relu = nn.ReLU(inplace = CONFIG.RESBLOCK.RELU_INPLACE)


    def forward(self, x):
        y = self.in_conv(x)
        y = self.mid_conv(y)
        y = self.out_conv(y)

        if self.res_conv is not None:
            y = y + self.res_conv(x)

        y = self.relu(y)

        return y




class ResStage(nn.Module):
    """
    This class implments the one stage in the RegNet architecture.

    Block means a res block.
    """

    def __init__(self, stage_in_channels, stage_out_channels, num_blocks, mid_conv_channels, mid_conv_groups, in_conv_stride):
        super().__init__()

        self.blocks = nn.ModuleList([
            ResBlock(
                # First block uses stage_in_channels and rest uses prev stage_out_channels...
                block_in_channels  = stage_in_channels if block_idx == 0 else stage_out_channels,

                block_out_channels = stage_out_channels,
                mid_conv_channels  = mid_conv_channels,
                mid_conv_groups    = mid_conv_groups,

                # First block uses in_conv_stride and rest uses 1...
                in_conv_stride     = in_conv_stride  if block_idx == 0 else 1,
            )
            for block_idx in range(num_blocks)
        ])


    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x




class ResNet50(nn.Module):
    """
    This class implements single channel ResNet50 using the RegNet
    nomenclature.

    ResNet50 architecture reference: [NEED URL]
    """

    def __init__(self):
        super().__init__()

        self.stem = ResStem(stem_in_channels = 1, stem_out_channels = 64)

        stage_in_channels  = 64
        stage_out_channels = 256
        num_stages         = 3
        self.layer1 = nn.ModuleList([
            ResStage(stage_in_channels  = stage_in_channels if stage_idx == 0 else stage_out_channels,
                     stage_out_channels = stage_out_channels,
                     num_blocks         = 3,
                     mid_conv_channels  = 64,
                     mid_conv_groups    = 1,
                     in_conv_stride     = 1,)
            for stage_idx in range(num_stages)
        ])


    def forward(self, x):
        x = self.stem(x)

        for layer in self.layer1:
            x = layer(x)

        return x