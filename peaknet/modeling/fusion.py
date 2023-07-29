import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trans import center_crop
from .common import DoubleConv


class CropAndCat(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, x_upscaled):
        H, W = x_upscaled.shape[-2:]
        x = center_crop(x, H, W)
        x_upscaled = torch.cat([x, x_upscaled], dim = 1)

        return x_upscaled




class AttentionGatedFusion(nn.Module):
    ''' The fusion layer.
    '''
    def __init__(self, channels_low, channels_high, channels_intmd, scale_factor = 2):
        super().__init__()

        self.scale_factor = scale_factor

        self.low_to_intmd = nn.Sequential(
            nn.Conv2d(in_channels  = channels_low,
                      out_channels = channels_intmd,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0,),
            nn.BatchNorm2d(channels_intmd),
        )

        self.high_to_intmd = nn.Sequential(
            nn.Conv2d(in_channels  = channels_high,
                      out_channels = channels_intmd,
                      kernel_size  = scale_factor,
                      stride       = scale_factor,
                      padding      = 0,),
            nn.BatchNorm2d(channels_intmd),
        )

        self.rearrange_channels = nn.Sequential(
            nn.Conv2d(in_channels  = channels_intmd,
                      out_channels = 1,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0,),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.upconv = nn.ConvTranspose2d(channels_low, channels_high, kernel_size = 2, stride = 2)

        self.concat = CropAndCat()

        self.relu = nn.ReLU()


    def forward(self, x_low, x_high):
        '''
        x_high   # hi  res feature
        x_low    # low res feature
        x_low has a smaller dimension size than x_high by a factor of 2 along each edge.
        '''
        # ___/ CALCULATE GATE \___
        scale_factor = self.scale_factor

        # Center crop to ensure matching dimensions...
        H_low, W_low = x_low.shape[-2:]
        H_high = scale_factor * H_low
        W_high = scale_factor * W_low
        x_high = center_crop(x_high, H_high, W_high)

        x_intmd_high  = self.high_to_intmd(x_high)    # DownConv by scale_factor
        x_intmd_low   = self.low_to_intmd(x_low)
        x_intmd_fused = self.relu(x_intmd_high + x_intmd_low)
        x_fused       = self.rearrange_channels(x_intmd_fused)

        # Upsample by interpolation (grid resampling)...
        gate_high = F.interpolate(x_fused,
                                  scale_factor  = scale_factor,
                                  mode          = 'bilinear',
                                  align_corners = False)

        x_gated_high = gate_high * x_high

        # ___/ CONCAT \___
        x_upscaled_high = self.upconv(x_low)
        x_concat        = self.concat(x_gated_high, x_upscaled_high)    # Channel = channels_low

        return x_concat




class FusionBlock(nn.Module):

    def __init__(self, channels_low, channels_high, channels_intmd, scale_factor = 2, uses_skip_connection = True):
        super().__init__()

        # Define the attention gated fusion layer...
        self.fusion_layer = AttentionGatedFusion(channels_low, channels_high, channels_intmd, scale_factor = scale_factor)

        # Define the double convolution layer to rearrange the output channels...
        self.double_conv_layer = DoubleConv(in_channels          = channels_low,
                                            out_channels         = channels_high,
                                            stride               = 1,
                                            uses_skip_connection = uses_skip_connection)


    def forward(self, x_low, x_high):
        x_concat = self.fusion_layer(x_low, x_high)
        x_new    = self.double_conv_layer(x_concat)

        return x_new
