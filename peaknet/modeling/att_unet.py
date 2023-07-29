import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder    import ImageEncoder
from .att_gated_fusion import FusionBlock


class PeakNet(nn.Module):

    def __init__(self, config_channels, uses_skip_connection = True):
        super().__init__()

        # Create the image encoder...
        self.image_encoder = ImageEncoder(config_channels,
                                          uses_skip_connection    = uses_skip_connection,
                                          saves_feature_per_layer = True)

        # Create the fusion blocks...
        fusion_layers = config_channels["fusion_layers"]
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(channels_low,
                        channels_high,
                        channels_high,
                        scale_factor         = 2,
                        uses_skip_connection = True)
            for channels_low, channels_high in fusion_layers
        ])

        # Create the prediction head...
        in_channels, out_channels = config_channels["head_segmask_layer"]
        self.head_segmask  = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 1)

        return None


    def forward(self, x):
        # Calculate and save feature maps in multipl resolutions...
        fmap_in_encoder_layers = self.image_encoder(x)

        # Fuse low-res and high-res features...
        x_low = fmap_in_encoder_layers[-1]
        for enum_idx, x_high in enumerate(fmap_in_encoder_layers[::-1][1:]):
            fusion_block = self.fusion_blocks[enum_idx]
            x_low = fusion_block(x_low, x_high)

        # Predict segmentation in the head...
        segmask = self.head_segmask(x_low)

        return segmask
