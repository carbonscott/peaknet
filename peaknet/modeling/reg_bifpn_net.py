import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CONFIG

from .resnet_encoder import ImageEncoder
from .bifpn          import BiFPN


class PeakNet(nn.Module):

    def __init__(self, num_blocks = 1, num_features = 64):
        super().__init__()

        # Create the image encoder...
        self.image_encoder = ImageEncoder(saves_feature_per_layer = True)

        # Create the adapter layer between encoder and bifpn...
        self.bifpn_adapter = nn.ModuleList([
            nn.Conv2d(in_channels  = in_channels,
                      out_channels = num_features,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0)
            for _, in_channels in CONFIG.RESNET_ENCODER.OUTPUT_CHANNELS.items()
        ])

        # Create the fusion blocks...
        self.bifpn = BiFPN(num_blocks   = num_blocks,
                           num_features = num_features,
                           num_levels   = len(CONFIG.RESNET_ENCODER.OUTPUT_CHANNELS))

        # Create the prediction head...
        in_channels, out_channels = CONFIG.SEG_HEAD.CHANNELS
        self.head_segmask  = nn.Conv2d(in_channels  = in_channels,
                                       out_channels = out_channels,
                                       kernel_size  = 1,
                                       padding      = 0,)

        return None


    def forward(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_encoder_layers = self.image_encoder(x)

        # Apply the BiFPN adapter...
        bifpn_input = []
        for idx, fmap in enumerate(fmap_in_encoder_layers):
            fmap_bifpn = self.bifpn_adapter[idx](fmap)
            bifpn_input.append(fmap_bifpn)

        # Apply the BiFPN layer...
        bifpn_output = self.bifpn(bifpn_input)

        # Upsample all bifpn output from high res to low res...
        fmap_upscale_list = []
        for idx, fmap in enumerate(bifpn_output):
            scale_factor = CONFIG.SEG_HEAD.UP_SCALE_FACTOR[idx]
            fmap_upscale = F.interpolate(fmap,
                                         scale_factor  = scale_factor,
                                         mode          = 'bilinear',
                                         align_corners = False)
            fmap_upscale_list.append(fmap_upscale)

        # Concatenate all output...
        fmap_upscale_final = torch.cat(fmap_upscale_list, dim = 1)    # B, C, H, W

        # Predict segmentation in the head...
        segmask = self.head_segmask(fmap_upscale_final)

        return segmask
