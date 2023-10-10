import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configurator import Configurator

from .resnet_encoder import ImageEncoder
from .bifpn          import BiFPN, DepthwiseSeparableConv2d


class PeakNet(nn.Module):

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG.BACKBONE = ImageEncoder.get_default_config()
            CONFIG.BACKBONE.OUTPUT_CHANNELS = {
                "stem"   : 64,
                "layer1" : 256,
                "layer2" : 512,
                "layer3" : 1024,
                "layer4" : 2048,
            }

            CONFIG.BIFPN = BiFPN.get_default_config()

            CONFIG.SEG_HEAD.UP_SCALE_FACTOR = [
                2,  # stem
                4,  # layer1
                8,  # layer2
                16, # layer3
                32, # layer4
            ]
            CONFIG.SEG_HEAD.Q3_UP_SCALE_FACTOR = 2
            CONFIG.SEG_HEAD.Q3_IN_CHANNELS     = 64
            CONFIG.SEG_HEAD.FUSE_IN_CHANNELS   = 64 * 5
            CONFIG.SEG_HEAD.OUT_CHANNELS       = 3
            CONFIG.SEG_HEAD.USES_Q3            = True

        return CONFIG


    def __init__(self, num_blocks = 1, num_features = 64, config = None):
        super().__init__()

        self.config = PeakNet.get_default_config() if config is None else config

        # Create the image encoder...
        self.backbone = ImageEncoder(config = self.config.BACKBONE)

        # Create the adapter layer between encoder and bifpn...
        self.backbone_to_bifpn = nn.ModuleList([
            DepthwiseSeparableConv2d(in_channels  = in_channels,
                                     out_channels = num_features,
                                     kernel_size  = 1,
                                     stride       = 1,
                                     padding      = 0)
            for _, in_channels in self.config.BACKBONE.OUTPUT_CHANNELS.items()
        ])

        # Create the fusion blocks...
        self.bifpn = BiFPN(num_blocks   = num_blocks,
                           num_features = num_features,
                           num_levels   = len(self.config.BACKBONE.OUTPUT_CHANNELS),
                           config       = self.config.BIFPN,)

        # Create the prediction head...
        in_channels  = self.config.SEG_HEAD.Q3_IN_CHANNELS if self.config.SEG_HEAD.USES_Q3 else \
                       self.config.SEG_HEAD.FUSE_IN_CHANNELS
        out_channels = self.config.SEG_HEAD.OUT_CHANNELS
        self.head_segmask  = nn.Conv2d(in_channels  = in_channels,
                                       out_channels = out_channels,
                                       kernel_size  = 1,
                                       padding      = 0,)

        return None


    def seg_from_fused(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_encoder_layers = self.backbone(x)

        # Apply the BiFPN adapter...
        bifpn_input_list = []
        for idx, fmap in enumerate(fmap_in_encoder_layers):
            bifpn_input = self.backbone_to_bifpn[idx](fmap)
            bifpn_input_list.append(bifpn_input)

        # Apply the BiFPN layer...
        bifpn_output_list = self.bifpn(bifpn_input_list)

        # Upsample all bifpn output from high res to low res...
        fmap_upscale_list = []
        for idx, fmap in enumerate(bifpn_output_list):
            scale_factor = self.config.SEG_HEAD.UP_SCALE_FACTOR[idx]
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


    def seg_from_q3(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_encoder_layers = self.backbone(x)

        # Apply the BiFPN adapter...
        bifpn_input_list = []
        for idx, fmap in enumerate(fmap_in_encoder_layers):
            bifpn_input = self.backbone_to_bifpn[idx](fmap)
            bifpn_input_list.append(bifpn_input)

        # Apply the BiFPN layer...
        bifpn_output_list = self.bifpn(bifpn_input_list)

        # Only upsample q3 output ...
        q3_fmap = bifpn_output_list[0]
        scale_factor = self.config.SEG_HEAD.Q3_UP_SCALE_FACTOR
        fmap_upscale = F.interpolate(q3_fmap,
                                     scale_factor  = scale_factor,
                                     mode          = 'bilinear',
                                     align_corners = False)

        # Predict segmentation in the head...
        segmask = self.head_segmask(fmap_upscale)

        return segmask


    def forward(self, x):
        return self.seg_from_q3   (x) if self.config.SEG_HEAD.USES_Q3 else \
               self.seg_from_fused(x)
