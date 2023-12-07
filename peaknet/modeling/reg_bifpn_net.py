import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List

from .resnet_encoder import ImageEncoderConfig, ImageEncoder
from .bifpn          import BiFPNConfig, BiFPN, DepthwiseSeparableConv2d


class SegLateralLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, num_layers, base_scale_factor = 2):
        super().__init__()

        self.enables_upsample = num_layers > 0

        # Strange strategy, but...
        num_layers = max(num_layers, 1)

        # 3x3 convolution with pad 1, group norm and relu...
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels  = (in_channels if idx == 0 else out_channels),
                          out_channels = out_channels,
                          kernel_size  = 3,
                          padding      = 1,),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(),
            )
            for idx in range(num_layers)
        ])

        self.base_scale_factor = base_scale_factor


    def forward(self, x):
        for layer in self.layers:
            # Conv3x3...
            x = layer(x)

            # Optional upsampling...
            if self.enables_upsample:
                x = F.interpolate(x,
                                  scale_factor  = self.base_scale_factor,
                                  mode          = 'bilinear',
                                  align_corners = False)

        return x




@dataclass
class SegHeadConfig:
    UP_SCALE_FACTOR: List[int] = field(
        default_factory = lambda : [
            ## 2,  # stem
            4,  # layer1
            8,  # layer2
            16, # layer3
            32, # layer4
        ]
    )
    NUM_GROUPS           : int  = 32
    OUT_CHANNELS         : int  = 128
    NUM_CLASSES          : int  = 3
    BASE_SCALE_FACTOR    : int  = 2
    USES_LEARNED_UPSAMPLE: bool = False


@dataclass
class PeakNetConfig:
    BACKBONE: ImageEncoderConfig = ImageEncoder.get_default_config()
    BACKBONE.OUTPUT_CHANNELS = {
        "layer1" : 256,
        "layer2" : 512,
        "layer3" : 1024,
        "layer4" : 2048,
    }

    BIFPN   : BiFPNConfig   = BiFPN.get_default_config()
    SEG_HEAD: SegHeadConfig = SegHeadConfig()


class PeakNet(nn.Module):

    @staticmethod
    def get_default_config():
        return PeakNetConfig()


    def __init__(self, config = None):
        super().__init__()

        self.config = PeakNet.get_default_config() if config is None else config

        # Create the image encoder...
        self.backbone = ImageEncoder(config = self.config.BACKBONE)

        # Create the adapter layer between encoder and bifpn...
        backbone_output_channels = self.config.BACKBONE.OUTPUT_CHANNELS
        if is_dataclass(backbone_output_channels):
            backbone_output_channels = asdict(backbone_output_channels)
        num_bifpn_features = self.config.BIFPN.NUM_FEATURES
        self.backbone_to_bifpn = nn.ModuleList([
            nn.Conv2d(in_channels  = in_channels,
                      out_channels = num_bifpn_features,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0)
            for _, in_channels in backbone_output_channels.items()
        ])

        # Create the fusion blocks...
        self.bifpn = BiFPN(config = self.config.BIFPN)

        # Create the prediction head...
        base_scale_factor         = self.config.SEG_HEAD.BASE_SCALE_FACTOR
        max_scale_factor          = self.config.SEG_HEAD.UP_SCALE_FACTOR[0]
        num_upscale_layer_list    = [ int(log(i/max_scale_factor)/log(2)) for i in self.config.SEG_HEAD.UP_SCALE_FACTOR ]
        lateral_layer_in_channels = self.config.BIFPN.NUM_FEATURES
        self.seg_lateral_layers = nn.ModuleList([
            # Might need to reverse the order (pay attention to the order in the bifpn output)
            SegLateralLayer(in_channels       = lateral_layer_in_channels,
                            out_channels      = self.config.SEG_HEAD.OUT_CHANNELS,
                            num_groups        = self.config.SEG_HEAD.NUM_GROUPS,
                            num_layers        = num_upscale_layers,
                            base_scale_factor = base_scale_factor)
            for num_upscale_layers in num_upscale_layer_list
        ])

        self.head_segmask  = nn.Conv2d(in_channels  = self.config.SEG_HEAD.OUT_CHANNELS,
                                       out_channels = self.config.SEG_HEAD.NUM_CLASSES,
                                       kernel_size  = 1,
                                       padding      = 0,)

        if self.config.SEG_HEAD.USES_LEARNED_UPSAMPLE:
            self.head_upsample_layer = nn.ConvTranspose2d(in_channels  = self.config.SEG_HEAD.NUM_CLASSES,
                                                          out_channels = self.config.SEG_HEAD.NUM_CLASSES,
                                                          kernel_size  = 6,
                                                          stride       = 4,
                                                          padding      = 1,)

        self.max_scale_factor = max_scale_factor

        return None


    def extract_features(self, x):
        # Calculate and save feature maps in multiple resolutions...
        fmap_in_encoder_layers = self.backbone(x)

        # Apply the BiFPN adapter...
        bifpn_input_list = []
        for idx, fmap in enumerate(fmap_in_encoder_layers):
            bifpn_input = self.backbone_to_bifpn[idx](fmap)
            bifpn_input_list.append(bifpn_input)

        # Apply the BiFPN layer...
        bifpn_output_list = self.bifpn(bifpn_input_list)

        return bifpn_output_list


    def seg(self, x):
        # Extract features from input...
        bifpn_output_list = self.extract_features(x)

        # Fuse feature maps at each resolution (from low res to high res)...
        for idx, (lateral_layer, bifpn_output) in enumerate(zip(self.seg_lateral_layers[::-1], bifpn_output_list[::-1])):
            fmap = lateral_layer(bifpn_output)

            if idx == 0:
                fmap_acc  = fmap
            else:
                fmap_acc += fmap

        # Make prediction...
        pred_map = self.head_segmask(fmap_acc)

        # Upscale...
        pred_map = F.interpolate(pred_map,
                                 scale_factor  = self.max_scale_factor,
                                 mode          = 'bilinear',
                                 align_corners = False)                   \
                   if not self.config.SEG_HEAD.USES_LEARNED_UPSAMPLE else \
                   self.head_upsample_layer(pred_map)

        return pred_map


    def forward(self, x):
        return self.seg(x)
