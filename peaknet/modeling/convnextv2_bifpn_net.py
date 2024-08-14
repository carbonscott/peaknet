import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import os

from math import log

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Optional

from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Backbone

from .bifpn        import BiFPN
from .bifpn_config import BiFPNConfig
from .utils_build  import BackboneToBiFPNAdapterConfig, BackboneToBiFPNAdapter

import logging
logger = logging.getLogger(__name__)

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
                nn.GELU(),
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
                x_dtype = x.dtype
                x = F.interpolate(
                    x.to(torch.float32),
                    scale_factor  = self.base_scale_factor,
                    mode          = 'bilinear',
                    align_corners = False
                ).to(x_dtype)

        return x


@dataclass
class SegHeadConfig:
    up_scale_factor: List[int] = field(
        default_factory = lambda : [
            4,  # stage1
            8,  # stage2
            16, # stage3
            32, # stage4
        ]
    )
    num_groups           : int  = 32
    out_channels         : int  = 256
    num_classes          : int  = 2
    base_scale_factor    : int  = 2
    uses_learned_upsample: bool = False


@dataclass
class PeakNetConfig:
    """
    ConvNextV2Config params:
        num_channels      = 1,
        patch_size        = 4,
        num_stages        = 4,
        hidden_sizes      = None,  # [96, 192, 384, 768]
        depths            = None,  # [3, 3, 9, 3]
        hidden_act        = "gelu",
        initializer_range = 0.02,
        layer_norm_eps    = 1e-12,
        drop_path_rate    = 0.0,
        image_size        = 224,
        out_features      = None,  # out_features = ['stage1', 'stage2', 'stage3', 'stage4']
        out_indices       = None,
    """
    backbone: ConvNextV2Config = ConvNextV2Config(
        num_channels = 1,
        out_features = ['stage1', 'stage2', 'stage3', 'stage4'],
    )
    bifpn             : BiFPNConfig              = BiFPN.get_default_config()
    seg_head          : SegHeadConfig            = SegHeadConfig()


class PeakNet(nn.Module):
    def __init__(self, config = None):
        super().__init__()

        self.config = config

        # Create the image encoder...
        backbone_config = self.config.backbone
        self.backbone = ConvNextV2Backbone(config = backbone_config)

        # Create the adapter layer between encoder and bifpn...
        backbone_output_channels = backbone_config.hidden_sizes
        num_bifpn_features = self.config.bifpn.block.num_features
        self.backbone_to_bifpn = nn.ModuleList([
            nn.Conv2d(in_channels  = in_channels,
                      out_channels = num_bifpn_features,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0)
            for in_channels in backbone_output_channels
        ])

        # Create the fusion blocks...
        self.bifpn = BiFPN(config = self.config.bifpn)

        # Create the prediction head...
        base_scale_factor         = self.config.seg_head.base_scale_factor
        max_scale_factor          = self.config.seg_head.up_scale_factor[0]
        num_upscale_layer_list    = [ int(log(i/max_scale_factor)/log(2)) for i in self.config.seg_head.up_scale_factor ]
        lateral_layer_in_channels = self.config.bifpn.block.num_features
        self.seg_lateral_layers = nn.ModuleList([
            # Might need to reverse the order (pay attention to the order in the bifpn output)
            SegLateralLayer(in_channels       = lateral_layer_in_channels,
                            out_channels      = self.config.seg_head.out_channels,
                            num_groups        = self.config.seg_head.num_groups,
                            num_layers        = num_upscale_layers,
                            base_scale_factor = base_scale_factor)
            for num_upscale_layers in num_upscale_layer_list
        ])

        self.head_segmask = nn.Conv2d(in_channels  = self.config.seg_head.out_channels,
                                      out_channels = self.config.seg_head.num_classes,
                                      kernel_size  = 1,
                                      padding      = 0,)

        if self.config.seg_head.uses_learned_upsample:
            # [NOTE] output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
            # - stride  := max_scale_factor
            # - padding := 1
            # - kernel  := stride+2
            self.head_upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels    = self.config.seg_head.num_classes,
                    out_channels   = self.config.seg_head.num_classes,
                    kernel_size    = max_scale_factor+2,
                    stride         = max_scale_factor,
                    padding        = 1,
                ),
                nn.GroupNorm(1, self.config.seg_head.num_classes),
                nn.GELU(),
            )

        # Refine the final prediction
        self.final_conv = nn.Conv2d(
            self.config.seg_head.num_classes,
            self.config.seg_head.num_classes,
            kernel_size=3,
            padding=1
        )

        self.max_scale_factor = max_scale_factor

        return None


    def init_weights(self):
        # Backbone has its own _init_weights
        self.backbone.apply(self.backbone._init_weights)

        # Initialize backbone_to_bifpn
        for m in self.backbone_to_bifpn:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # BiFPN has its own _init_weights
        self.bifpn._init_weights()

        # Initialize seg_lateral_layers
        for layer in self.seg_lateral_layers:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Initialize head_segmask
        nn.init.kaiming_normal_(self.head_segmask.weight, mode='fan_in', nonlinearity='linear')
        if self.head_segmask.bias is not None:
            nn.init.constant_(self.head_segmask.bias, 0)

        # Initialize head_upsample_layer if it exists
        if self.config.seg_head.uses_learned_upsample:
            for m in self.head_upsample_layer:
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Initialize final_conv
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='linear')
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, 0)


    def extract_features(self, x):
        # Calculate and save feature maps in multiple resolutions...
        # The output attributes are defined by https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/models/convnextv2/modeling_convnextv2.py#L568
        backbone_output = self.backbone(x)
        fmap_in_encoder_layers = backbone_output.feature_maps

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

        # Direct upscale as skip connection
        pred_map_dtype = pred_map.dtype
        pred_map = F.interpolate(
            pred_map.to(torch.float32),
            scale_factor  = self.max_scale_factor,
            mode          = 'bilinear',
            align_corners = False
        ).to(pred_map_dtype) \

        if self.config.seg_head.uses_learned_upsample:
            # Learnable upscale
            residual_map = self.head_upsample_layer(pred_map)

            # Skip connection
            pred_map = pred_map + residual_map

        pred_map = self.final_conv(pred_map)

        return pred_map


    def forward(self, x):
        return self.seg(x)
