import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import os

from math import log

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Optional

from .convnextv2_encoder import ConvNextV2BackboneConfig, ConvNextV2Backbone

from .bifpn        import BiFPN
from .bifpn_config import BiFPNConfig
from .utils_build  import BackboneToBiFPNAdapterConfig, BackboneToBiFPNAdapter

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
                x = F.interpolate(x.to(torch.float32),
                                  scale_factor  = self.base_scale_factor,
                                  mode          = 'bilinear',
                                  align_corners = False).to(torch.bfloat16)

        return x


@dataclass
class SegHeadConfig:
    up_scale_factor: List[int] = field(
        default_factory = lambda : [
            4,  # stage0
            8,  # stage1
            16, # stage2
            32, # stage3
        ]
    )
    num_groups           : int  = 32
    out_channels         : int  = 256
    num_classes          : int  = 2
    base_scale_factor    : int  = 2
    uses_learned_upsample: bool = False


@dataclass
class PeakNetConfig:
    backbone          : ConvNextV2BackboneConfig = ConvNextV2Backbone.get_default_config()
    bifpn             : BiFPNConfig              = BiFPN.get_default_config()
    seg_head          : SegHeadConfig            = SegHeadConfig()
    channels_in_stages: Optional[Dict[str, int]] = None


class PeakNet(nn.Module):
    @staticmethod
    def get_default_config():
        return PeakNetConfig()


    @staticmethod
    def estimate_output_channels(model_name):
        """ Estimate the output channels for an input backbone.

            Only rank 0 will perform the calculation.

            The output will be saved under the home directory.
        """
        rank = int(os.environ.get("RANK", 0))

        cache_dir  = os.environ.get("HOME", os.getcwd())
        cache_dir  = os.path.join(cache_dir, '.cache/peaknet')
        cache_file = os.path.join(cache_dir, f'output_channels.{model_name}.pt')

        if not os.path.exists(cache_file):
            if rank == 0:
                print(f"[RANK {rank}] Creating cache for building the model...")
                os.makedirs(cache_dir, exist_ok=True)

                device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

                B, C, H, W = 2, 1, 1*(2**4)*4, 1*(2**4)*4
                batch_input = torch.rand(B, C, H, W, dtype = torch.float, device = device)

                config = ConvNextV2BackboneConfig()
                config.model_name = model_name

                model = ConvNextV2Backbone(config)
                model.to(device)
                model.eval()
                with torch.no_grad():
                    stage_feature_maps = model(batch_input)
                model.train()

                # Explicitly delete the model and input tensor
                del model
                del batch_input
                if device != 'cpu':
                    torch.cuda.empty_cache()

                output_channels = { f"stage{enum_idx}" : stage_feature_map.shape[1] for enum_idx, stage_feature_map in enumerate(stage_feature_maps) }    # (B, C, H, W)

                torch.save(output_channels, cache_file)
            else:
                print(f"[RANK {rank}] Waiting...")

        if dist.is_initialized():
            dist.barrier()

        if os.path.exists(cache_file):
            print(f"[RANK {rank}] Loading the model building cache...")
            return torch.load(cache_file, map_location='cpu')    # CPU is fine for loading python dict

        raise RuntimeError(f"Cache file '{cache_file}' not found. This should not happen after synchronization barrier.")


    def __init__(self, config = None):
        super().__init__()

        self.config = PeakNet.get_default_config() if config is None else config

        # Create the image encoder...
        backbone_config = self.config.backbone
        self.backbone = ConvNextV2Backbone(config = backbone_config)

        # Adjust dependent settings based on the chosen pre-trained model...
        # ...Estimate the output channels
        backbone_output_channels = self.config.channels_in_stages
        if backbone_output_channels is None:
            backbone_output_channels = PeakNet.estimate_output_channels(backbone_config.model_name)

        # Create the adapter layer between encoder and bifpn...
        num_bifpn_features = self.config.bifpn.block.num_features
        self.backbone_to_bifpn = nn.ModuleList([
            nn.Conv2d(in_channels  = in_channels,
                      out_channels = num_bifpn_features,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0)
            for _, in_channels in backbone_output_channels.items()
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

        self.head_segmask  = nn.Conv2d(in_channels  = self.config.seg_head.out_channels,
                                       out_channels = self.config.seg_head.num_classes,
                                       kernel_size  = 1,
                                       padding      = 0,)

        if self.config.seg_head.uses_learned_upsample:
            self.head_upsample_layer = nn.ConvTranspose2d(in_channels  = self.config.seg_head.num_classes,
                                                          out_channels = self.config.seg_head.num_classes,
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
        pred_map = F.interpolate(pred_map.to(torch.float32),
                                 scale_factor  = self.max_scale_factor,
                                 mode          = 'bilinear',
                                 align_corners = False).to(torch.bfloat16) \
                   if not self.config.seg_head.uses_learned_upsample else  \
                   self.head_upsample_layer(pred_map)

        return pred_map


    def forward(self, x):
        return self.seg(x)
