import torch
import torch.nn as nn

import timm

from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ConvNextV2BackboneConfig:
    in_channels         : int  = 1
    model_name          : str  = 'convnextv2_atto.fcmae'
    downloads_weights   : bool = True
    path_pretrain_chkpt : Optional[str] = None


class ConvNextV2Backbone(nn.Module):
    @staticmethod
    def get_default_config():
        return ConvNextV2BackboneConfig()

    @staticmethod
    def show_supported_models():
        return [
            "convnextv2_atto.fcmae",
            "convnextv2_atto.fcmae_ft_in1k",
            "convnextv2_base.fcmae",
            "convnextv2_base.fcmae_ft_in1k",
            "convnextv2_base.fcmae_ft_in22k_in1k",
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "convnextv2_femto.fcmae",
            "convnextv2_femto.fcmae_ft_in1k",
            "convnextv2_huge.fcmae",
            "convnextv2_huge.fcmae_ft_in1k",
            "convnextv2_huge.fcmae_ft_in22k_in1k_384",
            "convnextv2_huge.fcmae_ft_in22k_in1k_512",
            "convnextv2_large.fcmae",
            "convnextv2_large.fcmae_ft_in1k",
            "convnextv2_large.fcmae_ft_in22k_in1k",
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
            "convnextv2_nano.fcmae",
            "convnextv2_nano.fcmae_ft_in1k",
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            "convnextv2_nano.fcmae_ft_in22k_in1k_384",
            "convnextv2_pico.fcmae",
            "convnextv2_pico.fcmae_ft_in1k",
            "convnextv2_tiny.fcmae",
            "convnextv2_tiny.fcmae_ft_in1k",
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        ]

    def __init__(self, config = None):
        super().__init__()

        self.config         = ConvNextV2Backbone.get_default_config() if config is None else config
        in_channels         = self.config.in_channels
        model_name          = self.config.model_name
        downloads_weights   = self.config.downloads_weights
        path_pretrain_chkpt = self.config.path_pretrain_chkpt

        model = timm.create_model(model_name, pretrained = downloads_weights)
        if path_pretrain_chkpt is not None:
            pretrain_chkpt = torch.load(path_pretrain_chkpt)
            print(f"-- Loading weights from {path_pretrain_chkpt}...")
            model.load_state_dict(pretrain_chkpt, strict = False)
            print(f"-- Done loading weights.")

        out_channels = model.stem[0].out_channels
        kernel_size  = model.stem[0].kernel_size
        stride       = model.stem[0].stride

        ave_weight_patch_embd     = model.stem[0].weight.data.mean(dim = in_channels, keepdim = True) # [40, 3, 4, 4] -> [40, 1, 4, 4]
        model.stem[0]             = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)
        model.stem[0].weight.data = ave_weight_patch_embd

        model.head = nn.Identity()    # Remove unsed parameters to save memory

        self.stem   = model.stem
        self.stages = model.stages


    def forward(self, x):
        # Process input through embeddings
        embedding_output = self.stem(x)

        # Initialize a list to hold the feature maps from each stage
        stage_feature_maps = []

        # Forward through each stage
        hidden_states = embedding_output
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            stage_feature_maps.append(hidden_states)

        return stage_feature_maps
