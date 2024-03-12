import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class BackboneToBiFPNAdapterConfig:
    num_bifpn_features      : int  = 256
    backbone_output_channels: Dict[str, int] = field(default_factory=lambda: {
        "stage0": 40,
        "stage1": 80,
        "stage2": 160,
        "stage3": 320,
    })

class BackboneToBiFPNAdapter(nn.Module):
    """
    A class to create an adapter layer that connects the output of a backbone network
    to the input of a BiFPN (Bidirectional Feature Pyramid Network) layer.
    """
    @staticmethod
    def get_default_config():
        return BackboneToBiFPNAdapterConfig()

    def __init__(self, config = None):
        super().__init__()

        self.config              = BackboneToBiFPNAdapter.get_default_config() if config is None else config
        num_bifpn_features       = self.config.num_bifpn_features
        backbone_output_channels = self.config.backbone_output_channels

        self.adapters = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=num_bifpn_features,
                      kernel_size=1,
                      stride=1,
                      padding=0)
            for _, in_channels in backbone_output_channels.items()
        ])


    def forward(self, x_in_stages):
        """
        Forward pass through the adapter layers. Assumes x is a dictionary where keys
        match those in backbone_output_channels, and values are the corresponding feature maps.

        Parameters:
        - x_in_stages: A list of tensors from the backbone layers.

        Returns:
        - A list of feature maps processed to match the BiFPN input dimensions.
        """
        return [ adapter(x_in_stage) for adapter, x_in_stage in zip(self.adapters, x_in_stages) ]
