import torch
import torch.nn as nn

from dataclasses import dataclass, field

from typing import Dict

from .regnet import ResNet50Config, ResNet50


@dataclass
class ImageEncoderConfig:
    RESNET50: ResNet50Config = ResNet50.get_default_config()

    SAVES_FEATURE_AT_LAYER: Dict[str, bool] = field(
        default_factory = lambda : {
            "stem"   : False,
            "layer1" : True,
            "layer2" : True,
            "layer3" : True,
            "layer4" : True,
        }
    )

    OUTPUT_CHANNELS: Dict[str, int] = field(
        default_factory = lambda : {
            "stem"   : 64,
            "layer1" : 256,
            "layer2" : 512,
            "layer3" : 1024,
            "layer4" : 2048,
        }
    )


class ImageEncoder(nn.Module):

    @staticmethod
    def get_default_config():
        return ImageEncoderConfig()


    def __init__(self, config = None):
        super().__init__()

        self.config = ImageEncoder.get_default_config() if config is None else config

        # Use the ResNet50 as the encoder...
        self.encoder = ResNet50(config = config.RESNET50)


    def forward(self, x):
        fmap_in_layers = []

        # Going through the immediate layers and collect feature maps...
        for name, encoder_layer in self.encoder.named_children():
            x = encoder_layer(x)

            # Save fmap from all layers except these excluded...
            if self.config.SAVES_FEATURE_AT_LAYER.get(name, False):
                fmap_in_layers.append(x)

        ret = fmap_in_layers if len(fmap_in_layers) else x
        return ret
