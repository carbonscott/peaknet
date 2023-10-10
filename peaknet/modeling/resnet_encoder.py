import torch
import torch.nn as nn

from ..configurator import Configurator

from .regnet import ResNet50


class ImageEncoder(nn.Module):

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG = ResNet50.get_default_config()

            CONFIG.SAVES_FEATURE_AT_LAYER = {
                "stem"   : True,
                "layer1" : True,
                "layer2" : True,
                "layer3" : True,
                "layer4" : True,
            }

        return CONFIG


    def __init__(self, config = None):
        super().__init__()

        self.config = ImageEncoder.get_default_config() if config is None else config

        # Use the ResNet50 as the encoder...
        self.encoder = ResNet50(config = config)


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
