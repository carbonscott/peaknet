import torch
import torch.nn as nn

from ..configurator import Configurator

from .regnet import ResNet50


class ImageEncoder(nn.Module):

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG.RESSTEM.BN.EPS       = 1e-5
            CONFIG.RESSTEM.BN.MOMENTUM  = 1e-1
            CONFIG.RESSTEM.RELU_INPLACE = False

            CONFIG.RESSTAGE.BN.EPS       = 1e-5
            CONFIG.RESSTAGE.BN.MOMENTUM  = 1e-1
            CONFIG.RESSTAGE.RELU_INPLACE = False
            CONFIG.RESSTAGE.USES_RES_V1p5 = True

            CONFIG.USES_RES_V1p5 = True

        return CONFIG


    def __init__(self, saves_feature_per_layer = True, config = None):
        super().__init__()

        if config is None: config = ImageEncoder.get_default_config()

        self.saves_feature_per_layer = saves_feature_per_layer

        # Use the ResNet50 as the encoder...
        self.encoder = ResNet50(config = config)


    def forward(self, x):
        saves_feature_per_layer = self.saves_feature_per_layer
        fmap_in_layers = []

        # Going through the immediate layers and collect feature maps...
        for name, encoder_layer in self.encoder.named_children():
            x = encoder_layer(x)

            # Save fmap from all layers except these excluded...
            if saves_feature_per_layer and CONFIG.RESNET_ENCODER.SAVES_LAYER[name]:
                fmap_in_layers.append(x)

        ret = fmap_in_layers if saves_feature_per_layer else x
        return ret
