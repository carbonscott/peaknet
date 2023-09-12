import torch
import torch.nn as nn

from ..config import CONFIG

from .regnet import ResNet50


class ImageEncoder(nn.Module):

    def __init__(self, saves_feature_per_layer = True):
        super().__init__()

        self.saves_feature_per_layer = saves_feature_per_layer

        # Use the ResNet50 as the encoder...
        self.encoder = ResNet50()


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
