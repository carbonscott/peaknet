import torch
import torch.nn as nn

from .common import DoubleConv


class ImageEncoder(nn.Module):

    def __init__(self, config_channels, uses_skip_connection, saves_feature_per_layer = True):
        super().__init__()

        self.config_channels         = config_channels
        self.uses_skip_connection    = uses_skip_connection
        self.saves_feature_per_layer = saves_feature_per_layer

        # Create the input layer...
        in_channels, out_channels = config_channels["input_layer"]
        self.input_layer = DoubleConv(in_channels          = in_channels,
                                      out_channels         = out_channels,
                                      stride               = 1,
                                      uses_skip_connection = uses_skip_connection,)

        # Create the encoder layers...
        layer_channels = config_channels["encoder_layers"]
        self.encoder_layers = nn.ModuleList([
            self._make_layer(in_channels  = in_channels,
                             out_channels = out_channels,)
            for in_channels, out_channels in layer_channels
        ])



    def _make_layer(self, in_channels, out_channels):
        """
        Make the backbone encoder layer.
        - Pooling.
        - Double Conv Block.

        Args:
        """
        uses_skip_connection = self.uses_skip_connection

        layer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            DoubleConv(in_channels          = in_channels,
                       out_channels         = out_channels,
                       stride               = 1,
                       uses_skip_connection = uses_skip_connection),
        )

        return layer


    def forward(self, x):
        saves_feature_per_layer = self.saves_feature_per_layer
        fmap_in_layers = []

        # Pass the input through the input layer...
        x = self.input_layer(x)
        if saves_feature_per_layer: fmap_in_layers.append(x)

        # Pass the feature map through a series of encoding layer...
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            if saves_feature_per_layer: fmap_in_layers.append(x)

        ret = fmap_in_layers if saves_feature_per_layer else x
        return ret
