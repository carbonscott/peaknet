import pytest
import torch
import numpy as np
from peaknet.modeling.convnextv2_encoder  import ConvNextV2Backbone, ConvNextV2BackboneConfig
from peaknet.modeling.convnextv2_bifpn_net import PeakNetConfig, PeakNet
from peaknet.trans import Resize

@pytest.mark.parametrize("model_name", ConvNextV2Backbone.show_supported_models())
def test_model_support(model_name):
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Initialize the model with the given configuration
        backbone_config = ConvNextV2BackboneConfig(model_name = model_name, uses_pretrained = False)
        peaknet_config  = PeakNetConfig(BACKBONE = backbone_config)
        model           = PeakNet(config = peaknet_config).to(device)

        model.eval()

        # Setup fake data
        B, C, H, W = 2, 1, 1024, 1024
        batch_input = torch.rand(B, C, H, W, dtype = torch.float, device = device)

        # Pass the input through the model
        with torch.no_grad():
            segmask = model(batch_input)

        # Explicitly delete the model and input tensor
        del model
        del batch_input
        if device != 'cpu':
            torch.cuda.empty_cache()

        # If the code reaches this point without error, the test passes for this model
        assert segmask is not None, f"Model {model_name} produced None output."

    except Exception as e:
        pytest.fail(f"Unexpected error occurred for model {model_name}: {e}")

