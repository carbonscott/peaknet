import pytest
import torch
import numpy as np
from peaknet.modeling.convnextv2_encoder import ConvNextV2Backbone, ConvNextV2BackboneConfig

@pytest.mark.parametrize("model_name", ConvNextV2Backbone.show_supported_models())
def test_model_support(model_name):
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Initialize the model with the given configuration
        config = ConvNextV2BackboneConfig(model_name = model_name, uses_pretrained = False)
        model  = ConvNextV2Backbone(config).to(device)
        model.eval()

        # Setup fake data
        B, C, H, W = 2, 1, 1024, 1024
        batch_input = torch.rand(B, C, H, W, dtype = torch.float)

        # Pass the input through the model
        with torch.no_grad():
            batch_input_tensor = batch_input.to(device)
            stage_feature_maps = model(batch_input_tensor)

        # Explicitly delete the model and input tensor
        del model
        del batch_input
        if device != 'cpu':
            torch.cuda.empty_cache()

        # If the code reaches this point without error, the test passes for this model
        assert stage_feature_maps is not None, f"Model {model_name} produced None output."

    except Exception as e:
        pytest.fail(f"Unexpected error occurred for model {model_name}: {e}")

