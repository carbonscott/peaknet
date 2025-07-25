"""
FLOPS estimation utilities for peaknet models.

All arithmetic assumes the use of torch tensor.
"""

import torch
import torch.nn as nn


def estimate_conv_flops(
        H_input, W_input, C_input, C_output,
        H_kernel, W_kernel, H_stride, W_stride,
        H_padding, W_padding, count_multiply_add_as=1
    ):
    """
    Estimate the number of FLOPs for a convolutional layer.

    This function implements the following logic:
    - Calculate the number of times the kernel is applied (n_H * n_W).
    - Calculate the FLOPs for a single 2D kernel application.
    - Multiply by the number of input and output channels.

    The calculation takes into account:
    - Input dimensions (height, width, channels)
    - Output channels
    - Kernel dimensions
    - Stride
    - Padding

    Count each multiply-add as this many FLOPs (default is 1)
    """
    # Calculate flops for a single 2D kernel application
    flops_kernel2d = H_kernel * W_kernel * count_multiply_add_as

    # Calculate n_H and n_W (number of times kernel is applied in each dimension)
    n_H = (H_input + 2*H_padding - H_kernel) // H_stride + 1
    n_W = (W_input + 2*W_padding - W_kernel) // W_stride + 1

    # Calculate total number of kernel applications
    num_kernel_travel = n_H * n_W

    # Calculate flops for all channels
    total_flops = C_input * C_output * num_kernel_travel * flops_kernel2d

    return total_flops


def estimate_linear_flops(in_features, out_features, count_multiply_add_as=2):
    """Estimate the number of FLOPs for a linear layer."""
    return count_multiply_add_as * in_features * out_features


def estimate_transformer_flops(model_hidden_size, num_heads, num_layers, context_length):
    """
    Estimate FLOPs for transformer architecture with detailed attention calculation.
    
    | Operation                   | Input shape           | Output shape | Ops         | Reshape                                   | FLOPs         |
    |-----------------------------|-----------------------|--------------|-------------|-------------------------------------------|---------------|
    | Embedding (deepmind style)  | (B,T,V);(V,E)         | (B,T,E)      | matmul      | N/A                                       | (2V)(BTE)     |
    | Embedding (lookup table)    | (B,T)                 | (B,T,E)      | look-up     | N/A                                       | 0             |
    | KQV Proj                    | (B,T,1,E);(B,T,E,3E)  | (B,T,1,3E)   | matmul      | (B,T,1,3E) -> (B,T,1,3HN) -> (B,N,T,3H)   | (2E)(3HNBT)   |
    | KQ                          | (B,N,T,H);(B,N,H,T)   | (B,N,T,T)    | matmul      | N/A                                       | (2H)(BNTT)    |
    | Softmax                     | (B,N,T,T)             | (B,N,T,T)    | elementwise | N/A                                       | 3(BNTT)       |
    | Update V                    | (B,N,T,T);(B,N,T,H)   | (B,N,T,H)    | matmul      | N/A                                       | (2T)(BNTH)    |
    | Final Linear (Proj V)       | (B,T,NH);(NH,E)       | (B,T,E)      | matmul      | (B,N,T,H)->(B,T,NH)                       | (2NH)(BTE)    |
    | Feedforward (FF)            | (B,T,E);(E,4E)        | (B,T,4E)     | matmul      | N/A                                       | (2E)(4EBT)    |
    """
    head_hidden_size = model_hidden_size / num_heads

    flop_in_kqv_proj = (2*model_hidden_size)*(3*head_hidden_size*num_heads)               # (2E)(3HNBT)
    flop_in_kq       = (2*head_hidden_size)*(num_heads*context_length*context_length)     # (2H)(BNTT)
    flop_in_softmax  = 3*(num_heads*context_length*context_length)                        # 3(BNTT)
    flop_in_update_v = (2*context_length)*(num_heads*context_length*head_hidden_size)     # (2T)(BNTH)
    flop_in_proj_v   = (2*num_heads*head_hidden_size)*(context_length*model_hidden_size)  # (2NH)(BTE)
    flop_in_ff       = (2*model_hidden_size)*(4*model_hidden_size*context_length)         # (2E)(4EBT)

    return num_layers * (flop_in_kqv_proj+flop_in_kq+flop_in_softmax+flop_in_update_v+flop_in_proj_v+flop_in_ff)


def estimate_flops_per_token(model, dummy_shape, patch_size, count_multiply_add_as=2, includes_backward=True, device='cpu'):
    """
    Estimate FLOPs per token for any model architecture using forward hooks.
    
    Args:
        model (nn.Module): The convolutional neural network model.
        dummy_shape (List):
            - B (int): Batch size.
            - C (int): Number of channels in the input image.
            - H (int): Height of the input image.
            - W (int): Width of the input image.
        patch_size (int): Size of the patch to calculate FLOPs per token.
        count_multiply_add_as (int): Count each multiply-add as this many FLOPs (default is 2).
        includes_backward (bool): Whether to include backward pass FLOPs.
        device (str): Device to run the dummy forward pass on.

    Forward/Backward:
    - num_flops_bwd    = 2 * num_flops_fwd
    - num_flops_fwdbwd = num_flops_fwd + num_flops_bwd
                       = 3 * num_flops_fwd
    """
    hooks = []
    layer_flops = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            H_in, W_in = input[0].size(2), input[0].size(3)
            C_in = module.in_channels
            C_out = module.out_channels
            H_kernel, W_kernel = module.kernel_size
            H_stride, W_stride = module.stride
            H_padding, W_padding = module.padding

            flops = estimate_conv_flops(H_in, W_in, C_in, C_out, H_kernel, W_kernel, H_stride, W_stride, H_padding, W_padding, count_multiply_add_as)
            layer_flops.append(flops)
        elif isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            flops = estimate_linear_flops(in_features, out_features, count_multiply_add_as)
            layer_flops.append(flops)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))

    # Use dummy data to capture all parameters for estimating conv mfu
    B, C, H, W = dummy_shape
    dummy_input = torch.randn(B, C, H, W, device=device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()

    model.train()

    total_flops = sum(layer_flops)

    # Include backward pass FLOPs if specified
    if includes_backward:
        total_flops *= 3  # Fwd pass + 2x bwd pass

    # Flops per pixel
    num_pixels = torch.tensor(dummy_shape).prod().item()
    flops_per_pixel = total_flops / num_pixels

    # Flops per token
    token_size = patch_size**2
    flops_per_token = flops_per_pixel * token_size

    return flops_per_token


def estimate_mfu_per_iteration(num_flops_per_token, total_num_tokens_per_iteration, t_delta, peak_flops_per_sec):
    """
    Estimate model flops utilization (MFU) in units of peak FLOPS of the GPUs.
    
    Args:
        num_flops_per_token (float): Number of FLOPs per token.
        total_num_tokens_per_iteration (float): Total number of tokens processed in this iteration.
        t_delta (float): Time taken for this iteration in seconds.
        peak_flops_per_sec (float): Peak FLOPS per second of the hardware.
        
    Returns:
        float: Model FLOPS utilization ratio (0.0 to 1.0).
    """
    # Flops per iteration
    num_flops_per_iteration = num_flops_per_token * total_num_tokens_per_iteration

    # MFU per iteration
    num_flops_per_iteration_per_sec = num_flops_per_iteration / t_delta
    mfu = num_flops_per_iteration_per_sec / peak_flops_per_sec

    return mfu