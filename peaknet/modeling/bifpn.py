import torch
import torch.nn            as nn
import torch.nn.functional as F

import math

from .bifpn_config import BiFPNBlockConfig, BiFPNConfig

def variance_scaling_initializer(tensor, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    """
    It's a near copy and paste from the TF implementation:
    https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/initializers/random_initializers.py#L273
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:  # 'fan_avg'
        scale /= max(1., (fan_in + fan_out) / 2.)

    if distribution == 'truncated_normal':
        std = math.sqrt(scale) / .87962566103423978  # Constant from TF implementation
                                                     # The magic number is probably caused by
                                                     # truncating random values outside [-2, 2]:
                                                     # std in this region becomes
                                                     # data = torch.empty(n_samples).normal_()
                                                     # data[(-2 <= data) & (data <= 2)].var().sqrt()
        with torch.no_grad():
            # Fold distribution to be within 2 sigma.  Basically, an in-place operation of
            # tensor = tensor % 2
            # tensor = tensor * std
            tensor.normal_().fmod_(2).mul_(std).add_(0)
    elif distribution == 'normal':
        std = math.sqrt(scale)
        with torch.no_grad():
            tensor.normal_(0, std)
    else:
        limit = math.sqrt(3.0 * scale)
        with torch.no_grad():
            tensor.uniform_(-limit, limit)

    return tensor


class DepthwiseSeparableConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d done in two steps:
    - Spatial only conv, no inter-channel communication.
    - Inter-channel communication, no spatial communication.
    """

    def __init__(self, in_channels,
                       out_channels,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       padding_mode = 'zeros',
                       device       = None,
                       dtype        = None):
        super().__init__()

        # Depthwise conv means channels are independent, only spatial bits communicate
        # Essentially it simply scales every tensor element
        self.depthwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = in_channels,
                                        kernel_size  = kernel_size,
                                        stride       = stride,
                                        padding      = padding,
                                        dilation     = dilation,
                                        groups       = in_channels,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)

        # Pointwise to facilitate inter-channel communication, no spatial bits communicate
        self.pointwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1,
                                        stride       = 1,
                                        padding      = 0,
                                        dilation     = 1,
                                        groups       = 1,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)


    def _init_weights(self):
        """ """
        nn.init.kaiming_uniform_(self.depthwise_conv.weight, mode='fan_in', nonlinearity='relu')
        if self.depthwise_conv.bias is not None:
            nn.init.zeros_(self.depthwise_conv.bias)

        nn.init.kaiming_uniform_(self.pointwise_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.pointwise_conv.bias is not None:
            nn.init.zeros_(self.pointwise_conv.bias)


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class BiFPNLayerNorm(nn.Module):
    """
    Design is similar to https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/convnextv2/modeling_convnextv2.py#L111
    """
    def __init__(self, normalized_shape, eps = 1e-6):
        super().__init__()

        self.normalized_shape = normalized_shape

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def _init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)



class BiFPNBlock(nn.Module):
    """
    One BiFPN block takes feature maps at L different scales(L>=3):
    (p_i, p_(i+1), ..., p_(i+L-1))

    where p_i = 2**(-i).

    Notice that the input to BiFPNBlock should have the same channel size,
    which can be achieved with a lateral conv layer in the upstream.
    """

    @staticmethod
    def get_default_config():
        return BiFPNBlockConfig()


    def __init__(self, config = None):
        super().__init__()

        self.config = BiFPNBlock.get_default_config() if config is None else config

        # Define shortcut variables that access the config values...
        num_features = self.config.num_features
        num_levels   = self.config.num_levels
        base_level   = self.config.base_level

        # Confusingly, there should be at least 3 levels in total...
        num_levels = max(num_levels, 3)

        # Decide the min max level...
        min_level  = base_level
        max_level  = base_level + (num_levels - 1)

        # Create conv2d layers for fusion stage M...
        # min_level, ..., max_level - 1
        m_conv = nn.ModuleDict({
            f"m{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                BiFPNLayerNorm(normalized_shape = (num_features,)),
                nn.GELU(),
            )
            for level in range(min_level, max_level)
        })

        # Create conv2d layers for fusion stage Q...
        # min_level + 1, max_level
        q_conv = nn.ModuleDict({
            f"q{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                BiFPNLayerNorm(normalized_shape = (num_features,)),
                nn.GELU(),
            )
            for level in range(min_level + 1, max_level + 1)
        })

        self.conv = nn.ModuleDict()
        self.conv.update(m_conv)
        self.conv.update(q_conv)

        # Define the weights used in fusion
        num_level_stage_m = max_level - min_level
        num_level_stage_q = num_level_stage_m
        self.w_m = nn.Parameter(torch.randn(num_level_stage_m, 2))    # Two-component fusion at stage M
        self.w_q = nn.Parameter(torch.randn(num_level_stage_q, 3))    # Three-component fusion at stage Q

        # Keep these numbers as attributes...
        self.min_level  = min_level
        self.max_level  = max_level


    def _init_weights(self):
        # Initialize fusion weights
        nn.init.constant_(self.w_m, 1e-4)
        nn.init.constant_(self.w_q, 1e-4)

        # Initialize DepthwiseSeparableConv2d and BiFPNLayerNorm in conv layers
        for module in self.conv.values():
            for layer in module:
                if isinstance(layer, DepthwiseSeparableConv2d):
                    layer._init_weights()
                elif isinstance(layer, BiFPNLayerNorm):
                    layer._init_weights()


    def forward(self, x):
        # Keep these numbers as attributes...
        min_level  = self.min_level
        max_level  = self.max_level
        num_levels = max_level - min_level + 1

        # Unpack feature maps into dict...
        # x is 0-based index
        # (B, C, [H], [W])
        p = { level : x[idx] for idx, level in enumerate(range(min_level, min_level + num_levels)) }

        # ___/ Stage M \___
        # Fuse features from low resolution to high resolution (pathway M)...
        m = {}
        for idx, level_low in enumerate(range(max_level, min_level, -1)):
            # Specify two feature maps for fusion
            level_high = level_low - 1
            m_low   = p[level_low ] if idx == 0 else m[level_low]
            p_high  = p[level_high]

            # Fusion...
            # ...Upscaling
            orig_dtype = m_low.dtype
            m_low_up = F.interpolate(
                m_low.to(torch.float32),
                scale_factor  = self.config.up_scale_factor,
                mode          = 'bilinear',
                align_corners = False
            ).to(orig_dtype)

            # ...Normalized weighted sum with learnable summing weights
            w1, w2 = self.w_m[idx]
            m_fused  = w1 * p_high + w2 * m_low_up
            m_fused /= (w1 + w2 + self.config.fusion.eps)

            # ...Pass through learnable layer to refine and produce better features
            m_fused  = self.conv[f"m{level_high}"](m_fused)

            # Add skip connection to allow adaptive learning
            # [NOTE] If fused feature is not helpful, it's okay to bypass it
            # entirely through the skip connection.  ¯\_(ツ)_/¯
            m_fused = m_fused + p_high

            # Track the new feature map
            m[level_high] = m_fused

        # ___/ Stage Q \___
        # Fuse features from high resolution to low resolution (pathway Q)...
        q = {}
        for idx, level_high in enumerate(range(min_level, max_level)):
            level_low = level_high + 1
            q_high = m[level_high] if idx == 0              else q[level_high]
            m_low  = m[level_low ] if level_low < max_level else p[level_low ]
            p_low  = p[level_low ]

            # Fusion
            # ...Downscaling
            orig_dtype = q_high.dtype
            q_high_down = F.interpolate(
                q_high.to(torch.float32),
                scale_factor  = self.config.down_scale_factor,
                mode          = 'bilinear',
                align_corners = False
            ).to(orig_dtype)

            # ...Normalized weighted sum with learnable summing weights
            w1, w2, w3 = self.w_q[idx]
            q_fused  = w1 * p_low + w2 * m_low + w3 * q_high_down
            q_fused /= (w1 + w2 + w3 + self.config.fusion.eps)

            # ...Pass through learnable layer to refine and produce better features
            q_fused  = self.conv[f"q{level_low}"](q_fused)

            # Add skip connection to allow adaptive learning
            q_fused = q_fused + p_low

            # Track the new feature map
            q[level_low] = q_fused

            # Track the highest level q feature
            if idx == 0: q[level_high] = q_high

        return [ q[level] for level in range(min_level, min_level + num_levels) ]




class BiFPN(nn.Module):
    """
    This class provides a series of BiFPN blocks.
    """

    @staticmethod
    def get_default_config():
        return BiFPNConfig()


    def __init__(self, config = None):
        super().__init__()

        self.config = BiFPN.get_default_config() if config is None else config

        num_blocks = self.config.num_blocks
        self.blocks = nn.Sequential(*[
            BiFPNBlock(config = self.config.block)
            for block_idx in range(num_blocks)
        ])


    def _init_weights(self):
        for block in self.blocks:
            block._init_weights()


    def forward(self, x):
        return self.blocks(x)
