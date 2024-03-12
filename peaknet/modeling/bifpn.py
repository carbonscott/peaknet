import torch
import torch.nn            as nn
import torch.nn.functional as F

from .bifpn_config import BiFPNBlockConfig, BiFPNConfig

class DepthwiseSeparableConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d doen in two steps:
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


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x




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
                nn.BatchNorm2d(num_features = num_features,
                               eps          = self.config.bn.eps,
                               momentum     = self.config.bn.momentum),
                nn.ReLU(),
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
                nn.BatchNorm2d(num_features = num_features,
                               eps          = self.config.bn.eps,
                               momentum     = self.config.bn.momentum),
                nn.ReLU(),
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
            level_high = level_low - 1
            m_low   = p[level_low ] if idx == 0 else m[level_low]
            p_high  = p[level_high]

            w1, w2 = self.w_m[idx]
            m_low_up = F.interpolate(m_low,
                                     scale_factor  = self.config.up_scale_factor,
                                     mode          = 'bilinear',
                                     align_corners = False)
            m_fused  = w1 * p_high + w2 * m_low_up
            m_fused /= (w1 + w2 + self.config.fusion.eps)
            m_fused  = self.conv[f"m{level_high}"](m_fused)

            m[level_high] = m_fused

        # ___/ Stage Q \___
        # Fuse features from high resolution to low resolution (pathway Q)...
        q = {}
        for idx, level_high in enumerate(range(min_level, max_level)):
            level_low = level_high + 1
            q_high = m[level_high] if idx == 0              else q[level_high]
            m_low  = m[level_low ] if level_low < max_level else p[level_low ]
            p_low  = p[level_low ]

            w1, w2, w3 = self.w_q[idx]
            q_high_up = F.interpolate(q_high,
                                      scale_factor  = self.config.down_scale_factor,
                                      mode          = 'bilinear',
                                      align_corners = False)
            q_fused  = w1 * p_low + w2 * m_low + w3 * q_high_up
            q_fused /= (w1 + w2 + w3 + self.config.fusion.eps)
            q_fused  = self.conv[f"q{level_low}"](q_fused)

            if idx == 0: q[level_high] = q_high
            q[level_low] = q_fused

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


    def forward(self, x):
        return self.blocks(x)




#####################################################################
# RAW BIFPN IMPLEMENTATION BELOW IS FOR LEARNING/DEBUGGING PURPOSES #
#####################################################################
'''
class BiFPNBlockEDU(nn.Module):
    """
    !!!
    THIS CLASS IS MEANT FOR EDUCATING MYSELF ON BIFPN.  ALL LAYERS ARE
    HARD-CODED.
    !!!

    One BiFPN block takes feature maps at five different scales:
    (p3, p4, p5, p6, p7).

    Notice that the input to BiFPNBlock should have the same channel size, 
    which can be achieved with a conv layer in the upstream.
    """

    def __init__(self, num_features = 64):
        super().__init__()

        # Create conv2d layers for fusion stage M...
        m_conv = nn.ModuleDict({
            f"m{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                nn.BatchNorm2d(num_features = num_features,
                               eps          = CONFIG.BIFPN.BN.EPS,
                               momentum     = CONFIG.BIFPN.BN.MOMENTUM),
                nn.ReLU(),
            )
            for level in (6, 5, 4, 3)
        })

        # Create conv2d layers for fusion stage Q...
        q_conv = nn.ModuleDict({
            f"q{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                nn.BatchNorm2d(num_features = num_features,
                               eps          = CONFIG.BIFPN.BN.EPS,
                               momentum     = CONFIG.BIFPN.BN.MOMENTUM),
                nn.ReLU(),
            )
            for level in (4, 5, 6, 7)
        })

        self.conv = nn.ModuleDict()
        self.conv.update(m_conv)
        self.conv.update(q_conv)

        # Define the weights used in fusion
        self.w_m = nn.Parameter(torch.randn(4, 2))    # Two-component fusion at stage M
        self.w_q = nn.Parameter(torch.randn(4, 3))    # Three-component fusion at stage Q


    def forward(self, x):
        # Unpack the feature maps at five different scales...
        p3, p4, p5, p6, p7 = x    # (B, C, [H], [W])

        # ___/ Stage M \___
        # Fuse M6 = P7 + P6
        w1, w2 = self.w_m[0]
        p7_up = F.interpolate(p7,
                              scale_factor  = CONFIG.BIFPN.UP_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        m6  = w1 * p6 + w2 * p7_up
        m6 /= (w1 + w2 + CONFIG.BIFPN.FUSION.EPS)
        m6  = self.conv.m6(m6)

        # Fuse M5 = M6 + P5
        w1, w2 = self.w_m[1]
        m6_up = F.interpolate(m6,
                              scale_factor  = CONFIG.BIFPN.UP_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        m5  = w1 * p5 + w2 * m6_up
        m5 /= (w1 + w2 + CONFIG.BIFPN.FUSION.EPS)
        m5  = self.conv.m5(m5)

        # Fuse M4 = M5 + P4
        w1, w2 = self.w_m[2]
        m5_up = F.interpolate(m5,
                              scale_factor  = CONFIG.BIFPN.UP_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        m4  = w1 * p4 + w2 * m5_up
        m4 /= (w1 + w2 + CONFIG.BIFPN.FUSION.EPS)
        m4  = self.conv.m4(m4)

        # Fuse M3 = M4 + P3
        w1, w2 = self.w_m[3]
        m4_up = F.interpolate(m4,
                              scale_factor  = CONFIG.BIFPN.UP_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        m3  = w1 * p3 + w2 * m4_up
        m3 /= (w1 + w2 + CONFIG.BIFPN.FUSION.EPS)
        m3  = self.conv.m3(m3)

        # ___/ Stage Q \___
        # Fuse Q4 = Q3 + M4 + P4
        w1, w2, w3 = self.w_q[0]
        q3 = m3
        q3_up = F.interpolate(q3,
                              scale_factor  = CONFIG.BIFPN.DOWN_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        q4  = w1 * p4 + w2 * m4 + w3 * q3_up
        q4 /= (w1 + w2 + w3 + CONFIG.BIFPN.FUSION.EPS)
        q4  = self.conv.q4(q4)

        # Fuse Q5 = Q4 + M5 + P5
        w1, w2, w3 = self.w_q[1]
        q4_up = F.interpolate(q4,
                              scale_factor  = CONFIG.BIFPN.DOWN_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        q5 = w1 * p5 + w2 * m5 + w3 * q4_up
        q5 /= (w1 + w2 + w3 + CONFIG.BIFPN.FUSION.EPS)
        q5  = self.conv.q5(q5)

        # Fuse Q6 = Q5 + M6 + P6
        w1, w2, w3 = self.w_q[2]
        q5_up = F.interpolate(q5,
                              scale_factor  = CONFIG.BIFPN.DOWN_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        q6 = w1 * p6 + w2 * m6 + w3 * q5_up
        q6 /= (w1 + w2 + w3 + CONFIG.BIFPN.FUSION.EPS)
        q6  = self.conv.q6(q6)

        # Fuse Q7 = Q6 + P7
        w1, w2, w3 = self.w_q[3]
        q6_up = F.interpolate(q6,
                              scale_factor  = CONFIG.BIFPN.DOWN_SCALE_FACTOR,
                              mode          = 'bilinear',
                              align_corners = False)
        q7 = w1 * p7 + w2 * p7 + w3 * q6_up
        q7 /= (w1 + w2 + w3 + CONFIG.BIFPN.FUSION.EPS)
        q7  = self.conv.q7(q7)

        return q3, q4, q5, q6, q7
'''
