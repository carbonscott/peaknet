import torch
import torch.nn            as nn
import torch.nn.functional as F

from ..config import CONFIG

from .blocks import conv2d, pool2d


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
        p7, p6, p5, p4, p3 = x    # (B, C, [H], [W])

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
