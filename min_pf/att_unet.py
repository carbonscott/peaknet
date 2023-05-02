import torch
import torch.nn as nn
import torch.nn.functional as F

from peaknet.datasets.transform import center_crop

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, uses_skip_connection = False):
        super().__init__()

        self.stride = stride
        self.uses_skip_connection = uses_skip_connection

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        if uses_skip_connection:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.uses_skip_connection:
            # Do not use out += as it's an 'in-place' operation
            out = out + self.res(x)

        return out




class CropAndCat(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x_prev, x_up):
        H, W = x_up.shape[-2:]
        x_prev = center_crop(x_prev, H, W)
        x_up = torch.cat([x_prev, x_up], dim = 1)

        return x_up




class AttentionGate(nn.Module):
    def __init__(self, F_prev, F_now, F_intmd, downsample_factor = 2):
        super().__init__()

        self.downsample_factor = downsample_factor

        self.x_to_intmd = nn.Sequential(
            nn.Conv2d(in_channels  = F_prev,
                      out_channels = F_intmd,
                      kernel_size  = downsample_factor,
                      stride       = downsample_factor,
                      padding      = 0,),
            nn.BatchNorm2d(F_intmd),
        )

        self.g_to_intmd = nn.Sequential(
            nn.Conv2d(in_channels  = F_now,
                      out_channels = F_intmd,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0,),
            nn.BatchNorm2d(F_intmd),
        )

        self.intmd_to_a = nn.Sequential(
            nn.Conv2d(in_channels  = F_intmd,
                      out_channels = 1,
                      kernel_size  = 1,
                      stride       = 1,
                      padding      = 0,),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()


    def forward(self, x, g):
        '''
        x = x_prev
        g = x_now
        g has a smaller dimension size than x by a factor of 2 along each edge.
        '''
        downsample_factor = self.downsample_factor

        # Center crop to ensure matching dimensions...
        Hg, Wg = g.shape[-2:]
        Hx = downsample_factor * Hg
        Wx = downsample_factor * Wg
        x = center_crop(x, Hx, Wx)

        x_intmd = self.x_to_intmd(x)    # DownConv by downsample_factor
        g_intmd = self.g_to_intmd(g)

        a_intmd = self.relu(x_intmd + g_intmd)
        a = self.intmd_to_a(a_intmd)

        # Upsample by interpolation (grid resampling)...
        a = F.interpolate(a, scale_factor  = downsample_factor,
                             mode          = 'bilinear',
                             align_corners = False)

        return a * x




class AttentionUNet(nn.Module):
    '''
    Naming convention.

    - fext: feature extraction path
    - fint: feature integration path
    - skip : skip connection
    - att  : attention gate
    '''

    def __init__(self, base_channels, in_channels, out_channels, att_gate_channels = None, depth = 5, uses_skip_connection = False):
        super().__init__()

        self.base_channels     = base_channels
        self.in_channels       = in_channels
        self.out_channels      = out_channels
        self.depth             = depth
        self.att_gate_channels = att_gate_channels
        self.uses_skip_connection = uses_skip_connection

        self.channel_fext_list, self.channel_fint_list = self.create_channel_list()

        self.module_list_double_conv_fext = self.create_module_list_double_conv(self.channel_fext_list[:-1])
        self.module_list_double_conv_fbot = self.create_module_list_double_conv(self.channel_fext_list[-2:], appends_placeholder = False)
        self.module_list_double_conv_fint = self.create_module_list_double_conv(self.channel_fint_list[1:], reverses_order = True)
        self.module_list_final_conv = self.create_module_list_final_conv()

        self.module_list_pool_fext = self.create_module_list_pool_fext()
        self.module_list_up_conv_fint = self.create_module_list_up_conv_fint(self.channel_fint_list)
        self.module_list_cat_fint = self.create_module_list_cat_fint()
        self.module_list_att_fint = self.create_module_list_att_fint()


    def create_channel_list(self):
        """
        Returns
        -------
        channel_fext_list : list
            - Level  0  : (in_channels, out_channels)
            - Level ... : (in_channels, out_channels)
            - Level -1  : (in_channels, out_channels)

        channel_fint_list : list
            - Level -1  : (in_channels, out_channels)
            - Level ... : (in_channels, out_channels)
            - Level  0  : (in_channels, out_channels)

        """
        base_channels = self.base_channels
        in_channels   = self.in_channels
        out_channels  = self.out_channels
        depth         = self.depth

        # Comupte the size for each down channel that grows exponentially with a base of 2...
        channel_list = [ base_channels * (2**i) for i in range(depth) ]
        channel_fext_list = [in_channels] + channel_list
        channel_fint_list =                 channel_list[::-1] + [out_channels]

        return channel_fext_list, channel_fint_list[::-1]


    def create_module_list_double_conv(self, channel_list, appends_placeholder = True, reverses_order = False):
        module_list = nn.ModuleList()
        for in_channels, out_channels in zip(channel_list[:-1], channel_list[1:]):
            if reverses_order: in_channels, out_channels = out_channels, in_channels
            module = DoubleConv(in_channels, out_channels, uses_skip_connection = self.uses_skip_connection)
            module_list.append(module)
        if appends_placeholder: module_list.append(nn.Identity())

        return module_list


    def create_module_list_final_conv(self):
        channel_fint_list = self.channel_fint_list

        in_channels, out_channels = reversed(channel_fint_list[:2])
        module = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 1)
        module_list = nn.ModuleList([module])

        return module_list


    def create_module_list_pool_fext(self):
        depth = self.depth

        module_list = nn.ModuleList()
        for i in range(depth)[:-1]:
            pool_module = nn.MaxPool2d(kernel_size = 2, stride = 2)
            module_list.append(pool_module)
        module_list.append(nn.Identity())

        return module_list


    def create_module_list_up_conv_fint(self, channel_list):
        up_conv_channel_list = channel_list[1:]
        module_list = nn.ModuleList([nn.Identity(),])
        for in_channels, out_channels in zip(up_conv_channel_list[:-1], up_conv_channel_list[1:]):
            module = nn.ConvTranspose2d(out_channels, in_channels, kernel_size = 2, stride = 2)
            module_list.append(module)

        return module_list


    def create_module_list_cat_fint(self):
        depth = self.depth

        module_list = nn.ModuleList()
        for i in range(depth)[:-1]:
            module = CropAndCat()
            module_list.append(module)
        module_list.append(nn.Identity())

        return module_list


    def create_module_list_att_fint(self):
        channel_fint_list = self.channel_fint_list[1:]    # Exclude the outgoing seg channels
        depth = len(channel_fint_list)

        module_list = nn.ModuleList()
        for i in range(depth)[:-1]:
            F_prev = channel_fint_list[i]
            F_now  = channel_fint_list[i+1]

            att_gate_channels = F_prev if self.att_gate_channels is None else self.att_gate_channels
            module = AttentionGate(F_prev, F_now, att_gate_channels)
            module_list.append(module)
        module_list.append(nn.Identity())

        return module_list


    def forward(self, x):
        depth = self.depth

        # ___/ FEATURE EXTRACTION PATH \___
        fmap_fext_list = []
        for i in range(depth):
            double_conv = self.module_list_double_conv_fext[i]
            pool        = self.module_list_pool_fext[i]

            fmap = double_conv(x)
            fmap_fext_list.append(fmap)

            x = pool(fmap)

        double_conv = self.module_list_double_conv_fbot[0]    # Single element though, may not be a good design.
        x_now = double_conv(x)

        # ___/ FEATURE INTEGRATION PATH \___
        depth_list = list(reversed(range(depth)))    # e.g., 4, 3, 2, 1, 0
        for depth_now, depth_prev  in zip(depth_list[:-1], depth_list[1:]):
            x_prev = fmap_fext_list[depth_prev]

            # Attention gate...
            att_gate = self.module_list_att_fint[depth_prev]
            a_prev = att_gate(x_prev, x_now)

            # Up conv...
            up_conv = self.module_list_up_conv_fint[depth_now]
            x_up = up_conv(x_now)

            # Concatenate...
            cat = self.module_list_cat_fint[depth_prev]
            x = cat(a_prev, x_up)

            # Double conv...
            double_conv = self.module_list_double_conv_fint[depth_prev]
            x_now = double_conv(x)

        final_conv = self.module_list_final_conv[0]
        x_now = final_conv(x_now)

        return x_now
