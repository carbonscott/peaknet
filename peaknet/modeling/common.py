import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trans import center_crop


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
