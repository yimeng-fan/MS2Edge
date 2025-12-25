from torch import nn
import torch

from spikingjelly.activation_based import layer

from .model_utils import batch_norm_2d, MembraneAverageDecoding
from .neurons import mem_update


class SpikingUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, scale_factor=2, step_mode='m',
                 backend='cupy', args=None):
        super().__init__()

        self.up_function = nn.Sequential(
            mem_update(),
            layer.MultiStepContainer(nn.Upsample(scale_factor=scale_factor, mode='nearest')),
        )
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                  step_mode=step_mode, bias=False)
        self.conv2 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                  step_mode=step_mode, bias=False)
        self.conv3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2,
                                  step_mode=step_mode, bias=False)
        self.conv4 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4,
                                  step_mode=step_mode, bias=False)
        self.conv5 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6,
                                  step_mode=step_mode, bias=False)

        self.conv6 = nn.Sequential(
            batch_norm_2d(out_channels * 5),
            mem_update(),
            layer.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                         step_mode=step_mode, bias=False),
            batch_norm_2d(out_channels),
        )

    def forward(self, x):
        x = self.up_function(x)
        # x = F.interpolate(x[0], size=size[-2:], mode='nearest').unsqueeze(0)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        xa = torch.cat((x1, x2, x3, x4, x5), 2)

        xa = self.conv6(xa)

        return xa


class SpikingPredUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, scale_factor=2, step_mode='m',
                 backend='cupy', args=None):
        super().__init__()
        self.T = args.T
        cardinality = 1

        self.up_function1 = nn.Sequential(
            mem_update(),
            layer.Conv2d(in_channels,
                         in_channels // 2,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=cardinality,
                         bias=False,
                         step_mode=step_mode),
            layer.MultiStepContainer(nn.Upsample(scale_factor=scale_factor, mode='nearest')),
            MembraneAverageDecoding(T=4),
        )

        self.up_function2 = nn.Sequential(
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      groups=cardinality,
                      bias=False),
        )

    def forward(self, x):
        assert x.shape[0] == self.T

        x = self.up_function1(x)
        # x = F.interpolate(x, size=size[-2:], mode='nearest')
        x = self.up_function2(x)

        return x

