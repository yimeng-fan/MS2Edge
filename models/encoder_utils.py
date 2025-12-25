import torch
from spikingjelly.activation_based import layer
from torch import nn

from .MS2ResNet import ms2resnet14, ms2resnet26, ms2resnet42, ms2resnet112
from .model_utils import batch_norm_2d, batch_norm_2d1
from .neurons import mem_update


class SpikingPlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, step_mode='m', backend='cupy', args=None):
        super().__init__()
        self.T = args.T
        cardinality = 1
        self.plain_function = nn.Sequential(
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=1,
                         padding=padding,
                         groups=cardinality,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(out_channels),
        )

        self.out_function = nn.Sequential(
            mem_update(),
            layer.Conv2d(out_channels,
                         out_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         groups=cardinality,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T

        down = self.plain_function(x)

        return down, self.out_function(down)


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, step_mode='m', backend='cupy',
                 args=None):
        super(SkipBlock, self).__init__()
        self.T = args.T
        self.skip_function = nn.Sequential(
            mem_update(),
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )

    def forward(self, x):
        assert x.shape[0] == self.T
        return self.skip_function(x)


class MDSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=.5, step_mode='m', backend='cupy'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        c_ = int(out_channels * e)  # hidden channels
        pad = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(),
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=k_size,
                         stride=stride,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            mem_update(),
            layer.Conv2d(c_,
                         out_channels,
                         kernel_size=k_size,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential(
            mem_update(),
            layer.Conv2d(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(out_channels),
        )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


# Model for MDS-ResNet
class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=.5, step_mode='m', backend='cupy'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        c_ = int(out_channels * e)  # hidden channels
        pad = None
        if k_size == 3:
            pad = 1
        if k_size == 1:
            pad = 0
        self.residual_function = nn.Sequential(
            mem_update(),
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=k_size,
                         stride=stride,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            mem_update(),
            layer.Conv2d(c_,
                         out_channels,
                         kernel_size=k_size,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


def get_model(args):
    family, version = args.backbone.split('-')
    if family == "ms2resnet":
        if int(version) == 14:
            return ms2resnet14(num_classes=2, backend='cupy')
        elif int(version) == 26:
            return ms2resnet26(num_classes=2, backend='cupy')
        elif int(version) == 42:
            return ms2resnet42(num_classes=2, backend='cupy')
        elif int(version) == 112:
            return ms2resnet112(num_classes=2, backend='cupy')
