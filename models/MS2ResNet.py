import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

from .model_utils import batch_norm_2d, batch_norm_2d1
from .neurons import mem_update


class MultiScaleGroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, dilations, bias, stride, step_mode='m'):
        super().__init__()

        assert groups == len(kernel_size) == len(padding) == len(dilations)

        self.groups = groups
        self.convs = nn.ModuleList()
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups

        for i in range(groups):
            if kernel_size[i] == 0:
                self.convs.append(
                    nn.Identity() if stride == 1 else layer.MaxPool2d(3, stride=stride, padding=1, step_mode=step_mode)
                )
            else:
                self.convs.append(
                    layer.Conv2d(group_in_channels, group_out_channels, kernel_size[i], padding=padding[i],
                                 dilation=dilations[i], bias=bias, step_mode=step_mode, stride=stride)
                )

    def forward(self, x):
        # 将输入分成几组
        x_groups = torch.chunk(x, self.groups, dim=2)

        # 对每组应用相应的卷积
        out_groups = [conv(x_group) for conv, x_group in zip(self.convs, x_groups)]

        # 合并所有组的输出
        return torch.cat(out_groups, dim=2)


# Model for MS2-ResNet
class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=.5, step_mode='m', backend='cupy',
                 attention=False):
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
            mem_update() ,
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=k_size,
                         stride=stride,
                         padding=pad,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            mem_update() ,
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


class MS2Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, e=.5, step_mode='m', backend='cupy',
                 attention=False):
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
            mem_update() ,
            layer.Conv2d(in_channels,
                         c_,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d(c_),
            mem_update() ,
            MultiScaleGroupConv2d(c_,
                                  c_,
                                  kernel_size=[1, 3, 5, 7],
                                  padding=[0, 1, 2, 3],
                                  stride=stride,
                                  bias=False,
                                  groups=4,
                                  dilations=[1, 1, 1, 1],
                                  step_mode=step_mode),
            batch_norm_2d(c_),
            mem_update() ,
            layer.Conv2d(c_,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         bias=False,
                         step_mode=step_mode),
            batch_norm_2d1(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                mem_update() ,
                layer.Conv2d(in_channels,
                             out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False,
                             step_mode=step_mode),
                batch_norm_2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential(
                mem_update() ,
                layer.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=False,
                             step_mode=step_mode),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet_origin_18(nn.Module):
    # Channel:
    def __init__(self, block, num_block, step_mode='m', backend='cupy', num_classes=1000, attention=False, base_width=64):
        super().__init__()
        k = 1
        self.nz, self.numel = {}, {}
        self.out_channels = []
        self.in_channels = 64 * k
        self.step_mode = step_mode
        self.backend = backend
        self.attention = attention
        self.base_width = base_width

        self.out_channels = [128 * k, 256 * k, 512 * k]

        self.conv1 = nn.Sequential(
            # mem_update() ,
            layer.Conv2d(32,
                         64 * k,
                         kernel_size=7,
                         padding=3,
                         bias=False,
                         stride=1,
                         step_mode=self.step_mode),
            batch_norm_2d(64 * k),
        )

        self.conv2_x = self._make_layer(block, 128 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 256 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 512 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(in_channels=self.in_channels, out_channels=out_channels, stride=stride,
                      step_mode=self.step_mode,
                      backend=self.backend, attention=self.attention))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output_list = []
        output = self.conv1(x)
        output_list.append(output)

        output = self.conv2_x(output)
        output_list.append(output)

        output = self.conv3_x(output)
        output_list.append(output)

        output = self.conv4_x(output)
        output_list.append(output)

        output = self.conv5_x(output)
        output_list.append(output)

        return output_list

    def add_hooks(self, instance):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}

        for name, module in self.named_modules():
            if isinstance(module, instance):
                self.nz[name], self.numel[name] = 0, 0
                self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel


class ResNet_origin_34(nn.Module):
    # Channel:
    def __init__(self, block1, block2, num_block, step_mode='m', backend='cupy', num_classes=1000, attention=False, base_width=64):
        super().__init__()
        k = 1
        self.nz, self.numel = {}, {}
        self.out_channels = []
        self.in_channels = 64 * k
        self.step_mode = step_mode
        self.backend = backend
        self.attention = attention
        self.base_width = base_width

        self.out_channels = [128 * k, 256 * k, 512 * k]

        self.conv1 = nn.Sequential(
            # mem_update() ,
            layer.Conv2d(3,
                         64 * k,
                         kernel_size=7,
                         padding=3,
                         bias=False,
                         stride=1,
                         step_mode=self.step_mode),
            batch_norm_2d(64 * k),
        )

        self.conv2_x = self._make_layer(block1, block2, 128 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block1, block2, 256 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block1, block2, 512 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block1, block2, 512 * k, num_block[3], 2)

    def _make_layer(self, block1, block2, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            if (i == num_blocks // 2) or (i == 0):
                layers.append(
                    block1(in_channels=self.in_channels, out_channels=out_channels, stride=stride,
                           step_mode=self.step_mode,
                           backend=self.backend, attention=self.attention))
            else:
                layers.append(
                    block2(in_channels=self.in_channels, out_channels=out_channels, stride=stride,
                           step_mode=self.step_mode,
                           backend=self.backend, attention=self.attention))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output_list = []
        output = self.conv1(x)
        output_list.append(output)

        output = self.conv2_x(output)
        output_list.append(output)

        output = self.conv3_x(output)
        output_list.append(output)

        output = self.conv4_x(output)
        output_list.append(output)

        output = self.conv5_x(output)
        output_list.append(output)

        return output_list

    def add_hooks(self, instance):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()

            return hook

        self.hooks = {}

        for name, module in self.named_modules():
            if isinstance(module, instance):
                self.nz[name], self.numel[name] = 0, 0
                self.hooks[name] = module.register_forward_hook(get_nz(name))

    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0

    def get_nz_numel(self):
        return self.nz, self.numel


def ms2resnet14(num_classes, backend, fusion=False, attention=False):
    return ResNet_origin_18(MS2Block, [1, 1, 1, 1], num_classes=num_classes, backend=backend,
                            attention=attention)


def ms2resnet26(num_classes, backend, fusion=False, attention=False):
    return ResNet_origin_18(MS2Block, [2, 2, 2, 2], num_classes=num_classes, backend=backend,
                            attention=attention)


def ms2resnet42(num_classes, backend, fusion=False, attention=False):
    return ResNet_origin_34(MS2Block, MSBlock, [3, 4, 6, 3], num_classes=num_classes, backend=backend,
                            attention=attention)

def ms2resnet112(num_classes, backend, fusion=False, attention=False):
    return ResNet_origin_34(MS2Block, MSBlock, [3, 8, 32, 8], num_classes=num_classes, backend=backend,
                            attention=attention)
