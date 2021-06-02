# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/3/16
"""
import math
import torch
import torch.nn as nn
__all__ = ['ResNet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# -------------------------------- 1 backbone ------------------------------ #
def constant_init(module, constant, bias=0):            # 常量初始化
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, is_vd_mode=False, act='relu'):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self.is_vd_mode = is_vd_mode
        self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1 if is_vd_mode else stride,
                              padding=(kernel_size-1)//2,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.RELU = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.is_vd_mode:
            x = self.avg_pool(x)
        x = self.conv(x)
        out = self.bn(x)
        if self.act == 'relu':
            out = self.RELU(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, is_first=False, shortcut=True):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.convBNLayer1 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride)
        self.convBNLayer2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     is_vd_mode=not is_first and stride[0] != 1)

    def forward(self, x):
        identity = x
        out = self.convBNLayer1(x)
        out = self.convBNLayer2(out)

        if not self.shortcut:
            identity = self.short(x)

        out = torch.add(out, identity)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, is_first=False, shortcut=True):
        super(Bottleneck, self).__init__()

        self.convBNLayer1 = ConvBNLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.convBNLayer2 = ConvBNLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride)
        self.convBNLayer3 = ConvBNLayer(in_channels=out_channels, out_channels=4*out_channels, kernel_size=1, act='')

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=4*out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     is_vd_mode=not is_first and stride[0] != 1)

    def forward(self, x):
        identity = x
        out = self.convBNLayer1(x)
        out = self.convBNLayer2(out)
        out = self.convBNLayer3(out)

        if not self.shortcut:
            identity = self.short(x)

        out = torch.add(out, identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers=34, in_channels=1, **kwargs):
        super(ResNet, self).__init__()
        self.layers = layers
        support_layers = [18, 34, 50, 101, 152, 200]
        assert layers in support_layers, f"support layers are {support_layers},but input layer is {layers}"
        if self.layers == 18:
            self.depth = [2, 2, 2, 2]
        elif self.layers == 34 or layers == 50:
            self.depth = [3, 4, 6, 3]
        elif self.layers == 101:
            self.depth = [3, 4, 23, 3]
        elif self.layers == 152:
            self.depth = [3, 8, 36, 3]
        elif self.layers == 200:
            self.depth = [3, 12, 48, 3]
        self.in_channel_lists = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        self.out_channel_lists = [64, 128, 256, 512]

        # [B, N, W, H] --> [B, 64, W//2, H//2]
        self.convBNLayer1 = ConvBNLayer(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.convBNLayer2 = ConvBNLayer(in_channels=32, out_channels=64, kernel_size=3)
        self.convBNLayer3 = ConvBNLayer(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # [B, 64, W//2, H//2] --> [B, 512, W//16, H//16]
        self.block_list = []
        for block in range(len(self.depth)):
            layer_lists = self._make_layer(block)
            self.block_list += layer_lists
        self.blocked = nn.Sequential(*self.block_list)

        # [B, 64, W//2, H//2] --> [B, 512, W//32, H//32]
        self.out_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block):
        layer_lists = []
        if self.layers >= 50:
            shortcut = False
            for i in range(self.depth[block]):
                if i == 0 and block != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)
                bottleneck_block = Bottleneck(in_channels=self.in_channel_lists[block] if i == 0 else self.out_channel_lists[block]*4,
                                              out_channels=self.out_channel_lists[block],
                                              stride=stride,
                                              shortcut=shortcut,
                                              is_first=block == i == 0)
                shortcut = True
                layer_lists.append(bottleneck_block)
                self.out_channels = self.out_channel_lists[block]
        else:
            shortcut = False
            for i in range(self.depth[block]):
                if i == 0 and block != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)
                basic_block = BasicBlock(in_channels=self.in_channel_lists[block] if i == 0 else self.out_channel_lists[block],
                                         out_channels=self.out_channel_lists[block],
                                         stride=stride,
                                         shortcut=shortcut,
                                         is_first=block == i == 0)
                shortcut = True
                layer_lists.append(basic_block)
                self.out_channels = self.out_channel_lists[block]
        return layer_lists

    def forward(self, x):
        # C=64, W//2, H//2
        x = self.convBNLayer1(x)
        x = self.convBNLayer2(x)
        x = self.convBNLayer3(x)
        x = self.maxpool(x)

        # C=512, W//16, H//16
        x = self.blocked(x)

        # C=512, W//32, H//32
        out = self.out_maxpool(x)
        return out       # 返回多个尺度的特征图


class FPN_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN_block, self).__init__()
        self.Conv1 = conv1x1(in_planes=in_channels, out_planes=out_channels)
        self.Conv3 = conv3x3(in_planes=out_channels, out_planes=out_channels, stride=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv3(x)
        out = self.bn(x)
        return out


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)  # ? Why no bias


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class FPN(nn.Module):
    def __init__(self, **kwargs):
        super(FPN, self).__init__()
        self.channel_lists = [128, 256, 512]
        self.conv1 = FPN_block(in_channels=self.channel_lists[-1]+self.channel_lists[-2],
                               out_channels=self.channel_lists[-2])
        self.conv2 = FPN_block(in_channels=self.channel_lists[-3]+self.channel_lists[-2],
                               out_channels=self.channel_lists[-3])
        self.conv3 = nn.Conv2d(in_channels=self.channel_lists[-3],
                               out_channels=512,
                               kernel_size=1,
                               bias=False)
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out


if __name__ == '__main__':
    pass
