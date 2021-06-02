# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50',
           'resnet152', 'atros_resnet18', 'atros_resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# -------------------------------- network tools ------------------------------ #
# -------------------------------- 1 backbone
def constant_init(module, constant, bias=0):            # 常量初始化
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)  # ? Why no bias


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, bias=False, padding=1)
        self.bn1 = norm_layer(planes)
        self.activate1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.activate2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, bias=False, padding=1)
        self.bn3 = norm_layer(planes)
        self.activate3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate3(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.activate1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.activate2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=4*planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(4*planes)
        self.activate3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate3(out)
        return out


# --------------------------------- 1.1 resnet ----------------------------------- #
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, dcn=None):
        self.dcn = dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.out_channels = []
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def get_channels(self):
        return self.out_channels

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dcn=dcn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5       # 返回多个尺度的特征图


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        # print('load from imagenet')
        print("Load pretrain model from imagenet.")
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        # print('load from imagenet')
        print("Load pretrain model from imagenet.")
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
# --------------------------------- 1.1 resnet ------------------------------------- #


# --------------------------------- 1.2 atros_resnet ------------------------------- #
class Atros_Resnet(nn.Module):
    def __init__(self, block, layers, output_stride=16, BatchNormal=None):
        super(Atros_Resnet, self).__init__()
        self.inplanes = 64
        if output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
        elif output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 4]
        elif output_stride == 4:
            stride = [1, 1, 1, 1]
            dilation = [1, 2, 4, 8]
        else:
            raise NotImplementedError
        self.out_channels = []
        if BatchNormal is None:
            BatchNormal = nn.BatchNorm2d
        blocks = [1, 2, 4]
        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNormal(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0], dilation=dilation[0], BatchNorm=BatchNormal)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1], dilation=dilation[1], BatchNorm=BatchNormal)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2], dilation=dilation[2], BatchNorm=BatchNormal)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3], dilation=dilation[3], BatchNorm=BatchNormal)

        self._init_weight()

        # if pretrained:
        #     self._load_pretrained_model()
    def get_channels(self):
        return self.out_channels

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                BatchNorm(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=BatchNorm))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("start convolution: ", x.size())

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5  # 返回多个尺度的特征图

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def atros_resnet101(output_stride=4, BatchNorm=nn.BatchNorm2d, pretrained=True, **kwargs):
    model = Atros_Resnet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm)
    if pretrained is True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def atros_resnet18(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True, **kwargs):
    model = Atros_Resnet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm)
    if pretrained is True:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model
# --------------------------------- 1.2 atros_resnet ------------------------------- #


# ---------------------------------- 2 neck
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', inplace=True):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4

        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)

        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out
        self._init_weight()

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


class SPP(nn.Module):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out
        self._init_weight()

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        # p4 = self.reduce_conv_c4(c4)
        # p3 = self.reduce_conv_c3(c3)
        # p2 = self.reduce_conv_c2(c2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _add(self, x, y):
        if x.size()[-1] != y.size()[-1]:
            return F.interpolate(x, size=y.size()[2:]) + y
        else:
            return x + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        if p3.size()[2:3] != h:
            p3 = F.interpolate(p3, size=(h, w))
        if p4.size()[2:3] != h:
            p4 = F.interpolate(p4, size=(h, w))
        if p5.size()[2:3] != h:
            p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


# ---------------------------------- 3 head
class DBHead(nn.Module):
    def __init__(self, in_channels, smooth=False, k=50):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),      # 使用两次转置卷积来对特征图进行上采样
                                      nn.BatchNorm2d(in_channels // 4),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
                                      nn.Sigmoid()
                                      )
        self.thresh = self._init_thresh(in_channels, smooth=smooth)
        self._init_weight()

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)

        if self.training:
            # 1.shrink_maps:代表像素点是文本的概率, 2.threshold_maps:每个像素点的阈值, 3.binary_maps:由1,2计算得到，计算公式为DB公式
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_thresh(self, inner_channels, smooth=False, bias=False):
        in_channels = inner_channels
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
                                    nn.BatchNorm2d(inner_channels // 4),
                                    nn.ReLU(inplace=True),
                                    self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
                                    nn.BatchNorm2d(inner_channels // 4),
                                    nn.ReLU(inplace=True),
                                    self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
                                    nn.Sigmoid()
                                    )
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = {'upsample': nn.Upsample(scale_factor=2, mode='nearest'),
                           'conv1': nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)}
            if out_channels == 1:
                module_list['conv2'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True)
            return nn.Sequential(OrderedDict(module_list))
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


# ------------------ 设置学习策略 ---------------------- #
class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer,
                 target_lr=0,
                 max_iters=0,
                 power=0.9,
                 warmup_factor=1./3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        # super(WarmupPolyLR, self).__init__()
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # print label_trues.dtype, label_preds.dtype
        for lt, lp in zip(label_trues, label_preds):
            try:
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            except:
                pass

    def get_scores(self):
        """Returns accuracy score evaluation models.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu, }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# ------------------------ 计算文本框的准确率 ---------------------- #
def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


if __name__ == '__main__':
    model = atros_resnet18(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=4)
    input = torch.rand(1, 3, 416, 416)
    x2, x3, x4, x5 = model(input)
    print(x2.size())
    print(x3.size())
    print(x4.size())
    print(x5.size())
    print(model.get_channels())