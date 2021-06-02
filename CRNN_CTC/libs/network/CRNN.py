# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from CRNN_CTC.Config import configs
from CRNN_CTC.libs.network import BidirectionalLSTM, mobilenetv3_large, mobilenetv3_small
from CRNN_CTC.libs.network.backbone import ResNet

#
# class CRNN(nn.Module):
#     def __init__(self, args):
#         super(CRNN, self).__init__()
#         self.params = args
#         assert self.params.MODEL.img_size.h % 16 == 0, "Image can be divisible by 16"
#         self.n_hidden = self.params.MODEL.NUM_HIDDEN
#         self.n_classes = self.params.MODEL.NUM_CLASSES
#         self.BidirectionalLSTM = BidirectionalLSTM
#         # self.cnn = mobilenetv3_small()
#         self.cnn = ResNet()
#         self.rnn = nn.Sequential(self.BidirectionalLSTM(self.cnn.out_channels, self.n_hidden, self.n_hidden),
#                                  self.BidirectionalLSTM(self.n_hidden, self.n_hidden, self.n_classes+1)
#                                  )
#
#     def forward(self, _input):
#         conv = self.cnn(_input)
#         b, c, h, w = conv.size()
#         print(conv.size())
#         assert h == 1, 'the height of conv must be 1'
#         conv = conv.squeeze(2)
#         conv = conv.permute(2, 0, 1)
#         output = F.log_softmax(self.rnn(conv), dim=2).requires_grad_()
#         return output


class CRNN(nn.Module):
    def __init__(self, args):
        super(CRNN, self).__init__()
        self.params = args
        assert self.params.MODEL.img_size.h % 16 == 0, "Image can be divisible by 16"
        self.n_hidden = self.params.MODEL.NUM_HIDDEN
        self.n_classes = self.params.MODEL.NUM_CLASSES
        self.BidirectionalLSTM = BidirectionalLSTM
        cnn = nn.Sequential()

        def ConvBnRelu(i, in_channels, out_channels, kernel_size=3, stride=1, LeakyRelu=False,
                       bn=False, padding=1, bias=True, inplace=True):

            cnn.add_module(f'conv{i}',
                           nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=bias)
                           )
            if bn:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            if LeakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=inplace))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(inplace=inplace))

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # conv-0
        ConvBnRelu(i=0, in_channels=1, out_channels=64, bn=False)
        cnn.add_module('pool0', self.pool1)

        # conv-1
        ConvBnRelu(i=1, in_channels=64, out_channels=128, bn=False)
        cnn.add_module('pool1', self.pool1)

        # conv-2,3
        ConvBnRelu(i=2, in_channels=128, out_channels=256, bn=True)
        ConvBnRelu(i=3, in_channels=256, out_channels=256, bn=False)
        cnn.add_module('pool2', self.pool2)

        # conv-4,5
        ConvBnRelu(i=4, in_channels=256, out_channels=512,  bn=True)
        ConvBnRelu(i=5, in_channels=512, out_channels=512,  bn=False)
        cnn.add_module('pool3', self.pool2)

        # conv-6
        ConvBnRelu(i=6, in_channels=512, out_channels=512, kernel_size=2, padding=0, bn=True)

        self.cnn = cnn
        self.rnn = nn.Sequential(self.BidirectionalLSTM(512, self.n_hidden, self.n_hidden),
                                 self.BidirectionalLSTM(self.n_hidden, self.n_hidden, self.n_classes+1)
                                 )

    def forward(self, _input):
        conv = self.cnn(_input)
        b, c, h, w = conv.size()
        print(conv.size())
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        output = F.log_softmax(self.rnn(conv), dim=2).requires_grad_()
        return output


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_crnn(args):
    model = CRNN(args)
    model.apply(weight_init)
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser('load parameter file')
    parse.add_argument('--cfg', default=r'D:\workspace\OCR_server\CRNN_CTC\utils\ownData_config.yaml',
                       help='load file', type=str)
    args = parse.parse_args()
    args = configs(args)
    x = torch.zeros(2, 1, 32, 460)
    model = get_crnn(args)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    out = model(x).cuda()
    print(out.shape)        # [2,512,1,116] -->[116,2,6830]
