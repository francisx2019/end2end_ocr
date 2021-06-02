# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
"""
import os, sys
from DBnet.Config import configs
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from DBnet.libs.network import build_backbone, build_neck, build_head
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../"))
sys.path.append(base_dir)       # 设置项目根目录


class DBnet(nn.Module):
    def __init__(self, params):
        super(DBnet, self).__init__()
        self.backbone = build_backbone(backbone_type=params.NETWORK.backbone_type,
                                       pretrained=params.NETWORK.backbone_pretrained,
                                       in_channels=params.NETWORK.backbone_in_channels)
        backbone_channel_out = self.backbone.get_channels()             # resnet网络输出通道列表
        self.neck = build_neck(neck_type=params.NETWORK.neck_type, in_channels=backbone_channel_out,
                               inner_channels=params.NETWORK.neck_inner_channels)
        neck_channel_out = self.neck.out_channels
        self.head = build_head(head_type=params.NETWORK.head_type, in_channels=neck_channel_out,
                               smooth=params.NETWORK.head_smooth, k=params.NETWORK.head_k)
        self.name = f'{params.NETWORK.backbone_type}_{params.NETWORK.neck_type}_{params.NETWORK.head_type}'

    def forward(self, x):
        image_height, image_width = x.size()[2:]
        model_out = self.head(self.neck(self.backbone(x)))
        model_out = F.interpolate(model_out, size=(image_height, image_width), mode='bilinear', align_corners=True)
        return model_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model hyper_parameters')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default=os.path.join(base_dir, 'utils/params.yaml'))
    args = parser.parse_args()
    params = configs(args.cfg)
    x = torch.zeros(2, 3, 1024, 1024).to('cuda')
    model = DBnet(params=params).to('cuda')
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    out = model(x)
    # print(out.shape)
    # print(out[:, 0, :, :])
    # print(out[:, 1, :, :])
    # print(out[:, 2, :, :])
