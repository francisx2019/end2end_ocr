# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
"""
from .utils import *
from .utils import FPN, SPP, DBHead


__all__ = ['build_backbone', 'build_head', 'build_neck']


# ---------------------------------- network ------------------------------- #
# backbone
support_backbone = {"resnet18": resnet18,
                    "resnet34": resnet34,
                    "resnet50": resnet50,
                    "resnet101": resnet101,
                    "resnet152": resnet152,
                    "deformable_resnet18": deformable_resnet18,
                    "deformable_resnet50": deformable_resnet50,
                    "atros_resnet18": atros_resnet18,
                    "atros_resnet101": atros_resnet101,
                    }


def build_backbone(backbone_type, **kwargs):
    assert backbone_type in support_backbone.keys(), f'all support backbone is {support_backbone.keys()[:-1]}'
    backbone = support_backbone[backbone_type](**kwargs)
    return backbone


# neck
support_neck = {
    "FPN": FPN,
    "SPP": SPP,
}


def build_neck(neck_type, **kwargs):
    assert neck_type in support_neck.keys(), f'all support neck is {support_neck.keys()}'
    neck = support_neck[neck_type](**kwargs)
    return neck


# head
support_head = {
    "DBHead": DBHead
}


def build_head(head_type, **kwargs):
    assert head_type in support_head.keys(), f'all support head is {support_head.keys()}'
    head = support_head[head_type](**kwargs)
    return head

