# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
import yaml
from easydict import EasyDict as edict
from CRNN_CTC.utils.alphabets import alphabet


def configs(args):
    with open(args.cfg, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = edict(params)
    params.DATASET.ALPHABETS = alphabet
    params.MODEL.NUM_CLASSES = len(params.DATASET.ALPHABETS)
    # print(params)
    return params
