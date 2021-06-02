# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/12
"""
import PIL
import numpy as np
import torch


# --------------------------------- data pre process--------------------------- #
class OWNCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict


# ------------------------ post process ------------------------------- #
from .tools import SegDetectorRepresenter


def get_post_processing(config):
    try:
        cls = eval(config.EVAL.post_processing.type)(**config.EVAL.post_processing.args)
        return cls
    except Exception as e:
        print(e)
        return None


# ------------------------- QuadMetric ------------------------------- #
from .tools import QuadMetric


def get_metric(config):
    try:
        if 'args' not in config.EVAL.metric:
            args = {}
        else:
            args = config.EVAL.metric.args
        if isinstance(args, dict):
            cls = eval(config.EVAL.metric.type)(**args)
        else:
            cls = eval(config.EVAL.metric.type)(args)
        return cls
    except:
        return None