# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/13
"""
import os, sys
import yaml
from easydict import EasyDict as edict
# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录


# ------------------------ 定义超参数 --------------------- #
def configs(path):
    with open(path, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = edict(params)
    # print(params)
    return params


if __name__ == '__main__':
    path = r"D:\workspace\OCR_server\DBnet\utils\params.yaml"
    params = configs(path)
    print(params['DATASET']['img_mode'])