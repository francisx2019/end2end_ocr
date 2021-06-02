# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/12
"""
import os
import sys
import torch
import argparse
from DBnet.utils.train_cls import Train
from DBnet.Config import configs
from torchvision import transforms
from DBnet.libs.dataset.dataLoad import OWNDataset
from DBnet.libs.dataset.utils import ImageAug, MakeBorderMap, MakeShrinkMap, ResizeShortSize, EastRandomCropData
from DBnet.libs.network.DBNet import DBnet
from DBnet.libs.network.loss_func import DBLoss
from DBnet.libs import OWNCollectFN, get_post_processing, get_metric
from torch.utils.data import DataLoader
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../")) # 设置项目根目录
sys.path.append(base_dir)


# -------------------------- 主函数 ------------------------ #
def main(params, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- load train dataset ------------- #
    train_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_data = OWNDataset(params=params,
                            mode='train',
                            img_transform=transforms.Compose([ImageAug(), EastRandomCropData(params),
                                                             MakeBorderMap(params), MakeShrinkMap(params)]),
                            transforms=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]))
    train_data_batch = DataLoader(train_data,
                                  batch_size=params.TRAIN.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  **train_kwargs)

    # ----------- load test dataset -------------- #
    own_collect_fn = OWNCollectFN()      # 自定义Batch生成类
    test_kwargs = {'num_workers': 1,
                   'pin_memory': False,
                   "collate_fn": own_collect_fn} if torch.cuda.is_available() else {}
    test_data = OWNDataset(params=params,
                           mode='test',
                           img_transform=transforms.Compose([ResizeShortSize(params)]),
                           transforms=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])]))
    test_data_batch = DataLoader(test_data,
                                 batch_size=params.EVAL.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 **test_kwargs)
    model = DBnet(params=params)
    criterion = DBLoss(params=params).to(device)

    # 后处理文件
    post_process = get_post_processing(params)
    # 模型评价指标
    metrics = get_metric(params)
    train = Train(parmas=params,
                  args=args,
                  model=model,
                  criterion=criterion,
                  train_loader=train_data_batch,
                  validate_loader=test_data_batch,
                  metric_cls=metrics,
                  post_process=post_process)
    train.forward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model hyper_parameters')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='utils/params.yaml')
    parser.add_argument('--log_path', help='log日志储存的路径', type=str, default='models/logs')
    args = parser.parse_args()
    params = configs(args.cfg)
    main(params, args)
