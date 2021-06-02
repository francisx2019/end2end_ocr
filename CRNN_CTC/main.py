# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
import os, sys
import torch
import argparse
from torchvision import transforms as T
from CRNN_CTC.utils.tools import train, validate, create_log_folder
from CRNN_CTC.libs.tools import strLabelConverter
from CRNN_CTC.utils.tools import setup_logger, model_info, WarmupPolyLR
from CRNN_CTC.libs.network.CRNN import get_crnn
from CRNN_CTC.libs.dataset import OWNDataset
from CRNN_CTC.libs.dataset import randomSequentialSampler, alignCollate, ImageAug
from CRNN_CTC.Config import configs
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from CRNN_CTC.libs.dataset.DataLoader import data_prefetcher
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
print(base_dir)


def main(args):
    params = configs(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging = setup_logger(os.path.join(params.TRAIN.log_path, 'train.log'))
    output_dict = create_log_folder(params, phase='train')

    # 设置cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # ----------------------------------- 加载训练集和验证集 ------------------------------- #
    train_data = OWNDataset(params,
                            is_train=True,
                            )
    if not params.MODEL.random_sample:
        sampler = randomSequentialSampler(train_data, params.TRAIN.batch_size)
    else:
        sampler = None
    train_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'collate_fn': alignCollate(img_h=params.MODEL.img_size.h,
                                               img_w=params.MODEL.img_size.w)} if torch.cuda.is_available() else {}
    train_loader = DataLoader(dataset=train_data,
                              batch_size=params.TRAIN.batch_size,
                              shuffle=True,
                              drop_last=True,
                              sampler=sampler,
                              **train_kwargs)
    train_prefetcher = data_prefetcher(train_loader)
    val_kwargs = {'num_workers': 1,
                  'pin_memory': False,
                  'collate_fn': alignCollate(img_h=params.MODEL.img_size.h,
                                             img_w=params.MODEL.img_size.w)} if torch.cuda.is_available() else {}
    val_data = OWNDataset(params,
                          is_train=False,
                          )
    val_loader = DataLoader(dataset=val_data,
                            batch_size=params.VAL.batch_size,
                            shuffle=False,
                            drop_last=False,
                            **val_kwargs)
    val_prefetcher = data_prefetcher(val_loader)
    # ----------------------------------- 加载训练集和验证集 ------------------------------- #

    # load model
    model = get_crnn(params)
    model = model.cuda(device)
    model_info(model)  # 打印模型参数及结构

    # ------------------------- 预训练模型及中断模型加载 --------------------------- #
    # 1. 迁移学习
    if params.TRAIN.finetune_file != '':
        checkpoint = torch.load(params.TRAIN.finetune_file, map_location='cpu')  # map_location避免加载一个模型检查点时GPU内存激增
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if params.TRAIN.finetune_freeze:
            for p in model.cnn.parameters():
                p.requires_grad = False

    # 2. 中断，重新学习
    elif params.TRAIN.resume_file != '':
        checkpoint = torch.load(os.path.join(base_dir, params.TRAIN.resume_file), map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
    # ------------------------- 预训练模型及中断模型加载 --------------------------- #

    last_epoch = params.TRAIN.begin_epoch
    criterion = torch.nn.CTCLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=params.TRAIN.lr)
    # 设置学习策略
    if params.TRAIN.lr_scheduler_type == 'WarmupPolyLR':
        warmup_iters = params.TRAIN.lr_scheduler_warmup_epoch * len(train_data)
        lr_dict = {}
        if last_epoch > 1:
            params.TRAIN.lr_scheduler_last_epoch = (last_epoch - 1) * len(train_data)
            lr_dict['last_epoch'] = params.TRAIN.lr_scheduler_last_epoch
        lr_scheduler = WarmupPolyLR(optimizer=optimizer, max_iters=params.TRAIN.epochs * len(train_data),
                                    warmup_iters=warmup_iters, **lr_dict)
    elif params.TRAIN.lr_scheduler_type == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            params.TRAIN.lr_step,
                                                            params.TRAIN.lr_factor,
                                                            last_epoch=last_epoch-1)
    elif params.TRAIN.lr_scheduler_type == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=params.TRAIN.T_0,
                                                                            T_mult=1,
                                                                            eta_min=params.TRAIN.lr_min,
                                                                            last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       params.TRAIN.lr_step,
                                                       params.TRAIN.lr_factor,
                                                       last_epoch=last_epoch-1)

    best_acc = 0.5
    converter = strLabelConverter(params.DATASET.ALPHABETS)

    # 开始训练
    for epoch in range(last_epoch, params.TRAIN.epochs):
        # 训练
        train(params, train_prefetcher, train_data, converter, model, criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        # 验证
        acc = validate(params, val_prefetcher, val_data, converter, model, criterion, device, epoch, writer_dict, output_dict)

        if acc > best_acc:
            best_acc = acc
            print(f'best acc is: {best_acc}')
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch+1,
                        # 'optimizer': optimizer,
                        # 'lr_scheduler': lr_sc.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(output_dict['chs_dir'], 'checkpoint_{}_acc_{:.4f}.pth'.format(epoch, acc)))
            torch.save({'state_dict': model.state_dict(), 'best_acc': best_acc, 'epoch':epoch+1},
                       os.path.join(output_dict['chs_dir'], 'best_model.pth'))
    writer_dict['writer'].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRNN+CTC parameters')
    parser.add_argument('--cfg', default='{}/utils/ownData_config.yaml'.format(base_dir), type=str, help='配置文件的路径')
    args = parser.parse_args()
    main(args)
