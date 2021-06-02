# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
from __future__ import print_function, absolute_import
import os
import sys
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_dir)


class OWNDataset(Dataset):
    def __init__(self, args, is_train=True, transforms=None, img_transform=None):
        super(OWNDataset, self).__init__()
        self.params = args
        self.root = r"D:\workspace\DATA\recognizer\appSub\image"
        self.is_train = is_train
        self.input_h = self.params.MODEL.img_size.h
        self.input_w = self.params.MODEL.img_size.w
        self.transforms = transforms
        self.img_transform = img_transform

        self.mean = np.array(self.params.DATASET.mean, dtype=np.float32)
        self.std = np.array(self.params.DATASET.std, dtype=np.float32)

        # 加载字典、训练集、验证集
        self.data_dict = []
        data_file = self.params.DATASET.train if is_train else self.params.DATASET.val

        # 1.1 加载开源数据集
        if self.params.DATASET.dataname == '360':
            character_file = self.params.DATASET.character_file
            with open(character_file, 'rb') as f:
                char_dict = {n: char.strip().decode('gbk', 'ignore') for n, char in enumerate(f.readlines())}
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    img_name = line.strip().split(' ')[0]
                    indices = line.strip().split(' ')[1:]
                    strings = ''.join([char_dict[int(idx)] for idx in indices])
                    self.data_dict.append({img_name: strings})

        # 1.2 加载自己的数据集
        elif self.params.DATASET.dataname == 'own':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.data_dict.append({line.strip().split('\t')[0]: line.strip().split('\t')[-1]})
        print('load {} images!'.format(self.__len__()))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        img_name = list(self.data_dict[item].keys())[0]
        image = cv2.imread(os.path.join(self.root, img_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.img_transform:
            image = self.img_transform(image)
        if self.transforms:
            image = self.transforms(image)
        return image, item


class data_prefetcher():
    def __init__(self, loader):
        self.len = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
            self.next_target = torch.from_numpy(np.array(self.next_target))
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __len__(self):
        return self.len


if __name__ == '__main__':
    print(base_dir)
