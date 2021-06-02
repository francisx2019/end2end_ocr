# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
import os
from abc import ABC
import time
import glob
import cv2
import torch
import argparse
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CRNN_CTC.Config import configs
from CRNN_CTC.libs.tools import strLabelConverter
from CRNN_CTC.libs.network.CRNN import get_crnn
from CRNN_CTC.libs.dataset import alignCollate
base_dir = os.path.dirname(os.getcwd())
print(base_dir)


class TestSet(Dataset, ABC):
    def __init__(self, args):
        super(TestSet, self).__init__()
        self.params = args
        self.input_h = self.params.MODEL.img_size.h
        self.input_w = self.params.MODEL.img_size.w
        self.mean = np.array(self.params.DATASET.mean, dtype=np.float32)
        self.std = np.array(self.params.DATASET.std, dtype=np.float32)
        self.data_dict = []
        img_paths = glob.glob(f"{base_dir}/"+self.params.INFER.data_dir)
        for i, img in enumerate(img_paths):
            self.data_dict.append(img)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        img = cv2.imread(self.data_dict[item])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, item


class Recognizer(object):
    def __init__(self, args):
        super().__init__()
        self.params = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = get_crnn(self.params).cuda(self.device)
        self.convert = strLabelConverter(self.params.DATASET.ALPHABETS)
        if args.TRAIN.finetune_file != '':
            self.checkpoint = torch.load(args.TRAIN.finetune_file)
        elif args.TRAIN.resume_file != '':
            self.checkpoint = torch.load(args.TRAIN.resume_file)
        else:
            print("file error: no checkpoint file")

        if 'state_dict' in self.checkpoint.keys():
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)

    def forward(self):
        test_data = TestSet(self.params)
        test_kwargs = {'pin_memory': False,
                       'collate_fn': alignCollate(img_h=params.MODEL.img_size.h,
                                                  img_w=params.MODEL.img_size.w)} if torch.cuda.is_available() else {}

        test_loader = DataLoader(test_data,
                                 batch_size=self.params.INFER.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 **test_kwargs)
        self.model.eval()
        sum_preds = []
        start = time.time()
        for i, (img, _) in enumerate(test_loader):
            img = img.to(self.device)
            preds = self.model(img)
            batch_size = img.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.convert.decode(preds.data, preds_size.data, raw=False)
            if isinstance(sim_preds, list):
                sum_preds += sim_preds
            else:
                sum_preds.append(sim_preds)
        # sum_preds = [word.replace('犍', '0') for word in sum_preds]
        end = time.time()
        print('result: {}\n, spend time: {}'.format(sum_preds, end-start))

        # if box_lists:
        #     res = []
        #     for word, box in zip(sum_preds, box_lists):
        #         point1, point2 = box[0, 0]+1, box[0, 1]+1
        #         point3, point4 = box[2, 0], box[2, 1]
        #         res.append({'word': word,
        #                     'location': {"left": point1, "top": point2, "right": point3, "bottom": point4}
        #                     })
        #     return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='set path')
    parser.add_argument('--cfg', default=f'{base_dir}/CRNN_CTC/utils/ownData_config.yaml')
    parser.add_argument('--img_path', default=f'{base_dir}/CRNN_CTC/data/test_images/20436218_1024524228.jpg')
    args = parser.parse_args()
    params = configs(args)
    recog = Recognizer(params)
    recog.forward()
