# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/24
"""
from __future__ import division
import copy
import torch
import time
from PIL import Image
import numpy as np
from abc import ABC
import os, sys, glob
from aip import AipOcr
from utils.tools import configs1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CRNN_CTC.libs.network.CRNN import get_crnn
from CRNN_CTC.libs.dataset import alignCollate
from utils.tools import cut_texture, put_background, strLabelConverter
# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录
print(base_dir)
api_key = 'oXlLba7jc5MVGOYAMh9FIm2f'
app_id = '23633125'
secret_key = 'c9nrWrTKBwbenWLMXfo4qe8s37zV8kgV'
ids = ['甲', '伊', '丙', '顶', '戊', '及', '庚', '辛', '扔', '癸']


# --------------------------------------------- 通用图片识别 --------------------------------- #
class Recognizer(object):
    def __init__(self, args=None):
        self.params = args
        self.texts = None

    def __len__(self):
        return len(self.texts['words_result'])

    def forward(self, img_arr, box_list=None):
        box_list_copy = list(copy.deepcopy(box_list))
        imgs_name = cut_texture(box_list, img_arr)
        result = []
        for k, img_path in imgs_name.items():
            with open(img_path, 'rb') as f:
                client = AipOcr(app_id, api_key, secret_key)
                self.texts = client.basicGeneral(f.read())
            res = self.texts['words_result']
            n = 0
            for i, value in enumerate(res):
                if not box_list_copy:
                    break
                words = value['words']
                if words[0] == ids[n]:
                    if n >= 10:
                        n = 0
                    location = box_list_copy.pop(-1)
                    point1, point2 = location[0, 0]+1, location[0, 1]+1
                    point3, point4 = location[2, 0], location[2, 1]
                    result.append({'words': words, 'location': {"left":point1, "top":point2, "right":point3, "bottom":point4}})
                else:
                    continue
        return result


# --------------------------------------- 百度机器臂ocr识别 -------------------------------- #
class Recognizer1(object):
    def __init__(self, args=None):
        self.params = args
        self.texts = None

    def __len__(self):
        return len(self.texts['words_result'])

    def forward(self, file_dict):
        img_list = []
        keys = []
        for k, v in file_dict.items():
            img_list.insert(-1, v)
            keys.insert(-1, k)
        imgs_name = put_background(img_list)
        result = []
        for i, img_path in imgs_name.items():
            with open(img_path, 'rb') as f:
                client = AipOcr(app_id, api_key, secret_key)
                self.texts = client.basicGeneral(f.read())
            res = self.texts['words_result']
            for i, value in enumerate(res):
                words = value['words'][1:]
                result.append({'img_name': keys[i], 'words': words})
            if os.path.exists(img_path):
                os.remove(img_path)
        return result


# --------------------------------------------- 内部ocr识别 ------------------------------------ #
class TestSet(Dataset, ABC):
    def __init__(self, img, box_lists):
        super(TestSet, self).__init__()
        box_list = list(box_lists)
        h, w, c = img.shape
        num_box = len(box_list)
        self.data_dict = []
        for i in range(num_box-1, -1, -1):
            x_min, y_min, x_max, y_max = box_list[i][0, 0], box_list[i][0, 1], box_list[i][2, 0], box_list[i][2, 1]
            border = 2
            if x_min - border < 0 or x_max + border > w or y_min - border < 0 or y_max + border > h:
                cut_img = img[y_min:y_max, x_min:x_max, :]
            else:
                cut_img = img[y_min - border:y_max + border, x_min - border:x_max + border, :]
            self.data_dict.append({'cut_img': cut_img,
                                   'location': {'left':x_min, 'top':y_min, 'right':x_max, 'bottom':y_max}})

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        img = self.data_dict[item]['cut_img']
        # img = Image.fromarray(np.uint8(img))
        location = self.data_dict[item]['location']
        return img, location


class Recognizer2(object):
    def __init__(self, args):
        self.sum_locations = []
        self.sum_preds = []
        self.params = configs1(args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = get_crnn(self.params).cuda(self.device)
        self.convert = strLabelConverter(self.params.DATASET.ALPHABETS)
        self.checkpoint = torch.load(base_dir+self.params.TRAIN.resume_file, map_location='cpu')
        if 'state_dict' in self.checkpoint.keys():
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)

    def __len__(self):
        return len(self.sum_preds)

    def forward(self, image, box_lists):
        test_data = TestSet(image, box_lists)
        test_kwargs = {'pin_memory': False,
                       'collate_fn': alignCollate(img_h=self.params.MODEL.img_size.h,
                                                  img_w=self.params.MODEL.img_size.w)} if torch.cuda.is_available() else {}

        test_loader = DataLoader(test_data,
                                 batch_size=self.params.INFER.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 **test_kwargs)
        self.model.eval()
        start = time.time()
        for i, (img, loc) in enumerate(test_loader):
            img = img.to(self.device)
            preds = self.model(img)
            batch_size = img.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.convert.decode(preds.data, preds_size.data, raw=False)
            self.sum_preds += sim_preds
            self.sum_locations += list(loc)
        end = time.time()
        # print('result: {}\n locations: {}\n reg spend time: {}'.format(self.sum_preds,
        #                                                                self.sum_locations, end - start))
        result = []
        for word, loc in zip(self.sum_preds, self.sum_locations):
            result.append({'words': word, 'location': loc})
        return result, end-start

