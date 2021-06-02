# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/24
"""
from __future__ import division

import os, sys
import cv2
import time
import torch
import copy
import argparse
import numpy as np
from utils.predict_recong import Recognizer2
from utils.tools import draw_box_on_img
import torchvision.transforms as transforms
from DBnet.libs.network.DBNet import DBnet
from utils.tools import configs, ResizeShortSize, SegDetectorRepresenter
# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录


class Detector(object):
    def __init__(self, args):
        self.params = configs(args.cfg)
        self.resize_image = ResizeShortSize(self.params)

        self.transformer = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.model = DBnet(params=self.params)
        self.device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(args.model_path))  # 加载模型
        self.model.to(self.device)
        self.model.eval()

        self.post_processing = SegDetectorRepresenter(
            thresh=self.params.TRAIN.post_processing_thresh,
            box_thresh=self.params.TRAIN.post_processing_box_thresh,
            max_candidates=self.params.TRAIN.post_processing_max_candidates,
            unclip_ratio=self.params.TRAIN.post_processing_unclip_ratio
        )

    def forward(self, img, is_output_polygon=False):
        img_tensor, img_copy, batch = self.prepare_image(img)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(img_tensor)
            t = time.time() - start
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_processing(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t, img_copy

    def prepare_image(self, image):
        # assert os.path.exists(img_path), "file is not exists"
        if os.path.exists(image):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, dict):
            batch_img = []
            data = {}
            batch = {'shape': []}
            for k, v in image.items():
                img = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR)
                if self.params.DATASET.img_mode == 'RGB':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                resize_img = self.resize_image(data=img)
                img_tensor = self.transformer(resize_img)
                batch['shape'].append((h, w))
                batch_img.append(img)
            batch_images = np.asarray(batch_img)
            img_copies = copy.deepcopy(batch_images)

            return
        else:
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if self.params.DATASET.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        img_copy = copy.deepcopy(img)
        height, weight = img.shape[:2]
        data = {"img": img}
        batch = {'shape': [(height, weight)]}
        data = self.resize_image(data=data)
        img_tensor = self.transformer(data["img"])
        img_tensor = torch.unsqueeze(img_tensor, dim=0)  # 扩展batch的维度
        img_tensor = img_tensor.to(self.device)  # 数据拷贝
        return img_tensor, img_copy, batch


if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser(description='Model hyper_parameters')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default=f'{base_dir}/utils/params.yaml')
    parser.add_argument('--cfg1', help='recognizer parameters', type=str,
                        default=f'{base_dir}/utils/configs.yaml')
    parser.add_argument('--model_path', help='load pretrain model', type=str,
                        default=f'{base_dir}/models/detection/model_best_model.pth')
    args = parser.parse_args()
    pre = Detector(args)
    _, box_list, score_list, det_t, img = pre.forward(f'{base_dir}/migu/1.jpeg')
    print(img.shape)
    reg = Recognizer2(args)
    boxAndText, reg_t = reg.forward(img, box_list)
    print(len(box_list))
    print(boxAndText, reg.__len__())
    print(f'detection spend time: {det_t}, recognizer spend time: {reg_t}')
    draw_box_on_img(img, boxAndText)
