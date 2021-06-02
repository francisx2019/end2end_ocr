# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/19
"""
import io
import os
import sys
import cv2
import torch
import copy
import time
import argparse
import numpy as np
from PIL import Image
from DBnet.Config import configs
# from DBnet.libs.dataset.utils import *
from DBnet.libs.network.DBNet import DBnet
from DBnet.libs.tools import SegDetectorRepresenter
from DBnet.libs.tools import draw_box_on_img
import torchvision.transforms as transforms
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
print(base_dir)
sys.path.append(base_dir)       # 设置项目根目录


class ResizeShortSize:
    def __init__(self, config):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = config.DATASET.AUGMENTATION.resize_short_size
        self.resize_text_polys = config.DATASET.AUGMENTATION.resize_text_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
        data['img'] = im
        return data


class Predict(object):
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
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t, img_copy

    def prepare_image(self, image):
        # assert os.path.exists(img_path), "file is not exists"
        img = cv2.imread(image, cv2.IMREAD_COLOR)
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
    parser = argparse.ArgumentParser(description='Model hyper_parameters')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='utils/params.yaml')
    parser.add_argument('--model_path', help='load pretrain model', type=str,
                        default=f'{base_dir}/DBnet/result/DBNet_atros_resnet18_SPP_DBHead/checkpoints/model_latest_model.pth')
    args = parser.parse_args()
    predict = Predict(args)
    _, box_list, score_list, t, img = predict.forward(r'D:\workspace\OCR_server\web900_.jpg')
    print(box_list)
    print(score_list)
    print(t)
    draw_box_on_img(img, text_polys=box_list)
