# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/26
"""
import cv2
import torch
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import sampler


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.data_dict[idx].values())[0])
    return label


class resizeNormalize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()

    def __call__(self, image):
        # img = cv2.resize(img, self.size, interpolation=self.interpolation)
        h, w, _ = image.shape
        if w / h < 280 / 32:
            image_temp = cv2.resize(image, (int(32 / h * w), 32), interpolation=cv2.INTER_LINEAR)
            h_temp, w_temp, _ = image_temp.shape
            # image_temp = cv2.cvtColor(np.asarray(image_temp), cv2.COLOR_RGB2BGR)
            image_temp = cv2.copyMakeBorder(
                image_temp, 0, 0, 0, 280 - w_temp, cv2.BORDER_CONSTANT, value=0)
            # res_image = Image.fromarray(cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB))
            res_image = cv2.resize(image_temp, self.size, Image.BILINEAR)
            res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)      # 转为灰度图
            res_image = self.toTensor(res_image)
            res_image.sub_(0.5).div_(0.5)
            return res_image
        elif w / h > 280 / 32:
            image_temp = cv2.resize(image, (280, int(280 / w * h)), self.interpolation)
            h_temp, w_temp, _ = image_temp.shape
            # image_temp = cv2.cvtColor(np.asarray(image_temp), cv2.COLOR_RGB2BGR)
            image_temp = cv2.copyMakeBorder(
                image_temp, 0, 0, 0, 32 - h_temp, cv2.BORDER_CONSTANT, value=0)
            # res_image = Image.fromarray(cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB))
            res_image = cv2.resize(image_temp, self.size, self.interpolation)
            res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)
            res_image = self.toTensor(res_image)
            res_image.sub_(0.45).div_(0.225)
            return res_image
        elif w / h == 280 / 32:
            image = cv2.resize(image, self.size, Image.BILINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, self.size, self.interpolation)
            image = self.toTensor(image)
            image.sub_(0.5).div_(0.5)
            return image


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        super(randomSequentialSampler, self).__init__()
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples
    

class alignCollate(object):
    def __init__(self, img_h=32, img_w=160, keep_ratio=False, min_ratio=1):
        self.imgH = img_h
        self.imgW = img_w
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                h, w, _ = image.shape
                ratios.append(w / float(h))
            max_ratio = max(ratios)
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


def transform():
    transforms = torch.nn.Sequential(
        T.CenterCrop(10),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    return torch.jit.script(transforms)


class ImageAug(object):
    def __call__(self, data):
        self.data = data
        seq = iaa.Sequential([iaa.SomeOf((2, 3), [iaa.Affine(translate_px={'x': -5}),
                                                  iaa.AdditiveGaussianNoise(scale=5),
                                                  iaa.Sharpen(alpha=0.2)]),
                              ])
        seq_det = seq.to_deterministic()  # 固定变换序列
        img_aug = seq_det.augment_image(self.data)
        return img_aug