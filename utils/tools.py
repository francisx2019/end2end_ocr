# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/24
"""
import os, sys
import cv2
import yaml
import copy
import pyclipper
import numpy as np
import uuid
import torch
from CRNN_CTC.utils.alphabets import alphabet
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict as edict
from shapely.geometry import Polygon
# root目录设置
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录
# print(base_dir)


def configs(path):
    with open(path, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = edict(params)
    # print(params)
    return params


def configs1(args):
    with open(args.cfg1, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = edict(params)
    params.DATASET.ALPHABETS = alphabet
    params.MODEL.NUM_CLASSES = len(params.DATASET.ALPHABETS)
    # print(params)
    return params


def draw_box_on_img(img, text_polys):
    img_copy = copy.deepcopy(img)
    text_polys_copy = copy.deepcopy(text_polys)
    for i, vals in enumerate(text_polys_copy):
        box = vals['location']
        # box_reshape = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.rectangle(img_copy, (box['left'], box['top']), (box['right'], box['bottom']), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_copy = draw_string(img_copy, (box['left']-5, box['top']-5), vals['words'])
    # img_copy = cv2.resize(img_copy, (600, 800))
    cv2.imwrite(f'{base_dir}/test.jpg', img_copy)
    cv2.imshow("image_before", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


# ------------------------------- 检测 ------------------------------- #
class ResizeShortSize:
    def __init__(self, config):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = config.DATASET.AUGMENTATION.resize_short_size
        self.resize_text_polys = config.DATASET.AUGMENTATION.resize_text_polys

    def __call__(self, data):
        """
        对图片进行缩放
        """
        if isinstance(data, dict):
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
        else:
            h, w, _ = data.shape
            short_edge = min(h, w)
            if short_edge < self.short_size:
                scale = self.short_size / short_edge
                im = cv2.resize(data, dsize=None, fx=scale, fy=scale)
            return im


# ------------------------------ post process ------------------------ #
class SegDetectorRepresenter():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        pred = pred[:, 0, :, :]
        # 语义分割后图
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            # print(batch['img'][batch_index].shape)
            # height, width = batch['img'][batch_index].shape[1:]
            height, width = batch['shape'][batch_index]
            # print(batch['img'][batch_index].shape)
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


# ----------------------------------- 识别 ------------------------------- #
ids = ['甲', '伊', '丙', '顶', '戊', '及', '庚', '辛', '扔', '癸']
# ids = ['壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾']

def draw_string(img, pts, string):
    """
    img: read by cv;
    box:[x, y];
    string: what you want to draw in img;
    return: img
    """
    x, y = pts
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simhei.ttf", 32, encoding="utf-8")
    draw.text((x, y - 16), string, (255, 0, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def cut_texture(box_list=None, src=None):
    global dst
    box_list = list(box_list)
    h, w, c = src.shape
    num_box = len(box_list)
    n_img = int(num_box/10 + 1)
    background = np.ones((n_img, h, w, c), dtype=np.uint8) * 255
    imgs_name = {}
    for i, v in enumerate(background):
        n = 0
        resize_img_list = []
        while n < 10:
            if not box_list:
                break
            else:
                cut_list = box_list.pop(-1)
                x_min, y_min, x_max, y_max = cut_list[0, 0], cut_list[0, 1], cut_list[2, 0], cut_list[2, 1]
                border = 3
                if x_min-border < 0 or x_max+border > w or y_min-border < 0 or y_max+border > h:
                    cut_img = src[y_min:y_max, x_min:x_max, :]
                else:
                    cut_img = src[y_min-border:y_max+border, x_min-border:x_max+border, :]
                cut_h, cut_w = cut_img.shape[:2]
                r = 42/float(cut_h)
                if 100+int(r*cut_w) > w-1:
                    resize_image = cv2.resize(cut_img, (int(r*cut_w)-100, 42), interpolation=cv2.INTER_CUBIC)
                else:
                    resize_image = cv2.resize(cut_img, (int(r*cut_w), 42), interpolation=cv2.INTER_CUBIC)
                resize_h, resize_w = resize_image.shape[:2]
                new_cor = {'y_min': 140*(n+1),
                           'y_max': 140*(n+1)+resize_h,
                           'x_min': 100,
                           'x_max': 100+resize_w}
                background[i][new_cor['y_min']:new_cor['y_max'], new_cor['x_min']:new_cor['x_max'], :] = resize_image
                resize_img_list.append(resize_image)
                n += 1
        dst = background[i]
        num = 0
        for j, value in enumerate(resize_img_list):
            pts = (100-42, 140*(j+1)+value.shape[0]//2)
            dst = draw_string(dst, pts, ids[j])
            num += 1
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        r = uuid.uuid4()
        cv2.imwrite(f'{base_dir}/images/{r}.jpg', dst)
        imgs_name[f'{num}_{i}'] = f'{base_dir}/images/{r}.jpg'
    return imgs_name


def put_background(files):
    num = len(files)
    print(num)
    n_img = int(num / 10 + 1)
    background = np.ones((n_img, 2400, 1080, 3), dtype=np.uint8) * 255
    img_names = {}
    for i, back in enumerate(background):
        n = 0
        resize_img_list = []
        while n < 10:
            if not files:
                break
            img = files.pop(0)
            h, w, _ = img.shape
            r = 40 / float(h)
            if 100 + int(r * w) > 1080 - 1:
                resize_image = cv2.resize(img, (int(r * w) - 100, 40), interpolation=cv2.INTER_CUBIC)
            else:
                resize_image = cv2.resize(img, (int(r * w), 40), interpolation=cv2.INTER_CUBIC)
            resize_h, resize_w = resize_image.shape[:2]
            background[i][150*(n+1):150*(n+1)+resize_h, 100:100+resize_w, :] = resize_image
            resize_img_list.append(resize_image)
            n += 1
        dst = background[i]
        num = 0
        for j, v in enumerate(resize_img_list):
            pts = (100 - 40, 150 * (j + 1) + v.shape[0] // 2)
            dst = draw_string(dst, pts, ids[j])
            num += 1
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        r = uuid.uuid4()
        cv2.imwrite(f'{base_dir}/only_reg_images/{r}.jpg', dst)
        img_names[f'{num}_{i}'] = f'{base_dir}/only_reg_images/{r}.jpg'
    return img_names


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


if __name__ == '__main__':
    shape = (300, 500, 3)
    dst = np.ones(shape, dtype=np.uint8)*255
    cv2.imshow('res', dst)
    cv2.waitKey(0)
