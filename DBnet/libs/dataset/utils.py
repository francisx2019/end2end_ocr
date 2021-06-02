# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
function: 数据增强与前处理
"""
import cv2
import PIL
import torch
import imgaug as ia
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from imgaug import augmenters as iaa
np.seterr(divide='ignore', invalid='ignore')


class OWNCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict


def order_points_clockwise(pts):
    """
    按顺时针对点进行排列
    :param points: 坐标点
    :return: 排序后坐标点
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ------------------------- data augmentation ---------------------- #
class ImageAug(object):
    def __call__(self, data):
        img = data['img']                   # 获取图片
        text_polys = data['text_polys']     # 文本框对应点

        # 绘制box在原图中
        # draw_box_on_img(img, text_polys)

        # imgaug进行数据增强
        seq = iaa.Sequential([iaa.SomeOf((2, 3), [iaa.Flipud(0.5),
                                                  iaa.Affine(rotate=(-10, 10)),
                                                  iaa.Resize(size=(0.5, 3))])
                              ])
        seq_det = seq.to_deterministic()    # 固定变换序列
        img_aug = seq_det.augment_image(img)
        text_polys_seq = []
        for box_points in text_polys:
            # 依次变换各个Box
            keypoints = ia.KeypointsOnImage([ia.Keypoint(point[0], point[1]) for point in box_points], shape=img.shape)
            keypoints = seq_det.augment_keypoints([keypoints])[0].keypoints
            poly = np.array([(p.x, p.y) for p in keypoints])
            text_polys_seq.append(poly)

        # 绘制box在增强后图片中
        # draw_box_on_img(img, text_polys)

        data['img'] = img_aug
        data['text_polys'] = np.array(text_polys_seq)

        return data


class EastRandomCropData(object):
    """
    从图中选择一块区域并缩放到固定尺寸,min_crop_side_ratio设置为1,只进行尺度变换
    """
    def __init__(self, config):
        self.size = config.DATASET.AUGMENTATION.random_crop_size
        self.max_tries = config.DATASET.AUGMENTATION.random_crop_max_tries
        self.min_crop_side_ratio = config.DATASET.AUGMENTATION.random_crop_min_crop_side_ratio
        self.require_original_image = config.DATASET.AUGMENTATION.random_crop_require_original_image
        self.keep_ratio = config.DATASET.AUGMENTATION.random_crop_keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['text']
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
            padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['text'] = texts_crop
        # draw_box_on_img(img, data['text_polys'])
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


class MakeBorderMap(object):
    def __init__(self, config):
        self.shrink_ratio = config.DATASET.AUGMENTATION.make_border_shrink_ratio  # 收缩因子,经验设置为0.4
        self.thresh_min = config.DATASET.AUGMENTATION.make_border_thresh_min  # 生成的概率图的最小值
        self.thresh_max = config.DATASET.AUGMENTATION.make_border_thresh_max  # 生成的概率图的最大值

    def __call__(self, data:dict) -> dict:
        """
            从scales中随机选择一个尺度，对图片和文本框进行缩放
            :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
            :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        # print(ignore_tags)
        # draw_box_on_img(im, text_polys)

        canvas = np.zeros(im.shape[:2], dtype=np.float32)
        mask = np.zeros(im.shape[:2], dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]:
                # 忽略不清楚的文本
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        # cv2.imshow("threshold_map", canvas)
        # cv2.imshow("threshold_mask", mask)
        # cv2.waitKey(0)
        data['threshold_map'] = canvas
        data['threshold_mask'] = mask

        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2  # 判断维度
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)  # 基于当前点构建多边形
        if polygon_shape.area <= 0:
            return
        # 根据面积与周长计算收缩偏移量  D=A(1-r*r)/L   见DB论文
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)

        # https://github.com/fonttools/pyclipper 参考连接
        padded_polygon = np.array(padding.Execute(distance)[0])  # 向外扩展
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)  # padded_polygon收缩之后的点坐标

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin  # 将原始文本框移动到图片的左上角
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))  # 生成坐标系
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]  # 依次处理线段
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
            # cv2.imshow("test", distance_map[i])
            # cv2.waitKey(0)
        distance_map = distance_map.min(axis=0)
        # cv2.imshow("distance_map", distance_map)
        # cv2.waitKey(0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])  # 移动处理好的文本框到图像原本的位置

        # cv2.imshow("canvas", canvas)
        # cv2.waitKey(0)

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
                    2 * np.sqrt(square_distance_1 * square_distance_2))  # 三角余弦公式  a^2=b^2+c^2-2bc*cosA  边a的对角为A
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(
            square_distance_1 * square_distance_2 * square_sin / square_distance)  # 计算三角形的面积之后再计算 距离   b*sinA就是高,乘以c就是面积
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]  # 注意三角余弦公式中的负号
        # self.extend_line(point_1, point_2, models)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2


def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrink = padding.Execute(-distance)       # 向内收缩
    if not shrink:
        shrink = np.array(shrink)
    else:
        shrink = np.array(shrink[0]).reshape(-1, 2)
    return shrink


class MakeShrinkMap(object):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, config):
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[config.DATASET.AUGMENTATION.make_shrink_shrink_type]
        self.min_text_size = config.DATASET.AUGMENTATION.make_shrink_min_text_size
        self.shrink_ratio = config.DATASET.AUGMENTATION.make_shrink_shrink_ratio     # 收缩因子

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:       # 对忽略的文本与过小的文本进行忽略
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask      # mask就是将那些不合格的文本框mask掉,比如文本框过小,无法收缩的文本等等
        # cv2.imshow("shrink_map", data['shrink_map'])
        # cv2.imshow("shrink_mask", data['shrink_mask'])
        # cv2.waitKey(0)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)        # 将文本框坐标限制在w,h中
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])           # 计算当前文本框的面积
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]          # 对点进行反序
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        return cv2.contourArea(polygon)


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
        text_polys = data['text_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_text_polys:
                # text_polys *= scale
                text_polys[:, 0] *= scale[0]
                text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data
