# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
"""
import requests
import os
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="OCR demo")
parser.add_argument('--file', type=str, help='test image file', default=r'D:\workspace\DATA\det\imgs/imagine_579.jpg')
args = parser.parse_args()

# 主函数
if __name__ == "__main__":
    url = "http://127.0.0.1:5001/"
    imageFilePath = args.file.strip()

    # --------------------------- 数据流方式 --------------------- #
    # with open(imageFilePath, 'rb') as f:
    #     image = f.read()
    # image_bs64 = str(base64.b64encode(image), encoding='utf-8')
    # file_dict = {"imgString": image_bs64}
    # result = requests.post(url, data=json.dumps(file_dict)).text
    # predict_result = result
    # print('预测结果为:%s\n' % predict_result)

    # --------------------- 路径方式 --------------------- #
    imageFileName = os.path.split(imageFilePath)[1]
    file_dict = {'imgString': (imageFileName, open(imageFilePath, 'rb'), 'image/jpg')}
    result = requests.post(url, files=file_dict)
    predict_result = result.text
    print('图片路径:%s 预测结果为:%s\n' % (imageFilePath, predict_result))

