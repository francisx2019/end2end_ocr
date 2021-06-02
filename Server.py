"""
Author:xufei
Date:2021/1/8
"""
import os
import cv2
import copy
import uuid
import json
import time
import base64
import argparse
from utils.predict_det import Detector
from utils.tools import draw_string
from utils.predict_recong import Recognizer2
from flask import Flask, request
app = Flask('OCR_Server1')
base_dir = os.path.dirname(os.path.abspath(__file__))
import zipfile

# 设置参数
parser = argparse.ArgumentParser(description='Model hyper_parameters')
parser.add_argument('--cfg', help='load detection parameters file ', type=str,
                    default='utils/params.yaml')
parser.add_argument('--cfg1', help='load recognizer parameters file ', type=str,
                    default='utils/configs.yaml')
parser.add_argument('--model_path', help='load pretrain model path', type=str,
                    default='models/detection/model_best_model.pth')
args = parser.parse_args()


# ----------------------------- 服务返回 ------------------------------- #
# 定义回调函数，接收来自/的post请求，并返回预测结果
@app.route("/", methods=['POST'])
def return_result():
    # ------------------------ 文件路径方式 ----------------------- #
    received_file = request.files['imgString']
    imageFileName = received_file.filename
    # if request.method == 'POST':
    if received_file:
        received_dirPath = './'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        predict = Detector(args)
        # 预测结果， 后处理后框， 框的得分， 预测时间， 预测图片
        _, box_list, score_list, t, img = predict.forward(imageFilePath)
        reg = Recognizer2(args)
        result = reg.forward(img, box_list)
        print('完成对接收图片的识别，总共耗时%.2f秒' % t)
        print("test_res:{}, sum:{}".format(result, len(result)))
        return str(result)
    # --------------------------- 文件路径方式 ------------------------- #

    # ------------------------- 数据流方式 ---------------------------- #
    # # file_dict = parse_zip(request.data)     # zip包文件
    # # 单个图片数据流
    # res = json.loads(request.data)
    # img = base64.b64decode(bytes(res['imgString'], encoding='utf-8'))
    # if img:
    #     predict = Detector(args)
    #     # 预测结果， 后处理后框， 框的得分， 预测时间， 预测图片
    #     _, box_list, score_list, t, img = predict.forward(img)
    #     reg = Recognizer()
    #     result = reg.forward(img, box_list)
    #     result = str(result)
    #     print('完成对接收图片的识别，总共耗时%.2f秒' % t)
    #     print("testtest" ,result)
    #     return result
    # else:
    #     return 'failed'
    # ------------------------- 数据流方式 ---------------------------- #


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run('0.0.0.0', port=5001, debug=False, threaded=True, processes=1)
