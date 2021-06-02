# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/28
"""
import os, sys, cv2
from CRNN_CTC.utils.alphabets import alphabet
from sklearn.model_selection import train_test_split
base_dir = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(base_dir)
print(base_dir)


def split_data(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        images = []
        labels = []
        for line in lines:
            # print(line.strip().split('\t'))
            img = line.strip().split('\t')[0]
            label = line.strip().split('\t')[1]
            images.append(img)
            labels.append(label)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, shuffle=True, random_state=2021)
    print(f'x_train:{len(x_train)}, x_test:{len(x_test)}, y_train:{len(y_train)}, y_test:{len(y_test)}')
    with open(os.path.join(base_dir, 'data/train.txt'), 'a', encoding='utf-8') as f:
        for i in range(len(x_train)):
            f.write(x_train[i]+'\t')
            for j in range(len(y_train[i])):
                f.write(str(y_train[i][j]))
            f.write('\n')
    with open(os.path.join(base_dir, 'data/test.txt'), 'a', encoding='utf-8') as f:
        for i in range(len(x_test)):
            f.write(x_test[i]+'\t')
            for j in range(len(y_test[i])):
                f.write(str(y_test[i][j]))
            f.write('\n')


def rm_ill(txt_path, data_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    del_flag = False
    for line in lines:
        img = line.split('\t')[0]
        img_arr = cv2.imread(os.path.join(data_path, 'images', img))
        print(os.path.join(data_path, 'images', img))
        h, w, _ = img_arr.shape
        label = line.split('\t')[1]
        for i in label:
            if i not in alphabet:
                print(i)
                del_flag = True
                break
            else:
                del_flag = False
        if del_flag:
            # print(os.path.join(data_path, 'images', img))
            os.remove(os.path.join(data_path, 'images', img))
            os.remove(os.path.join(data_path, 'labels', img.split('.')[0] + '.txt'))


if __name__ == '__main__':
    path = r"img_text.txt"
    data_path = r"D:\workspace\DATA\recognizer\04_13"
    split_data(path)
    # rm_ill(path, data_path)