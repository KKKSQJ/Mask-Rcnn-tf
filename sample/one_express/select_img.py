#!/usr/bin/env python
#conding:utf-8

import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

img_path = '/home/kingqi/proj/data/mask-rcnn/train_data/20190325_express/labels'


def select_img():
    for i in (os.listdir(img_path)):
        label_path = os.path.join(img_path, i)
        image_path = label_path.replace('labels', 'images').replace('txt', 'jpg')

        # img = cv2.imread(image_path)
        # plt.imshow(img[:,:,::-1])
        # plt.show()

        with open(label_path, 'r') as f:
            info = f.read().strip().split()

            # os.remove(label_path)
        label = info[0]
        x = []
        y = []
        for ii in range(1, 5):
            x.append(int(float((info[ii * 2 - 1]))))
            y.append(int(float((info[ii * 2]))))
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)

        if (min_x < 10 or min_y < 50 or max_x > 1750 or max_y > 1100) and label in ["0","1"]:
            print(x, y)
            print(label_path)
            print(image_path)
            img = cv2.imread(image_path)
            # info = [int(float(x)) for x in info[1:]]
            # cv2.line(img, (info[0], info[1]), (info[2], info[3]), (0, 255, 0))
            # cv2.line(img, (info[0], info[1]), (info[6], info[7]), (0, 255, 0))
            # cv2.line(img, (info[4], info[5]), (info[2], info[3]), (0, 255, 0))
            # cv2.line(img, (info[4], info[5]), (info[6], info[7]), (0, 255, 0))
            # plt.imshow(img[:, :, ::-1])
            # plt.show()
            os.remove(image_path)
            os.remove(label_path)

def one2img():
    for i in (os.listdir(img_path)):
        label_path = os.path.join(img_path, i)
        image_path = label_path.replace('labels', 'images').replace('txt', 'jpg')

        with open(label_path, 'r') as f:
            p = f.readlines()
            if len(p)>1:
                print(label_path)
                img = cv2.imread(image_path)
                plt.imshow(img[:, :, ::-1])
                plt.show()
                os.remove(label_path)
                os.remove(image_path)


if __name__ == '__main__':
    one2img()
    select_img()

