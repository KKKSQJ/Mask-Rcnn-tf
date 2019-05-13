#coding:utf-8
import os
import cv2
from tqdm import tqdm

def get_list(img_fold, txt_out):
    f = open(txt_out, 'w')
    for img in tqdm(os.listdir(img_fold)):
        path = img_fold + os.sep + img
        img = cv2.imread(path)
        f.write(path + '\n')
        print(path)

    f.close()


if __name__ == '__main__':
    bg_img_fold = '/home/kingqi/proj/data/mask-rcnn/BG_img'
    bg_txt_out = 'bg_im.list'
    get_list(bg_img_fold, bg_txt_out)

    fg_img_fold = '/home/kingqi/proj/data/mask-rcnn/train_data/20190325_express/labels'
    fg_txt_out = 'org_label.list'
    get_list(fg_img_fold, fg_txt_out)
