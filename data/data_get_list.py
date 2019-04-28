#coding:utf-8
import os
import cv2

def get_list(img_fold, txt_out):
    f = open(txt_out, 'w')
    for img in os.listdir(img_fold):
        path = img_fold + os.sep + img
        img = cv2.imread(path)
        if img.shape:
            f.write(path + '\n')
            print(img)
        else:
            print("No image in %s " % img_fold)
    f.close()


if __name__ == '__main__':
    img_fold = '/home/kingqi/proj/data/mask-rcnn/BG_img'
    txt_out = '/home/kingqi/proj/Mask_RCNN/gen_data/bg_im.list'
    get_list(img_fold, txt_out)
