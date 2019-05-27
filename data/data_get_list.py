#coding:utf-8
import os
import cv2
from tqdm import tqdm

def get_bg_list(img_fold, txt_out):
    f = open(txt_out, 'w')
    for img in tqdm(os.listdir(img_fold)):
        path = img_fold + os.sep + img
        img = cv2.imread(path)
        f.write(path + '\n')
        print(path)

    f.close()

def get_one_fg_list(dir, txtout):
    f = open(txtout, 'w')
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        f.writelines(path+'\n')
        print(path)
    f.close()


def traversPath(rootDir, fileList):
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            fileList.append(os.path.join(root,file))
        for dir in dirs:
            traversPath(dir, fileList)



def get_fg_list(img_fold, txt_out):
    dirs = []
    f = open(txt_out,'w')
    for dir in os.listdir(img_fold):
        if dir.split('_')[-1] == 'express':
            dirs.append(os.path.join(img_fold, dir))
    for i in dirs:
        path = os.path.join(i, 'labels')
        for ii in tqdm(os.listdir(path)):
            f.writelines(os.path.join(path, ii)+'\n')
            print(os.path.join(path, ii))
    f.close()


def changname():
    path = input('请输入文件路径(结尾加上/)：')

    # 获取该目录下所有文件，存入列表中
    f = os.listdir(path)

    n = 0
    for i in f:
        # 设置旧文件名（就是路径+文件名）
        oldname = path + f[n]

        # 设置新文件名
        newname = path + 'a' + str(n + 1) + '.JPG'

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)
        print(oldname, '======>', newname)

        n += 1


if __name__ == '__main__':

    # changname()


    bg_img_fold = '/home/kingqi/proj/data/mask-rcnn/train_data/BG_img'
    bg_txt_out = '/home/kingqi/proj/data/mask-rcnn/train_data/bg_im.list'
    get_bg_list(bg_img_fold, bg_txt_out)

    fg_img_fold = '/home/kingqi/proj/data/mask-rcnn/train_data'
    fg_txt_out = '/home/kingqi/proj/data/mask-rcnn/train_data/org_label.list'
    get_fg_list(fg_img_fold, fg_txt_out)

    # one_fg_dir = '/home/kingqi/proj/data/mask-rcnn/train_data/20190325_express/labels'
    # fg_txt_out = '/home/kingqi/proj/data/mask-rcnn/train_data/org_label.list'
    # get_one_fg_list(one_fg_dir, fg_txt_out)
