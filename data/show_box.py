#conding:tuf-8
import cv2
import os
import matplotlib.pyplot as plt


label_path = '/home/kingqi/proj/data/mask-rcnn/labels'
img_path = '/home/kingqi/proj/data/mask-rcnn/images'

for label in os.listdir(label_path):
    if label not in ['1553066274T0_1_.txt']:
        continue
    print(label)
    name = label.strip().split('.')
    if name[-1] =='json':
        continue
    path = img_path+os.sep+str(name[0])+".jpg"
    img = cv2.imread(path)
    # cv2.imshow('a.jpg',img)
    # cv2.waitKey()
    f = open(os.path.join(label_path, label), 'r')
    # if len(f.readlines())!=1:
    #     print(label)
    for i in f.readlines():
        info = i.strip().split()
        print(info)
        n = info[0]
        info = [int(float(x)) for x in info[1:]]
        cv2.line(img,(info[0],info[1]),(info[2],info[3]),(0,255,0))
        cv2.line(img, (info[0], info[1]), (info[6], info[7]), (0, 255, 0))
        cv2.line(img, (info[4], info[5]), (info[2], info[3]), (0, 255, 0))
        cv2.line(img, (info[4], info[5]), (info[6], info[7]), (0, 255, 0))
        print(info)
    img2 = img[:, :, ::-1]
    # plt.subplot(111)
    plt.xticks([]), plt.yticks([])  # 隐藏x和y轴
    plt.imshow(img2)
    plt.show()


