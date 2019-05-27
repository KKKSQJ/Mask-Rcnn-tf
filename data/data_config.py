#!/usr/bin/env python
#conding:utf-8

class Config:
    # seed_data_list set param
    dataset_classes_name = ['box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']
    dataset_out_path = '/home/kingqi/proj/data/mask-rcnn/test_d'
    #Background image 背景必须跟设置的图片大小相同
    bg_list = '/home/kingqi/proj/data/mask-rcnn/train_data/bg_im.list'#'/home/kingqi/proj/data/mask-rcnn/bg_im.list'
    fg_list = '/home/kingqi/proj/data/mask-rcnn/train_data/org_label.list'#'/home/kingqi/proj/data/mask-rcnn/org_label.list'
    #Image param
    Image_num = 20
    Image_width = 1100
    Image_height = 800
    Image_channels = 3
    Image_objects = 15
    Image_background = 1#只支持一张 多张还不支持

    #Image augmentation
    Image_rotate = (0, 360)#图片旋转 0~360度
    Image_affine_scale = (0.8, 1)#缩放 0.6~1倍
    Image_beta = 50
    Image_alpha = 0.25


    #mask threshod
    im_mask_ratio = 0.8 #重叠后剩余80%的被保存被留下

    #use nms flag
    use_nms_flag = False #使用重叠前和重叠后的比例来区域一些干扰加大的家伙们.
    #use mask nms
    use_mask_nms_flag = True