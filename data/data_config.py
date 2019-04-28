#!/usr/bin/env python
#conding:utf-8

class Config:
    # seed_data_list set param
    dataset_classes_name = ['box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']
    dataset_out_path = '/home/kingqi/proj/data/mask-rcnn/test_d'
    #Background image 背景必须跟设置的图片大小相同
    bg_list = '/home/kingqi/proj/Mask_RCNN/gen_data/bg_im.list'
    fg_list = '/home/kingqi/proj/Mask_RCNN/gen_data/org_label.list'
    #Image param
    Image_num = 8
    Image_width = 1100
    Image_height = 1100
    Image_channels = 3
    Image_objects = 20
    Image_background = 1#只支持一张 多张还不支持

    #Image augmentation
    Image_rotate = (0, 360)#图片旋转 0~360度
    Image_affine_scale = (0.8, 1)#缩放 0.6~1倍
    Image_brightness = (0.8, 1.5)#明亮度
    Image_hue_saturation = (-10, 10)#hue 和饱和度
    Image_dropout = 0 # 0.02的像素点被设置为0
    Image_beta = 50
    Image_alpha = 0.25

    #nms threshold
    im_nms = 0.6 #值越大,被删除的就越少
    #union threshod
    im_union_ratio = 1.1#值大于1 越小删除的越少
    #mask threshod
    im_mask_ratio = 0.8 #重叠后剩余80%的被保存被留下

    #use nms flag
    use_nms_flag = False #使用重叠前和重叠后的比例来区域一些干扰加大的家伙们.
    #use mask nms
    use_mask_nms_flag = True