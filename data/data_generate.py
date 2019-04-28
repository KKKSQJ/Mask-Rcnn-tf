#!/usr/bin/env python
#conding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from data_config import Config as cfg
from functools import wraps
from data_utils import *


#时间函数
def func_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print("generate an example %06f seconds" %(end - start))
        return result
    return function_timer


#初始化org_image,label信息
class seed_fg:
    def __init__(self, org_list, classname_str):
        self.__list_file = org_list
        self.__get_name_list()
        self.classname_str = classname_str
        self.__init_seed_data()
        self.__classname_ids()


    def __get_name_list(self):
        """
        获取org_list（标签路径文件）中每一行标签文件内容
        label_list[]:数据标签路径列表
        image_list[]:数据图片路径列表
        :return:
        """
        self.label_list = []
        self.image_list = []
        try:
            flist = open(self.__list_file, 'r')
            lines = flist.readlines()
            for line in lines:
                self.label_list.append(line.replace('\n', ''))
                self.image_list.append(line.replace('labels', 'images').replace('txt', 'jpg').replace('\n',''))
        except:
            print("open the file error", self.__list_file)

    def __init_seed_data(self):
        """
        加载图片和标签信息路径，并编号
        :return:
        """
        #seed_data = {
        #               id:(label_path, image_path)
        #               id: (label_path, image_path)
        #            }
        self.seed_data = {}
        for i, info in enumerate(zip(self.label_list, self.image_list)):
            self.seed_data.update({i: info})

    def __get_label_info(self, path):
        """
        读取标签信息,path:标签路径
        label = {'class_id': class_id, 'shape': shape}
        :param path:标签路径
        :return:返回label字典
        """
        try:
            f = open(path, 'r')
            line = f.readline()
            line = line.split()
            line = [int(float(a)) for a in line]
            class_id = line[0]
            shape = line[1:]
            label = {'class_id': class_id, 'shape': shape}
        except:
            print("open %s error" %path)
            label = False
        return label

    def __get_image_info(self, path):
        """
        读取图片
        :param path: 图片路径
        :return: 图片
        """
        try:
            im = cv2.imread(path)

            # import matplotlib.pyplot as plt
            # plt.imshow(im)
            # plt.show()
        except:
            print("open %s error" %path)
            im = False
        return im

    def __classname_ids(self):
        """
        1.把每条标签的索引（key）保存到对应的class下面
        即每一类对应的标签索引
        temp_list = [
                    第一类[1,5,6,7,8]
                    第二类[11,256,841,344]
                    第三类[98,84,94,31]
                    第n类[]
                    ]
        2.把temp_list对应到相对应的class下面
        即每个类对应着标签索引
        str(类):[该类标签索引]
        :return:
        """
        temp_list = []
        for i, value in enumerate(self.classname_str):
            t_l = []
            temp_list.append(t_l)

        for key, value in self.seed_data.items():
            label = self.__get_label_info(value[0])
            class_id = label['class_id']
            temp_list[class_id].append(key)

        self.classname_ids = {}
        for i, str in enumerate(self.classname_str):
            self.classname_ids.update({str: temp_list[i]})

    def __len__(self):
        return len(self.label_list)

    #最终输出images, class_id, shapes
    def __interpret_example(self, ids):
        """
        从seed_data通过id获得数据信息存到examples
        把数据转换为np.array输出
        :param ids:np.ndarray[1,2,3,4,5]
        :return:np.array(images), np.array(class_ids), np.array(shapes)
        """
        examples = []
        if isinstance(ids, np.ndarray):
            for id in ids:
                info = self.seed_data[id]
                lapath, impath = info
                label = self.__get_label_info(lapath)
                img = self.__get_image_info(impath)
                example = {'label': label, 'image': img}
                examples.append(example)
        else:
            print("id must be a np.ndarray")

        images = []
        class_ids = []
        shapes = []
        for _example in examples:
            images.append(_example['image'])
            class_ids.append(_example['label']['class_id'])
            shapes.append(_example['label']['shape'])
        return np.array(images), np.array(class_ids), np.array(shapes)

    def interpret_example_all_class(self, num):
        """
        从所有样本中随机选取,不根据类名
        :param num:选取样本的个数
        :return:
        """
        ids = np.random.choice(np.arange(self.__len__()), num)
        return self.__interpret_example(ids)

    def interpret_example_class(self, str, num):
        """
        根据类名随机选择num个样本
        :param c:类别字符串
        :param num:选这个类的num的样本
        :return:
        """
        ids = np.random.choice(self.classname_ids[str], num)
        return self.__interpret_example(ids)



#初始化背景信息
class seed_bg():
    def __init__(self, list):
        self.__list_file = list
        self.__get_list()

    def __get_list(self):
        self.seed_bg = []
        try:
            flist = open(self.__list_file, 'r')
            lines = flist.readlines()
            for line in lines:
                self.seed_bg.append(line.replace('\n', ''))
        except:
            print("open %s error", self.__list_file)

    def __len__(self):
        return len(self.seed_bg)

    def interpret_background(self, n):
        assert isinstance(n, int) == True
        if n <= 1:
            n = 1
            path = np.random.choice(self.seed_bg, n)
            path = path[0]
            try:
                bg = cv2.imread(path)

                # import matplotlib.pyplot as plt
                # bg = bg[:, :, ::-1]
                # plt.imshow(bg)
                # plt.show()

            except:
                print("interpret_background  read %s error" %path)
        else:
            path = np.random.choice(self.seed_bg, n)
            bg = []
            for pth in path:
                bg.append(pth)
        return bg





class Sample(object):
    def __init__(self, im_h, im_w, classname_list):
        self.__classname_list = classname_list #注意与下面的id相互对应
        self.__im_h = im_h
        self.__im_w = im_w

    def set_object_info(self, images, class_ids, shapes, bg):
        """images:[batch, im_h, im_w, im_c]
            class_ids:[batch, class_id]
            shapes:[batch, shape] shape:[x0,y0, x1,y1, x2,y2, x3,y3]
            bg: background image
        """
        self.__images=images #获得图像列表
        self.__class_ids = class_ids #获得类标签
        self.__shapes=self.__shapes2points(shapes) #把标注信息转换为点信息
        self.__bg=bg #获得背景列表
        self.__bboxes = self.__shapes2boxs(self.__shapes) #把点信息转化为框信息
        self.__aug_objects() #数据增强(旋转+缩放)
        self.__simplify_object_info() #把objects统一移到图像左上角
        self.__transform_objects() #随机放置objects位置
        self.__generate_image() #生成图片
        self.__filter_masks() #过滤掉不符合要求的objects
        self.__ConBri_imgs() #改变图像明亮和色域

    #############################################
    #key function
    #############################################
    def __aug_objects(self):
        #图像旋转增强
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=cfg.Image_rotate,
                scale=cfg.Image_affine_scale)
        ])
        seq_det = seq.to_deterministic()
        #code key points
        keypoints_on_images = []
        for i, image in enumerate(self.__images):
            keypoints = []
            for p in self.__shapes[i]:
                keypoints.append(ia.Keypoint(x=p[0], y=p[1]))
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

        self.__images = seq_det.augment_images(self.__images)
        keypoints_on_images=seq_det.augment_keypoints(keypoints_on_images)

        n_shapes =[]
        bboxs=[]
        for shape_on_image in keypoints_on_images:
            shape = []
            for p in shape_on_image.keypoints:
                shape.append((int(p.x), int(p.y)))
            n_shapes.append(shape)
            bboxs.append(self.__shape2box(shape))
        #update shapes
        self.__shapes=n_shapes
        #update bboxs
        self.__bboxes = bboxs

    def __simplify_object_info(self):
        """
        把图片的目标区域, shape, box的左上角都移动到图像左上角,以便于统一进行平移
        :return:
        """
        images = []
        shapes = []
        bboxs = []
        for image, shape, bbox in zip(self.__images, self.__shapes, self.__bboxes):
            images.append(image[bbox[1]:bbox[3], bbox[0]:bbox[2],:])
            shape=np.array(shape)-np.array([bbox[0], bbox[1]])
            shape_list = []
            for p in shape:
                shape_list.append((p[0], p[1]))
            shapes.append(shape_list)
            bbox = bbox-np.array([bbox[0], bbox[1],bbox[0], bbox[1]])
            bboxs.append(bbox)
        self.__bboxes = bboxs
        self.__images=images
        self.__shapes=shapes

    def __transform_objects(self):
        """
        随机目标位置信息,并对位置信息处理以便于对应到一张图片上,
        随机位置对应图片的任意位置
        即清洗objects
        :return:
        """
        self.__random_location()
        new_bboxs= []
        for tl, bbox in zip(self.__tls, self.__bboxes):
            new_bboxs.append(self.__box_translation(bbox, tl))

        if cfg.use_nms_flag:
            keep_ixs = non_max_suppression_sort(
                np.array(new_bboxs), np.arange(len(new_bboxs)), cfg.im_nms, cfg.im_union_ratio)

            cls = [c for i, c in enumerate(self.__class_ids) if i in keep_ixs]
            images = [im for i, im in enumerate(self.__images) if i in keep_ixs]
            shapes = [s for i, s in enumerate(self.__shapes) if i in keep_ixs]
            tls = [t for i, t in enumerate(self.__tls) if i in keep_ixs]
            bboxes = [b for i, b in enumerate(new_bboxs) if i in keep_ixs]

            # update class_ids, images, shapes, tls. only index
            self.__class_ids = cls
            self.__images = images
            self.__shapes = shapes
            self.__tls = tls
            # update bboxs. index and location
            self.__bboxes = bboxes
        else:
            #update bboxs location
            self.__bboxes = new_bboxs

    def __generate_image(self):
        bg_im = self.__bg
        bg_im = cv2.resize(bg_im, (self.__im_h, self.__im_w))#归一化到统一大小
        new_shapes = []
        for image, shape, tl in zip(self.__images, self.__shapes, self.__tls):
            #update bg
            #这里的mask只提供索引
            mask =np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask = self.__draw_mask(mask, shape)
            mask_index = np.where(mask>0)
            bg_im[mask_index[0]+tl[1], mask_index[1]+tl[0], :] = image[mask_index[0], mask_index[1],:]

            # update shape
            n_shape = []
            for xy in shape:
                n_shape.append((xy[0]+tl[0], xy[1]+tl[1]))
            new_shapes.append(n_shape)

        #update shapes
        self.__shapes = new_shapes
        self.__sample = bg_im
        return self.__sample, self.__shapes

    def __filter_masks(self):
        masks=[]
        for shape in self.__shapes:
            mask = np.zeros((self.__im_h,self.__im_w), dtype=np.uint8)
            mask = self.__draw_mask(mask, shape)
            masks.append(mask)

        self.__masks = masks.copy() #no_overly_mask
        count=len(masks)
        # Handle occlusions
        occlusions = np.logical_not(masks[count-1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            masks[i] = masks[i] * occlusions
            occlusions = np.logical_and(
                occlusions, np.logical_not(masks[i]))

        if cfg.use_mask_nms_flag:

            keep_ixs = self.__filter_idexs_mask(masks, self.__masks)

            cls = [c for i, c in enumerate(self.__class_ids) if i in keep_ixs]
            new_masks = [m for i, m in enumerate(masks) if i in keep_ixs]
            tls = [t for i, t in enumerate(self.__tls) if i in keep_ixs]
            # update class_ids, shapes, tls. mask, bboxes only index
            self.__class_ids = cls
            self.__masks = new_masks
            self.__tls = tls

            def __from_mask_update_info(masks, keep_ixs):
                # 这里把shape和box统一了.....因为从mask上不好寻找这几个点
                bboxes = []
                shapes = []
                for mask in masks:
                    indexs = np.where(mask > 0)
                    xmin, xmax = min(indexs[1]), max(indexs[1])
                    ymin, ymax = min(indexs[0]), max(indexs[0])
                    bboxes.append(np.array([xmin, ymin, xmax, ymax]))
                    shapes.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
                return shapes, bboxes

            shapes, bboxes=__from_mask_update_info(new_masks, keep_ixs)
            self.__shapes = shapes
            self.__bboxes = bboxes
        else:
            self.__masks = masks
        return self.__masks

    def __ConBri_imgs(self):
        blank = np.zeros(self.__sample.shape, self.__sample.dtype)
        # dst = alpha * img + beta * blank
        alpha = cfg.Image_alpha + np.random.uniform()
        beta = np.random.randint(0, cfg.Image_beta)
        dst = cv2.addWeighted(self.__sample, alpha, blank, 1 - alpha, beta)
        image1 = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        random_bright = cfg.Image_alpha + np.random.uniform()
        image1[:, :, 0] = image1[:, :, 0] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
        self.__sample = image1
        return self.__sample

    def generate_ann(self):
        self.__anns=[]

        for id, shape in zip(self.__class_ids, self.__shapes):
            ann = []
            ann.append(self.__classname_list[id])
            ann.append(shape)
            ann.append(None)
            ann.append(None)
            ann.append(False)#dif
            self.__anns.append(ann)
        return self.__anns

    def generate_bboxs(self):
        return self.__bboxes

    def generate_class_ids(self):
        return self.__class_ids

    def get_sample(self):
        self.generate_ann()
        return self.__sample, self.__anns, self.__masks
    #############################################
    #tool function
    #############################################
    def __filter_idexs_mask(self, mask_after, mask_before):
        """
        过滤mask, 经过叠加处理的
        :return:
        """
        rate = []
        for mbf ,maf in zip(mask_before, mask_after):
            rate.append(np.sum(maf)/np.sum(mbf))
        rate_ay = np.array(rate)
        keep_idx = np.where(rate_ay>cfg.im_mask_ratio)
        return keep_idx[0]

    def __box_translation(self,box, tl):
        """box translation tl org point=(0,0)"""
        box[0] = box[0] + tl[0]
        box[2] = box[2] + tl[0]
        box[1] = box[1] + tl[1]
        box[3] = box[3] + tl[1]
        return box

    def __draw_mask(self, image, shape, color=1):
        points = np.array([shape]).astype(np.int)
        image = cv2.fillPoly(image, points, color)
        return image

    def __shape2box(self, shape):
        """one shape to one box"""
        shape = np.array(shape)
        x = shape[:, 0]
        y = shape[:, 1]
        bbox = np.array([min(x), min(y), max(x), max(y)])
        return bbox

    def __shapes2boxs(self, shapes):
        """a list shape to a list box"""
        self.__bboxes = []
        for shape in shapes:
            bbox = self.__shape2box(shape)
            self.__bboxes.append(bbox)

    def __random_location(self):
        tls = []
        for bbox in self.__bboxes:
            tl_x = np.random.randint(0, (self.__im_w - bbox[2] - 1))
            tl_y = np.random.randint(0, (self.__im_h - bbox[3] - 1))
            tls.append(np.array([tl_x, tl_y]))
        self.__tls = tls

    def __shapes2points(self, shapes):
        points = []
        for shape in shapes:
            shape = np.array(shape)
            x = shape[[0, 2, 4, 6]]
            y = shape[[1, 3, 5, 7]]
            shape_point = []
            for i in range(4):
                shape_point.append((x[i], y[i]))
            points.append(shape_point)
        return points
    #############################################
    #display function
    #############################################
    def show_info(self):
        import matplotlib.pyplot as plt
        for image, shape, bbox in zip(self.__images, self.__shapes, self.__bboxes):
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (208,220,0),3)
            for p in shape:
                cv2.circle(image, p, 10, (0, 255, 0), -1)
            plt.imshow(image)
            plt.show()

    def  show_sample(self):
        import matplotlib.pyplot as plt
        im = self.__sample
        masks = im
        for ann, mask, bbox, class_id in zip(self.__anns, self.__masks, self.__bboxes, self.__class_ids):
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (208, 220, 0), 3)
            for p in ann[1]:
                cv2.circle(im, p, 10, (0, 255, 0), -1)
            cv2.putText(im, self.__classname_list[class_id], ann[1][0],
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,128,128), lineType=cv2.LINE_AA )
            mask = cv2.cvtColor(mask*255,cv2.COLOR_GRAY2RGB)
            masks = np.concatenate((masks,mask), axis=1)
        plt.imshow(im)
        plt.show()
        plt.imshow(masks)
        plt.show()


class Generate(cfg):
    def __init__(self):
        self.__foreground = seed_fg(self.fg_list, self.dataset_classes_name)
        self.__background = seed_bg(self.bg_list)
        self.__sample = Sample(self.Image_height, self.Image_width, self.dataset_classes_name)
        self.__show_data_info()
        self.__set_out_path()

    #创建Image,Annotation,mask的输出路径
    def __set_out_path(self):
        self.image_path = os.path.join(self.dataset_out_path,"Images")
        self.xml_path = os.path.join(self.dataset_out_path, "Annotation")
        self.mask_path = os.path.join(self.dataset_out_path, "Mask")
        if not os.path.exists(self.dataset_out_path):
            os.mkdir(self.dataset_out_path)
        else:
            import shutil
            shutil.rmtree(self.dataset_out_path)
            os.mkdir(self.dataset_out_path)
        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
        if not os.path.exists(self.xml_path):
            os.mkdir(self.xml_path)
        if not os.path.exists(self.mask_path):
            os.mkdir(self.mask_path)

    #输出数据信息
    def __show_data_info(self):
        print('seed_data_list set save path ', self.dataset_out_path)
        print('foreground image numbers:',self.__foreground.__len__())
        print('dataset class name list:', self.__foreground.classname_str)
        print('background image numbers:', self.__background.__len__())
        print('dataset will generate %d samples'%self.Image_num)
        print('a sample max %d object'%self.Image_objects)
        print('a sample shape (h,w,c)=', (self.Image_height, self.Image_width, self.Image_channels))
        print('etc.')
        print('see details Config.py')


    @func_timer
    def __generate_example(self):
        #获得图片,类别索引,ann以及bg_image信息

        images, class_ids, shapes = self.__foreground.interpret_example_all_class(self.Image_objects)
        bg_image = self.__background.interpret_background(self.Image_background)

        #生成新的图片,输出图片,anns,masks信息
        self.__sample.set_object_info(images, class_ids, shapes, bg_image)
        sample, anns, masks = self.__sample.get_sample()
        #self.__sample.show_sample()
        return (sample, anns, masks)

    @func_timer
    def __generate_example_one_class(self, bshape):
        # 获得图片,类别索引,ann以及bg_image信息
        # for bshape in self.dataset_classes_name:
            images, class_ids, shapes = self.__foreground.interpret_example_class(bshape, self.Image_objects)
            bg_image = self.__background.interpret_background(self.Image_background)

            # 生成新的图片,输出图片,anns,masks信息
            self.__sample.set_object_info(images, class_ids, shapes, bg_image)
            sample, anns, masks = self.__sample.get_sample()
            # self.__sample.show_sample()
            return (sample, anns, masks)

    #保存生成的图片,标签,掩码信息
    def __save_generate_example(self, dataset_id):
        #image
        (sample, anns, masks) = self.__generate_example()
        #(sample, anns, masks) = self.__generate_example_one_class()
        fimage = os.path.join(self.image_path, "%06d" % dataset_id + '.jpg')
        cv2.imwrite(fimage, sample)

        #label
        fxml = os.path.join(self.xml_path, '%06d' % dataset_id + '.xml')
        writer = PascalVocWriter(self.xml_path, '%06d'%dataset_id, (self.Image_height, self.Image_width, self.Image_channels),
                                 localImgPath=fxml)
        print("this is No.%06d" % dataset_id)
        for ann in anns:
            writer.addBndBox(ann[1], ann[0], ann[4])
            writer.save(fxml)

        #mask
        n=0
        for mask in masks:
            ma_path = os.path.join(self.mask_path, "%06d" % dataset_id + '_' + str(n) + '.jpg')

            mask = cv2.cvtColor(mask*255,cv2.COLOR_GRAY2RGB)
            #masks = np.concatenate((masks,mask), axis=1)
            cv2.imwrite(ma_path, mask)
            n+=1



    def generate_dataset_one_class(self):
        for i in range(self.Image_num):
            for bshape in self.dataset_classes_name:
                images, class_ids, shapes = self.__foreground.interpret_example_class(bshape, self.Image_objects)
                bg_image = self.__background.interpret_background(self.Image_background)

                # 生成新的图片,输出图片,anns,masks信息
                self.__sample.set_object_info(images, class_ids, shapes, bg_image)
                sample, anns, masks = self.__sample.get_sample()
                # self.__sample.show_sample()

                fimage = os.path.join(self.image_path, "%s_%06d" %(bshape, i) + '.jpg')
                cv2.imwrite(fimage, sample)

                # label
                fxml = os.path.join(self.xml_path, '%s_%06d' %(bshape, i) + '.xml')
                writer = PascalVocWriter(self.xml_path, '%s_%06d' %(bshape, i),
                                     (self.Image_height, self.Image_width, self.Image_channels),
                                     localImgPath=fxml)
                print("this is No .%s_%06d" %(bshape, i))
                for ann in anns:
                    writer.addBndBox(ann[1], ann[0], ann[4])
                    writer.save(fxml)

                # mask
                n = 0
                for mask in masks:
                    ma_path = os.path.join(self.mask_path, "%s_%06d" %(bshape, i) + '_' + str(n) + '.jpg')

                    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
                    # masks = np.concatenate((masks,mask), axis=1)
                    cv2.imwrite(ma_path, mask)
                    n += 1


    #生成多张新图,并保存
    def generate_dataset(self):
        for i in range(self.Image_num):
            self.__save_generate_example(i)


    #生成一张新图,返回图片,标签,掩码
    def generate_an_example(self):
        return self.__generate_example()

    def generate_an_example_one_class(self, bshape):
        return self.__generate_example_one_class(bshape)

if __name__ == '__main__':
    gen = Generate()
    gen.generate_dataset()
    gen.generate_dataset_one_class()