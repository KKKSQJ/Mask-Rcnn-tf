#!/usr/bin/env python
#coding:utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import imgaug

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from data.data_generate import Generate
#from auto_gener_dataset.generate_samples import Generate


MODEL_DIR = os.path.join(ROOT_DIR, "logs-50")
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ExpressConfig(Config):
    NAME = "express"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6
    IMAGE_MIN_DIM = 1024 #256
    IMAGE_MAX_DIM = 1024
    LEARNING_RATE = 0.0001
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detecti
    DETECTION_NMS_THRESHOLD = 0.2
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)



config = ExpressConfig()
config.display()


dataset_classes_name = ['box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']
train_class_name = ['box_0', 'box_1', 'soft_1', 'envelope_0', 'envelope_1']

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ExpressGenerate(Generate):
    def generate__sample(self,n):
        return self.generate_an_example(n)

    def generate__sample_one_class(self, bshape,n):
        return self.generate_an_example_one_class(bshape,n)

    def get_image_size(self):
        return self.Image_height, self.Image_width


class ExpressDataset(utils.Dataset):
    # str='train' or 'infere'
    def set_dataset_class(self, str):
        self.__dataset_flag = str

    # True or False
    def set_dataset_exit_flag(self, flag):
        self.__dataset_exit = flag


    def out_path(self, path):
        self.data_directory = path
        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)


    def set_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)




    def load_shapes(self, count, num, n1=0 , index=0, pn=4):

        # Add classes
        self.add_class("express", 1, "box_0")
        self.add_class("express", 2, "box_1")
        self.add_class("express", 3, "soft_0")
        self.add_class("express", 4, "soft_1")
        self.add_class("express", 5, "envelope_0")
        self.add_class("express", 6, "envelope_1")
        generator = ExpressGenerate()
        self.__image_height, self.__image_width = generator.get_image_size()


        #data_directory = '/home/kingqi/proj/data/mask-rcnn/my_data'
        path_img = os.path.join(self.data_directory, "images")
        masks_path = os.path.join(self.data_directory, "masks")
        class_ids_path = os.path.join(self.data_directory, "class_ids")
        if not self.__dataset_exit:
            self.set_dir(path_img)
            self.set_dir(masks_path)
            self.set_dir(class_ids_path)


        if self.__dataset_exit:
            for i in tqdm(range(count+num*len(train_class_name))):
                image_path = os.path.join(self.data_directory, "images", '%06d%s.jpg' % (i, self.__dataset_flag))
                masks_path = os.path.join(self.data_directory, "masks", 'masks_%06d%s.npy' % (i, self.__dataset_flag))
                class_ids_path = os.path.join(self.data_directory, "class_ids", 'class_ids_%06d%s.npy' % (i, self.__dataset_flag))

                self.add_image("express", image_id=i,path=None, class_ids_path=class_ids_path,
                               image_path=image_path, info_path=masks_path)


        else:
            def gen_data(index=0):
                n = int(index)
                #print(n)
                cc = int(count/pn)
                for i in (range(cc)):
                    image_path = os.path.join(self.data_directory, "images", '%06d%s.jpg' % (n, self.__dataset_flag))
                    masks_path = os.path.join(self.data_directory, "masks",
                                              'masks_%06d%s.npy' % (n, self.__dataset_flag))
                    class_ids_path = os.path.join(self.data_directory, "class_ids",
                                                  'class_ids_%06d%s.npy' % (n, self.__dataset_flag))

                    # sample, masks, class_ids = generator.generate__sample()
                    sample, class_ids, masks = generator.generate__sample(n1+n)
                    self.add_image("express", image_id=n, path=None, class_ids_path=class_ids_path,
                                   image_path=image_path, info_path=masks_path)

                    masks_index = []
                    for mask in masks:
                        masks_index.append(np.where(mask == 1))

                    cv2.imwrite(image_path, sample)
                    np.save(masks_path, masks_index)
                    np.save(class_ids_path, class_ids)
                    n += 1

                # n = count
                cc1 = int(num/pn)
                for i in (range(cc1)):
                    for bshape in (train_class_name):
                        image_path = os.path.join(self.data_directory, "images",
                                                  '%06d%s.jpg' % (n, self.__dataset_flag))
                        masks_path = os.path.join(self.data_directory, "masks",
                                                  'masks_%06d%s.npy' % (n, self.__dataset_flag))
                        class_ids_path = os.path.join(self.data_directory, "class_ids",
                                                      'class_ids_%06d%s.npy' % (n, self.__dataset_flag))

                        # sample, masks, class_ids = generator.generate__sample()
                        sample, class_ids, masks = generator.generate__sample_one_class(bshape,n1+n)
                        self.add_image("express", image_id=n, path=None, class_ids_path=class_ids_path,
                                       image_path=image_path, info_path=masks_path)

                        masks_index = []
                        for mask in masks:
                            masks_index.append(np.where(mask == 1))

                        cv2.imwrite(image_path, sample)
                        np.save(masks_path, masks_index)
                        np.save(class_ids_path, class_ids)
                        n += 1
            gen_data(index)



            ## n = int(index)
            ## print(n)
            # for i in tqdm(range(count)):
            #     image_path = os.path.join(self.data_directory,  "images", '%06d%s.jpg' % (i, self.__dataset_flag))
            #     masks_path = os.path.join(self.data_directory, "masks", 'masks_%06d%s.npy' % (i, self.__dataset_flag))
            #     class_ids_path = os.path.join(self.data_directory, "class_ids", 'class_ids_%06d%s.npy' % (i, self.__dataset_flag))
            #
            #     # sample, masks, class_ids = generator.generate__sample()
            #     sample, class_ids, masks = generator.generate__sample()
            #     self.add_image("express", image_id=n, path=None, class_ids_path=class_ids_path,
            #                    image_path=image_path, info_path=masks_path)
            #
            #     masks_index = []
            #     for mask in masks:
            #         masks_index.append(np.where(mask == 1))
            #
            #     cv2.imwrite(image_path, sample)
            #     np.save(masks_path, masks_index)
            #     np.save(class_ids_path, class_ids)
            #     n += 1
            #
            # # n = count
            # for i in tqdm(range(num)):
            #     for bshape in (train_class_name):
            #         image_path = os.path.join(self.data_directory, "images", '%06d%s.jpg' % (n, self.__dataset_flag))
            #         masks_path = os.path.join(self.data_directory, "masks",
            #                                   'masks_%06d%s.npy' % (n, self.__dataset_flag))
            #         class_ids_path = os.path.join(self.data_directory, "class_ids",
            #                                       'class_ids_%06d%s.npy' % (n, self.__dataset_flag))
            #
            #         # sample, masks, class_ids = generator.generate__sample()
            #         sample, class_ids, masks = generator.generate__sample_one_class(bshape)
            #         self.add_image("express", image_id=n, path=None, class_ids_path=class_ids_path,
            #                        image_path=image_path, info_path=masks_path)
            #
            #         masks_index = []
            #         for mask in masks:
            #             masks_index.append(np.where(mask == 1))
            #
            #         cv2.imwrite(image_path, sample)
            #         np.save(masks_path, masks_index)
            #         np.save(class_ids_path, class_ids)
            #         n += 1



    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_path = info['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # plt.imshow(image)
        # plt.show()
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['info_path']
        masks_list = np.load(mask_path,mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        class_ids_path = info['class_ids_path']
        class_ids = np.load(class_ids_path,mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
        mask_h, mask_w = self.__image_height, self.__image_width
        count = len(class_ids)
        masks = np.zeros((mask_h, mask_w, count), dtype=np.uint8)
        _class_ids = []
        for i in class_ids:
            _id = i[0]
            _class_ids.append(_id)
        for i, mask in enumerate(masks_list):
            masks[mask[0], mask[1], i] = 1

            # plt.imshow(masks[:,:,i])
            # plt.show()
        _class_ids = np.array([dataset_classes_name.index(s) for s in _class_ids], dtype=np.uint32)
        _class_ids = _class_ids + 1
        return masks.astype(np.bool), _class_ids.astype(np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "express":
            return info["express"]
        else:
            super(self.__class__).image_reference(self, image_id)

def set_thread(num1, num2, do):
    import multiprocessing
    pn = 4
    index = []
    for i in range(pn):
        index.append(int(i/pn*(len(train_class_name)*num2+num1)))
    pool = multiprocessing.Pool(processes=pn)
    for i in range(pn-1):
        pool.apply_async(do,(index[i], index[i+1]))
    pool.apply_async(do, (index[pn-1], num1+len(train_class_name)*num2))
    pool.close()
    pool.join()
    print("finished")


def train(out_path, num1_1, num1_2, num2_1, num2_2, show = True):

    dataset_train = ExpressDataset()
    dataset_train.out_path(out_path)
    dataset_train.set_dataset_class('train')
    dataset_train.set_dataset_exit_flag(True)
    dataset_train.load_shapes(num1_1, num1_2)
    dataset_train.prepare()

    dataset_val = ExpressDataset()
    dataset_val.out_path(out_path)
    dataset_val.set_dataset_class('val')
    dataset_val.set_dataset_exit_flag(True)
    dataset_val.load_shapes(num2_1, num2_2)
    dataset_val.prepare()


    if show:
        image_ids = np.random.choice(dataset_train.image_ids, num1_1+num1_2*len(train_class_name))
        for image_id in image_ids:
            t0 = time.time()
            image = dataset_train.load_image(image_id)
            mask, class_ids = dataset_train.load_mask(image_id)
            t1 = time.time()
            print("dataset cost time %06f " % (t1 - t0))
            bbox = utils.extract_bboxes(mask)
            # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
            visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)



    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)


    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(init_with, by_name=True)

    #augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='heads',
                    #augmentation=augmentation
                )

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='4+',
                    #augmentation=augmentation
                )

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=70,
                    layers='all',
                    #augmentation=augmentation
                )



def detection(out_path, num2_1, num2_2):
    dataset_val = ExpressDataset()
    dataset_val.out_path(out_path)
    dataset_val.set_dataset_class('val')
    dataset_val.set_dataset_exit_flag(True)
    dataset_val.load_shapes(num2_1, num2_2,0)
    dataset_val.prepare()

    image_ids = np.random.choice(dataset_val.image_ids, 22)
    for image_id in image_ids:
        t0 = time.time()
        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)
        t1 = time.time()
        print("dataset cost time %06f " % (t1 - t0))
        bbox = utils.extract_bboxes(mask)
        # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
        visualize.display_instances(image, bbox, mask, class_ids, dataset_val.class_names)

    class InferenceConfig(ExpressConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # Test on a random image


    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                       image_id, use_mini_mask=False)

    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_val.class_names, figsize=(8, 8))

    start = time.time()
    results = model.detect([original_image], verbose=1)

    r = results[0]
    end = time.time()
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_val.class_names, r['scores'], ax=get_ax())
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])

    print("检测时间:%.2f秒" % (end - start))


def run_map(out_path, num2_1, num2_2):
    dataset_val = ExpressDataset()
    dataset_val.out_path(out_path)
    dataset_val.set_dataset_class('val')
    dataset_val.set_dataset_exit_flag(True)
    dataset_val.load_shapes(num2_1, num2_2, 0)
    dataset_val.prepare()

    class InferenceConfig(ExpressConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    #model_path = '/home/kingqi/proj/Mask_RCNN/log/express20190424T1751/mask_rcnn_express_0006.h5'
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    image_ids = np.random.choice(dataset_val.image_ids, num2_1+num2_2*6)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                  image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

        image1 = dataset_val.load_image(image_id)
        mask1, class_ids1 = dataset_val.load_mask(image_id)

        bbox = utils.extract_bboxes(mask1)
        # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
        #visualize.display_instances(image1, bbox, mask1, class_ids1, dataset_val.class_names)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                            dataset_val.class_names, r['scores'])
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))



def thread_gen_data(out_path, num1_1, num1_2, num2_1, num2_2, flag):
    #多线程,必须写在main函数下
    import multiprocessing
    pn = 4
    index = []
    for i in range(pn):
        index.append(int(i / pn * (len(train_class_name) * num1_2 + num1_1)))
    pool = multiprocessing.Pool(processes=pn)

    dataset_train = ExpressDataset()
    dataset_train.out_path(out_path)
    dataset_train.set_dataset_class('train')
    dataset_train.set_dataset_exit_flag(flag)
    if flag:
        dataset_train.load_shapes(num1_1, num1_2)
    else:
        for i in range(pn):
            n = np.random.randint(0, 2019)
            pool.apply_async(dataset_train.load_shapes, (num1_1, num1_2, n, index[i], pn,))

        pool.close()
        pool.join()
        print("finished")


    index1 = []
    for i in range(pn):
        index1.append(int(i / pn * (len(train_class_name) * num2_2 + num2_1)))
    pool = multiprocessing.Pool(processes=pn)

    dataset_val = ExpressDataset()
    dataset_val.out_path(out_path)
    dataset_val.set_dataset_class('val')
    dataset_val.set_dataset_exit_flag(flag)
    if flag:
        dataset_val.load_shapes(num2_1, num2_2)
    else:
        for i in range(pn):
            n1 = np.random.randint(0, 2019)
            pool.apply_async(dataset_val.load_shapes, (num2_1, num2_2, n1, index1[i], pn,))

        pool.close()
        pool.join()
        print("finished")



if __name__ == '__main__':
    out_path = '/home/kingqi/proj/data/mask-rcnn/my_data1'
    #num 必须是4(pn)的倍数
    num1_1 = 12#3000
    num1_2 = 8#200
    num2_1 = 0#500
    num2_2 = 0#80
    flag = True
    show = True

    t0 = time.time()
    thread_gen_data(out_path, num1_1, num1_2, num2_1, num2_2, flag)
    t1 = time.time()
    print("dataset cost time %06f " % (t1 - t0))
    train(out_path, num1_1, num1_2, num2_1, num2_2, show)


    #detection(out_path, num2_1, num2_2)

    #run_map(out_path, num2_1, num2_2)
