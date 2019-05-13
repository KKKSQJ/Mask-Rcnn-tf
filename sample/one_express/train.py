#!/usr/bin/env python
#conding:utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs-1")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ExpressConfig(Config):
    NAME = "express"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 448
    LEARNING_RATE = 0.02
    STEPS_PER_EPOCH = 500
    GRADIENT_CLIP_NORM = 1.0
config = ExpressConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class ExpressDataset(utils.Dataset):

    def load_shapes(self, seed_data):
        self.add_class("express", 1, "box_0")
        self.add_class("express", 2, "box_1")
        self.add_class("express", 3, "soft_0")
        self.add_class("express", 4, "soft_1")
        self.add_class("express", 5, "envelope_0")
        self.add_class("express", 6, "envelope_1")

        count = len(seed_data)
        for i in tqdm(range(count)):
            self.add_image("express", image_id=i, path=None,
                           img_info=seed_data[i])

    def load_image(self, image_id):
        info = self.image_info[image_id]["img_info"]
        img_path = info[0]
        #print(img_path)
        img = cv2.imread(img_path)[:,:,::-1]
        return img

    def load_mask(self, image_id):
        info = self.image_info[image_id]["img_info"]
        height = cv2.imread(info[0]).shape[0]
        width = cv2.imread(info[0]).shape[1]
        label_path = info[1]
        with open(label_path, 'r',encoding='utf8') as f:
            shapes = f.readlines()
        shape_num = len(shapes)
        mask = np.zeros((height, width, shape_num), np.uint8)
        class_ids = []
        for i, shape in enumerate(shapes):
            shape = [int(float(xx)) for xx in shape.strip().split()]
            class_ids.append(shape[0]+1)
            points = self.shape2point(shape)
            mask[:,:,i:i+1] = self.draw_mask(mask[:,:,i:i+1].copy(), points, 1)
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(shape_num - 2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_mask(self, image, point, color):
        points = np.array([point]).astype(np.int)
        cv2.fillPoly(image, points, color)
        return image

    def shape2point(self, shape):
        points = []
        for ii in range(4):
            x = shape[(ii + 1) * 2 - 1]
            y = shape[(ii + 1) * 2]
            points.append((x, y))
        return points


def seed_data(img_list, label_list):
    __seed_data = {}
    for i, info in enumerate(zip(img_list, label_list)):
        __seed_data.update({i: info})
    return __seed_data

def get_train_val(dirpath):
    img_list = [os.path.join(dirpath, img) for img in os.listdir(dirpath)]
    img_len = len(img_list)
    np.random.seed(2019)
    np.random.shuffle(img_list)
    train_percent = 0.85
    train_number = int(img_len * train_percent)
    train_img_list = img_list[:train_number]
    train_label_list = [label.strip().replace('images','labels').replace('jpg','txt') for label in train_img_list]
    val_number = img_len-train_number
    val_img_list = img_list[int(train_number):]
    val_label_list = [label.strip().replace('images','labels').replace('jpg','txt') for label in val_img_list]
    print('训练集样本数量:%d，验证集样本数量:%d' % (train_number, val_number))
    train_seed_data = seed_data(train_img_list, train_label_list)
    val_seed_data = seed_data(val_img_list, val_label_list)
    return train_seed_data, val_seed_data

# get_train_val('/home/kingqi/proj/data/mask-rcnn/images')

def train(img_dir):
    train_seed_data, val_seed_data = get_train_val(img_dir)

    dataset_train = ExpressDataset()
    dataset_train.load_shapes(train_seed_data)
    dataset_train.prepare()

    dataset_val = ExpressDataset()
    dataset_val.load_shapes(val_seed_data)
    dataset_val.prepare()

    # image_ids = np.random.choice(dataset_train.image_ids, 10)
    # for image_id in image_ids:
    #     #print(image_id)
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
        #visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        model.load_weights(model.find_last(), by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=40,
                layers="all")

def detection(img_dir):
    train_seed_data, val_seed_data = get_train_val(img_dir)
    dataset_val = ExpressDataset()
    dataset_val.load_shapes(val_seed_data)
    dataset_val.prepare()

    class InferenceConfig(ExpressConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    #model_path = '/home/kingqi/proj/MASKRCNN/logs-1/express20190506T1604/mask_rcnn_express_0007.h5'
    model_path = model.find_last()

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)
    #
    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_val.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())



def eval(img_dir):
    train_seed_data, val_seed_data = get_train_val(img_dir)
    dataset_val = ExpressDataset()
    dataset_val.load_shapes(val_seed_data)
    dataset_val.prepare()

    class InferenceConfig(ExpressConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    #model_path = '/home/kingqi/proj/MASKRCNN/logs-1/express20190506T1604/mask_rcnn_express_0007.h5'
    model_path = model.find_last()

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_ids = np.random.choice(dataset_val.image_ids, 100)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                   dataset_val.class_names, r['scores'])

        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))
if __name__ == '__main__':
    img_dir = '/home/kingqi/proj/data/mask-rcnn/images'
    #train(img_dir)
    #detection(img_dir)
    eval(img_dir)



