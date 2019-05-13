#!/usr/bin/env python
# coding: utf-8


import os
import sys
import skimage.io


ROOT_DIR = os.path.abspath("../../")


sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR))

MODEL_DIR = os.path.join(ROOT_DIR, "logs-1")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


#IMAGE_DIR = os.path.join(ROOT_DIR, "images_test")
IMAGE_DIR = '/home/kingqi/proj/data/org1'
#IMAGE_DIR = '/home/kingqi/proj/data/2018-12-20'


class ExpressConfig(Config):
    NAME = "express"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 448
    LEARNING_RATE = 0.01
    GRADIENT_CLIP_NORM = 1



class InferenceConfig(ExpressConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
inference_config.display()

model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)


# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = '/home/kingqi/proj/MASKRCNN/logs-1/express20190506T1604/mask_rcnn_express_0006.h5'
#model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)





class_names = ['BG', 'box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']





# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
#
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#
# # Run detection
# results = model.detect([image], verbose=1)
#
# # Visualize results
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])


for i in os.listdir(IMAGE_DIR):
    #file_names = next(os.walk(IMAGE_DIR))[2]

    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, i))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])





