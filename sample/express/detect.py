#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR))  # To find local version


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs-50")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images_test")
IMAGE_DIR = '/home/kingqi/proj/data/org'
#IMAGE_DIR = '/home/kingqi/proj/data/2018-12-20'


class ExpressConfig(Config):
    # NAME = "express"
    # BACKBONE = "resnet101"
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # NUM_CLASSES = 1 + 6
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 448
    # LEARNING_RATE = 0.01
    # TOP_DOWN_PYRAMID_SIZE = 256
    # FPN_MASK_GRAPH_CONV_SIZE = 256
    # #STAGE5 = True
    # GRADIENT_CLIP_NORM = 1


    NAME = "express"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6
    # IMAGE_MIN_DIM = 256 #256
    # IMAGE_MAX_DIM = 1024
    LEARNING_RATE = 0.02
    STEPS_PER_EPOCH = 500
    GRADIENT_CLIP_NORM = 1.0
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.4



class InferenceConfig(ExpressConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = '/home/kingqi/proj/MASKRCNN/logs-50/express20190507T1821/mask_rcnn_express_0022.h5'

model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)




# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']


# ## Run Object Detection



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
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, i))
    image = cv2.imread(os.path.join(IMAGE_DIR, i))
    image1 = image.copy()
    h, w = image.shape[:2]
    max_dim = max(h, w)
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    image = image[:,:,::-1]
    print(image.shape)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(image1[:, :, ::-1])
    # plt.axis('off')
    # plt.show()

    #padding org
    #image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image[:,:,::-1], r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])





