#!usr/bin/env python
# -*-coding: utf -8 -*-

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def COCOWrite(image_name, image_id, annos, masks,shapes, height, width, cls):
    json_dict = {
        "images": [], "type": "instance", "annotations": [], "categories": []
    }

    image = {
        "file_name" :image_name, "id": image_id, "height": height, "width": width
    }
    json_dict['images'].append(image)
    print(image)

    for i, anno in enumerate(annos):
        cls_id = cls.index(anno[0])
        (x1, y1) = anno[1][0]
        (x2, y2) = anno[1][2]
        h = max(0, int(y2)-int(y1))
        w = max(0, int(x2)-int(x1))
        mask = masks[i]
        shape = [(int(ii[0]), int(ii[1])) for ii in shapes[i]]
        indexs = []
        index = np.where(mask>0)
        indexs.append((index[0].tolist(),index[1].tolist()))
        json_dict['annotations'].append(
            {
                'area': int(w)*int(h),
                'bbox': [int(x1), int(y1), w, h],
                'category_id': int(cls_id),
                'id': i,
                'image_id': image_id,
                'iscrowd': 0,
                # 'segmentation': indexs   #直接写入mask,过于大
                'segmentation': shape      #写入shape,后面转为mask
            }
        )


    json_name = image_name
    with open(json_name, 'w') as f:
        json.dump(json_dict, f)



def COCORead(json_file):
    with open(json_file,'r') as f:
        json_dict = json.load(f)

    print(json_dict)
    file_info = json_dict['images']
    annotations = json_dict['annotations']
    json_path = file_info[0]['file_name']

    image_path = json_path.replace('Annotations','JPEGImages').replace('json','jpg')
    img = cv2.imread(image_path)
    # plt.imshow(img[:,:,::-1])
    # plt.axis('off')
    # plt.show()

    image_id = file_info[0]['id']
    height = file_info[0]['height']
    width = file_info[0]['width']
    count = len(annotations)

    masks = []
    labels = []
    bboxs = []

    for i, info in enumerate(annotations):
        area = info['area']
        bbox = info['bbox']
        bboxs.append(bbox)

        #label
        category_id = info['category_id']
        labels.append(category_id)

        segmentation = info['segmentation']

        #bbox
        x1 = bbox[0]
        y1 = bbox[1]
        b_width = bbox[2]
        b_height = bbox[3]

        #mask
        mask = np.zeros((height, width),dtype=np.uint8)
        points = np.array([segmentation]).astype(np.int)
        mask = cv2.fillPoly(mask, points, 1)
        masks.append(mask)
        # plt.imshow(mask)
        # plt.show()

    #masks
    occlusions = np.logical_not(masks[count-1]).astype(np.uint8)
    for i in range(count - 2, -1, -1):
        masks[i] = masks[i] * occlusions
        occlusions = np.logical_and(
            occlusions, np.logical_not(masks[i]))

    # for i, x in enumerate(masks):
    #     a = np.where(x>0)

    return img, labels, bboxs, masks
    #return file_info,annotations



if __name__ == '__main__':
    COCORead('/home/kingqi/proj/data/mask-rcnn/test_d/Annotations/000000.json')








