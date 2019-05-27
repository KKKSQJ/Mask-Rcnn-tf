import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np

class_name = ['box_0', 'box_1', 'soft_0', 'soft_1', 'envelope_0', 'envelope_1']

def shape2box(xmlbox):
    shape = []
    for i in range(int(xmlbox.find('len').text)):
        shape.append(float(xmlbox.find('x%d'%i).text))
        shape.append(float(xmlbox.find('y%d'%i).text))
    return shape

def convert(box):
    #return (xmin,ymin,xmax,ymin,xmin,ymax,xmax,ymax)
    return (box[0],box[2], box[1], box[2], box[0], box[3], box[1], box[3])

def gen_labels():
    #cwd = os.getcwd()
    cwd = '/home/kingqi/proj/data/mask-rcnn/train_data/20190401_express'
    ann_dir = os.path.join(cwd, 'Annotations')
    labels = os.path.join(cwd, 'labels')
    if not os.path.exists(labels):
            os.makedirs(labels)
    for walk in os.walk(ann_dir):
        for file in walk[2]:
            xml_path = os.path.join(ann_dir, file)
            if xml_path.split('.')[-1] == 'json':
                continue
            print(xml_path)
            out_path = os.path.join(labels, file.replace('xml', 'txt'))
            out_file = open(out_path, 'w')
            in_file = open(xml_path)
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            # print(w, h)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in class_name or int(difficult) == 1:
                    continue
                cls_id = class_name.index(cls)
                xmlbox = obj.find('bndbox')
                b = shape2box(xmlbox)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')
            out_file.close()
            in_file.close()


if __name__ == '__main__':
    gen_labels()
