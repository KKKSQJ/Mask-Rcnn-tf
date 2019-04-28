#!/usr/bin/env python
# -*- coding: utf8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np

def compute_iou_sort(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.

    :param box: 1D vector [y1, x1, y2, x2]
    :param boxes: [boxes_count, (y1, x1, y2, x2)]
    :param box_area: float. the area of 'box'
    :param boxes_area: array of length boxes_count.
    Note: The areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate
    duplicate work.
    :return: iou
    """
    #Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0)* np.maximum(y2 -y1, 0)

    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection/union

    return iou, union


def non_max_suppression_sort(boxes, scores, threshold, union_ratio):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)].Notice that (y2, x2) lays outside the box.
    在原有的基础上增加了删除排序后比检测的box小的(即iou通过阈值,但是在贴图时出现问题),用union和box area比较.
    用Intersection和box area比较的话是删除较大的.(这里是由于贴图的顺序是依照shape的顺序的,
    所以删除shape中靠前,但是较小的object)
    scores: 1-D array of box scores.
    threshold:Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    area = (y2 - y1)*(x2 - x1)

    #Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs)>0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the pick box with the rest
        iou, union = compute_iou_sort(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This returns indices
        # into ixs[1:], so add 1 to get indices into ixs.
        # 添加内容,为了删除被覆盖的object目标
        cover_ixs = np.where(union < union_ratio * area[i])[0]+1


        # 大于阈值就会被认为是一个物体,所以把小score的box给删除,这里加1是为了留下比较的那个box
        remove_ixs = np.where(iou>threshold)[0]+1

        #合并cover_ixs和remove_ixs,并去重
        remove_ixs = np.concatenate((cover_ixs, remove_ixs), axis=0)
        remove_ixs = np.array(list(set(remove_ixs)))
        #remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)#这边在删除,以便留下的第二个重新进行循环

    return np.array(pick, dtype=np.int32)







XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, points , name, difficult):
        bndbox = {}
        i=0
        for point in points:
            keyx = 'x'+ str(i)
            keyy = 'y'+ str(i)
            bndbox[keyx] = point[0]
            bndbox[keyy] = point[1]
            i+=1
        bndbox['len'] = i
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            '''if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin'])== 1):
                truncated.text = "1" # max == height or min
            elif (int(each_object['xmax'])==int(self.imgSize[1])) or (int(each_object['xmin'])== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"'''
            truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            bndbox = SubElement(object_item, 'bndbox')

            len = SubElement(bndbox,'len')
            len.text = str(each_object['len'])

            for i in range(0,each_object['len']):
                keyx = 'x'+ str(i)
                keyy = 'y' + str(i)
                x = SubElement(bndbox,keyx)
                x.text = str(each_object[keyx])
                y = SubElement(bndbox,keyy)
                y.text = str(each_object[keyy])


    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        points = []
        for i in range(0, int(bndbox.find('len').text)):
            keyx = 'x' + str(i)
            keyy = 'y' + str(i)
            x = int(bndbox.find(keyx).text)
            y = int(bndbox.find(keyy).text)
            points.append((x,y))

        self.shapes.append((label, points, None, None, difficult))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True
