# Mask-Rcnn-tf
基于tensorflow,keras的Mask-Rcnn

# 环境搭配
ubuntu16.04 cuda9.0 cudnn7.1 tensorflow-gpu==1.8.0 keras==2.0.8

# 小工具使用
* 模型训练完成后，会自动生成logs文件夹，并在此文件夹下生成events.out. 和.h5模型
* source activate --name keras
* cd Mask-Rcnn-tf
* tensorboard --logdir logs
* 打开出现的网站，可以观察到loss等信息

# 模型下载
coco预训练模型：COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

# demo.py 测试
python run demo.py

# 模型训练
* 见sample文件夹
* balloon, coco, nucleus, shape的使用参见Mask-RCNN官网，其中shape/train_shape.py 从数据产生->训练->测试>MAP

# 训练自己的模型
* aff_data:对图像进行形态的数据增强,即多形态!!!!
* 见sample/express/train_express.py文件
* data/data_generate.py:产生训练数据
* data/data_config.py:数据的配置信息
* sample/express/train_express.py:模型训练，测试，map
* sample/express/detect.py:测试实际场景图
* 训练数据的格式参考data/express_data

# 参考网站
https://github.com/matterport/Mask_RCNN #Mask_RCNN官网

