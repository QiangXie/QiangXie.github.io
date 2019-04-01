---
layout: post
title: "用自己的数据训练Faster-RCNN"
subtitle: "玩faster-rcnn手记"
author: "Johnny"
date: 2016-05-17 21:59:08
header-img: "img/bg-3.jpg"
tags:
    - Object Detection
---

Faster-RCNN玩了一段时间了，一直想写一篇文章总结一下，但拖延症一直也没有下笔，今天就写了吧，做个总结。

![这里写图片描述](http://img.blog.csdn.net/20160509221439591)

Faster-RCNN是RCNN系列检测方法中最新的方法了（Yolo除外，因为Yolo走的是另一个路子，有机会再说这个），简单来说Faste-RCNN把Fast-RCNN方法用到的proposal方法也弄到了CNN网络里（rbg大神称此网络为RPN网络），省去了额外的诸如selective search之类的proposal方法，后面的分类直接把RPN网络生成的proposal映射回卷基层生成的feature map得到固定纬的特征，进行分类，这就跟之前Fast-RCNN一样了。唯一不同的是，对同一幅图片的多个疑似目标Faster-RCNN只需要进行一次forward，这样就避免了对同一个图片的多个proposal进行多次forward，这样节省的时间就显而易见了，在TitanX上我的实验结果每张图片差不多170ms-200ms, 这个速度还是比Fast-RCNN快了很多。

具体的内容可以参看论文，这里我介绍一下如何用自己的数据训练Faster-RCNN，至于如何安装我就不赘述了，可以参考rgb的[github](https://github.com/rbgirshick/fast-rcnn)。

**1. 准备数据**

在rbg的原版Faster-RCNN里，他是用的Pascal voc 2007的数据集，为了使用rbg的代码，我在这里没有改变自己的数据格式，而是把标注数据整理成和Pascal voc一样的数据格式，简单来说就是新建一个文件夹，可以任意命名，我在这里假设`$pascal_format`是我新建的自己放pascal voc格式的数据文件夹。

`$pascal_format`下应该有一个叫做VOC2007的文件夹，VOC2007下应该有Annotations、ImageSets和JPEGImages三个文件夹，其中：Annotations是标注数据xml文件，JPEGImages是和Annotations中的xml文件一一对应的图片文件，两个文件除了后缀分别是.xml和.jpg，其他部分应该一样，都应该是六位数字，比如：000001.jpg和000001.xml。ImageSets下有一个Main文件夹，文件夹内有一个trainval.txt文件，里面是训练数据设置，格式如下：


    000000
    000001
    000002
    000003
    000004
    000005
    000006
    ……


一般我们手头的数据不是Pascal voc格式的，Pascal voc数据里面的标注数据是如下的xml文件：


    <?xml version="1.0"?>

    -<annotation>

    <folder>VOC2007</folder>

    <filename>013120.jpg</filename>


    -<source>

    <database>ChongQing Database</database>

    <annotation>ChongQing</annotation>

    <image>flickr</image>

    <flickrid>QiangXie</flickrid>

    </source>


    -<owner>

    <flickrid>QiangXie</flickrid>

    <name>QiangXie</name>

    </owner>


    -<size>

    <width>1664</width>

    <height>1200</height>

    <depth>3</depth>

    </size>

    <segmented>0</segmented>


    -<object>

    <name>car</name>

    <pose>Unspecified</pose>

    <truncated>1</truncated>

    <difficult>0</difficult>


    -<bndbox>

    <xmin>438</xmin>

    <ymin>348</ymin>

    <xmax>840</xmax>

    <ymax>774</ymax>

    </bndbox>

    </object>


    -<object>

    <name>car</name>

    <pose>Unspecified</pose>

    <truncated>1</truncated>

    <difficult>0</difficult>


    -<bndbox>

    <xmin>33</xmin>

    <ymin>9</ymin>

    <xmax>282</xmax>

    <ymax>189</ymax>

    </bndbox>

    </object>


    -<object>

    <name>car</name>

    <pose>Unspecified</pose>

    <truncated>1</truncated>

    <difficult>0</difficult>


    -<bndbox>

    <xmin>471</xmin>

    <ymin>0</ymin>

    <xmax>651</xmax>

    <ymax>105</ymax>

    </bndbox>

    </object>

    </annotation>


这就需要我们把标注数据转成xml文件，我写了几行程序把手头的数据转成xml文件，程序如下：


    #!/usr/bin/env python
    '''
    This file is a tool to parse json file and generate voc format xml file.
    '''
    import json
    import xml.etree.ElementTree as ET
    import cv2
    import os

    out_xml_path = "/home/qxie/Data/out/"  
    #jpg files folder
    src_path = "/home/qxie/Data/src/"
    json_file = open("./daytime_res.json",'r')


    counter = 0

    while True:
    file_path = json_file.readline()
    while file_path == "\n":
    file_path = json_file.readline()
    json_obj = json_file.readline()
    while(json_obj == "\n"):
    json_obj = json_file.readline()
    print counter
    jpg_name = file_path[43:53]
    xml_file_name = os.path.splitext(jpg_name)[0] + ".xml"
    print xml_file_name
    jpg_path = os.path.join(src_path,jpg_name)
    print jpg_path
    im = cv2.imread(jpg_path)
    im_height = im.shape[0]
    im_width = im.shape[1]
    im_ch = im.shape[2]
    counter += 1

    #create a xml
    out = ET.Element('annotation')
    #folder
    folder = ET.SubElement(out,"folder")
    folder.text = "VOC2007"
    #filename
    filename = ET.SubElement(out,"filename")
    filename.text = jpg_name
    #filesource
    file_source = ET.SubElement(out,"source")
    database = ET.SubElement(file_source,"database")
    database.text = "ChongQing Database"
    annotation = ET.SubElement(file_source,"annotation")
    annotation.text = "ChongQing"
    image = ET.SubElement(file_source,"image")
    image.text = "flickr"
    flickid = ET.SubElement(file_source,"flickrid")
    flickid.text = "QiangXie"

    #file owner
    owner = ET.SubElement(out,"owner")
    flickid = ET.SubElement(owner,"flickrid")
    flickid.text = "QiangXie"
    name = ET.SubElement(owner,"name")
    name.text = "QiangXie"

    #file size
    file_size = ET.SubElement(out,"size")
    file_width = ET.SubElement(file_size,"width")
    file_width.text = str(im_height)
    file_height = ET.SubElement(file_size,"height")
    file_height.text = str(im_width)
    file_depth = ET.SubElement(file_size,"depth")
    file_depth.text = str(im_ch)

    #file segmented
    file_segmented = ET.SubElement(out,"segmented")
    file_segmented.text = "0"



    json_data = json.loads(json_obj)
    if json_data == None:
    print "Open json obj failed"

    if 'RecVehicles' not in json_data:
    continue

    vehicles = json_data['RecVehicles']
    for vehicle in vehicles:
    roi = vehicle['Cutboard']
    bbox_x = roi['PosX']
    bbox_y = roi['PosY']
    bbox_width = roi['Width']
    bbox_height = roi['Height']
    #create a car obj
    obj = ET.SubElement(out,'object')
    obj_name = ET.SubElement(obj,"name")
    obj_name.text = "car"

    obj_pose = ET.SubElement(obj,"pose")
    obj_pose.text = "Unspecified"

    obj_truncated = ET.SubElement(obj,"truncated")
    obj_truncated.text = "1"

    obj_difficult = ET.SubElement(obj,"difficult")
    obj_difficult.text = "0"

    #create boundingbox
    bndbox = ET.SubElement(obj,"bndbox")
    xmin = ET.SubElement(bndbox,'xmin')
    xmin.text = str(bbox_x)

    ymin = ET.SubElement(bndbox,'ymin')
    ymin.text = str(bbox_y)

    xmax = ET.SubElement(bndbox,'xmax')
    xmax.text = str(bbox_x + bbox_width)

    ymax = ET.SubElement(bndbox,'ymax')
    ymax.text = str(bbox_y + bbox_height)

    out_tree = ET.ElementTree(out)
    out_tree.write(out_xml_path + xml_file_name)

    print "Process done"



**2. 训练前的一些设置**


rbg提供了两种训练方法，分别是在NIPS2015中的paper所描述的alternating optimization，另一个是end-to-end的训练方法，第一种训练方法具体可以参考论文，我在这里主要以end-to-end的训练方法为例。在训练之前需要先下载预训练模型的权重数据：

```
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
然后把训练数据链接到Faster-RCNN：

```
ln –s $pascal_format $FASTER_RCNN_ROOT/data/VOCdevkit2007
```
另外值得注意的一点是，如果你在用自己的数据训练Faster-RCNN之前按照github上rbg的方法用Pascal_voc数据训练过，一定要删除cache文件：

```
rm $FASTER_RCNN_ROOT/data/cache/voc_2007_trainval_gt_roidb.pkl
```
否则，会提示错误。

**3.启动训练**


之前已经把需要的数据和预训练权重准备好了，这一步就相对简单一些，只需要输入指令启动训练：

```
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
```

GPU_ID 是训练要用的 GPU，NET 是要用的训练网络可选 ZF, VGG_CNN_M_1024,
VGG16，一般用 VGG16，其他 model我试验了一下， 效果不是很好。假设用 GPU0，VGG16，则命令为：

```
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc
```
训练默认输出模型权重到`$FASTER_RCNN_ROOT/output`,70000次迭代，每10000次迭代保存一次权重。fine-tune时可以通过修改`$FASTER_RCNN_ROOT/experiment/scripts/faster_rcnn_end2end.sh`来修改初试权重和迭代次数。

**4.参考资料**

1. [rbg的github：rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)
1. [RCNN, Fast-RCNN, Faster-RCNN的一些事](http://closure11.com/rcnn-fast-rcnn-faster-rcnn%E7%9A%84%E4%B8%80%E4%BA%9B%E4%BA%8B/)
