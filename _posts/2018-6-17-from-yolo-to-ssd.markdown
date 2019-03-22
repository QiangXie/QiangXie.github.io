---
layout: post
title: "从YOLO到YOLO3"
subtitle: "非region proposal检测方法回顾"
author: "谢强"
date: 2018-12-17 14:03:54
header-img: "img/caffe_install.jpg"
tags:
    - Detection
    - Paper Reading

---

#### 前言 ####

&#160; &#160; &#160; &#160;自从RCNN算法诞生以来，经过不断改进优化，该系列算法相较于之前的目标检测算法检测的准确率提升了很多，例如Faster-RCNN在VOC07数据集上mAP值能达到73.2%，这比RCNN系列算法没出来之前提高了近一倍。但是，这一系列算法在准确率不断提升的同时却始终存在一个致命的问题——速度慢。这一系列中最快的目标检测算法是Faster-RCNN，经过GPU加速之后也只有5fps。目标检测在计算机视觉领域是一个基础技术，很多问题都需要在目标检测之后进行解决，比如识别、跟踪、检索和ReID等等，这些问题无不需要更快更强的目标检测算法。RCNN系列算法更强做到了，但是更快怎么办呢？

&#160; &#160; &#160; &#160;分析一下RCNN系列算法就可以获得一些启发：最初的RCNN和Fast-RCNN算法需要依靠外部的Region Proposal算法来提取ROI，但是Fast-RCNN相较于RCNN算法的改进是对整张图片提取一次CNN特征，使用ROI Pooling在特征图上投影获得ROI区域特征，这样算法避免了多次对同一张图提取CNN特征，节省了计算时间。而Faster-RCNN则是把Region Proposal也做到了算法里，RPN只在Fast-RCNN上增加了少量的计算开支，却节约了别的Region Proposal算法的时间，速度也得到了很大提升。现在Faster-RCNN系列算法的速度瓶颈在哪里呢？了解CNN的都知道，全连接层是一个很耗计算资源的地方。而Faster-RCNN最后要提取大约2000个ROI，每一个都要过一遍Fast-RCNN进行边框回归和类别预测，虽然一个ROI消耗的时间不多，但是乘以2000之后这还是一个很大的时间开销。能不能想办法节省一下这些时间呢？下面要介绍的YOLO系列算法给出了答案。

#### Yon Only Look Once  ###

&#160; &#160; &#160; &#160;针对Region Proposal算法速度慢的问题，Joseph Redmon提出了YOLO算法。顾名思义，YOLO只需要一次全图卷积加回归，不需要使用Region Proposal算法提取多个ROI依次进行单独的ROI区域的分类和位置信息回归。算法整体流程见下图：

![java-javascript](/img/in-post/yolo/yolo1-architecture.png)

&#160; &#160; &#160; &#160;YOLO人为把一张输入图片划分为S\*S 个方格，每个方格预测B个Boundingbox和置信度，每一个Boundingbox用 \\(x,y,w,h\\) 表示，分别表示Boundingbox的中心点相对于方格边缘的坐标和宽度高度相对于整图的比例。置信度代表的是预测的矩形框和任意一个ground-truth之间的IOU值。除了前面提到的B\*5 个分别代表位置信息和方格位置是否有物体的置信度，每一个方格还要预测C类的条件概率 \\(P_{r}(Class_{i}\mid Object)\\) ,所谓条件概率，就是如果前面的置信度预测这个方格位置存在物体之后，再预测一个是C类每一类物体的概率，这个C就是检测时物体的类别数，例如Pascal VOC 20类，那这个C就等于20。在测试的时候，单个方格预测是某类的置信度等于是物体的置信度和是某一类物体条件概率的乘积。所以综合下来，经过CNN backbone网络提取完特征之后，需要预测 (S\*S\*(B\*5+C) 大小的张量。针对Pascal VOC检测任务，论文里 \\(S=7,B=2，C=20\\) ，所以最后预测时张量的维度是 7\*7\*30，网络结构见下图：

![java-javascript](/img/in-post/yolo/yolo1-net-architecture.png)





**参考资料**


 1.[You Only Look Once: Unified, Real-Time Object Detection][1]

 2.[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition][2]

 3.[Fast R-CNN][3]

 4.[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks][4]

 5.[Mask R-CNN][5]


  [1]: https://arxiv.org/pdf/1506.02640.pdf
  [2]: https://arxiv.org/pdf/1406.4729.pdf
  [3]: https://arxiv.org/pdf/1504.08083.pdf
  [4]: https://arxiv.org/pdf/1506.01497.pdf
  [5]: http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
