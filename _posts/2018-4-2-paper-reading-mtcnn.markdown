---
layout: post
title: "论文阅读Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks "
subtitle: "一个快速准确的人脸检测算法"
author: "Johnny"
date: 2018-3-26 11:03:54
header-img: "img/caffe_install.jpg"
tags: 
    - Face Detection 
    - Paper Reading
---

1. 算法核心内容 

&#160; &#160; &#160; &#160;本论文提出了一种分步、多任务（同时预测人脸和人脸关键点）的人脸检测算法，算法的整体Pipeline如下图所示：

![java-javascript](/img/in-post/mtcnn/pipeline.png)

&#160; &#160; &#160; &#160;拿到一张图片后，首先对图片进行resize到不同尺寸得到图形金字塔。然后作者设计一个全卷积神经网络P-Net（Proposal Network）用于产生人脸候选框和回归具体位置坐标的回归向量，然后使用NMS合并重合度较高的候选框。把P-Net产生的所有候选人脸
送入到另一个叫做Refine Network (R-Net)的卷积网络用来去掉第一步产生的候选疑似人脸中大部分的false example。R-Net同时对候选人脸进行一定的回归得到较为准确的人脸位置信息，同P-Net一样，R-Net之后同样适用NMS进行候选框去重。最后一个阶段，适用和R-Net相似的卷积网络对人脸进行更准确的回归，同时获得人脸的关键点信息。

2.CNN网络结构

&#160; &#160; &#160; &#160;对于网络结构作者分析如下：（1）卷积层中的filter缺乏差异性限制了它们的识别能力。**（这句话没看懂，也没看出作者以这个论点做出什么改进，如果有人看懂这句话什么意思，求告知。）**（2）和多目标检测相比，人脸检测问题是一个二分类问题，所以它每一层可能需要更少的卷积核。**（这一句也没看懂，这个原因跟结果之间感觉没有什么联系。）**针对以上两个问题，作者减少了卷积核的个数并且把5x5的卷积核改为3x3的卷积核。上述改进使得算法获得又快又准的准确率。具体网络结构如下：

![java-javascript](/img/in-post/mtcnn/cnn_arc.png)

3.训练

&#160; &#160; &#160; &#160;该算法一共需要训练三个部分：一个人脸/非人脸的二分类器、一个bounding box回归器和一个人脸关键点定位回归。对是不是人脸的判别使用经典的交叉熵loss：

$$ L^{det}_{i} =-\left( y^{det}_{i} log( p_{i}) +\left( 1-y^{det}_{i}\right)( 1-log( p_{i}))\right) $$

&#160; &#160; &#160; &#160;对于bounding box的回归使用 Euclidean loss度量训练误差：

$$ L^{box}_{i} =||\hat{y}^{box}_{i} -y^{box}_{i} ||^{2}_{2} $$

&#160; &#160; &#160; &#160;脸部关键点定位同样使用Euclidean loss来度量训练误差：

$$ L^{landmark}_{i} =||\hat{y}^{landmark}_{i} -y^{landmark}_{i} ||^{2}_{2} $$

&#160; &#160; &#160; &#160;以上三个loss函数分别针对不同的任务，在训练阶段P-Net、R-Net和O-Net三个网络训练的目的不同，所以训练时的loss略有差别，简单来说就是不同网络训练时三个任务的loss权重不同，总的loss可以用如下公式进行定义:

$$ min\sum\nolimits ^{N}_{i=1}\sum\nolimits _{j\in \{det,box,landmark\}} \alpha _{j} \beta ^{j}_{i} L^{j}_{i} $$

&#160; &#160; &#160; &#160;在论文中，作者对P-Net和R-Net使用\\(\alpha _{det} =1,\alpha _{box} =0.5,\alpha _{landmark} =0.5\\),在O-Net分别设置\\(\alpha _{det} =1,\alpha _{box} =0.5,\alpha _{landmark} =1\\)增加关键点loss的权重以获得更为准确的关键点位置信息。

&#160; &#160; &#160; 为了使算法在hard sample上获得更好的表现，这篇论文还增加了Online Hard sample mining。具体做法如下：在一次mini batch训练的时候选取获得loss最高的70%的样本进行反向传播，忽略剩下30%的样本。这样就提高了算法在难分样本上的检测能力，后续的实验也证明了以上结论。

**参考资料**


 1. [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks[1]

 


  [1]: https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf