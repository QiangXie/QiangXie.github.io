---
layout: post
title: "论文阅读Seeing Small Faces from Robust Anchor’s Perspective"
subtitle: "如何解决人脸检测小脸的漏检问题"
author: "Johnny"
date: 2018-3-26 11:03:54
header-img: "img/caffe_install.jpg"
tags: 
    - Computer Vision 
---

## 1. 问题阐述 ##
&#160; &#160; &#160; &#160;现有的人脸检测算法基本都是受Faster-rcnn和SSD启发的基于anchor机制的检测算法，这种算法有一个很大的技术特点，那就是：使用default box和人脸之间的IOU（Intersection-over-Union）overlap评估某个人脸和default box之间的相似度，从而决定图片中某个人脸应该由哪个default box来预测其精确的位置信息。IOU这个计算方法的特点使得比较小的人脸在此类算法中普遍表现不好，为什么呢？原因如下图：

![java-javascript](/img/in-post/seeing-small-faces/iou.png)

&#160; &#160; &#160; &#160;上图中红色的框是人脸的正确的位置，黄色框是default box的位置信息，从上图可以看到，由于卷积神经网络的特点，default box在某一层一般有对应的感受野，也就是一个default box在固定的卷积层上映射到原图有一个固定的位置，这个区域一般比较大，按照IOU的计算方式，小脸跟default box之间的IOU值比较小。在基于anchor机制的目标检测算法中，训练或者预测的时候一定会设置一个过滤无用的预测框的一个IOU阈值。这种小脸可能在训练的时候就被过滤掉了，所以在训练过程中模型是学习不到怎么预测小脸的位置信息，同样的，预测的时候小脸也因为和default box之间的IOU值较低，同样被当作negative过滤掉了，所以这一类算法普遍存在小脸检测不到的问题。论文中作者专门统计了不同大小的脸和default box之间的IOU值，数据也验证了以上猜想，如下图所示，小脸（16x16个像素以下）的脸获得的平均IOU值只有0.27。相应的，小脸获得的召回率（recall）也是最低的，只有0.44，而获得高IOU的大脸获得的召回率普遍较高，普遍在0.9以上，实验证明不同大小的脸的召回率和其平均IOU值基本成正相关。

![java-javascript](/img/in-post/seeing-small-faces/recall.png)

## 2.Expected Max Overlapping Scores ##

&#160; &#160; &#160; &#160;为了验证上述论证结果，论文作者设计了一个EMO Score（Expected Max Overlapping）从理论上进一步论证了小脸在基于anchor机制的检测器中获得EMO Score较低，以及如何才能提高EMO Score。论证过程如下如下：

&#160; &#160; &#160; &#160;由于卷积神经网络的池化操作（pooling)，经过层卷积-池化之后获得的特征图（feature maps）一般维度是等比例缩小的，比如原图尺寸是WxH，经过卷积之后的特征图是hxw，则相邻两个特征图上的点映射回原图有一个尺度变换，这个变换因子是：

$$ \frac{H}{h} =\frac{W}{w} =s_{F} $$

&#160; &#160; &#160; &#160;相应的，anchor的步长和这个变换因子是相等的：

$$ s_{F} =s_{a} $$

&#160; &#160; &#160; &#160;假设p( x,y)是图片中人脸的概率密度，它满足：

$$ \int ^{H}_{0}\int ^{W}_{0} p( x,y) dxdy=1 $$

&#160; &#160; &#160; &#160;则EMO可以定义为:

$$ EMO=\int ^{H}_{0}\int ^{W}_{0} p( x,y)\max_{a\in A}\frac{| B_{f} \cap B_{a}| }{| B_{f} \cup B_{a}| } dxdy=1 $$

&#160; &#160; &#160; &#160;如下图所示：

![java-javascript](/img/in-post/seeing-small-faces/face-anchor.png)

&#160; &#160; &#160; 人脸的中心点被4个anchor的中心点包围，距离人脸最近的中心点所对应的anchor肯定能获得最大的IOU值。现举例分析如果左上侧的anchor和人脸之间的IOU值最大，则人脸的中心点肯定在下图的蓝色区域。因为如果人脸的中心点不在图示的蓝色区域内，则人脸的外界矩形框肯定跟其他三个anchor中至少一个之间的IOU值左上边那个anchor的大。则人脸外接矩形和anchor box的IOU值定义如下：

$$ IOU=\frac{( l-x')( l-y')}{2l^{2} -( l-x')( l-y')} $$

&#160; &#160; &#160; 可以看出IOU值是\\((x',y')\\)的函数，anchor box的中心离人脸外界矩形框的中心越近IOU值越大。相应的，EMO公式可以定义如下：

$$ EMO=\int ^{\frac{s_{A}}{2}}_{0}\int ^{\frac{s_{A}}{2}}_{0}\left(\frac{2}{sA}\right)^{2}\frac{( l-x')( l-y')}{2l^{2} -( l-x')( l-y')} dx'dy' $$

&#160; &#160; &#160;从这个公式可以看出人脸越大越能获得更高的EMO score，当人脸是固定大小的时候，\\( s_{A}\\)越小EMO score越大。针对这个结论，作者同样做了实验验证，作者实验\\( s_{A}\\)分别等于8和16时的EMO score，结果如下图，同样支持了上述结论。

![java-javascript](/img/in-post/seeing-small-faces/sa-exp.png)

## 3.Strategies of New Anchor Design ##

&#160; &#160; &#160;知道问题出现在哪，下一步就好办了。上面的理论推导以及实验告诉我们，人脸检测任务中基于anchor机制的算法获得的recall和EMO Score高度相关，EMO Score值越大，recall越高。而EMO Score的值又和\\( s_{A}\\)高度相关，\\( s_{A}\\)越小EMO Score的值越大。

### 3.1 Stride Reduction with Enlarged Feature Maps ###
&#160; &#160; &#160; 第一种方法是增大feature map的尺寸，增大feature map的尺寸之后，相邻特征点之间映射回原图的距离就减小了，也即减小了\\( s_{A}\\)的值，具体方法如下图：

![java-javascript](/img/in-post/seeing-small-faces/increase-feature-maps.png)

&#160; &#160; &#160;如上图所示，分别采取a.对要进行检测的图进行上采样；b.对上采样之后的特征图和前一层的特征图进行element-wise相加，针对通道数不一致的问题使用1x1卷积使相加的两个特征图通道一致；c.使用Dilated convolution对特征图进行上采样。

### 3.2 Extra Shifted Anchors ###

&#160; &#160; &#160;前面分析知道EMO Score值越大对小脸的检测效果也好，为了增加EMO Score需要减小\\(s_{A}\\)，而\\(s_{A}=s_{F}\\)，那是不是可以考虑不改变\\(s_{F}\\)只改变\\(s_{A}\\)呢？答案是肯定的，作者考虑使用另外一种方法减小\\(s_{A}\\)而不改变\\(s_{F}\\)，那就是所谓的Extra Shifted Anchors。具体采用方法如下：anchor的中心点可以不一定在stride的中心，可以在stride周围增加anchor点，这样\\(s_{A}\\)就相应减小了，如下图所示：

![java-javascript](/img/in-post/seeing-small-faces/increase-anchor.png)

&#160; &#160; &#160;a.不增加额外anchor，\\(s_{A}=s_{F}\\)；b.在原有的每个anchor右下角增加一个anchor，这个时候\\(s_{A} =\sqrt{2} s_{F}\\)；c.在每个原有anchor的右方、下方和右下方各增加一个anchor，此时\\(s_{A} =\frac{s_{A}}{2}\\)。实验结果表明，增加额外的anchor可以提高小脸的平均IOU值，如下图所示：

![java-javascript](/img/in-post/seeing-small-faces/increase_anchor_result.png)

### 3.3 Face Shift Jittering & Hard Face Compensation ###

&#160; &#160; &#160;基于anchor的目标检测算法中anchor的位置一般是固定的，映射回输入的原图中每个anchor之间有一定距离。但是人脸在输入图片中并不是固定的，它可能在图片任何一个位置，那些中心点离anchor的中心点远的人脸和anchor之间总是会获得较小的IOU值。为了克服这个缺点，在训练的时候随机对图片进行一定的平移，这样理论上就能解决上述问题。

&#160; &#160; &#160;还存在一种情况，那就是：经过以上种种策略之后，还是有一些人脸因为和所有anchor的位置相距太远，它和任何一个anchor的IOU值依然小于匹配anchor的阈值，这种情况怎么办呢？论文中采取的策略是：对于这种人脸，把它当做hard face，对这种hard face计算它和所有anchor之间的IOU值，然后进行排序，选取IOU最高的N个anchor都认为和这个hard face匹配上了，N值在后续实验中确定。

&#160; &#160; &#160;实验结果证明经过上述策略改进之后的算法确实对小脸检测的准确率有很大提升，这些策略对于目标检测算法的设计具有借鉴意义。

**参考资料**


 1. [arxiv：Seeing Small Faces from Robust Anchor's Perspective][1]

 


  [1]: https://arxiv.org/abs/1802.09058v1