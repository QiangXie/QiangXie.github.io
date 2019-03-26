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

&#160; &#160; &#160; &#160;YOLO输入图片尺寸固定到448\*448，经过多次卷积池化之后最后得到7\*7\*1024维的特征，经过两个全连接得到7\*7\*30的张量，然后对检测结果进行NMS，就可以得到检测结果。YOLO相较于RCNN系列算法快了很多，在mAP值为63.4%时全尺寸YOLO可以达到45fps。如果牺牲一些检测精度，YOLO可以达到155fps，速度是Faster-RCNN的9倍到31倍。

#### The Single Shot Detector ###

&#160; &#160; &#160; &#160;通过改进，YOLO获得不错的运行速度，但是相应的算法准确率大幅下降。在Pascal VOC数据集上，mAP从74.3%下降到63.4%。分析YOLO的算法细节不难看出，YOLO天生存在一些短板：YOLO把一张图片划分为S\*S个格子，所以理论上一个格子只能预测一个物体，对于论文中的网络结构来说，一个格子的感受野映射回输入的448\*448上是64\*64的区域，如果这个感受野里面有两个较小的物体，YOLO预测时已经假定一个格子里只有一个物体存在，对于多个小物体YOLO是无能为力的（论文里作者有提到这个缺点）。另外一个问题，YOLO在最后预测种类以及特征时使用的是全图特征，但是在目标检测任务时最后定位物体的边缘位置信息需要很精细的语义信息，这个时候YOLO7\*7的特征显得有些粗粒度，而且这些特征里面对于一个格子中的物体来说可能大部分都是无用的。实际试用YOLO时也可以看出，YOLO检测物体时对外接矩形框的定位并不是很准确。有没有办法兼顾速度的同时让检测准确率和定位精度也保持在一个较高的水平呢？答案是有，SSD很好地完成了这个任务。

&#160; &#160; &#160; &#160;Single Shot Detector(SSD)参考了过往的目标检测算法，包括Faster-RCNN和YOLO，取其精华，去其糟粕，使得检测速度较快地同时，检测准确率也没有明显下降。SSD的核心思想也是抛弃Region Proposal方法，直接回归。**（可以这么说，在算力没有明显进步或者Region Proposal方法没有革命性地变化之前，要想检测速度快，抛弃proposal直接使用回归是不可避免的。）** YOLO已经证明，使用全图特征会降低检测的准确率，而且一个特征图划分成固定个数区域会使得算法对目标尺度变化表现非常差。针对上述问题，SSD采用了比较有针对性的解决办法：

+ 使用3\*3卷积核在特征图上滑动卷积，使用局部特征回归目标位置信息使得定位更加准确；

+ 使用不同深度卷积层的输出作为回归使用的特征图，同时使用Faster-RCNN类似的anchor机制，可以应对尺度变化的问题；

![java-javascript](/img/in-post/yolo/YOLO-vs-SSD.png)

&#160; &#160; &#160; &#160;上图是SSD和YOLO的网络结构对比，可以看到，SSD在VGG-16的基础上后面使用1\*1的通道卷积以及3\*3步长为2的卷积对特征图进行降维增大特征图的感受野，并使用其中6个特征图进行3\*3滑动卷积，使得同样是3\*3的卷积核但是对应的感受野不同，这样就可以兼顾不同的尺度。除了不同深度卷积层输出的特征图上3\*3卷积核区域感受野大小不同，SSD借鉴Faster-RCNN预设了多个不同大小和宽高比的预设矩形框，这使得算法对尺度变化更加敏感，对宽高比的变化处理得也更好。具体原理见下图：

![java-javascript](/img/in-post/yolo/SSD-conv.png)

&#160; &#160; &#160; &#160;图中狗和猫大小不同，这个可以使用不同尺度的特征图分别负责感知预测，狗的外接矩形框是一个矩形，猫的外接矩形是一个近似的正方形，这可以通过预设的不同宽高比的矩形框来解决。这样一些措施就使得SSD相较于YOLO来说对尺度的变化更加鲁棒。

&#160; &#160; &#160; &#160;和Faster-RCNN中RPN的loss函数类似，SSD在最优化求解时采用的Loss函数也包含两部分，一部分是分类loss，一部分是位置损失loss：

$$
L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc}(x, l,g))
$$

&#160; &#160; &#160; &#160;其中， \\(l\\) 是预测的目标的外接矩形框位置信息，\\(g\\) 是ground-truth矩形框位置信息，这里 \\(l\\) 和 \\(g\\) 也是相对偏移量，和Faster-RCNN位置损失函数使用的也是 \\(smooth_{L1}\\) ：

$$
L_{loc}(x,l,g)=\sum_{i\in Pos}^{N}\sum_{m\in\{cx,cy,w,h\}}x_{ij}^{k}smooth_{L1}(l_{i}^{m}-\hat{g}_{j}^{m}) \\
\hat{g}_{j}^{cx}=(g_{j}^{cx}-d_{i}^{cx})/d_{i}^{w} \\ \hat{g}_{j}^{cy}=(g_{j}^{cy}-d_{i}^{cy})/d_{i}^{h}
\\ \hat{g}_{j}^{w}=log(\frac{g_{j}^{w}}{d_{i}^{w}}) \\ \hat{g}_{j}^{h}=log(\frac{g_{j}^{h}}{d_{i}^{h}})
$$

&#160; &#160; &#160; &#160;分类loss函数就是softmax loss，其定义如下：

$$
L_{conf}(x,c)=-\sum_{i\in Pos}^{N}x_{i}^{p}log(\hat{c}_{i}^{p})-\sum_{i\in Neg}log(\hat{c}_{i}^{0}) \qquad where \quad \hat{c}_{i}^{p}=\frac{exp(c_{i}^{p})}{\sum_{p}exp(c_{i}^{p})}
$$

**参考资料**


 1.[You Only Look Once: Unified, Real-Time Object Detection][1]

 2.[SSD: Single Shot MultiBox Detector][2]

 3.[Fast R-CNN][3]

 4.[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks][4]

 5.[Mask R-CNN][5]


  [1]: https://arxiv.org/pdf/1506.02640.pdf
  [2]: https://arxiv.org/pdf/1512.02325.pdf
  [3]: https://arxiv.org/pdf/1504.08083.pdf
  [4]: https://arxiv.org/pdf/1506.01497.pdf
  [5]: http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf
