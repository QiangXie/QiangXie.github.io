---
layout: post
title: "Deep Learning 之 batch normalization"
subtitle: "batch-normalization"
author: "Johnny"
date: 2017-08-07 18:47:09
header-img: "img/caffe_install.jpg"
tags: 
    - deep learning 
---

一直使用的各种卷积神经网络模型一般都会使用Batch Normalization，虽然经常使用，但是我对其中的细节一直不是很清楚。最近花了点儿时间仔细看了这篇论文[《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》][1]和几篇相关的博客，在这里做一下总结。

**why batch normalization?**

首先考虑为什么要使用batch normalization。自从卷积神经网络（Convolutional Neural Network,CNN）在图像分类任务大放异彩之后，现在CV界基本被
CNN一统江山了。在AlexNet之后涌现出了VGG、GoogLeNet、ResNet等一大批新的网络结构，这些网络结构无一例外越来越深，要不然也对不起deep learning这个称号嘛。但是，随着网络深度的加深，卷积神经网络也越来越难以训练。尤其是使用随机梯度下降（Stochastic Gradient Descent，SGD）的训练方法，超参数特别难调。所以很多研究者开玩笑说：调CNN是一门玄学。

对于CNN来说，训练过程无非是不断进行前向传播计算loss和进行后向传播进行权重更新。但是，由于每次迭代获取到的数据都是随机的，而且在训练过程中参数一直在变化，所以导致CNN中隐藏层的输入的分布也在不停地变化。随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近，如下图所示。激活函数靠近两端的梯度基本为0，后面层会出现梯度消失的问题，这也是深度神经网络收敛越来越慢的本质原因，作者将这个问题称之为“Internal Covariate Shift”，。要解决这个问题，那就要把输入分布拉到相对激活函数曲线来说较为合适的位置。

![java-javascript](/img/in-post/batch-normalization/ac.jpg)

其实解决这个问题的一般方法还有另一个名字，叫做白化（whitening）。白化的方式有好几种，常用的有PCA白化：即对数据进行PCA操作之后，在进行方差归一化。这样数据基本满足0均值、单位方差、弱相关性。那我们要问了，既然已经有解决办法了为什么不采用呢？那是因为：白化过程需要计算协方差矩阵、求逆等操作，计算量很大，此外，反向传播时，白化操作不一定可导。这就导致在每一层隐藏层的输入上使用白化是不现实的。作者分析了以上种种问题后得出的结论就是：要使用batch normalization来改变隐藏层输入的分布，解决“Internal Covariate Shift”的问题。

**How batch normalization？**

Normalization方法其实很简单，就是要输入分布具有0均值和单位方差,如下式：

$$ \hat{x}^{(k)}=\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}} $$

但是如果使用这种简单的Normalization方法又会引入另一个问题，如上图激活函数图所示，这种Normalization方法有可能把分布限制在激活函数的线性区域。本来引入激活函数就是为了使用激活函数的非线性特性，如果仅仅使用线性区域显然会降低整个模型的表达能力，这样就得不偿失了。怎么解决这个问题呢？作者的回答是再引入两个参数，如下式所示：

$$ y^{(k)}=\gamma ^{(k)}\hat{x}^{(k)}+\beta ^{(k)} $$

其中，\\(\gamma\\)和\\(\beta\\)分别是scale和shift两个参数。有的同学看到这个公式可能会大叫：如果这两个参数分别是\\(\sqrt{Var[x^{(k)}]}\\)和\\(E[x^{(k)}]\\),那这个公式不是又变回Normalization之前的样子了吗，那batch normalization的意义何在呢？

其实这个操作还是有意义的，问题的重点就在于\\(\gamma\\)和\\(\beta\\)不是固定的参数，在训练过程中它们会根据实际的BP过程进行变化，最终获得一个较为合适的值，既能使得隐藏层的输入满足具有0均值和单位方差的白化要求，同时又满足在激活函数的输入曲线上均匀分布，不至于只使用激活函数的线性部分导致模型表达能力下降。

道理讲完了，怎么实现呢？请看下图：

![java-javascript](/img/in-post/batch-normalization/formula.jpg)

这是从论文中截取的算法流程图，计算均值和方差时没办法使用全部数据进行，只能使用一个batch的数据计算，这也是batch normalization的batch的由来。BP的求导过程使用链式法则，如下图所示：

![java-javascript](/img/in-post/batch-normalization/bp.jpg)

**实验结果**

如下图所示，实验结果表明，batch normalization可以极大地加速训练过程。

![java-javascript](/img/in-post/batch-normalization/result0.jpg)

其实，作者总结了多个使用batch normalization的好处，包括可以使用更大的学习率，可以不适用LRN等其他normalization方法。实际实验效果也表明，使用batch normalization确实对训练过程有比较大的好处，它也成为现有CNN网络的标配。

**参考资料**

 1. [arxiv-《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》][2]
 2. [ufldl:白化][3]
 3. [解读Batch Normalization][4]
 4. [Batch Normalization导读][5]
 5. [《Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift》阅读笔记与实现][6]
 6. [知乎-深度学习中 Batch Normalization为什么效果好？][7]


  [1]: https://arxiv.org/pdf/1502.03167.pdf
  [2]: https://arxiv.org/pdf/1502.03167.pdf
  [3]: http://ufldl.stanford.edu/wiki/index.php/%E7%99%BD%E5%8C%96
  [4]: http://blog.csdn.net/shuzfan/article/details/50723877
  [5]: http://blog.csdn.net/malefactor/article/details/51476961
  [6]: http://blog.csdn.net/happynear/article/details/44238541
  [7]: https://www.zhihu.com/question/38102762?utm_campaign=webshare&utm_source=weibo&utm_medium=zhihu