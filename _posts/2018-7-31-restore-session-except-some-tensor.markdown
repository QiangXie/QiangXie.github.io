---
layout: post
title: "如何让TensorFlow restore时step为0"
subtitle: "restore except global step"
author: "Johnny"
date: 2018-7-31 19:56:33
header-img: "img/caffe_install.jpg"
tags: 
    - TensorFlow 
---

&#160; &#160; &#160; &#160;在使用TensorFlow训练模型时，常常需要反复修改参数进行多次训练，训练时常常需要对学习率进行衰减，比如使用`tf.train.exponential_decay`对初始学习率进行衰减。但是这种方法存在一个弊端：训练模型时，由于数据多少、模型大小等因素的影响，一般无法提前确定训练模型使用多大的学习率，学习率在训练多少次迭代后衰减，衰减多少。大多数时候需要“调参师”看着loss下降曲线，发现loss不再下降之后自己手动修改学习率。这种情况下，需要对之前训练的结果进行保存，修改学习率，重新载入之前保存的模型，启动训练。这个时候就出现一个问题，使用TensorFlow的`tf.train.Saver`对模型进行载入的时候会一并载入global_step这个Tensor，global是上一次保存训练结果时的全局训练次数，如果想让后续的训练按照一定的规则衰减就不可能了，因为训练step不是从0开始的，虽然可以通过修改decay steps解决这个问题，但是还是想探寻一下是否能把这个global step设置为0呢？

&#160; &#160; &#160; &#160;查了点资料发现这个问题是可以解决的。`tf.train.saver.restore`保存session为checkpoint时是可以指定保存的Tensor列表的，一般保存时使用如下代码：

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

这其中tf.global_variables()返回的是所有variable，包括global step，只需要把global step从列表中剔除再保存，重新载入模型之后global step就会被初始化为0，代码如下：

    all_variables_list = tf.global_variables()
    restore_variables_list = []
    for item in all_variables_list: 
    	if item.name != "global_step:0":
    		restore_variables_list.append(item) 
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

使用上面代码的saver就可以保存除global step之外的变量。