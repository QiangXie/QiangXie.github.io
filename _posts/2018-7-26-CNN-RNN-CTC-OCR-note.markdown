---
layout: post
title: "CNN+RNN+CTC OCR "
subtitle: "TensorFlow实现笔记"
author: "Johnny"
date: 2018-7-26 19:56:33
header-img: "img/caffe_install.jpg"
tags: 
    -  OCR
---


## RNN、CTC和OCR ##

&#160; &#160; &#160; &#160;OCR（Optical Character Recognition，光学字符识别）它是利用光学技术和计算机技术把印在或写在纸上的文字读取出来，并转换成一种计算机能够接受、人又可以理解的格式。OCR用处很广，例如车牌识别，身份证等证件的识别，文档扫描，等等。早期的OCR技术一般使用传统的图像处理技术，比如灰度化、二值化、找轮廓、分割、识别。这些技术的应用高度依赖算法工程师对数据的理解，而且设计的算法通用性差，基本不可能从一个场景迁移到另一个场景，这些缺点都说明了OCR技术还存在很大的问题。

&#160; &#160; &#160; &#160;随着大数据和并行计算技术的兴起，深度神经网络强大的学习能力开始应用到模式识别的各个子领域。因为其自身的特点，深度神经网络特别适合从大量数据中学习到特定的特征用于特定问题的解决，比如图像识别、目标检测、语音识别等。其中语音识别和OCR识别有很多相似之处：语音和文字一样都具有不定长的特点，而且两者某一位置的预测结果有可能跟前面某一位置的输入信息相关（这里“位置”在语音里对应时刻）。经过证明，RNN在用于语音识别时具有很不错的效果，尤其是LSTM，它通过一个遗忘门使得LSTM能够解决长时依赖的问题。但是RNN应用于序列学习时有一个很大的缺点，它需要对训练数据进行预分割，这使得RNN的应用收到了很大的限制。直到CTC（Connectionist Temporal Classification，连续时间序列分类）的出现。CTC能够直接对未分割的序列进行预测，然后对预测结果按照一定的规则进行映射得到最终的输出结果，实验表明CTC用于语音识别具有非常好的效果。

![java-javascript](/img/in-post/CRNN-TensorFlow/ctc.jpg)

&#160; &#160; &#160; &#160;上面说了OCR和语音识别有着一定的相似性，那是否可以用RNN+CTC的方法进行OCR呢？答案是肯定的。但是，存在一个问题，语音信息是天然的时间序列，文本不是时间序列怎么适应RNN的序列输入要求呢？解决方法是卷积神经网络。卷积神经网络本来就是为图像处理专门设计的人工神经网络结构，使用CNN提取抽象的特征，对特征沿纵向案列切分就可以构造和时间序列类似的特征序列。论文《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition》提出了一种使用CNN提取特征构造序列并使用LSTM+CTC的模型进行文本识别，获得了相当不错的结果。

![java-javascript](/img/in-post/CRNN-TensorFlow/crnn.jpg)

## TensorFlow实现CRNN+CTC的一些经验 ##

&#160; &#160; &#160; &#160;本文是我使用TensorFlow实现一个文本识别模型的记录，对于碰到的一些问题和解决的办法做一个总结。

&#160; &#160; &#160; &#160;如上面图中所示，一个基于CRNN+CTC的文本识别模型一共包含三个部分：CNN特征提取网络，RNN和CTC。其中，基本的卷积神经网络设计要根据实际需要进行设计，这其中涉及到输入图像尺寸，卷积层、池化层、激活函数和BN层的堆叠方式，还有使用多少层的网络。在实际应用中，这个基础网络的设计没有定法，但是有一些需要遵循的原则。网络的复杂度要和训练数据量相适应，如果数据量不够多，那设计的卷积网络应该相应地简单一些，比如使用更少的层数，减少卷积层的使用，以防止过拟合。如果输入图像尺寸比较小，应该较少池化层的使用，防止网络最后输出的特征图尺寸不够，进行分割构造特征序列时导致特征序列的数目小于文本长度就不行了。针对我的识别任务，我的输入图片resize到384*64,网络由四个基础网络模块组成，前三个模块相同，都是卷积层、BN层、ReLu、池化层的方式堆叠，最后一个模块使用卷积层、BN层、ReLu、卷积层、BN层、ReLu、池化层的方式堆叠。这样经过卷积之后得到原图1/16的特征图。在实验过程中，我尝试使用过更多卷积层的网络，但是训练不收敛。关于特征图尺寸，有些技术博客里说最后构造的特征序列长度为要识别的输出结果的长度两倍以上识别结果才比较好，我的实验说明这个要求不是必须的，比如这里我的识别结果长度在20个字符以下，我是用24的序列长度识别结果也很不错。

&#160; &#160; &#160; &#160;使用TensorFlow实现的CNN输出结果时是这样一个顺序：[batch_size, feature_w, feature_h, out_channels]，为了构造序列需要对特征图进行一些转换：

    x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
    x = tf.reshape(x, [batch_size, feature_w, feature_h * self.out_channels])

&#160; &#160; &#160; &#160;这样操作之后获得batch_size*feature_w*(feature_h*out_channels)的特征序列。

&#160; &#160; &#160; &#160;在论文中以及大部分的CRNN实现中都使用了双向RNN，也就是BRNN，这些实现说使用双向RNN不仅能够学到前向（->)的信息，还能学到后向（<-）的信息。但是OCR中有些后面的序列输出跟前面的序列没有明显的关系，比如验证码识别，而且双向RNN的使用会增加计算开销，所以就不需要使用双向RNN。在RNN Cell的选择中，一般使用LSTM，使用GRU也可以获得不错的结果，据说使用GRU还能降低训练的难度。可能跟数据集有关，我使用GRU和LSTM可以获得相似的识别结果。上面的特征序列输入到LSTM之后输出为`[batch_size, feature_w,output_size]`，其中output_size就是构建LSTM时的`num_hidden`。对于LSTM的输出需要做一个线性变换以适应`tf.nn.ctc_loss`的输入，其实就是一个全连接，需要定义一个num_hidden*num_classes的W和一个大小为num_classes的b。使用`A*W+b`并reshape得到[batch_size, feature_w, num_classes]的输入送入`tf.nn.ctc_loss`。代码如下：

    outputs = tf.reshape(outputs, [-1, self.num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]
    W = tf.get_variable(name='W_out',
    	shape=[self.num_hidden, self.num_classes],
    	dtype=tf.float32,
    	initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
    b = tf.get_variable(name='b_out',
    	shape=[self.num_classes],
    	dtype=tf.float32,
    	initializer=tf.constant_initializer())
    
    self.logits = tf.matmul(outputs, W) + b # Reshaping back to the original shape
    shape = tf.shape(x)
    self.logits = tf.reshape(self.logits, [shape[0], -1, self.num_classes])# Time major
    self.logits = tf.transpose(self.logits, (1, 0, 2))

## tf.nn.ctc\_beam\_search\_decoder和tf.nn.ctc\_greedy\_decoder	 ##

![java-javascript](/img/in-post/CRNN-TensorFlow/ctc-decoder.jpg)

&#160; &#160; &#160; &#160;在TensorFlow CTC解码的时候有两种解码方式，分别是beam search和greedy search，其中两个函数接口分别如下：

    tf.nn.ctc_beam_search_decoder(
    inputs,
    sequence_length,
    beam_width=100,
    top_paths=1,
    merge_repeated=True
    )
    tf.nn.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True
    )

beam search时在每一个时间点选择beam\_width个最大的可能类别，然后在每个时间点beam\_width个类别组成的空间里寻找整体概率最大的一条路径，得到最后得识别输出。而greedy search则直接在每个时间点寻找概率最大的类别，然后依次组成这个路径。也就是说，greedy search是beam\_width=1版本的beam search。上图是CTC论文里greedy search示意图。





**参考资料**


 1.[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks][1]

 2.[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition][2]

 3.[tensorflow LSTM+CTC实现端到端的不定长数字串识别][3]

 4.[tensorflow LSTM+CTC/warpCTC使用详解][4]

 5.[github-watsonyanghx-CNN-LSTM-CTC-Tensorflow][5]

 6.[github-ypwhs-baiduyun-deeplearning-competition][6]

 


  [1]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.6306&rep=rep1&type=pdf

  [2]: https://arxiv.org/pdf/1507.05717.pdf

  [3]: https://www.jianshu.com/p/45828b18f133

  [4]: http://ilovin.me/2017-04-23/tensorflow-lstm-ctc-input-output/

  [5]: https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow

  [6]: https://github.com/ypwhs/baiduyun_deeplearning_competition