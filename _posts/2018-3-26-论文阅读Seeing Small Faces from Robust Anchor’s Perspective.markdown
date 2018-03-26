---
layout: post
title: "RSA-Net C++复现笔记"
subtitle: "C++实现matlab的左除运算"
author: "Johnny"
date: 2018-1-20 11:03:54
header-img: "img/caffe_install.jpg"
tags: 
    - Computer Vision 
---

&#160; &#160; &#160; &#160;最近一段时间一直在复现一篇论文的代码,论文是ICCV 2017中的一篇论文：《Recurrent Scale Approximation for Object Detection in CNN》[1]。

![java-javascript](/img/in-post/RSA-net-reputation/algorithm_flow_graph.png)

&#160; &#160; &#160; &#160;如上图所示，RSA-Net使用Scale-forecast Network对所给图像的人脸size进行预测，按照Scale-forecast Net预测结果使用RSA（Recurrent Scale Approximation）Unit对图片进行多次前向传播得到对应的尺度的特征图，然后把特征图送入到Landmark Retracing Network预测最后的人脸关键点，然后对关键点按照规则对应得到人脸的矩形框，具体细节参见论文。论文作者在github上放出了源码[2]，但是作者的代码是使用Matlab调用caffe，由于版权问题或者别的其他原因，使用Matlab代码在实际应用中有诸多不便，所以我尝试使用C++对原版的Matlab代码进行复现。由于对于Matlab不甚熟悉，再实现过程中基本是遇到语法问题就查一下，对于源码中意图不甚了解的地方需要运行Matlab代码打印出来猜测代码的意图，中间颇费了一番周折，现对实现过程中遇到的一些问题及解决办法进行记录，以留作后来查阅。

## Eigen求广义逆矩阵 ##


&#160; &#160; &#160; &#160;使用C++复现RSA-Net遇到最大的一个问题就是使用前述三个网络预测到的关键点得到人脸的矩形框过程。Matlab代码在RSA-for-object-detection/predict/utils/get_rect_from_pts.m里如下：

    function rects = get_rect_from_pts(allpoints)
    %from 5 points to rect -- l, t, r, b
    if false
    	rects = fast_get_rect_from_pts(allpoints);
    else
    	std_points = [
    		0.2 0.2
    		0.8 0.2
   			0.5 0.5
    		0.3 0.75
    		0.7 0.75];
    
    	rects = nan(size(allpoints, 1), 4);
    
    	tic;
    	for i = 1:size(allpoints, 1)
    		try
    			points = allpoints(i, :);
    			points = reshape(points, 2, [])';
    			points = points(1:5, :);
    			t = cp2tform(double(points), std_points, 'similarity');
    
   				[xc, yc] = tforminv(t, 0.5, 0.5);
    			[xtl, ytl] = tforminv(t, 0, 0);
    			[xtr, ytr] = tforminv(t, 1, 0);
    
    			w = sqrt((xtl-xtr).^2+(ytl-ytr).^2);
    			rect = [xc-w/2, yc-w/2, xc+w/2, yc+w/2];
    			rect = round(rect);
    			rects(i, :) = rect;
    			if (mod(i, 10000) == 0)
    				fprintf('%d/%d... ', i, size(allpoints, 1));
    				toc;
    			end
    		catch
    			fprintf('Fail to process: %d!\n', i);
    			continue;
    		end
    	end
    end

	end

&#160; &#160; &#160; &#160;上面代码里`t = cp2tform(double(points), std_points, 'similarity');`求网络预测到的五个关键点的坐标到代码里

    std_points = [
    0.2 0.2
    0.8 0.2
    0.5 0.5
    0.3 0.75
    0.7 0.75];

这五个标准点坐标的相似矩阵，这五个点其实就是标准的人脸上面五个关键点坐标。然后利用相似变换求得由（0.5,0.5）、（0,0）和（1,0）三个点对应的由关键点确定的人脸矩形框的位置。这三个点分别是人脸矩形框的中心、左上和右上三个点的坐标。本来以为很简单一个变换，查了一下很多都用opencv的仿射变换实现，照例也使用opencv实现，试过StackOverflow上说的好几种方法，但是变换后的结果就是不一样，求出来的变换矩阵跟Matlab代码求出来的矩阵不一样，一直不知道问题在哪。既然现有的函数变换错误那就尝试自己实现，最开始不知道Matlab的源码能查看，我就查了开源的Octave的实现[3]，Octave是GNU的开源项目，和Matlab的语法函数等使用方法一样。但是，按照Octave的实现依然不对。后来发现Matlab的代码能查看，于是查看Matlab安装路径下的toolbox/images/images/cp2tform.m的代码，严格按照Matlab的代码进行实现，里面涉及矩阵计算的部分使用Eigen这个线性代数库进行计算。然而，这样实现之后结果依然不对，在这个问题上我浪费了四天的时间，一行一行比对C++的代码和原作者Matlab的代码，终于发现问题所在了。

&#160; &#160; &#160; &#160;在cp2tform.m里最重要的一个一步就是`r = X \ U;`，这一句是使用左除求r,通常左除都使用X的逆矩阵和矩阵U相乘，所以这里我使用Eigen里的`inverse`函数求X的逆矩阵，也即：

    r = X.inverse() * U;

&#160; &#160; &#160; &#160;但是求X的逆矩阵要求X是个方阵，而且求相似变换的变换矩阵都是使用三个点进行变换，所以我就也提供了三个点，作者的Matlab代码里则提供了五个点。最大的不同就是这里，所以我查了Matlab的左除实现方法（参考[4]）,发现事情并不是像我想的那样。Matlab里左除的实现方法是求矩阵的广义逆矩阵，并不要求矩阵X必须是方阵，所以这里求相似变换矩阵用到了五个点，而我用了三个点，求得的结果自然不一样。知道问题出在哪，接下来就好办了。

- 在配置libx264编译选项时，需要加入--enable-shared选项；
- 在编译libx265时需要把编译选项中的-DENABLE_SHARED:bool=off改为-DENABLE_SHARED:bool=true；
- 在编译libvpx时在配置编译选项时，加入选项--enable-shared --enable-pic；
- 在编译libfdk-aac时把编译选项中的--disable-shared去掉，改为--enable-shared；
- 在编译libmp3lame时把编译选项中的--disable-shared去掉，改为--enable-shared；
- 在编译libopus时把编译选项中的--disable-shared去掉，改为--enable-shared；
- 在最后编译FFmpeg配置编译选项时，加入--enable-shared选项。

### 2）libvpx的源码需要FQ下载 ###

&#160; &#160; &#160; &#160;如果使用源码编译安装的方式安装以上各个依赖库，其中有一个库libvpx，是VP8/VP9格式的视频的编码器和解码器，它的下载链接是googlesource.com。由于众所周知的原因，它这个链接在大陆是打不开的，需要你使用一些其他的方法才能下载到，下载方法我就不赘述了，FQ应该是一个技术人员的必备技能。

## 2.FFmpeg解码视频保存图片 ##

&#160; &#160; &#160; &#160;笔者主要研究的也不是视频编解码方向，做这个东西主要是用来解码图片方便后续项目使用，所以也就没打算对FFmpeg做深入的研究。经过笔者在GitHub上费了点儿时间的搜索，找到了一个FFMpeg解码视频保存图片的项目，项目地址见参考资料[3]。这个项目是Windows下的一个项目，并且使用的是较为老版本的FFmpeg，有些函数的名字已经有所改变，为了能在Linux下使用还是需要进行一些修改。比如`PIX_FMT_BGR24`需要改为`AV_PIX_FMT_BGR24`,`av_close_input_file`改为`avformat_close_input`等。而且这个项目保存的是bmp格式的图片，里面调用了Windows的API，我要的是Linux下的代码，所以需要对以上这些进行了修改。保存.jpg图片最常见方法是把解码出的YUV图像转为RGB图片，再使用libjpeg库把RGB数据按照JPEG格式写成图片。但是呢，我比较懒，我不想写这么一大坨东西，我选择使用OpenCV的`cv::imread`函数把RGB图像写入硬盘保存为.jpg图片，该函数的优点是可以根据写入文件名的后缀自动选择编码格式，.jpg和.bmp图片都可以保存，所以我参考文献[4]和文献[5]把保存图片函数改为：



    bool Save(const std::string & pFileName, AVFrame * frame, const int & w, const int & h){
		cv::Mat img_mat(h, w , CV_8UC3, frame->data[0]);
		cv::imwrite(pFileName, img_mat);
		return true;
	}

&#160; &#160; &#160; &#160;当然了，要想使用FFmpeg最好还是要了解一点FFmpeg的知识，比如解码的流程，需要调用哪些函数，上面那个工程里的代码为什么这样写，哪些自己用不到是可以删掉的，哪些是必须的…… 了解这些知识最佳途径是看雷神雷霄骅的博客[6]。说到这里不得不提一下雷神，雷神是做视频编解码的尤其是做FFmpeg的大家对雷霄骅的敬称，雷霄骅生前是中国传媒大学的通信与信息系统专业博士生，把雷神称为是国内FFmpeg的第一人一点儿都不为过，他在当今音视频编解码封闭技术领域，专注，勤奋，分享，传播，奉献。在中文音视频技术圈留下他深深的印记。可惜天妒英才，2016年7月17日凌晨雷神猝死在学校主楼五层，知乎上有人说：**上帝也想玩一把直播，就把雷神招进去了....** 

&#160; &#160; &#160; &#160;愿雷神在天堂安息！

## 3.源代码 ##

&#160; &#160; &#160; &#160;代码我已经整理好放到GitHub上了，[地址在这里][7]。如果这个小工程帮到你了，还望你在GitHub上给个star。




**参考资料**


 1. [Recurrent Scale Approximation for Object Detection in CNN-arxiv][1]
 2. [RSA-for-object-detection Matlab 版本 --github][2]
 3. [Octave cp2tform.m源码][3]
 4. [matlab的矩阵左除（A\B）是如何实现的？--知乎][4]
 5. [How to capture frame from RTSP Stream witg FFMPEG Api, OpenCV][5]
 6. [FFmpeg源代码简单分析--雷霄骅][6]

 


  [1]: https://arxiv.org/pdf/1707.09531.pdf
  [2]: https://github.com/sciencefans/RSA-for-object-detection
  [3]: https://sourceforge.net/p/octave/image/ci/default/tree/inst/cp2tform.m#l121
  [4]: https://www.zhihu.com/question/25036509
  [5]: http://hasanaga.info/tag/ffmpeg-avframe-to-opencv-mat/
  [6]: http://blog.csdn.net/leixiaohua1020/article/details/44064715
  [7]: https://github.com/QiangXie/FFmpeg-Decoder-Linux