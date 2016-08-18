---
layout: post
title: "caffe 安装"
subtitle: "包括CUDA opencv"
author: "Johnny"
date: 2016-07-20 16:00:09
header-img: "img/caffe_inatall.jpg"
tags: 
    - deep learning
    - caffe
---

caffe是深度学习方面一个框架，主要应用于计算机视觉方面，科研上用caffe的有很多，但是caffe有一个很大的弊端：依赖的库多，安装麻烦。学习用caffe已经有很长一段时间了，也折腾安装过好几次，每次安装过程都是一段血泪史，经常碰到各种各样的问题，每次安装都是网上一顿乱搜。这一次为一台4个Titan X的机器装caffe，装的过程中也是各种错误，系统重装了三次，又咨询了NVIDIA的支持工程师才顺利安装完成。为了自己以后再配置部署环境能有一个参考，这篇博客就把自己的安装过程记录下来，也希望给安装过程中遇到同样问题的其他同学提供一些有用的东西。
#1.安装系统#
我安装的系统版本是Ubuntu 14.04.4 Desktop 64-bit，下载链接在这里：[http://www.ubuntu.com/download/alternative-downloads](http://www.ubuntu.com/download/alternative-downloads)。
具体系统地安装方式在这里就不赘述了。
# 2.安装依赖库 #
caffe里用到很多依赖库，包括ProtoBuffer、GFLAGS、GLOG、BLAS、HDF5、LMDB等等,安装这些库很简单，只需要apt-get install就行了。

    sudo apt-get install --no-install-recommends libboost-all-dev
    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
# 3.安装blas #
BLAS (Basic Linear Algebra Subprograms)是一个基本线性代数运算库，除了BLAS还有Intel的MKL（Math Kernel Library），这个库需要购买或者用edu邮箱申请，相应的这个库比BLAS运算效率高，如果有GPU的情况下这个库就不是很重要了，所以这里我们用免费的BLAS，安装命令如下。

    sudo apt-get install libatlas-base-dev
# 4.opencv #
OpenCV的全称是：Open Source Computer Vision Library。OpenCV是一个基于BSD许可（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows和Mac OS操作系统上。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。caffe里用到很多caffe的函数，包括基本的图像处理，需要执行以下命令：
## 4.1安装cmake等编译opencv需要用到的工具 ##

    sudo apt-get install build-essential
    sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
## 4.2 下载opencv ##

    mkdir ~/opencv
    cd ~/opencv
    wget https://github.com/Itseez/opencv/archive/3.0.0-alpha.zip -O opencv-3.0.0-alpha.zip
    unzip opencv-3.0.0-alpha.zip
	cd opencv-3.0.0-alpha
## 4.3 编译安装opencv ##
	mkdir build
	cd build
	cmake ..
	sudo make
	sudo make install
	sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
	sudo ldconfig
# 5.安装CUDA #
CUDA(Compute Unified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。 CUDA™是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。 它包含了CUDA指令集架构（ISA）以及GPU内部的并行计算引擎。 开发人员现在可以使用C语言来为CUDA™架构编写程序，C语言是应用最广泛的一种高级编程语言。caffe中几乎所有的layer都有GPU实现的版本，利用cuda使得训练速度大大提升。CUDA相应也是比较难以安装的一个依赖包了，这次安装的过程
中出现最多问题的就是CUDA的安装。

在安装之前需要说明一些情况，Ubuntu14.04安装的时候自带了一个名为nouveau的驱动，这个驱动为安装带来了很多麻烦。如果你原本的显示设备是NVIDIA的卡的话，尤其是单GPU的话，那你安装的时候需要先关闭GUI，执行`Ctrl+Alt+F1`后进入tty1，执行`sudo stop lightdm`关闭显示器管理器，禁用旧的显卡驱动：

    cd /etc/modprobe.d/
	sudo vim nvidia-installer-disable-nouveau.conf
输入以下内容，保存并退出：

	blacklist nouveau
	options nouveau modeset=0
打开`/etc/default/grub`，在文件末尾添加：

	rdblacklist nouveau
	nouveau.modeset=0
然后再安装NVIDIA驱动：

	sudo sh ./NVIDIA-Linux-x86_64-352.30.run #把这里替换成你要安装的驱动
安装过程中，根据提示，选择accept, ‘yes’或默认选项。
接着安装CUDA（驱动和CUDA都可以去NVIDIA官网下载），安装CUDA成功之后再执行`sudo start lightdm`,这是比较正统的安装方式。

我也试过NVIDIA普通显卡加两个Titan X不禁用lightdm，直接安装CUDA也成功的经历。（感觉安装CUDA跟拼人品似的）

这次安装就相对曲折很多，这次安装的环境是4个Titan X和主板上自带的非NVIDIA集显。
1 安装时禁用OpenGL，用.run文件安装
2 如果安装错误，进BIOS，更改显示输出口,例如用Titan X作输出。

# 6 caffe安装 #

caffe安装相对简单，可以参考官方安装方法，此处略。