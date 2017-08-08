---
layout: post
title: "Linux下非root用户安装GCC 5.4"
subtitle: "非root用户如何安装软件"
author: "Johnny"
date: 2017-07-31 16:27:09
header-img: "img/Linux.jpg"
tags: 
    - Linux
---


我使用的服务器上的GCC版本是4.7，但是我编译一个项目的源码需要GCC 5.3以上的版本，而我又没有root权限，而且因为别人都用着4.7的版本没有问题，不可能因为我一个人升级GCC，这种情况下有没有办法为自己单独安装一个GCC呢？答案是肯定的。只需要稍微修改一下安装的配置文件就行了。

首先需要下载新的GCC源码，我这里安装的是GCC 5.4，下载之后解压并进入目录。命令如下：

    wget http://mirrors-usa.go-parts.com/gcc/releases/gcc-5.4.0/gcc-5.4.0.tar.bz2
    bzcat gcc-5.4.0.tar.bz2|tar xvf -
    cd gcc-5.4.0

安装GCC需要依赖三个库：GMP, MPFR 和 MPC，只需要在GCC目录下运行`./contrib/download_prerequisites`命令就可以自动下载这三个组件。这里说一下，安装软件配置系统时最好按照官方的安装文档安装，好多二手的教程要么不对，要么会走很多冤枉路，比如这三个组件，有一个教程里一通wget，然后手动编译，费时费力，使用前面这个命令一键解决问题。

接下来是配置编译选项，配置之前要记住很重要的一点，安装GCC不能在它的源码目录下直接配置，需要在它的目录下新建一个文件夹，然后进入这个文件夹配置并编译安装：

    mkdir objdir
    cd objdir
    ../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/path/to/install/gcc-5.4 --enable-threads=posix
    make -j32 && make install

其中，`path/to/install`就是要安装GCC的目录。GCC编译的比较慢，需要稍等一会儿。安装完成后还要进行设置以使系统默认的GCC版本从之前的4.7转到自己安装的版本上。方法是在~/.bashrc或者~/.zshrc（取决于你自己用的shell种类）中添加如下两行并重新启动shell：

    export PATH=/path/to/install/gcc-5.4/bin:/path/to/install/gcc-5.4/lib64:$PATH
    export LD_LIBRARY_PATH=/path/to/install/gcc-5.4/lib/:$LD_LIBRARY_PATH

一定要确保安装路径在`$LD_LIBRARY_PATH`和`$PATH`之前，这样安装的程序才能取代之前系统默认的程序。同样地，也可以安装别的软件到自己的目录下并采用以上方式指定默认程序。

**参考文献**

 1. [Installing GCC][1] 
 2. [gcc和boost的升级步骤(非root权限)][2]
 3. [linux无root权限安装软件 ][3]

  [1]: https://gcc.gnu.org/wiki/InstallingGCC
  [2]: http://blog.csdn.net/u010246947/article/details/42099021
  [3]: http://www.cnblogs.com/yukaizhao/archive/2012/09/03/linux_no_root_make_install.html