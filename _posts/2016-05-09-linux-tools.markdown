---
layout: post
title: "some linux tools"
subtitle: "zsh tmux vim htop"
author: "Johnny"
date: 2016-05-09 16:10:08
header-img: "img/bg-2.jpg"
tags: 
    - Linux
---

因为笔者本科是学电子的，所以大概去年的这个时候，我还对Linux一点儿也不了解，我一直以为操作系统就是Windows那样的，写程序就该是用VS那样的IDE。所以，刚接触Linux的时候我极端不适应，我可以说是很排斥用Linux，我当时很赞成曾经看到的一篇文章：使用Linux的程序员都是受虐狂。尤其是用shell，这怎么用啊？难道操作电脑不该是点击鼠标吗？还有写代码，vim更是变态啊，竟然有人用这种东西，简直无法理解。虽然Linux下现在也有了图形界面，但是跟Windows比这是渣渣好吗……

后来，因为一些原因不得不用Linux，慢慢的，从开始连cd命令都不知道，后来慢慢地适应了Linux，开始习惯用命令行操作，尤其是在无法用GUI的情况下，慢慢体会到shell的强大。当然了，要想适应Linux还是需要一段时间的，尤其是从没接触过Linux的用户从Windows转过来。不过，经过一段时间的使用我觉得程序员是会喜欢上Linux的。就像《The Unix-Haters Handbook》里的一句话：“The fundamental difference between Unix and the Macintosh operating system is that **Unix was designed to please programmers**, whereas the Mac was designed to please users. (Windows, on the other hand, was designed to please accountants.”Linux就是一种类Unix的操作系统，Linux之所以说是为程序员设计的是因为它非常灵活，你可以用它实现任何东西，如果你不爽某个工具，你可以把它换掉，自己造一个换上去。对于具有Geek精神的程序员来说，他们很乐意开发一些轮子来方便自己也造福后来者。与此同时，因为这些好用而略显杂乱的工具的存在，使得Linux的使用学习成本也更大一些，可以选择的更多人越不知道怎么选择。

所谓工欲善其事，必先利其器。所以，借鉴别人的经验挑选一些好用的工具是很重要的，下面是一些我觉得好用的工具，虽然很多我用得还不是很熟，而且很多强大的功能我还没有用到，但是还是记录下来，为了方便自己（因为自己如果换一个环境就要重新配一遍，还要翻别人的博客，挺麻烦的），所以按照我的使用习惯记录下来。另外，如果别人觉得有用的话也可以略微看一下，以后如果遇到好用的工具我还会记录到这篇博客里。

**1.htop**

htop 是Linux系统中的一个互动的进程查看器，一个文本模式的应用程序(在控制台或者X终端中)，需要ncurses。

与Linux传统的top相比，htop更加人性化。它可让用户交互式操作，支持颜色主题，可横向或纵向滚动浏览进程列表，并支持鼠标操作。

与top相比，htop有以下优点：

可以横向或纵向滚动浏览进程列表，以便看到所有的进程和完整的命令行。htop还可以查看cpu占用率之类的详细信息，总之我觉得htop比top好用。
htop的界面如下：

![这里写图片描述](http://img.blog.csdn.net/20160509142504878)

htop的安装比较简单，直接可以用apt-get安装：


    sudo apt-get install htop


**2.zsh**

Ubuntu默认的shell是bash，但是zsh是一个更加强大的shell，但是因为配置太复杂，虽然很好用但是用的人不多，直到有一个叫做[oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)的开源项目，它把使用zsh的难度降了下来，接下来我们看看zsh怎么安装和配置。
Mac默认直接有zsh的，但是Ubuntu默认没有安装zsh，需要我们自己安装：


    sudo apt-get install zsh

然后安装oh-my-zsh：


    git clone git://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

更改默认shell：


    chsh -s /bin/zsh


然后退出重新进入shell，就可以使用zsh。zsh的补全功能非常强大，切换目录时甚至可以不用cd直接输入想要切换的目录，当然了zsh的功能远不止此，具体使用可以参阅[终极 Shell](http://macshuo.com/?p=676)，里面有更详细的zsh功能介绍。

**3.tmux**

[tmux](https://tmux.github.io/)是一个优秀的终端复用软件，即使非正常掉线，也能保证当前的任务运行，这一点对于 远程SSH访问特别有用，网络不好的情况下仍然能保证工作现场不丢失!此外，tmux完全使用键盘 控制窗口，实现窗口的切换功能。来看一个tmux的使用截图：

![这里写图片描述](http://img.blog.csdn.net/20160509145314180)

这里，我把窗口切分为三个窗口，一个打开vim，其他的进行浏览文件，运行调试程序之用。当然了，如果你乐意，你可以把窗口切分成任意个，只要你屏幕足够大。在没用tmux之前我一直是在图形界面上开一堆终端窗口，然后等到使用时半天找，tmux极大地方便了我们使用shell。
tmux另外一个非常赞的功能是保存工作现场，如果你用shh连接服务器工作，网络不是那么好，有时会掉线，掉线之后之前进行的工作可能就丢失了，有了tmux之后你就不用担心了，掉线之后tmux依然在后台运行，重新连接之后你只要运行tmux attach，工作现场就回来了。
当然，tmux还有其他一些功能，同样的你也可以个性化定制自己使用tmux的习惯，只需要配置~/.tmux.conf文件，具体的使用方法请参阅[tmux的使用方法和个性化配置](http://mingxinglai.com/cn/2012/09/tmux/)

**4.vim插件**


在Linux下写程序，尤其是在没有图形界面时，一般我们使用vim写程序，当然了Emacs也是非常优秀的编辑器，这个看个人习惯选择自己喜欢的编辑器。vim是一个比较经典的编辑器（这里强烈推荐一个vim教程：[简明 Vim 练级攻略](http://coolshell.cn/articles/5426.html)），但是如果不对vim进行一些配置的话，可能会觉得vim用起来很不方便，很反人类。好在有很多大神为大家制作好用的轮子（也就是各种插件），把这些插件用上之后，vim一点儿也不比IDE差，先上一个vim的配置图：
![这里写图片描述](http://img.blog.csdn.net/20160509151840436)

这个配置可以在[这里](https://github.com/humiaozuzu/dot-vimrc)找到。更强大的配置还有[spf13-vim](https://github.com/spf13/spf13-vim)。
这些配置很强大，插件也很多，但是有些我并不需要，我最需要的是以下四个插件：vundle，NERD Tree，YouCompleteMe，Vim Powerline。

**vundle**

[vundle](https://github.com/VundleVim/Vundle.vim)是一个vim插件管理工具，它能够搜索、安装、更新和移除vim插件，再也不需要手动管理vim插件。
安装vundle：


    git clone https://github.com/gmarik/vundle.git ~/.vim/bundle/vundle


在.vimrc里面加上如下配置：


    set nocompatible  " be iMproved, required
    filetype off  " required
    
    " set the runtime path to include Vundle and initialize
    set rtp+=~/.vim/bundle/Vundle.vim
    call vundle#begin()
    " alternatively, pass a path where Vundle should install plugins
    "call vundle#begin('~/some/path/here')
    
    " let Vundle manage Vundle, required
    Plugin 'VundleVim/Vundle.vim'
    
    " The following are examples of different formats supported.
    " Keep Plugin commands between vundle#begin/end.
    " plugin on GitHub repo
    Plugin 'tpope/vim-fugitive'
    " plugin from http://vim-scripts.org/vim/scripts.html
    Plugin 'L9'
    " Git plugin not hosted on GitHub
    Plugin 'git://git.wincent.com/command-t.git'
    " git repos on your local machine (i.e. when working on your own plugin)
    Plugin 'file:///home/gmarik/path/to/plugin'
    " The sparkup vim script is in a subdirectory of this repo called vim.
    " Pass the path to set the runtimepath properly.
    Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
    " Install L9 and avoid a Naming conflict if you've already installed a
    " different version somewhere else.
    Plugin 'ascenator/L9', {'name': 'newL9'}
    
    " All of your Plugins must be added before the following line
    call vundle#end()" required
    filetype plugin indent on" required
    " To ignore plugin indent changes, instead use:
    "filetype plugin on
    "
    " Brief help
    " :PluginList   - lists configured plugins
    " :PluginInstall- installs plugins; append `!` to update or just :PluginUpdate
    " :PluginSearch foo - searches for foo; append `!` to refresh local cache
    " :PluginClean  - confirms removal of unused plugins; append `!` to auto-approve removal
    "
    " see :h vundle for more details or wiki for FAQ
    " Put your non-Plugin stuff after this line


当然，以上Plugin后面不是必须的，这是举例说明使用不同类型的插件如何添加，你可以选择你需要的插件加到上面这些行里面。配置好.vimrc文件之后，打开vim，运行`:BundleInstall`或在shell中直接运行`vim +BundleInstall +qall`，就开始安装你的各种插件了：

![这里写图片描述](http://img.blog.csdn.net/20160509153852740)

具体使用方法还可以参阅[Vim配置、插件和使用技巧](http://www.jianshu.com/p/a0b452f8f720)。

    NERD Tree

[NERD Tree](https://github.com/scrooloose/nerdtree)是一个树形目录插件，方便浏览当前目录有哪些目录和文件。

![这里写图片描述](http://img.blog.csdn.net/20160509154427295)

我在~/.vimrc文件中配置NERD Tree，设置一个启用或禁用NERD Tree的键映射`nmap <F5> :NERDTreeToggle<cr>`，这样就可以在vim里点击F5打开或者关闭NERD Tree了。

**YouCompleteMe**

用过VS的大家可能都对VS的代码补全功能记忆深刻，其实vim下一样可以进行代码补全，YouCompleteMe就是这样一个插件。[YouCompleteMe](http://valloric.github.io/YouCompleteMe/)是一个快速、支持模糊匹配的vim代码补全引擎。

![这里写图片描述](http://img.blog.csdn.net/20160509155018762)

YouCompleteMe的安装稍微麻烦一些，它需要在vundle插件执行插件安装之后对YouCompleteMe进行编译，执行以下命令：


    cd ~/.vim/bundle/YouCompleteMe
    ./install.py


当然，对于不同的的语言支持需要不同的方法，具体可以参考[github](https://github.com/Valloric/YouCompleteMe)。

**Vim Powerline**

另一个插件是Vim Powerline，[Vim Powerline](https://github.com/Lokaltog/vim-powerline)是一个显示vim状态栏插件，它能够显示vim模式、操作环境、编码格式、行数/列数等信息。

![这里写图片描述](http://img.blog.csdn.net/20160509160512838)

暂时我觉得好用的工具就是这些，以后觉得别的工具还会在这里更新，最后感谢那些造这些轮子并开源贡献给大家的那些大神们，如果你也用了这些工具，不妨在github上给他们点个star，算是对他们的感谢。

**5.参考资料**


1. [终极 Shell](http://macshuo.com/?p=676)
1. [tmux的使用方法和个性化配置](http://mingxinglai.com/cn/2012/09/tmux/)
1. [Vim配置、插件和使用技巧](http://www.jianshu.com/p/a0b452f8f720)
1. [spf13/spf13-vim](https://github.com/spf13/spf13-vim)
1. [VundleVim/Vundle.vim](https://github.com/VundleVim/Vundle.vim)
1. [Valloric/YouCompleteMe](https://github.com/Valloric/YouCompleteMe)