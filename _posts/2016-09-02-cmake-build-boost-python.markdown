---
layout: post
title: "用cmake编译boost.python库"
subtitle: "boost.python"
author: "Johnny"
date: 2016-09-02 12:15:09
header-img: "img/boost_python.jpg"
tags: 
    - python
    - boost.python
    - C++
---


Python是一种动态，由于它语法简单，使用方便，也用作脚本语言，但是它也有缺点，越高级的语言效率越低，它不适合用作底层对计算效率有要求的语言。C++是一种静态语言，执行效率高，但是也有缺点，那就是它语法比较死板，不如Python灵活。但是，往往实际应用中我们希望底层实现效率高，而上层接口使用简单方便，这样就能兼顾动态语言和静态语言的优点。很多框架中都采用了这种设计思想，比如深度学习框架Caffe等。

Boost.Python是这样一个工具，它能把C++代码封装成Python库供Python调用。至于怎么封装，我看了一些资料发现有的资料提供的是用指令生成Python库，用起来不方便。Boost.Python推荐的使用bjam，它用起来很麻烦。我一直都是用CMake来编译C/C++程序，所以就去找一些cmake的资料，在这里总结如下。

这里有几个要求：


- 编译器用g++
- 系统安装了CMake
- 安装了Boost库
- 安装了Python

# 1. 新建文件夹，设置路径 #

新建一个文件夹命名为src，在src下新建一个子文件夹命名为build，用来放编译过程的中间文件。


    mkdir -p src/build


进入文件夹：

    cd src


为了在生成库之后运行Python代码，并确保"."在环境变量中，执行如下操作：

    $PATH=.:$PATH
    $export PATH


执行上面操作后允许Python导入我们编译好的库。为了防止每次打开一个shell都要设置一遍，可以把以上两行加入到~/.bashrc里面。

# 2. 新建一个CMakeLists.txt #

现在需要新建一个CMakeLists.txt用来让CMake把C++编译成一个Python库，这里我们生成的是.so(Shared Object)文件。下面是一个CMakeLists.txt文件的范本：

    CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
    IF(NOT CMAKE_BUILD_TYPE)
      SET(CMAKE_BUILD_TYPE "DEBUG")
      #SET(CMAKE_BUILD_TYPE "RELEASE")
      #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
      #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
    ENDIF()
    
    FIND_PACKAGE(Boost 1.45.0)
    IF(Boost_FOUND)
      INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
      SET(Boost_USE_STATIC_LIBS OFF)
      SET(Boost_USE_MULTITHREADED ON)
      SET(Boost_USE_STATIC_RUNTIME OFF)
      FIND_PACKAGE(Boost 1.45.0 COMPONENTS python)
    
      ADD_LIBRARY(test SHARED test.cpp)
      TARGET_LINK_LIBRARIES(test ${Boost_LIBRARIES})
    ELSEIF(NOT Boost_FOUND)
      MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
    ENDIF()
    
    IF(CMAKE_COMPILER_IS_GNUCXX)
      ADD_DEFINITIONS("-Wall")
    ELSE()
      MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
    ENDIF()

Linux下生成的库通常命名为libxx.so,xx是CMakeLists.txt文件中指定的名字，比如test，在这里生成的库命名为libtest.so。为了正确编译出我们需要的库文件，必须在INCLUDE_DIRECTORIES包括Boost的include路径和Python的include路径。


# 3. 新建一个test.cpp #


新建一个test.cpp:

    #include <boost/python.hpp>
    
    char const* test()
    {
      return "Hello,world!";
    }
    
    BOOST_PYTHON_MODULE(libtest)
    {
      using namespace boost::python;
      def("test", test);
    }

这里注意：`BOOST_PYTHON_MODULE(libtest)`里的lib一定要和CMakeLists.txt文件中指定的生成名字一样。（也即`ADD_LIBRARY(test SHARED test.cpp)`中的test，生成的库会在指定名字前自动加lib前缀）


# 4. 编译Python库  #

由于编译过程中会生成很多中间文件，所以在最开始才建立一个build文件夹，用来存放这些生成的文件。首先我们需要根据CMakeLists.txt生成gcc编译要用的Makefile：

    $ cd build
    $ cmake ..

生成Makefile只需要执行cmake命令，接着执行make命令生成我们需要的库文件：

    $ make

这样就生成了我们需要的libtest.so文件。

# 5. 新建一个Python文件测试 #

现在我们可以新建一个Python程序测试我们封装的C++程序，新建lib_test.py:

    #!/usr/bin/python
    import libtest
    print libtest.test()

运行以上程序文件会输出：
    
    Hello，world！

当然libtest.so需要和Python文件在同一个文件夹。

这里只是简单的封装一个简单的函数，没有参数传入，没有类等复杂的C++程序，要想封装以上复杂的C++程序可以参考Boost.Python官方提供的tutorial：[http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/tutorial/index.html](http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/tutorial/index.html "Boost.Python Tutorial")

# 6. 参考资料 #


1. [Boost.Python](http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/index.html )
2. [Using CMake to Build C++ Boost Python Libraries](https://www.preney.ca/paul/archives/107#comment-74315 )
3. [boost::python的使用](http://www.cnblogs.com/gaoxing/p/4335148.html )
4. [Boost.Python hello world example using CMake](https://feralchicken.wordpress.com/2013/12/07/boost-python-hello-world-example-using-cmake/ )