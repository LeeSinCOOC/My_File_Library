# OpenCV-Python教程简介
>参考自[官方文档](https://docs.opencv.org/4.1.1/da/df6/tutorial_py_table_of_contents_setup.html)

为了对过往的使用作一个总结，也方便以后自己复习，对官方的示例一一进行了复现学习，第一遍先用python来大致了解一下API，整体熟悉后，准备再用C++过一遍。
## OpenCV
OpenCV由加里·布拉德斯基（Gary Bradsky）于1999年在英特尔创立，第一版于2000年发布。瓦迪姆 ·皮萨列夫斯基（Vadim Pisarevsky）与加里·布拉德斯基（Gary Bradsky）一起管理英特尔的俄罗斯软件OpenCV团队。2005年，OpenCV用于Stanley，该车赢得了2005年DARPA大挑战赛的冠军。后来，在Willow Garage的支持下，它的积极发展得以继续，Gary Bradsky和Vadim Pisarevsky领导了该项目。OpenCV现在支持与计算机视觉和机器学习有关的多种算法，并且正在日益扩展。

OpenCV支持多种编程语言，例如C ++，Python，Java等，并且可在Windows，Linux，OS X，Android和iOS等不同平台上使用。基于CUDA和OpenCL的高速GPU操作接口也在积极开发中。

OpenCV-Python是用于OpenCV的Python API，结合了OpenCV C ++ API和Python语言的最佳质量。

## OpenCV中的Python
OpenCV-Python是旨在解决计算机视觉问题的Python绑定库。

与C / C ++之类的语言相比，Python速度较慢。也就是说，可以使用C / C ++轻松扩展Python，这使我们可以用C / C ++编写计算密集型代码并创建可用作Python模块的Python包装器。这给我们带来了两个好处：首先，代码与原始C / C ++代码一样快（因为它是在后台运行的实际C ++代码），其次，在Python中比C / C ++编写代码更容易。OpenCV-Python是原始OpenCV C ++实现的Python包装器。

OpenCV-Python使用Numpy，这是一个高度优化的库，用于使用MATLAB风格的语法进行数值运算。所有OpenCV数组结构都与Numpy数组相互转换。这也使与使用Numpy的其他库（例如SciPy和Matplotlib）的集成变得更加容易。

## OpenCV-Python教程
本指南主要针对OpenCV 3.x版本，我这里使用的是4.1，如果实现示例的过程中遇到问题，会进行说明。

## OpenCV-Python安装
`pip`安装比较简单，一行命令，确保自己改了`pip`源，不会改的请访问我的`ubuntu16.04装机教程`
```shell
pip install opencv-python
```
在开始之前，请确认一下自己的版本：
```python
In [7]:cv2.__version__
Out[7]: '4.1.1'
```

---

# 教程目录

- [OpenCV简介](https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)

  了解如何在计算机上设置OpenCV-Python！

- [OpenCV中的Gui功能](https://docs.opencv.org/master/dc/d4d/tutorial_py_table_of_contents_gui.html)

  在这里，您将学习如何显示和保存图像和视频，控制鼠标事件以及创建轨迹栏。

- [核心运营](https://docs.opencv.org/master/d7/d16/tutorial_py_table_of_contents_core.html)

  在本部分中，您将学习图像的基本操作，例如像素编辑，几何变换，代码优化，一些数学工具等。

- [OpenCV中的图像处理](https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html)

  在本节中，您将学习OpenCV内部的不同图像处理功能。

- [特征检测与描述](https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html)

  在本节中，您将学习有关特征检测器和描述符的信息

- [视频分析（视频模块）](https://docs.opencv.org/master/da/dd0/tutorial_table_of_content_video.html)

  在本部分中，您将学习与对象跟踪等视频配合使用的不同技术。

- [相机校准和3D重建](https://docs.opencv.org/master/d9/db7/tutorial_py_table_of_contents_calib3d.html)

  在本节中，我们将学习有关相机校准，立体成像等的信息。

- [机器学习](https://docs.opencv.org/master/d6/de2/tutorial_py_table_of_contents_ml.html)

  在本节中，您将学习OpenCV内部的不同图像处理功能。

- [计算摄影](https://docs.opencv.org/master/d0/d07/tutorial_py_table_of_contents_photo.html)

  在本节中，您将学习不同的计算摄影技术，例如图像去噪等。

- [对象检测（objdetect模块）](https://docs.opencv.org/master/d2/d64/tutorial_table_of_content_objdetect.html)

  在本节中，您将学习对象检测技术，例如面部检测等。

- [OpenCV-Python绑定](https://docs.opencv.org/master/df/da2/tutorial_py_table_of_contents_bindings.html)

  在本节中，我们将了解如何生成OpenCV-Python绑定