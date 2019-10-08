# 图像的基本操作

## 目标

- 访问像素值并修改它们
- 访问图像属性
- 设置兴趣区（ROI）
- 分割和合并图像
  本节中的几乎所有操作都主要与Numpy相关，而不是与OpenCV相关。要使用OpenCV编写更好的优化代码，需要Numpy的丰富知识。

*（由于大多数示例都是单行代码，因此示例将在Python终端中显示）*

## 访问和修改像素值

让我们先加载彩色图像：

```python
>>> import numpy as np
>>> import cv2 as cv
>>> img = cv.imread('messi5.jpg')
```

您可以通过像素值的行和列坐标来访问它。对于BGR图像，它将返回一个蓝色，绿色，红色值的数组。对于灰度图像，仅返回相应的强度。

```python
>>> px = img[100,100]
>>> print( px )
[157 166 200]
# accessing only blue pixel
>>> blue = img[100,100,0]
>>> print( blue )
157
```

您可以用相同的方式修改像素值。

```python
>>> img[100,100] = [255,255,255]
>>> print( img[100,100] )
[255 255 255]
```

> 警告:Numpy是用于快速数组计算的优化库。因此，简单地访问每个像素值并对其进行修改将非常缓慢，因此不建议使用。

> 注意:上面的方法通常用于选择数组的区域，例如前5行和后3列。对于单个像素访问，Numpy数组方法`array.item()`和`array.itemset()`更好，但是它们始终返回标量。如果要访问所有B，G，R值，则需要分别调用所有的`array.item()`。

更好的像素访问和编辑方法：

```python
# accessing RED value
>>> img.item(10,10,2)
59
# modifying RED value
>>> img.itemset((10,10,2),100)
>>> img.item(10,10,2)
100
```

## 访问图像属性

图像属性包括行数，列数和通道数，图像数据类型，像素数等。

图像的形状可通过`img.shape`访问。它返回行，列和通道数的元组（如果图像是彩色的）：

```python
>>> print( img.shape )
(342, 548, 3)
```

> 注意:如果图像是灰度的，则返回的元组仅包含行数和列数，因此这是检查加载的图像是灰度还是彩色的好方法。

像素总数可通过访问`img.size`：

```python
>>> print( img.size )
562248
```

图像数据类型通过`img.dtype`获得：

```python
>>> print( img.dtype )
uint8
```

> 注意:`img.dtype`在调试时非常重要，因为OpenCV-Python代码中的大量错误是由无效的数据类型引起的。

## 图像投资回报率

有时，您将不得不使用图像的某些区域。为了在图像中进行眼睛检测，首先在整个图像上进行面部检测。当获得一张脸时，我们仅选择脸部区域并在其中搜索眼睛，而不是搜索整个图像。它提高了准确性（因为眼睛总是在脸上：D）和性能（因为我们在小范围内搜索）。

使用Numpy索引再次获得ROI。在这里，我要选择球并将其复制到图像中的另一个区域：

```python
>>> ball = img[280:340, 330:390]
>>> img[273:333, 100:160] = ball
```

检查以下结果：
![](images/roi.jpg)

## 分割和合并图像通道

有时您需要分别处理图像的B，G，R通道。在这种情况下，您需要将BGR图像拆分为单个通道。在其他情况下，您可能需要将这些单独的频道加入BGR图片。您可以通过以下方式简单地做到这一点：

```python
>>> b,g,r = cv.split(img)
>>> img = cv.merge((b,g,r))

```

要么

```python
>>> b = img[:,:,0]

```

假设您要将所有红色像素都设置为零，则无需先拆分通道。numpy索引更快：

```python
>>> img[:,:,2] = 0
```

> 警告:cv.split()是一项昂贵的操作（就时间而言）。因此，仅在需要时才这样做。否则请进行Numpy索引。

## 为图像设置边框（填充）

如果要在图像周围创建边框（如相框），则可以使用`cv.copyMakeBorder()`。但是它在卷积运算，零填充等方面有更多应用。此函数采用以下参数：

- `src`-输入图像
- `top, bottom, left, right`边界的宽度，以相应方向上的像素数为单位
- `borderType`-定义要添加哪种边框的标志。它可以是以下类型：
  - `cv.BORDER_CONSTANT`-添加恒定的彩色边框。该值应作为下一个参数给出。
  - `cv.BORDER_REFLECT`-边框将是边框元素的镜像，如下所示：fedcba|abcdefgh|hgfedcb
  - `cv.BORDER_REFLECT_101`或`cv.BORDER_DEFAULT`-与上述相同，但略有变化，例如：gfedcb|abcdefgh|gfedcba
  - `cv.BORDER_REPLICATE`-最后一个元素被复制，像这样：aaaaaa|abcdefgh|hhhhhhh
  - `cv.BORDER_WRAP`-无法解释，它看起来像这样：cdefgh|abcdefgh|abcdefg
  - `value` -边框颜色，如果边框类型为cv.BORDER_CONSTANT
    下面是一个示例代码，演示了所有这些边框类型，以便更好地理解：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv.imread('opencv-logo.png')
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

```

请参阅下面的结果。（图像与matplotlib一起显示。因此红色和蓝色通道将互换）：
![](images/border.jpg)

# 图像上的算术运算

## 目标

- 学习图像的几种算术运算，例如加法，减法，按位运算等。
- 您将学习以下功能：`cv.add()`，`cv.addWeighted()`等。

## 图像加法

您可以通过OpenCV函数`cv.add()`或仅通过numpy操作`（res = img1 + img2 ）`添加两个图像。两个图像应具有相同的深度和类型，或者第二个图像可以只是一个标量值。

> 注意:OpenCV加法和Numpy加法之间有区别。OpenCV加法是饱和运算，而Numpy加法是模运算。
> 例如，考虑以下示例：

```python
>>> x = np.uint8([250])
>>> y = np.uint8([10])
>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]
>>> print( x+y )          # 250+10 = 260 % 256 = 4
[4]
```

当添加两个图像时，它将更加可见。OpenCV功能将提供更好的结果。因此，始终最好坚持使用OpenCV功能。

## 图像融合

这也是图像加法，但是对图像赋予不同的权重，以使其具有融合或透明的感觉。

```python
img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

检查以下结果：
![](images/blending.jpg)

## 按位运算

这包括按位与，或，非和异或运算。在提取图像的任何部分（如我们将在后续章节中看到），定义和使用非矩形ROI等方面，它们将非常有用。下面我们将看到有关如何更改图像特定区域的示例。

我想在图像上方放置OpenCV徽标。如果添加两个图像，它将改变颜色。如果混合它，我将获得透明效果。但我希望它不透明。如果是矩形区域，则可以像上一章一样使用ROI。但是OpenCV徽标不是矩形。因此，您可以按如下所示进行按位操作：

```python
# Load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()
```

请参阅下面的结果。左图显示了我们创建的遮罩。右图显示了最终结果。为了进一步理解，请显示以上代码中的所有中间图像，尤其是img1_bg和img2_fg。
![](images/overlay.jpg)

# 绩效评估和改进技术

## 目标

在图像处理中，由于每秒要处理大量操作，因此必须使代码不仅提供正确的解决方案，而且还必须以最快的方式提供代码。因此，在本章中，您将学习

- 衡量代码的性能。
- 一些提高代码性能的技巧。
- 您将看到以下功能：`cv.getTickCount`，`cv.getTickFrequency`等。
  除了OpenCV，Python还提供了一个模块**时间**，这有助于衡量执行时间。另一个模块**配置文件**有助于获取有关代码的详细报告，例如代码中每个函数花费了多少时间，函数被调用了多少次等。但是，如果您使用的是IPython，则所有这些功能都集成在用户友好的界面中方式。我们将看到一些重要的信息，有关更多详细信息，请查看“ 其他资源”部分中的链接。

## 使用OpenCV衡量性能

`cv.getTickCount`函数返回从参考事件（如打开机器的那一刻）到调用此函数一刻之间的时钟周期数。因此，如果在函数执行之前和之后调用它，则会获得用于执行函数的时钟周期数。

`cv.getTickFrequency`函数返回时钟周期的频率或每秒的时钟周期数。因此，要找到执行时间（以秒为单位），您可以执行以下操作：

```python
e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
```

我们将通过以下示例进行演示。下面的示例应用中位数过滤，其内核的奇数范围为5到49。（不用担心结果会是什么样，这不是我们的目标）：

```python
img1 = cv.imread('messi5.jpg')
e1 = cv.getTickCount()
for i in xrange(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print( t )
# Result I got is 0.521107655 seconds
```

> 注意:您可以使用时间模块执行相同的操作。代替cv.getTickCount，使用time.time（）函数。然后取两次相差。

## OpenCV中的默认优化

许多OpenCV功能已使用SSE2，AVX等进行了优化。它还包含未优化的代码。因此，如果我们的系统支持这些功能，则应该加以利用（几乎所有现代处理器都支持它们）。默认在编译时启用。因此，OpenCV如果启用了优化代码，则将运行优化代码，否则它将运行未优化的代码。您可以使用`cv.useOptimized()`来检查是否启用/禁用它，并使用`cv.setUseOptimized()`来启用/禁用它。让我们看一个简单的例子。

```python
# check if optimization is enabled
In [5]: cv.useOptimized()
Out[5]: True
In [6]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 34.9 ms per loop
# Disable it
In [7]: cv.setUseOptimized(False)
In [8]: cv.useOptimized()
Out[8]: False
In [9]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 64.1 ms per loop
```

请参阅，优化的中值滤波比未优化的版本快2倍。如果检查其来源，则可以看到中值滤波已进行SIMD优化。因此，您可以使用它在代码顶部启用优化（请记住默认情况下已启用）。

## 在IPython中评估性能

有时您可能需要比较两个类似操作的性能。IPython为您提供了一个神奇的命令计时器来执行此操作。它会多次运行代码以获得更准确的结果。同样，它们适用于测量单行代码。

例如，您知道以下哪个加法运算更好，x = 5; y = x ** 2，x = 5; y = x * x，x = np.uint8（[5]）; y = x * x或y = np.square（x）？我们将在IPython shell中使用timeit找到它。

```python
In [10]: x = 5
In [11]: %timeit y=x**2
10000000 loops, best of 3: 73 ns per loop
In [12]: %timeit y=x*x
10000000 loops, best of 3: 58.3 ns per loop
In [15]: z = np.uint8([5])
In [17]: %timeit y=z*z
1000000 loops, best of 3: 1.25 us per loop
In [19]: %timeit y=np.square(z)
1000000 loops, best of 3: 1.16 us per loop
```

您可以看到x = 5; y = x * x最快，比Numpy快20倍左右。如果您还考虑阵列的创建，它可能会快100倍。酷吧？*（大量开发人员正在研究此问题）*

> 注意:Python标量运算比Numpy标量运算快。因此，对于包含一两个元素的运算，Python标量比Numpy数组要好。当数组大小稍大时，Numpy会占优势。

我们将再尝试一个示例。这次，我们将比较同一图片的`cv.countNonZero()`和`np.count_nonzero()`性能。

```python
In [35]: %timeit z = cv.countNonZero(img)
100000 loops, best of 3: 15.8 us per loop
In [36]: %timeit z = np.count_nonzero(img)
1000 loops, best of 3: 370 us per loop
```

可以看出，OpenCV功能比Numpy功能快25倍。

> 注意:通常，OpenCV函数比Numpy函数要快。因此，对于相同的操作，首选OpenCV功能。但是，可能会有例外，尤其是当Numpy处理视图而不是副本时。

## 更多IPython魔术命令

还有其他一些魔术命令可以用来测量性能，性能分析，行性能分析，内存测量等。它们都有很好的文档记录。

## 性能优化技术

有几种技术和编码方法可以充分利用Python和Numpy的性能。此处仅记录相关内容，并链接到重要资源。这里要注意的主要事情是，首先尝试以一种简单的方式实现该算法。工作正常后，对其进行概要分析，找到瓶颈并对其进行优化。

- 尽量避免在Python中使用循环，尤其是双/三重循环等。它们本来就很慢。
- 由于Numpy和OpenCV已针对向量运算进行了优化，因此将算法/代码向量化到最大程度。
- 利用缓存一致性。
- 除非需要，否则切勿制作数组的副本。尝试改用视图。阵列复制是一项昂贵的操作。
  即使执行了所有这些操作后，如果您的代码仍然很慢，或者不可避免地需要使用大循环，请使用Cython等其他库来使其更快。