<h1 align="center">MNIST · 手写识别计算器</h1>

<p align="center">
<img src="https://img.shields.io/badge/made%20by-youjiaZhang-blue.svg" >

<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" >
</p>

- 扩展MNIST数据，新增 **+, -, ×, ÷** 以及 **(, )**
- CNN 可视化
- [English README.md](readme/README.en.md)  

**You can also read my [Blog :)](https://youjiazhang.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/mnist/)** 

<div align=center>
<img src="images/main.jpg" align=center/>
</div>

---

主要代码见 [main.ipynb](main.ipynb)，
- python==3.6.9
- Ubuntu 18.04.6 LTS         
- matplotlib == 3.3.4
- tensorflow == 2.6.2
- scikit-learn == 0.24.2
- numpy == 1.19.5
- tqdm == 4.62.3
- opencv-python == 4.5.4.58

## 1. 数据集扩展
MNIST数据集只包含 **0~9** 10个数字的图片。

<!-- <div align=center>
<img src="images/mnist.png" width = "560" height = "284" align=center/>
</div> -->

我们还需要准备四则运算符 **+, -, ×, ÷** 以及 **(, )** 的数据集，将搜集到的扩展数据集进行分类，每一个文件夹存放着一个符号的所有数据。**🌱扩展数据集位于 `models/cfs.tar.xz` 解压缩之后即可使用，包含6个文件夹一共 33895 个图片。数据集收集不易喜欢的话给一个✨吧~**

## 2. 模型训练
采用 CNN 进行预测，模型结构如下所示：
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 28)        280       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 28)        7084      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 22, 22, 28)        7084      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 11, 11, 28)        0         
_________________________________________________________________
flatten (Flatten)            (None, 3388)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               867584    
_________________________________________________________________
dense_1 (Dense)              (None, 16)                4112      
=================================================================
Total params: 886,144
Trainable params: 886,144
Non-trainable params: 0
_________________________________________________________________ 
```

## 3. 目标检测
我们写了一个算式，如下图所示：

<div align=center>
<img src="images/out.png" align=center/>
</div>

针对上面的图片，我们需要有效的识别出每一个符号。针对此场景的特殊性（只有黑白两种颜色）我们可以考虑使用无监督聚类算法进行类别划分。K-MEANS 是比较流行得到聚类算法，操作简单，运行快速，实现容易。但是这个算法并不适合我们这个场景，**因为 K-MEANS 需要事先确定分几个类**，这个是不太好实现的和判断。(不过也不是不可以)

为了解决这个问题我们可以使用 **DBSCAN**(Density-Based Spatial Clustering of Applications with Noise)算法，这是一种著名的基于密度聚类的算法。因为我们手写的字符，局
部的笔画是密集的；而字符与字符之间是相对稀疏的，所以这种算法可以使用。

<div align=center>
<img src="images/DBSCAN.png" width = "700" height = "185" align=center/>
</div>

分类效果还是不错的。然后我们需要将每一个识别出来的符号进行提取，并且处理为 28*28 尺寸的图片便于我们模型的输入。分割之后的图片效果如下：

|(|9|+|2|)|×|3|
|-|-|-|-|-|-|-|
|<img src="images/split/0.png" width = "100" height = "100" align=center/>|<img src="images/split/2.png" width = "100" height = "100" align=center/>|<img src="images/split/6.png" width = "100" height = "100" align=center/>|<img src="images/split/4.png" width = "100" height = "100" align=center/>|<img src="images/split/1.png" width = "100" height = "100" align=center/>|<img src="images/split/5.png" width = "100" height = "100" align=center/>|<img src="images/split/3.png" width = "100" height = "100" align=center/>|

我们将每一个分割后的图片输入模型进行预测，我们将预测的符号类别、预测概率以及符号在原图中的位置进行标注，结果为：
<div align=center>
<img src="images/current.png" align=center/>
</div>

## 4. 表达式计算

我们一般通过 **后缀表达式(逆波兰式)** 进行求值，因为对后缀表达式求值比直接对中缀表达式求值简单很多。**中缀表达式** 不仅依赖运算符的优先级，而且还要处理括号，而后缀表达式中已经考虑了运算符的优先级，且没有括号。主要分为两个步骤：
1. 把中缀表达式转换为后缀表达式
2. 对后缀表达式求值

### 4.1 中缀转后缀
利用一个 **栈** (存放操作符) 和一个**Output**，从左到右读入中缀表达式：
- 如果字符是操作数，将它添加到 Output。
- 如果字符是操作符，从栈中弹出操作符，到 Output 中，直到遇到左括号 或 优先级较低的操作符(并不弹出)。然后把这个操作符 push 入栈。
- 如果字符是左括号，无理由入栈。
- 如果字符是右括号，从栈中弹出操作符，到 Output 中，直到遇到左括号。(左括号只弹出，不放入输出字符串)
- 中缀表达式读完以后，如果栈不为空，从栈中弹出所有操作符并添加到 Output 中。

### 4.2 计算后缀表达式
使用一个栈，从左到右读入后缀表达式：
- 如果字符是操作数，把它压入堆栈。
- 如果字符是操作符，从栈中弹出两个操作数，执行相应的运算，然后把结果压入堆栈。(如果不能连续弹出两个操作数，说明表达式不正确)
- 当表达式扫描完以后，栈中存放的就是最后的计算结果。

## 5. CNN 可解释性
深度学习模型虽然具有很好的预测效果，但其内部的预测原理通常难以解释，往往被看作为是黑盒模型(Black box)。我们使用遮盖法对模型的决策依据进行量化，基本思想：如果将输入特征的值改变会增加模型输出结果的误差，则该特征认为是重要的，因为模型依赖该特征为依据执行预测。反之，如果将特征的值改变模型输出结果的误差很小(或与原结果基本相同)，则该特征不重要，模型不依赖该特征。

我们选择符号 **"9"** 进行解释性计算，其结果如下图所示：

<div align=center>
<img src="images/split/exp.png" width = "280" height = "280" >
</div>

将每一个像素点的 **预测贡献度(重要性)** 在原图上进行可视化展示, 总共有 28*28 个贡献度数值。数值越大图中 **小圆点** 的 size 就越大,颜色也越重。  

## 喜欢的话 给一个 star 吧
