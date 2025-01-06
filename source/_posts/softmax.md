---
title: softmax
tags:
  - softmax
categories: math
mathjax: true
date: 2024-03-06 16:37:12
---

之前学的时候把softmax当作在公式里的点缀，现在来研究一下整个函数
<!--more-->
# 来源
softmax是将各个输出节点的输出值范围映射到[0, 1]，并且约束各个输出节点的输出值的和为1的函数

比起hard(直接求最值)，我们更期望得到文章对于每个可能的文本类别的概率值（置信度），可以简单理解成属于对应类别的可信度。所以此时用到了soft的概念，Softmax的含义就在于不再唯一的确定某一个最大值，而是为每个输出分类的结果都赋予一个概率值，表示属于每个类别的可能性。

# 定义
softmax一般拿来当作激活函数
$$
Softmax(z_i)=\frac{e^{z_i}}{\sum_{c=1}^Ce^{z_c}}
$$
C是输出节点的个数，通过Softmax函数就可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布

softmax求得是一个元素(自变量)的值
# 指数函数
优点：放大变化
缺点：数值可能会溢出
# 损失函数
当使用Softmax函数作为输出节点的激活函数的时候，一般使用**交叉熵**作为损失函数。
由于Softmax函数的数值计算过程中，很容易因为输出节点的输出值比较大而发生数值溢出的现象，在计算交叉熵的时候也可能会出现数值溢出的问题。为了数值计算的稳定性，TensorFlow提供了一个统一的接口，将Softmax与交叉熵损失函数同时实现，同时也处理了数值不稳定的异常，使用TensorFlow深度学习框架的时候，一般推荐使用这个统一的接口，避免分开使用Softmax函数与交叉熵损失函数。
## 推导过程
$$
y_i=softmax(z_i)=\frac{e^{z_i}}{\sum e^{z_m}}
$$
要求$\frac{\partial y_i}{\partial z_m}$有两种情况：令foftmax(z_i)表示为pi，
$$
\frac{\partial y_{i}}{\partial z_{j}} = \begin{cases} p_{i}(1 - p_{j}) & j = i \\ -p_{j}\cdot p_{i} & j\ne i \\ \end{cases}
$$

## 交叉熵损失函数
orz我看不下去了
以后有机会再学吧
[推导参考](https://zhuanlan.zhihu.com/p/105722023)