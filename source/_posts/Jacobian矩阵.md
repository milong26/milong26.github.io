---
title: Jacobian矩阵
tags: 
categories: math
mathjax: true
date: 2024-04-09 16:34:31
---


在映射过程中，描述不同函数变量间变化速度的导数非常重要，Jacobian矩阵提供了一种表达局部输出对于输入的敏感度的方法。

<!-- more -->
神经网络BP反向传播依赖误差与权重的偏导数关系来训练权重的，神经网络的权重千千万，cost function对哪些权重的变化敏感，无疑那些权重是更重要的，Jacobian矩阵就提供了一个分析神经网络输入-输出行为的数学框架。
当然，Jocobian的应用是极其广泛的，机器学习只不过是冰山一角。
# jacobian矩阵
## 坐标变换
Jacobian矩阵可被视为是一种组织梯度向量的方法。
梯度向量可以被视为是一种组织偏导数的方法。0
故，Jacobian矩阵可以被视为一个组织偏导数的矩阵。

多变量的情况下，坐标变换描述的是从(u,v)到(x,y)连续的1对1变换，
此处 (x,y)是自变量，与上面的(u,v)为自变量的函数互为反函数，可见Jacobian可以是双向的， 一般从积分难度较大指向积分较容易的方向。
## 公式表示
$$
u=u(x,y);v=v(x,y)\\
\Delta u \approx \frac{\partial u}{\partial x}\Delta x+\frac{\partial u}{\partial y}\Delta y\\
\Delta v \approx \frac{\partial v}{\partial x}\Delta x+\frac{\partial v}{\partial y}\Delta y
$$

矩阵形式
$$
\begin{bmatrix}\Delta u \\ \Delta v\end{bmatrix}\approx {\begin{bmatrix}\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{bmatrix}}\begin{bmatrix}\Delta x \\ \Delta y\end{bmatrix}
$$
## 定义
$$
{J(x,y)=\begin{bmatrix}\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{bmatrix}}
$$
