---
title: quaternion四元数
mathjax: false
tags: 3d
categories: it
---

看genesus的代码看得头秃，分不清它的wxyz顺序，学习一下

<!--more-->
四元数定义：q=w+xi+yj+zk

w,x,y,z是实数，ijk是虚数

当用一个四元数乘以一个向量时，实际上就是让该向量围绕着这个四元数所描述的旋转轴，转动这个四元数所描述的角度而得到的向量。

w表示旋转角度 w=cos(theta/2)，其中theta是旋转角，(xyz)是旋转轴
