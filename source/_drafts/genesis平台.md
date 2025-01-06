---
title: genesis平台
mathjax: false
tags: [robotic]
categories: it
---

新物理平台https://genesis-world.readthedocs.io/en/latest/

<!--more-->
# install
版本0.20.0，要求python3.9+，我用3.10

1. conda create -n genesis python==3.10
2. pip install genesis-world
3. 

尝试导入robosuite的mjcf数据（xml格式），需要在geom那一行加 contype="0" conaffinity="0"

quat有bug阿啊啊啊啊