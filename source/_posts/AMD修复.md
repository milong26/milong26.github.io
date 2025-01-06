---
title: AMD修复
tags:
  - amd
  - cuda
mathjax: false
categories: it
date: 2024-10-17 11:31:18
---


想装cuda所以尝试看看电脑上到底有没有NVIDIA

平时用不到这个，以后有需要再整理吧。
<!--more-->


1. 到dell下对应的显卡驱动

2. 下载完安装，出现报错1603

3. 到amd[找](https://www.amd.com/zh-hans/support/kb/faq/gpu-kb1603)

   1. 修复注册表

      window+x 管理员执行，运行两个指令

      ```shell
      DISM /Online /Cleanup-Image /RestoreHealth
      sfc /scannow
      ```

   2. Microsoft Visual C++

      1. 打开程序和功能

         cp控制面板-所有控制功能-程序和功能

   3.  Windows®更新

      [下载](https://www.microsoft.com/en-us/download/details.aspx?id=5753)

