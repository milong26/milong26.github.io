---
title: latexocr
tags:
  - latex
  - tool
mathjax: false
categories: it
date: 2024-10-28 22:07:18
---


图片转latex，本地版。需要梯子，也可以搞镜像
<!--more-->
参考[github](https://github.com/lukas-blecher/LaTeX-OCR)

可以用conda搭建好环境，我没搞，python 3.7+，我直接本地搞的，如果conda的话参考[csdn](https://blog.csdn.net/ymzhu385/article/details/128757783)

1. `pip install "pix2tex[gui]"` 
2. 后面按道理来说应该能用了但是版本不一样会提示`ImportError: DLL load failed while importing QtCore: The specified procedure could not be found.`
   1. 参考[github issue](https://github.com/lukas-blecher/LaTeX-OCR/issues/330)
   2. 执行以下
   ```
   pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-WebEngine PyQt6-WebEngine-Qt6
   pip uninstall -y PySide6 PySide6-Addons PySide6-Essential shiboken6
   pip install PyQt6==6.5.1 PyQt6-Qt6==6.5.1 PyQt6-WebEngine-Qt6==6.5.1 PyQt6-WebEngine
   pip install PySide6-Essentials==6.5.1 PySide6==6.5.1 PySide6-Addons==6.5.1 shiboken6==6.5.1
   ```
   可能会报错说其中一个没版本，找最接近的我记得是6.5.1.1，然后其它的因为依赖这个版本所以都改成一样的就行
3. 然后命令行 `latexocr`会弹出一个ui界面，成功
4. bat脚本(但是失败了orz)。以后有思路了再搞
   1. 我本来的指令是cmd  /c latexocr
