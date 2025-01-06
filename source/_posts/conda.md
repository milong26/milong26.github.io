---
title: conda
tags: conda
mathjax: false
categories: it
date: 2024-12-19 18:40:56
---

conda
<!--more-->
# 安装
## linux
1. wsl上安装的，版本是18.04
2. 下载[地址](https://repo.anaconda.com/archive/index.html)
3. 安装
   1. 最好换个目录。指令：`wget -c https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh` 下载大概要4分钟
   2. 在放了conda文件的目录下输入命令安装，一路回车，直到他要求输入yes`bash Anaconda3-2023.03-1-Linux-x86_64.sh`，然后输入yes
   3. 退出conda
4. 环境配置
   1. 修改/etc/profile
      1. 因为它是只读的，系统不让修改，换`sudo gedit /etc/profile`
      2. 如果gedit说没有这个指令就`sudo apt-get install gedit`
      3. 在末尾添加环境变量`export PATH=~/anaconda3/bin:$PATH`
      4. 保存退出，gedit可能报错，直接保存
   2. 修改bashrc
      1. `vim ~/.bashrc`
      2. i，加上`export PATH=~/anaconda3/bin:$PATH`
      3. 保存:wq退出
   3. 刷新环境变量
      1. `source /etc/profile`
      2. `source ~/.bashrc`
5. 应该还有个
6. idea配置远程开发的时候有问题，根据https://www.cnblogs.com/hg479/p/17869109.html

