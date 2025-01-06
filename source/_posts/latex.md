---
title: latex
tags:
  - latex
mathjax: false
categories: it
date: 2024-07-09 10:19:41
---

写英文论文还是用latex比较好

<!--more-->

# 安装
## 下载
### texlive
1. 下载地址
   1. [清华镜像](https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/)，5.6G需要较多时间

### TexStudio
1. 下载地址
   1. [清华惊险](https://mirrors.tuna.tsinghua.edu.cn/github-release/texstudio-org/texstudio/LatestRelease/)

## 安装
需要先配置好texlive才能安装texstudio

## texlive
1. 在下载好的文件中找到 install-tl-windows.bat文件，进入安装页面
2. 修改安装路径installation root
3. advanced中可以进行更多安装配置
4. 点击安装，大概需要半小时，中途不要点击abort
5. 安装完成后会出现提示“欢迎进去TexLive的世界”
6. 控制台中 `latex -v`测试是否安装成功


## texstudio
直接安装即可

# 使用
## 基础使用
1. 打开texstudio，新建
2. 输入以下代码
   ```
   \documentclass{article}
 
   \begin{document}
   
   Hello, world!
   
   \end{document}
   ```
3. 构建并查看

## 导入模板
以爱思唯尔（ELSEVIER）为例子

1. 下载

## 常用语法
### 样式
1. \mathrm{}：使用\mathrm{...}可以将括号内的字母由数学斜体变为正体，即罗马体（roman type）。