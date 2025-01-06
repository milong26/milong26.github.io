---
title: nodejs
tags:
  - conf
  - nodejs
mathjax: false
categories: it
date: 2024-09-12 11:34:24
---


安装、配置、更新nodejs的流程

但是我现在更新不了……重新配置一遍算了

<!--more-->
# 安装

## 非侵入式

- 下载压缩包http://nodejs.cn/download/ 二进制文件
- 解压后在解压出来的文件夹里面新建两个目录：
  - node-global :npm全局安装位置
  - node-cache：npm 缓存路径
- 配置环境变量Path
  D:\conf\node\node-v14.16.0-win-x64
- cmd 中node -v有输出，🆗
- cmd进入node-v14.16.0-win-x64文件夹
  npm下载慢：更换源
  ```shell
  npm config set prefix "D:\conf\nodejs\node-v20.17.0-win-x64\node-global"
  npm config set cache "D:\conf\nodejs\node-v20.17.0-win-x64\node-cache"
  npm config set registry https://registry.npm.taobao.org
  npm install webpack -g 
  ```

# 更新
## 用n升级
- `node -v` 查看当前版本



# 常见报错



# 卸载nodejs
1、输入命令：npm cache clean --force  

2、从程序中卸载&使用卸载程序的特性（如：控制面板中卸载删除）

3、重新启动（或者您可以从任务管理器中删除所有与节点相关的进程）

4、查找这些文件夹并删除它们（及其内容）（如果还存在）。根据您安装的版本、UAC设置和CPU体系结构，这些设置可能存在，也可能不存在：

(1) C:\Program Files (x86)\Nodejs

(2) C:\Program Files\Nodejs

(3) C:\Users\{User}\AppData\Roaming\npm （或%appdata%\npm）

(4) C:\Users\{User}\AppData\Roaming\npm-cache（或%appdata%\npm-cache）

(5) C:\Users\{User}\AppData\Local\Temp\npm-*

5、检查您的%PATH%环境变量以确保没有引用Nodejs或npm存在。

6、如果是仍然未卸载，键入where node在命令提示符下，您将看到它所在的位置-也删除它(可能还有父目录)。

7、重新启动，很好的措施。