---
title: hexo建站
tags:
  - hexo
  - blog
categories: it
date: 2024-03-04 15:26:28
mathjax: true
---

个人blog生成日志，防备以后需要

<!--more-->

# prework
1. nodejs安装以及相关安装配置
2. git本地登录

# 安装hexo
`npm i hexo-cli -g`

将`D:\conf\nodejs\node-v20.17.0-win-x64\node-global`加入环境变量Path

`hexo -v`验证

# 初始化
新建文件夹，在这里`hexo init`，会生成若干文件夹。
`hexo g`+`hexo s` 可以预览网页

# 托管到github
新建一个名为`git用户名.github.io`的仓库

修改`_config.yml`最后的deploy
   ```yaml
   deploy:
     type: git
     repository: git@github.com:用户名/用户名.github.io
     branch: main
   ```

安装git插件
`npm install hexo-deployer-git --save`

确认git账号登陆了，执行`hexo d`

正常的话就能搞上去

# 域名配置（可选）
不设置的话可以在`https://用户名.github.io/`看到整个网站

仓库-settings-pages，Custom domain填入自己的域名

域名另外设置一下

/source文件夹里要新建一个CNAME的文件，填上域名

# 文稿
## 新建
`hexo new draft xxx` 生成草稿

修改后再`hexo p xxx`转换成post可见

也可以直接`hexo n xxx`

## 模板
对/scaffolds中的md文件修改

# 主题
默认的landscape

## 使用next
[主题仓库](https://github.com/next-theme/hexo-theme-next)

[主题配置文件](https://theme-next.js.org/docs/getting-started/)

### 安装
`git clone https://github.com/next-theme/hexo-theme-next themes/next`

_config.yml文件中更改theme:
`theme: next`

### 主题修改
   
# 配置
## 显示
1. 首页里面放的正文太多了：使用`<!--more-->`，可以直接放在模板里面。
2. 修改_config.yml：修改author、语言、标题等
3. 按更新顺序排：修改源目录下，\node_modules\hexo-generator-index\lib\generator.js，改成
   ```js
   'use strict';

   var pagination = require('hexo-pagination');

   module.exports = function(locals){
   var config = this.config;
   var posts = locals.posts.sort('-updated'); //修改这里！ 原代码为var posts = locals.posts.sort('-date');
   var paginationDir = config.pagination_dir || 'page';

   return pagination('', posts, {
      perPage: config.index_generator.per_page,
      layout: ['index', 'archive'],
      format: paginationDir + '/%d/',
      data: {
         __index: true
      }
   });
   };

   ```
   

# 其它功能
## 数学公式
测试
$$
e=mc^2
$$
你好$\frac{a}{b}$





# bug及解决方案
## 连接github问题
1. fatal: Could not read from remote repository.
   1. 挂了梯子
   2. git账号没有配置好
      1. 执行ssh-keygen –t rsa –C "git仓库邮箱"，重新生成密钥；
      2. 执行git config --global user.name "git用户名"，重新配置本地用户名；
      3. 执行git config --global user.email "git登录邮箱"，重新配置本地邮箱；
      4. 之后将生成在C:\Users\用户名\.ssh文件夹下的id_rsa.pub文件打开后复制到Git仓库设置—SSH配置—Key配置的地方粘贴即可

我在linux上重新配置了一下，先hexo init，然后再把以前的除了node_modules全拿过来。

1. 尝试hexo d出现ERROR Deployer not found: git：npm install hexo-deployer-git --save


# 多平台同步source
https://blog.csdn.net/qq_30105599/article/details/118302086。

1. 在github仓库里面创建分支，命名为`hexo`
2. 设置hexo分支为默认分支：将博客项目仓库的Settings->Branches->Default branch修改为hexo
3. 将创建的分支的远程仓库克隆到本地 （找个不会干扰的路径）`git clone https://github.com/xxxxx.github.io.git`
4. 删去除.git文件夹以外的所有内容，记得显示隐藏文件（linux可以ctrl+H）
5. 在克隆的仓库下分别执行以下命令更新删除操作到远程
   ```
   git add -A
   git commit -m "--"
   git push origin hexo

   ```

