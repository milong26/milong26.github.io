---
title: git
mathjax: false
tags: git
categories: it
---

<!--more-->

1. remote: Support for password authentication was removed on August 13, 2021.

根据[官方教程](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

1. Settings，最下面的Developer settings，展开Personal access tokens，选中classic那个，Generate new token
2. token名字随便，过期、用途选repo
3. 将令牌复制到剪贴板。关闭页面后不能再出现，除非重新生成！
4. 设置token，使用，填密码的时候粘贴token
5. 将分支克隆到本地的仓库（xxx.github.io）中的.git文件夹复制到博客文件夹中
6. 在博客目录下执行命令同步到远程的hexo分支
   ```
   git add -A
    git commit -m "备份Hexo(提交的描述)"
    git push origin hexo

   ``` 
