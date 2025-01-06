---
title: >-
  GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for
  Robot Manipulation
tags: agent
mathjax: false
categories: paper
date: 2024-11-24 20:39:30
---


<!--more-->
Project page: https://gr2-manipulation.github.io.


是什么：通用机器人agent 可用于多功能和可泛化的机器人操纵
怎么做：首先在大量互联网视频上进行预训练，然后针对使用机器人轨迹的视频生成和动作预测进行了微调



预训练在视频数据集：

1. 目标：给定文本描述和视频帧，模型可以根据文本预测后续帧
2. 基于GPT风格的transformer，将标记化的文本和图像序列作为输入，并输出未来图像的离散标记。


微调在机器人轨迹上

1. 机器人数据包含多个视图
2. GR-2用标记化的语言指令、从多个视图捕获的图像序列以及机器人状态序列作为输入。输出包括每个视图的未来图像和操作轨迹。
3. 全身控制算法(WBC)

