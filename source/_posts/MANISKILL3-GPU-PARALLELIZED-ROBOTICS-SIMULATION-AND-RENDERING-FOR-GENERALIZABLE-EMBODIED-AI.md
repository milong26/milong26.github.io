---
title: >-
  MANISKILL3: GPU PARALLELIZED ROBOTICS  SIMULATION AND RENDERING FOR 
  GENERALIZABLE EMBODIED AI
tags: generalizable
mathjax: false
categories: paper
date: 2024-11-25 20:55:43
---

一个机器人模拟平台的论文

<!--more-->

论文不重要，这一篇重点还在平台啊，[官网，需要梯子](http://maniskill.ai/)

# abstract
1. 问题：仿真框架不够通用化
2. 解决：ManiSkill3
   1. fastest
   2. 状态可视化 + GPU并行
   3. robotics simulator
   4. with contact-rich physics
   5. targeting generalizable manipulation


# introduction
1. GPU 并行模拟：通过强化学习 
2. 问题：不支持异构模拟，不支持快速并行渲染功能，所以用强化学习时会很慢
3. “ManiSkill3” 的优点
   1. 快速并行渲染和低开销
   2. 环境很全，全部GPU并行化
   3. 异构模拟：ManiSkill3 可以在每个并行环境中模拟和渲染完全不同的对象、关节，甚至整个房间规模的场景
   4. api好用
   5. 从少量演示中生成可扩展的数据生成管道


# realted works
ManiSkill 通过不同方法的组合来获取大规模演示demonstration。

- 对简单任务：使用 RL 的运动规划和奖励来生成演示。
- 对没有简单定义的运动规划脚本或奖励函数的更多复杂任务：依赖于 RLPD和 RFCL等演示算法的在线学习

# Maniskill3的核心特征
1. 支持开箱即用的统一 的GPU 并行化任务：有基于任务的API   这个和第五点差不多吧
2. GPU 并行模拟和渲染
3. 异构 GPU 模拟：这是能够在不同的并行环境中模拟不同对象几何图形、不同数量对象以及具有不同景深的不同关节的功能。
4. 用于机器人操作的 SIM2REAL 和 REAL2SIM：都可以通过数字孪生使用 ManiSkill3来完成
5. 用于构建 GPU 模拟机器人任务的简单统一 API
   1. 用于关节、链接、关节和角色的面向对象的 API
      1. 高级关节/角色，直至单个链接/关节和网格
      2. ManiSkill3 中的姿势信息是面向对象的，并存储为批处理的 Pose 对象
   2. 机器人和控制器：原生支持 URDF 和 Mujoco MJCF 定义格式，并直接基于 URDF/MJCF 构建关节机器人
6. 演示数据集
   1. 对于最简单的任务，我们编写并开源了一些基于运动规划的解决方案来生成演示数据。一些具有易于定义的奖励函数的任务定义了密集的奖励函数，并使用融合的 RL 策略来生成演示数据。
   2. 对于更困难的任务，我们通过远程操作工具收集演示数据（通常约为 10 个演示）。然后，我们使用 RFCL  或 RLPD  来运行快速的在线模仿学习，并从融合策略中生成数据。 