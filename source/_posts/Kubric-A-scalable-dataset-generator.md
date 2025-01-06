---
title: 'Kubric: A scalable dataset generator'
tags: 
- data-generate
mathjax: false
categories: paper
date: 2024-11-22 19:48:12
---


到了新环境，安排我看的论文，说是了解一下就行

<!--more-->

代码：https://github.com/google-research/kubric


# abstract
1. 研究啥：合成数据
2. 相关工作为什么不行：
3. kubric是什么：一个python框架
4. 特点：可以与PyBullet 和 Blender 连接

# introduction
kubric

   1. python pipeline
   2. 用于生成逼真的数据，图像和视频
   3. 支持大规模生成
   4. 多功能性

# related work
相比别的合成数据项目，kubric的优点：Kubric 会自动为每一帧生成图像提示，并轻松支持各种视角和照明条件。

1. 专门的合成数据pipeline：缺点是它们通常用于特定任务
2. 通用的数据集创建pipeline：主要区别在于 Kubric 专注于将工作负载扩展到许多工作线程，以及与 TensorFlow 数据集的集成。

# infrastructure
kubric需要充当渲染引擎、物理模拟器、数据导出设施这三者之间的链接

## 设计原则
1. 开源
2. 易于使用：在后台使用 PyBullet 和 Blender 提供一个简单的面向对象的 API 接口
3. 真实：用blender的Cycles 光线追踪引擎，这样生成的数据能和真实数据差不多
4. 可拓展性：可以到云上
5. 便携且可重复：分发 Kubric Docker 镜像
6. 导出数据：注释、SunDs

## kubric worker
kubric工作流程：写一个worker script，然后多次运行worker来生成完整数据集，然后收集生成数据

**scene structure** 每个worker包含：

1. 一个Scene object，跟踪全局设置（例如，分辨率、要渲染的帧数、重力）、
2. 一个 Camera 
3. 其它，统称为Assets。

将Assets添加到场景中时，将在每个视图中创建相应的对象。

**simulator** 物理模拟用PyBullet

**renderer** bpy作为blender的接口

**annotation** blender自带的功能

## assets
1. 问题：要使用集中asset collections的话，需要大量的清理和转换，以使他们与给定的pipeline兼容
2. kubric怎么解决：在一个public google cloud bucket种提供了多个经过预处理的collection of assests
3. 怎么处理collection的？每个数据集源都和一个manifest.json文件相关联

## Scene Understanding Datasets (SunDs)
1. 目的：便于将数据摄取到机器学习模型中
2. 所有 SunDs 数据集都由两个子数据集组成： 
   1. 场景数据集包含高级场景元数据（例如场景边界、整个场景的网格等）。 
   2. 帧数据集包含场景中的各个示例 （例如，RGB 图像、边界框等）。 
3. SunDs 抽象出特定于数据集的文件格式（json、npz、文件夹结构等），并返回机器学习模型（TF、Jax、Torch）可直接摄取的张量。


# Kubric 数据集和挑战
相当于实验设计

每个问题都依赖于不同的注释子集（流、分段、深度、摄像机姿势或对象姿势），使用不同的功能子集（例如，物理或绑定动画），并且需要控制不同的因素（背景、材质或照明）

## object discovery from video
对象发现方法旨在将场景分解为其组成组件，并在最少的监督下查找对象实例分割掩码。


## optical flow
光流是指视频中从一帧中的像素到下一帧的 2D 运动.

光流实际上是计算机视觉中第一个依赖合成数据进行评估的子领域 


## Texture-structure in NeRF


# conclusion
1. kubric
   1. 一个通用的 Python 框架，
   2. 其中包含用于大规模生成的工具，
   3. 集成了来自多个来源的资产、丰富的注释和通用的导出数据格式 （SunDS），用于将数据直接移植到训练管道中。
   4. Kubric 能够生成高质量的合成数据，解决了管理自然图像数据所固有的许多问题，并避免了构建特定于任务的一次性管道的费用。
2. 我们在 11 个案例研究中展示了我们的框架的有效性，并为一系列不同的视觉任务生成了不同复杂程度的数据集。在每种情况下，Kubric 都大大减少了生成所需数据的工程工作，并促进了重用和协作。我们希望它能通过降低生成高质量合成数据的门槛来帮助社区，减少碎片化，并促进管道和数据集的共享。
3. 虽然 Kubric 已经非常有用，但它仍在开发中，尚不支持 Blender 和 PyBullet 的许多功能。值得注意的示例包括体积效果（如雾或火）、软体和布料模拟，以及高级摄像机效果（如景深和运动模糊）。
4. 我们还计划预处理和统一来自更多来源的assets，。目前，Kubric 需要大量的计算资源，因为它依赖于路径追踪渲染器而不是光栅化渲染器。我们希望添加对光栅化后端的支持，允许用户在速度和渲染质量之间进行权衡。我们在补充材料的 A 部分讨论了围绕我们系统应用的潜在社会影响和道德影响。


# 最后总结一下
1. kubric是什么
   1. pythonfarmework
   2. interface with pyBullet and Blender
   3. to generate scenes
   4. with annotation
   5. can scale
2. 构成
   1. kubric worker
      1. scene structure 跟踪所有全局设置
      2. camera
      3. assets created from cloud like google
   2. renderer-blender还有一个annotation的功能
   3. simulator
   4. SunSs：数据集转化成tensor
3. 验证：用kubric生成的数据给这些模型进行预训练
