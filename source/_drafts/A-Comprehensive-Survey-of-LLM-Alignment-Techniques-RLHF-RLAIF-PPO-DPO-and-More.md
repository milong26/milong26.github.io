---
title: >-
  A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and
  More
mathjax: false
tags: alignment
categories: paper
---

百篇paper计划(9/100)，对齐这个方向的综述性文章，好好儿看，就是可能会有很多看不懂的

1003开始看，希望今天一天能看完

<!--more-->

- 论文标题：A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and
  More
- code：
- 打标签：对齐
- 时间：2024年7月23日(latest)

按照标题从头到尾翻了一遍，第一印象是这篇文章的公式会非常多orz，那这篇blog不要记太全了，看重点的部分就行了

# abstract
- 问题：训练数据的混合质量会导致结果不好
- 解决：对不同的对齐方法进行总结

# introduction
本文将分成4个主题

1. 奖励模型 Reward Model
   1. explicit RM vs. implicit RM
   2. pointwie RM vs. preference model
   3. Response-level reward vs. token-level reward
   4. negative preference optimization
2. 反馈
   1. Preference Feedback vs. Binary Feedback
   2. Pairwise Feedback vs. Listwise Feedback
   3. Human Feedback vs. AI Feedback
3. 强化学习 Reinforcement Learning
   1. Reference-Based RL vs. Reference-Free RL
   2. Length-Control RL
   3. Different Divergences in RL
   4. On Policy RL vs. Off-Policy RL
4. 优化
   1. Online/Iterative Preference Optimization vs. Offline/Non-iterative Preference Optimization
   2. Separating SFT and Alignment vs. Merging SFT and Alignment


# Categorical Outline
介绍LLM对齐的关键要素，包括4个方向（就是introduction说的4个分类和其下面的子分类）

## 奖励模型
定义：是一个微调的LLM，根据提示和产生的反应来分配分数

### 外显或内隐奖励模型
RLHF：

1. 收集一个由三元组组成的大型数据集，三元组包括一个提示x，一个期望响应yw和一个非期望响应yl
2. 基于收集到的偏好数据集，通过在预训练的LLM上微调来为每个提示信息和响应分配奖励，得到了以r φ ( x , y)表示的显式奖励模型。然后，将该奖励模型用于RL设置中，以对齐LLM策略。相反，以r θ ( x , y)为代表的隐式奖励模型则绕过了训练显式奖励模型的过程。例如，在DPO中，在RL中的最优奖励模型和最优策略之间建立映射，允许LLM对齐，而不需要直接推导奖励模型。

### 逐点奖励或偏好模型

### token级或反应级的奖励模型

### 仅使用消费偏好的训练奖励模型

## 反馈


## 强化学习


## 优化




# Individual Paper Reviews in Detail

## RLHF/PPO




# Future Directions


# 总结
## abstract
这篇文章干啥的：对齐的综述

## introduction
要概括哪些对齐方法：看figure1就可以了

## Categorical Outline
把introduction的4个分类详细说了一下
