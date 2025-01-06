---
title: 'Aligner: Efficient Alignment by Learning to Correct'
mathjax: true
tags: aligner
categories: paper
---

百篇paper计划(8/100)，对齐，新的方向，提高准确率的，还是微调上面的。

所以LLM的方向实际上就两种：微调和压缩？

<!--more-->

- 论文标题：Aligner: Efficient Alignment by Learning to Correct
- code：
- 打标签：对齐
- 时间：2024年6月24日(latest)

9/19开始看，希望9/20能看完第一遍

# abstract
1. llm需要alignment method
2. 现有的方法比较复杂，但是需求是快速迭代，所以需要一个可在这些限制条件下运行的与模型无关的alignment方法
3. aligner，是一个alignment paradigm，利用一个小模型，学习首选答案和非首选答案之间的correctional residual 修正残差
4. aligner是一个与模型无关的plug-and-play模块，可以用在各种开源、基于API的模型上面，满足快速迭代的需求，还有其他一堆应用好处
5. 实验是在11个不同的llm中部署相同的aligner模型评估，评估3H 有helpfulness,harmlessness,honesty


Q:

1. 什么是alignment method，有同行吗
   1. alignment是什么？简单的说就是对齐人类的意图，换句话就是人类希望AI产生怎样的行为，那么AI就应该产生什么行为。
1. 修正残差 correctional residual是啥
   1. 矫正残差（correctional residual）通常指的是在统计模型或预测模型中，通过某种方法对原始残差进行修正以提高模型精度的过程
   2. 原始残差？残差在数理统计中是指实际观察值与估计值（拟合值）之间的差


# introduction
1. 背景
   1. 同行有那些：supervised fine-tune(SFT)用人类演示来微调、RLHF根据人类偏好训练奖励模型，用强化学习来微调
   2. 好处：3H的结果不错，
   3. 坏处：消耗太多的训练资源，并且难以保证性能一致性，实际应用时需求是动态变化的，模型可能会遇到对齐训练之外的情况，并表现出不期望的行为，这些行为很难立即使用耗时的方法解决，例如SFT和RLHF。
2. 问题：尝试开发一种高效、轻量、模型无关的对齐方法?
3. 解决
   1. 思路来源：受残差学习的启发，我们通过关注复制和校正操作来简化对齐过程。
   2. aligner是什么：是一个对齐范式alignment paradigm，不涉及RL过程，具体见论文图1
   3. 特性：plug-and-play模块，堆叠在llm的上游
   4. 如何运行：aligner将来自上游模型的初始答案重新分配为更有帮助和无害的答案，从而使合成的LLM响应与人类意图保持一致。可以将aligner类比为LLMs在架构和能力上的残差学习增强器，就像一个通过捷径添加修改而不改变基础结构的残差块一样，aligner采用了复制和更正的方法来改进原始答案。
   5. aligner的优点：保留上游模型参数的同时，增强参数(还是增强模型？)这样能和期望的结果一致
4. 具体aligner怎么设计的
   1. 在偏好数据集 preference dataset上进行微调，以学习偏好和非偏好相应之间的校正残差
   2. 堆叠在上游模型上以实现校正对齐
   3. 这里的上游LLM指的是用于对齐的模型，并将其与RLHF过程中的源模型进行比较。
   4. 与需要训练和加载多个模型的RLHF方法不同，aligner只需要一个额外的模块堆叠在上游的LLM上。而且计算资源只取决于aligner，跟上游llm无关
5. aligner为什么好
   1. 从表示学习的角度看，aligner表现出可解释的残差行为
   2. 如图4所示，aligner在早期层根据原始答案的质量来决定对原始答案的参考程度和额外修正的程度，而它的中期和后期层则用于实现这一决策。
   3. 该机制比直接学习从输入查询到对齐答案的映射更简单。这种简单性表明，小的输纸装置也可以学习复杂的校正模式，证明了它们能够以相对较少的推理来驾驭强大的模型，这进一步强调了我们的输纸装置范式的优越性。
   4. 总结下它的优点
      1. 资源高效：在没有actor, critic, reward, and reference模型等额外模型的情况下，aligner就只是一个在偏好数据集上训练的小模型，用于学习校正残差。
      2. 即插即用：aligner的即插即用特性和模型不可知论使得它非常适合于没有参数访问的基于API的模型。一旦训练好，aligner可以应用于各种上游LLMs，而无需进行参数调整。



Q：

1. 难以保证性能一致性(ensure consistent performance)是什么样的结果？(后面好像有，第二遍看得时候应该可以解答了)
2. 残差学习怎么启发你了？
3. 经常说到的corrected是什么意思，还有校正残差
4. 这一章还不够精简


# aligner
## 在那之前
### SFT监督微调
目的是利用监督学习，特别是最大似然估计，在一个高质量的数据集上对预训练的LLM进行微调以生成目标答案

# 总结重点
1. aligner是什么？一个和模型无关的模块
2. 目的？
3. 好在哪？
4. 怎么做到的？
5. 结果？