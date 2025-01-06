---
title: 'CLLMs: Consistency Large Language Models'
tags:
  - consistency llm
mathjax: true
categories: paper
date: 2024-10-22 15:59:04
---


百篇paper计划(10/100)，关于持续性，不知道对我有没有启发，先看看吧。

这一篇的实验要求比较高，不精读了，看思想。它有个前提是jacobi方法，没这个基础的话看不太懂。
<!--more-->

- title: CLLMs: Consistency Large Language Models
- rank: In the proceedings of the 41st International Conference on Machine Learning (ICML) 2024
- code: [github](https://github.com/hao-ai-lab/Consistency_LLM)
- tag:
  - 更快
  - 推理过程
- time: 2024年2月28日(latest)
- 评价：感觉挺有意思的，做得实验可真够多。需要的资源太多，代码先不看了，这一篇结束⭐


# abstract
1. 背景：并行解码方法比如jacobi译码 有望实现更高效的llm推断。因为它打破了llm姨妈过程的顺序本质，并将其转化为可以并行的计算。
2. 问题：实际应用中，与传统的自回归( Autoregressive，AR )译码相比，Jacobi译码的加速效果并不明显，这主要是由于Jacobi译码在单个定点迭代步中很少能准确地预测多个令牌。
3. 解决：实现从任意状态快速收敛到雅克比轨迹上的不动点
4. 怎么做到的：**细化目标LLM，从而一致地预测固定点（什么样的：给定任意状态作为输入的）**
5. 实验：测试中，生成速度提高了2.4 ×到3.4 ×，同时保持了生成质量

# introduction
1. llm基础（和本文的解码有关的）
   1. llm的推理延迟不能高
   2. 但是，llm在AR范式下执行服务，由于注意力机制需要前面的令牌状态生成下一格，因此每次都要生成一个token
   3. 为了产生一个长回答，前向执行程序必须以生成的令牌数多次通过LLMs，导致高延迟。
2. 现有方案（其它不管，只谈和本文有关的）
   1. Jacobi解码
      1. 理论来源：求解非线性方程组的Jacobi和GaussSeidel不动点迭代
      2. 思路
         1. 首先从输入提示中随机猜测序列(以下简称n -令牌序列)中的后n个令牌。
         2. 然后将n - token序列连同提示符一起送入LLM进行迭代更新。
         3. 最终，在贪心策略下，n - token序列收敛于AR译码产生的相同输出。n - token序列的演化形成了从随机初始化序列到AR解码(即固定点)生成的n - token序列之间的Jacobi轨迹。
   2. jacobi解码的问题
      1. 问题：vanilla Jacobi译码相对于AR译码仅有边际加速比。
      2. 原因：注意机制使得LLM在其前一个令牌中存在错误（incorrection）的情况下，很少能产生正确的令牌。
      3. 其它解决方法：Lookahead decoding通过利用前几次Jacobi迭代产生的n元词串来提高效率，并在解码过程中进行并行验证。然而，这两项工作都无法完成Meudsa那样的加速。
3. 文本工作（写得有点乱，我尽量总结一下）
   1. 训练一个LLM，只需要一步就可以生成与AR解码(固定点)相同的n - token序列。
      1. 对LLM进行微调，使其可以一次性产生多个而不是一个前缀的后继标记。
      2. 初步实验表明，当n较大时，单步学习任务比较困难，导致模型收敛缓慢。
      3. 因此，我们还考虑了雅可比轨迹上的中间点，考虑了更多的正确标记，从而简化了学习过程。特别地，对于轨迹上的第二个到最后一点，其学习与AR建模相同，此时目标的LLM在没有自适应的情况下已经非常出色。
   2. 这种策略有利于模型收敛
      1. 怎样的学习策略：即通过调整单个模型来解决将轨迹上的任意点映射到固定点的一系列学习问题。
   3. 类比一致性模型
      1. 将n -令牌序列的演化想象为自然图像的去噪过程，我们惊讶地发现，上述学习过程与一致性模型( CMs )扩散模型的加速技术形成了鲜明的类比。
      2. CMs旨在通过最小化训练过程中沿着概率流常微分方程( ODE )轨迹的连续去噪步骤之间的距离，实现使用去噪目标的单步图像生成。
   4. 最终确定的方法
      1. 我们的方法和CMs共享了将求解过程(非线性系统或常微分方程组)的中间状态直接映射到其最终解的概念，用于推理加速。基于此，我们将训练好的模型称为一致性大语言模型( Consistency Large Language Models，CLLMs )。与先前的推测解码和Medusa等方法相比，CLLM没有引入额外的内存开销来容纳辅助模型组件，同时以最小的性能损失提供了显著的加速比。
      2. 为了实现这种学习策略，它只需要两个损失项的模型训练。通过CMs，我们可以将前述的学习目标转化为一致性损失，其中模型被要求将雅克比轨迹上的任意点映射到固定点。CLLMs还包含一个AR损失，以避免偏离目标LLM的分布，从而保证生成质量。
      3. CLLMs的微调代价适中，例如，在Spider数据集上，仅使用LLaMA - 7B的1M令牌进行训练，就可以获得3.4倍的加速比。我们进一步经验性地发现，这种加速很可能源于存在1 )快速转发，即在单次转发中正确地预测到多个连续的令牌；2 )平稳令牌，即在随后的迭代中被正确地预测并保持不变，尽管在此之前有不准确的令牌。
4. 总结贡献
   1. 提出一致性大语言模型( Consistency Large Language Models，CLLMs )，这是一类专门用于降低延迟的Jacobi解码方法的LLMs
   2. 通过实验观察到CLLMs的Jacobi解码中存在快速转发和固定令牌现象
   3. 在各种基准上证明了CLLM的有效性

introduction这一章主要是两个东西：以前的方法（特别指jacobi编码）不好，在于要生成太多token。本文就要解决这个问题，要一步生成那么多token。但是工作那里写得很模糊，我只觉得重点就是一个CLLMs，不知道为什么扯了这么多犊子，结合后文再看吧。


# related work
## 高效的llm推理
大致分为两个方面：需要额外培训的方法和不需要额外培训的方法。LLMs中高昂的AR推断成本引发了旨在高效LLM推断的研究热潮，主要集中在加速AR解码过程。

不需要训练的方法本文没有设计，略。

对于需要训练的方法，它们通常需要集成辅助组件，例如额外的LM或AR头，以促进更快的AR生成。CLLM算是这一部分的，CLLMs既不需要对预训练模型进行修改，也不需要任何辅助组件。这给用户在推理时带来了更高的记忆效率和适应性。

## LLM蒸馏
知识蒸馏( KD )是一种创建较小模型的技术，可以复制较大模型的功能。列举了一些与llm结合的（教师-学生模型这种）蒸馏方法。

CLLMs与这些工作不同，因为我们提出的方法可以看作是一种自蒸馏方法，具有与目标LLM的输出分布相匹配的Jacobi轨迹训练数据集。

## 一致性模型
扩散模型存在迭代采样过程缓慢的问题。一致性模型克服了这一限制，在单步中，沿着扩散过程的概率流ODE将任意点映射回原始点，对应于初始图像。

在本工作中，我们强调了CLLMs的少步生成能力与一致性模型的少步生成能力之间的并行性。

# Methodology
1. 先回顾Jacobi解码方法
2. 详细说明Cllms，它是对预训练LLMs的改进，可以从Jacobi解码中获得更高的加速比。在本文中，我们只考虑贪婪采样，并将其他采样策略留给未来的工作。
3. 实证识别了CLLMs中的快进现象和驻留令牌的存在，这些驻留令牌是这种加速的来源。

## Jacobi解码方法
1. 前任：传统的自回归(AR)一次前向传递（forward pass）只能生成一个token，因为每个y都是根据前面的求出来的，通常可以表示为
   $y_i=argmax_y p(y|y_{<i},x) for i=1,...,n$
2. 定义：Jacobi decoding 是一种用于稀疏图码的迭代译码算法。它的灵感来源于 Jacobi 方法，该方法最初用于求解非线性方程组，是一种并行更新的解码技术。它一步可以产生多个token
3. 具体计算过程
   1. 把前任的yi计算公式换成f=yi-argmax...的形式，就变成f=0的这个非线性方程组
   2. 用Jacobi不动点迭代法并行求解
      1. 从随机初始化一个n-token序列 $y^{(0)}=\{y_1^{(0)},...,y_n^{(0)}\}$开始
      2. 通过以下规则对其进行迭代更新
      $$\begin{cases}y_1^{(j+1)}
      &=\arg\max_yp(y|x)\\y_2^{(j+1)}
      &=\arg\max_yp(y|y_1^{(j)},x)\\
      &\vdots\\y_n^{(j+1)}
      &=\arg\max_yp(y|y_{<n}^{(j)},x).
      &\end{cases}$$
4. 对于LLM的特殊解法
   1. 上述n个最大化问题可以用一个因果注意力掩码来并行求解
   2. 只需要LLM的一个前向通道就可以基于y (j)得到y (j+1)。
   3. 迭代的过程中在存在k，使得y ( k ) = y ( k-1 )，这个时候可以退出了
   4. 我们定义y *：= y ( k )为不动点.令J：= { y ( 1 )，..，y ( k ) }表示雅克比轨迹。
   5. 可以证明，在贪心策略下，y*与AR译码是一致的。
   6. Jacobi译码的加速效果主要源于LLM的每个前向传递可能在n - token序列中产生多个固定的token，因此LLM的查询次数可能小于AR译码，即k≤n。
5. KV缓存下的jacobi解码
   1. LLMs的顺序特性保证了每个令牌的生成只依赖于前一个令牌。也就是说，我们有越来越多的固定令牌，它们与AR生成正确对齐。
   2. 得益于KV缓存技术，我们不需要迭代地更新它们，并重新计算它们的密钥和值，以便在后续的迭代中计算注意力。
   3. 因此有两步：
      1. 逐步减少至少一个令牌的迭代状态长度
      2. 将固定令牌的KV缓存随译码过程一起保存

## CLLMs
1. 问题
   1. 研究现状：在实际用，Jacobi解码对原始llms的加速效果比较差
   2. 原因：AR训练的LLMs在每次Jacobi迭代中通常只能产生一个正确的令牌，因为当前面的令牌不正确时，这些模型很少能产生一个正确的令牌。
2. 解决思路
   1. 使用预训练的LLMs一致地将Jacobi轨迹J上的任意点y映射到固定点y*
   2. 这样的目标类似于一致性模型，这是一种领先的扩散模型的加速方法。
3. 本节
   1. 调整Cllms的数据准备过程
   2. cllm的训练过程
   3. cllm加速的可能原因


### Jacobi轨迹采集
1. 数据收集过程
   1. 令p表示我们所要适应的目标LLM。令q_θ(·|x)表示参数θ用p初始化的CLLM。
   2. 为了实现上述自适应，我们通过运行雅可比解码算法，从某个兴趣域的提示上运行目标LLM p，收集一组雅可比轨迹，形成原始训练集D。
   3. 值得注意的是，为了产生N个( N 远大于 n)令牌的冗长响应l，我们可以依次对n个令牌的每一个截断进行Jacobi解码，以避免对冗长输入进行缓慢的模型评估。
   4. 因此，l相当于一组连续的固定点的串联。
2. 数据增强
   1. 为什么？在一个典型的Jacobi迭代过程中，正确的令牌往往相继出现，且n个token的序列通常表现出"正确，正确，错误，错误，错误"的模式。相比较而言，像"正确，正确，错误，正确，错误"这样的模式可以是罕见的。
   2. 怎么做：为了增强CLLMs的学习和泛化能力，我们通过随机纠正样本中错误预测的标记来扩充数据集D。
3. 数据后处理
   1. 由于目标LLM本身会对某些提示产生误差，往往会导致雅克比轨迹的低质量世代。
   2. 我们发现，训练一个包含n个令牌序列的CLLM，令牌级别的或句子级别的重复往往会导致重复的内容生成，并显著降低性能。
   3. 考虑到高质量数据集对于训练LLMs的重要性，用基于规则的检测器 对训练数据集D进行后处理以剔除低质量样本。
4. 算法
 $$
\begin{aligned}
&\overline{\textbf{Algorithm 1 Generate dataset to train a CLLM}} \\
&\overline{\textbf{Input: prompt set }\mathcal{O},\text{n-token sequence size }n,\max\text{new tokens}} \\
&{c}N\text{, target LLM }p\\
&\textbf{repeat} \\
&\qquad \text{Sample prompt }x\text{ from origin dataset }\mathcal{O}. \\
&\qquad \textbf{while <}\text{EOS> is not generated and length generated}<N \\
&\qquad \text{do} \\
&\qquad \qquad\mathcal{J}=\{\boldsymbol{y}^{(0)},\ldots,\boldsymbol{y}^*\}\leftarrow\text{Jacobi Decoding}(p,\boldsymbol{x}) \\
&\qquad \qquad x\leftarrow\operatorname{cat}(x,y^*) \\
&\qquad \qquad\text{if use data augmentation then} \\
&\qquad \qquad \qquad \textbf{for all }y\in\mathcal{J}\textbf{ do} \\
&\qquad \qquad \qquad \qquad \text{Augment y with false tokens corrected randomly} \\
&\qquad \qquad \qquad \text{end for} \\
&\qquad \qquad \text{endif} \\
&\qquad \qquad\ \mathrm{Append~}x\mathrm{~and~}\mathcal{J}\text{ to Training Dataset }\mathcal{D} \\
&\text{end while}\\
&\text{until all prompts in orgin dataset} \mathcal{O} \text{are used}
\end{aligned}
$$


### 训练
联合优化两个损失来调节CLLM，一个保证一次性预测多个令牌，另一个避免CLLM偏离目标LLM，从而保持生成质量。

#### 一致性损失
1. 全局一致性
   1. 对于雅克比轨迹J的提示x，令y和y *分别表示轨迹和固定点上的随机状态。以y为输入，通过最小化以下损失，可以直接推动CLLM输出y *：
   $$\begin{aligned}
   \mathcal{L}_{\mathrm{GC}}
   &=\mathbb{E}_{(\boldsymbol{x},\mathcal{J})\sim\mathcal{D},\boldsymbol{y}\sim\mathcal{J}}
   \left[\\
   \sum_{i=1}^nD
   \left(q_{\boldsymbol{\theta}^-}(\cdot|\boldsymbol{y}_{<i}^*,\boldsymbol{x})||q_\theta(\cdot|\boldsymbol{y}_{<i},\boldsymbol{x}))\right.
   \right]
   \end{aligned}$$

   2. 其中提到的记号：
      1. Θ - = Stopgrad ( Θ )
      2. E 表示从数据集中均匀采样
      3. D ( · | | · )表示两个分布之间的距离，向前KL，反向KL，以及他们的混合物(即Jensen - Shannon散度)作为流行的例子，实验里面首先是前向KL距离
2. 局部一致性
   1. 也可以实现CLLM在CMs之后以局部一致性( LC )损失一致地将所有中间状态映射到不动点，其中要求雅可比轨迹J中的相邻状态( y ( j )，y ( j + 1 ) )产生相同的输出：
   $$\begin{aligned}\mathcal{L}_{\mathrm{LC}}&=\mathbb{E}_{(\boldsymbol{x},\mathcal{J})\sim\mathcal{D},(\boldsymbol{y}^{(j)},\boldsymbol{y}^{(j+1)})\sim\mathcal{J}}\left[
   \sum_{i=1}^nD\left(q_{\boldsymbol{\theta}^-}(\cdot|\boldsymbol{y}_{<i}^{(j+1)},x)||q_\theta(\cdot|\boldsymbol{y}_{<i}^{(j)},\boldsymbol{x})\right)\right]\end{aligned}$$

3. GC和LC比较
   1. 在表6中对LGC和LLC进行了实证比较，结果表明全局一致性(GC)损失更有效地训练CLLM。
   2. 这可能是由于L_LC只是隐式地通过最小化连续点之间的距离来实现从任意点一致地映射到固定点。
   3. 然而，L_LC距离同时预测多个令牌的目标还有一定差距，因为在收集到的Jacobi轨迹中，y ( j + 1 )中通常只有一个比y ( j )多的正确令牌。

#### AR 损失
为了避免偏离目标LLM的分布，我们在目标LLM p的生成l的基础上加入了传统的AR损失：

$$\mathcal{L}_{\mathrm{AR}}=\mathbb{E}_{(\boldsymbol{x},\boldsymbol{l})\sim\mathcal{D}}\Big[-\sum_{i=1}^N\log q_\theta(l_i|\boldsymbol{l}_{<i},\boldsymbol{x})\Big]$$

这个公式有助于保持生成质量大幅度提升。


所以最后总的训练CLLM的损失为
$$
\mathcal{L}(  \theta  )=  \mathcal{L}_ {consistency}  +  w\mathcal{L}_ {AR} 
$$

其中，ω为权重系数，L的一致性既可以是LGC，也可以是LLC，我们在实验中采用LGC。

## CLLMs中间的加速机制
1. 做实验研究
   1. how：比较目标LLM和CLLM的Jacobi轨迹
   2. result
      1. 怎么做：在Spider上比较目标LLM和CLLM的Jacobi轨迹。
      2. 图像表示方法：沿Jacobi轨迹的每个点都是一个颜色编码序列：蓝色代表与AR结果匹配正确的标记，红色代表不准确的标记。
      3. 结论
         1. 收敛效率：CLLM比目标LLM更快地收敛到不动点2×。可以归因于一致性损失，它有助于学习给定前缀的每个n - token序列的结构。
         2. 快速转发：
            1. 目标LLMs在一次迭代中通常只生成一个正确的令牌；
            2. CLLMs中的快速转发现象，即在单次转发中正确预测了多个连续的令牌。如表3所示，CLLMs中每个前向通道的平均快速前向计数范围为2到6个令牌。
         3. 令牌正确：
            1. 在目标LLMs中，事先正确生成的令牌，在后续的迭代中往往会被不准确地替换。
            2. CLLMs在保证令牌保持不变的情况下，表现出先发制人地预测正确令牌的能力，即使之前有错误的令牌。我们称这样的令牌为平稳令牌，它的存在允许在n -令牌序列中同时扩展不连续的正确令牌。这两种现象都有助于CLLMs的Jacobi解码的快速收敛，从而导致相当大的生成加速比。
2. 脑子凭空想
   1. CLLMs通过训练获得了一个重要的语言概念——搭配collocations：一系列比随机机会所预期的更频繁地共现的词或术语。
   2. 语言不仅仅是由孤立的单词组成，它还在很大程度上依赖于特定的词对。搭配的例子在自然语言和编码语言中都很丰富。它们包括动词+介词组合，动词+名词结构，以及更多领域特定的句法结构。一致性生成目标允许CLLMs从雅可比轨迹中的任意点推断此类结构，从而激励CLLMs获得效率
3. 其它加速机制
   1. 前瞻区间解码(lookahead decoding)收集上一次Jacobi迭代产生的ngram作为候选令牌，并在下一次迭代中进行验证，以加速解码。
   2. CLLMs还可以与前瞻区间解码相结合，实现额外的加速(见表1和表2)，因为在CLLMs中学习到的搭配提高了n元词串的质量，从而提高了接受率。

# 实验
## 评估
### Benchmarks and Setup
评估标准和实验基础设置，不需要过脑子，只要知道下文分析的时候提到的一些仅仅是benchmark而已就行了。

1. 对3个领域特定任务的表现进行了评估
   1. text-to-SQL (Spider)
   2. Python code generation (Codesearch-Python)
   3. graduate school math (GSM8k)
2. 为了测试CLLMs在开放域会话交互和指令跟随场景下的泛化能力
   1. 在ShareGPT2数据上训练了CLLMs
   2. 在MTbench 上进行了评估。
3. 性能指标(metric)为GSM8K上的贪心答案问题解决率( test @ 1 )、MT - workbench评分、Spider上的执行准确率以及Human - Eval上的严格准确率( pass @ 1 )。
4. 在raw - WikiText2和PTB上对CLLMs的语言建模能力进行了评估。
5. 实验均使用预训练的编码器LLM：Deepseek - coder - 7B - instruct或LLaMA - 2 - 7B，视任务而定
6. 训练和评估均在配备8个NVIDIA A100 40GB GPU和128个AMD EPYC 7742 64核处理器的服务器上进行。

### baseline
评估怎么进行的：

1. 将CLLMs与**使用不同策略来加速推理过程**的替代模型进行了比较
   1. 修改底层架构的Medusa
   2. 使用蒸馏草案模型进行推测解码的方法 distilled draft models for speculative decoding
   3. 微调的基线模型
2. 我们的评估是在模型兼容的**不同解码范式**下对每个模型进行测试，以全面评估它们的推理质量和速度。解码算法包括
   1. vanilla AR解码
   2. Jacobi解码
   3. 推测解码
   4. 前瞻解码

### 结果
1. 为了评估CLLMs在不同任务上的性能和推理加速比，我们在3个特定领域(参考benchmark)的任务上与SOTA基线和开放领域的MT-workbench进行了广泛的比较。
2. 表1和表2
   1. 表1和表2干了什么？比较了三种不同生成模式下CLLMs与微调基线模型的差异：AR解码、Jacobi解码、前瞻解码，以及使用蒸馏草案模型的更强推测性解码基线。
      1. 表1：用llama2-7B的
         1. 将CLLMs与其他基线进行比较，包括使用蒸馏草稿模型、Medusa进行推测解码，以及使用LLaMA2 - 7B作为骨干模型的微调模型。
         2. 使用适用的生成技术对性能和推理速度进行了评估。
         3. 为了量化速度改进，加速比定义为每个模型的wall-clock速度与基线AR解码速度的比值。
         4. 结果以批次大小为1进行测量。
      2. 表2：用deepseek-7B的
         1. 使用Deepseek - Coder - 7B - Instruct作为主干模型，将CLLMs与其他基线(baseline一节第一个)进行比较。
         2. 其它基线包括fine-tunes medusa distilled
   2. 结论
      1. 无论是Jacobi解码还是前瞻解码，CLLMs均超越基线。
      2. （表2的结论）在Spider数据集上，使用Jacobi解码，CLLMs以可忽略的性能损失获得了3.4倍的加速比。
      3. 与其他有效的LLM推断SOTA方法相比，特别是那些需要训练的SOTA方法，CLLMs显示出快速一致性生成的能力，同时保持较低的内存和计算需求，（除了fine-tune之外）内存消耗最低。
      4. 在这些情况下，我们仍然可以看到CLLMs在Spider和GSM8K等搭配更常见的数据集上，始终优于蒸馏草案模型的推测性解码，并取得了较好的准确性和相当甚至更好的推理加速。CLLMs还可以与前瞻解码无缝集成，相比于前瞻解码应用于微调后的LLMs获得了更多的加速比。
3. 其它不怎么重点的结论
   1. CLLMs相对于使用蒸馏草案模型的推测解码的优势，Medusa是它的高适应性。这是因为CLLMs是为Jacobi解码量身定做的模型。Jacobi解码不需要对原模型进行任何修改。相反，推测解码和Meudsa都需要LM头、基于树的注意力掩码或草图模型等辅助组件，这通常伴随着搜索最优配置的成本。这一点在表7中进一步总结。
   2. 此外，表5中的语言建模结果表明，CLLMs能够在保持较低困惑度的同时，实现至少2倍的加速比，表明CLLMs有潜力被训练为预训练的LLM，具有更高的推理效率。


## Cllms中的加速机制
1. 目的：考察Jacobi译码中的快进现象和平稳令牌的出现，
2. 怎么做：在四个数据集上比较了目标LLMs和CLLMs中的快速转发和静态令牌计数。
3. 表3
   1. 表3干了啥
      1. 是什么：微调模型和CLLMs中快进和平稳令牌计数的分析结果。
      2. 每个n - token序列都报告了其编号，具有最佳的性能模型和伴随的n - gram大小。
      3. 表中报告的Fast-forwarded令牌计数包括即使没有快进也会被正确预测的一个令牌。
   2. 结论
      1. 在所有四个数据集中，快速转发令牌和固定令牌计数都有2.0 x到6.8 x的一致改进。
      2. 对于特定领域的数据集，这种改进比在MT - bench上构造的开放领域数据集更加显著。

## 消融实验
评估各种超参数选择对CLLMs性能的影响。

### 数据集规模和可推广性
1. CLLMs：收集Jacobi轨迹数据集进行训练，以实现高效的Jacobi解码。
2. 表4：
   1. 干了什么：比较不同大小Jacobi轨迹数据集训练的CLLMs在ShareGPT上的表现
   2. 结论
      1. Jacobi轨迹数据集规模越大，加速比越大，且随着数据集规模的增大，加速比逐渐趋于饱和。
      2. 使用更多数据训练的CLLMs即使在未训练的n - token序列长度下也能表现良好，并引入更多的部署时间鲁棒性。(?没看懂这一个结论是从哪来的)

### 不同长度的n - token序列
1. 要干啥：研究了Jacobi轨迹数据集中不同的n - token序列长度如何影响CLLMs在GSM8K上的性能
2. 怎么做：使用不同的长度来生成Jacobi数据集，并相应地训练CLLMs
3. 图3
   1. 是啥：
      1. 不同n - token序列长度训练的模型在GSM8K数据集上的准确率和加速比。
      2. 生成的序列长度与训练设置相匹配。
      3. 加速比是用Jacobi解码时产生的wall-clock吞吐量与基线AR解码时产生的wall-clock吞吐量的比值来衡量的。
   2. 结论：
      1. 当模型以不同长度训练时，CLLMs始终保持生成质量(准确率不变)。
      2. 在实际应用中，较长的序列长度是以增加推理过程中的计算开销为代价的。当n - token序列长度超过64时，可以观察到显著的退化推断速度。

### 损失设计
1. 怎么设计的：调整了 3.2.2训练 中描述的一致性损失和自回归损失的比例，并评估了不同损失比例在GSM8K上的性能。
2. 表6
   1. 是啥：比较不同损失设计训练的CLLMs的性能。所有模型均在GSM8K上进行训练。
   2. 结论：
      1. 增加对自回归损失的重视确实提高了准确性，尽管它略微降低了加速比的增益。
      2. 此外，比较了同时使用一致性全局损失和一致性局部损失的CLLMs的有效性。
      3. 全局损失在CLLMs的训练中更有效。

## Limitations and Discussion
1. 观察到CLLM在保持良好的生成质量的同时获得显著的加速比强烈依赖于拥有一个高质量的Jacobi轨迹数据集。
2. 因此需要数据清洗
3. 之前提到：数据集大小也发挥了作用，但程度较小。然而，对于像ShareGPT这样的开放域数据集，需要更多的数据来提高效率。
4. 在我们提出的方法和实验中，我们主要使用教师(teacher)的输出序列来收集Jacobi轨迹并训练CLLM。
5. 这与传统的模型训练相比引入了一些额外的开销。在策略GKD建议使用教师和学生样本的混合物甚至学生样本本身进行LLM蒸馏可以产生高性能的模型。
6. 因此，一种缓解方法是使用训练模型本身生成的n - token序列作为训练样本。这可以去除Jacobi轨迹收集开销，使得我们提出的方法在预训练方面具有潜在的可行性。
7. 如表5所示，我们的语言建模实验的结果证明了CLLM在预训练作业上训练时的鲁棒性，并且具有显著的加速比。通过加入在策略GKD，可以设想我们提出的方法的一个修改版本应用于Llm预训练。这样的修改将使预训练的模型既具有现有模型强大的语言建模能力，又能在使用Jacobi解码进行推理时具有较高的生成速度。我们为未来的工作留下了让CLLM适应预先培训的工作的机会。

# conclusion
1. 本工作中做了什么：CLLMs，它在高效并行译码方面表现优异，旨在显著提高Jacobi译码的效率。
2. 现有的高效LLM推断技术缺点：需要额外的架构组件或草案模型。
3. CLLMs：直接由目标预训练的LLM改编而来。降低了与额外的架构设计或在单个系统中管理两个不同模型相关的复杂性。
4. 此外，CLLMs还可以与其他技术无缝集成，实现高效的LLM推理，以获得更大的加速比。
5. 工作证明了CLLMs在特定域和开放域上的有效性，揭示了在保持生成质量的同时显著提高了生成速度。


# self
## 做啥
CLLMs是并行解码器

## 咋做
重点为第三节 methodology

### jacobi
1. 优点：可以并行求解，每步可以预测出多个token，并且在贪婪采样策略下雨ar解码结果一致。所以jacobi好啊，为啥我没看到有人解说它这个方法？
2. 过程（没有仔细阅读jacobi那篇论文，稍微说一下）
   1. 一句话说明一下，把自回归的过程看作是联立方程组求解（即从之前的输入中找概率最大的=输出，并将其加入进行下一步预测）自行迭代的方求解，因为是greedy decoing，所以每次迭代至少能获得一个稳定的token，这样迭代次数肯定小于等于方程的个数
   2. 从输入提示中随机猜测序列的下一个token（以下简称为n -token序列，除非另有说明）
   3. 将n -token序列连同提示一起馈送到LLM中，以进行迭代更新。这个过程会持续进行，直到n -token的序列稳定下来，不再发生变化，达到一个固定点。
   4. 终，n -token的序列会收敛到在贪婪策略下由AR解码生成的输出。
3. 从最初的随机猜测到最终的AR生成结果的这一过程被称为「Jacobi轨迹」。
4. 缺点
   1. 只能用于贪婪解码
   2. 不适合llm：因为当LLM在先前的token中存在错误时，很难产生正确的token
5. 所以要cllm

### cllms
1. 采集jacobi
   1. 收集jacobi轨迹形成D
   2. 数据增强：泛化性，随机纠正D种错误预测的标记
   3. 后处理：低质量，用一个基于规则的检测器处理D
2. 训练
   1. 一致性损失 
      1. 全局一致性
      2. 局部一致性
      3. 二者比较
   2. ar损失
   3. 就是让这两个损失分别最小

### 加速机制
1. 实验研究
2. 其它可能影响的结果

## 实证
第四节的内容：

1. 评估cllm的效果
   1. 任务
   2. 解码算法
   3. 模型骨架
   4. 指标
      1. 加速比
      2. metric
2. Cllms中的加速机制
   1. 目的：快进现象和平稳令牌
3. 消融实验
