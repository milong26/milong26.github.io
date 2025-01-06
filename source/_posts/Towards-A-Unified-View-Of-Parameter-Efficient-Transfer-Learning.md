---
title: Towards A Unified View Of Parameter-Efficient Transfer Learning
tags:
  - peft
categories: paper
mathjax: true
date: 2024-03-18 16:16:31
---

高效参数微调的综述类论文阅读

<!-- more -->

百篇paper计划(2/100)，对【高效参数】迁移学习的总结。

由于我也不怎么会选论文，就按照会议选了这篇比较近的(ps现在是2024年初)。光从标题看，这一篇应该是综述，正好我补一下几个经典的高效参数方法。

- 论文标题：Towards a Unified View of Parameter-Efficient Transfer Learning
- arxiv地址：[链接](https://arxiv.org/abs/2110.04366)
- code：[github](https://github.com/jxhe/unify-parameter-efficient-tuning)
- rank: ICLR 2022 (spotlight presentation)

# abstract
- 大背景：
  在**下游任务**上**微调**大规模的、预训练的模型 在**NLP**领域里面用得很多。但是传统方法**全参数微调**，当模型参数和任务个数增加时变得很难搞。
- 小背景（现在工作）：
  高效参数迁移学习(parameter-efficient transfer learning)方法只微调小规模参数，但是几种方法之间缺少系统的联系
- 贡献
  分析了几种最先进的高效参数迁移学习方案，给出一个统一的框架。 
- 具体：
  将这几种方法概括为修改预训练模型中特定的**隐藏状态**，并定义一组设计维度来总括性地设计方法。
  通过跨机器翻译、文本摘要、语言理解和文本分类基准的综合实证研究，我们利用统一的观点来识别以前方法中的重要设计选择。
- 实验：
  统一框架实现了设计元素在不同方法之间的迁移，因此我们能够实例化新的参数高效的微调方法，这些方法比以前的方法调整更少的参数，同时更有效，在所有四个任务上都能获得与所有参数微调相当的结果。

DDDDDDDDDDDDDDD# introduction
## 大背景
在预训练的语言模型PLMs基础上进行迁移学习在NLP领域比较流行。
为什么不要全参数微调：模型越来越大+需要对每个任务都微调。

## 小背景
一些方法更新小部分参数，让大部分参数不变。

### adapter tuning
在预训练网络每层中插入一些小的神经模块**adapters**
微调时只训练adapters

### prefix tuning 和 prompt tuning
从 通过文本提示控制PLM的提示方法(prompting methods that control PLMs through textual prompts) 出发，给输入或者隐含层预置了一个额外的可调前缀标记l，并且在下游任务中预训练时，只训练这些soft prompt
(这个soft prompt应该是刚刚预置的可调标记)

### LoRA（low-rank matrices to approximate parameter updates）
学习**低秩矩阵**来近似参数更新。

### 总结
这些方法在不同的任务集上表现出与全微调相当的性能，通常通过更新不到1%的原始模型参数来实现。
除了节省参数外，高效参数的调整使其能够快速适应新任务而不会遗忘。并且在分布外评估(out-of-distribution evaluation)中表现出优异的健壮性。


## 问题产生
为什么这些高效参数的调优方法会成功，这些方法之间有什么联系。3个问题：

1. 以上3种方法方法有什么联系
2. 这些方法对对有效性在设计上有没有共同之处
3. 能否将每种方法的有效成分迁移给其他方法，以产生更有效的变异体?

## 文章结构
1. 首先推导了前缀调优(prefix tuning)的一种替代方法，揭示了前缀调优与适配器(adapter)的密切联系
2. 贡献：设计统一框架
   1. 是什么：基于1，设计了统一的框架，将上述3个方法总结为不同的方式来修改固定PLMs的隐藏表示。
   2. 怎么做到的：我们的统一框架将以前的方法沿着一个共享的设计维度集合set进行分解，例如用于执行修改的函数、施加修改的位置、如何集成修改。
   3. 有什么用：这个框架允许我们在各种方法之间迁移设计选择，以提出新的变体，例如具有多个头的适配器
3. 实验
   1. 现有的PEFT方法在资源较高且有挑战性的任务上仍然落后于全参数微调
   2. 使用统一的框架来识别关键的设计选择，并验证变体
4. 结果
   我们在涵盖文本摘要、机器翻译、文本分类和通用语言理解的四个NLP基准上的实验表明，所提出的变体使用了比现有方法更少的参数，同时更有效，在所有四个任务上都匹配了完全微调的结果。

# 预备知识
## transformer结构
transformer(2017)：

1. transformer的构成
   L个堆起来的块，每个块有两种子层：多头自注意力+一个全连接前馈网络(FFN)
2. 传统注意力：
   queries$Q\in R^{n\times d_k}$,key-value pairs $K\in R^{m\times d_k},V\in R^{m\times d_v}$,
   $$
   Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
   $$
  其中n是queires的个数，m是key-value对的个数
3. 多头注意力
   在$N_h$个头上并行计算注意力函数，其中每个头分别被$W_q^{(i)},W_k^{(i)},W_v^{(i)}\in R^{d\times d_h}$参数化，为了将输入映射为queries、keys和values
4. 多头注意力的操作流程
   给一个序列，包含m个向量$C\in R^{m\times d}$,希望对其计算注意力；给一个查询向量query vector $x\in R^d$，多头注意力MHA计算每个头上的输出，并把它们串联起来
  $$MHA(C,x)=Concat(head_1,...head_h) W_o,\\
  head_i=Attention(xW_q^{(i)},CW_k^{(i)},CW_v^{(i)})
  $$
  其中$W_o\in R^{d\times d}$，d是模型的维度,MHA里的$d_h$一般被设置为$d/N_h$以保存参数，这表明每个注意力头都在低维空间上计算
5. 前馈神经网络FFN
   由2个线性变换组成，中间有个ReLU激活函数
   $$
   FFN(x)=ReLU(xW_1+b_1)W_2+b_2
   $$
   其中$W_1\in E^{d\times d_m},W_2\in R^{d_m\times d}$
   transformer模型通常用比较大的$d_m$
6. 残差连接+层归一化

## 之前的PEFT方法
注：除非另有规定，否则它们只在PLM被冻结时调整所添加的参数。

### Adapters

#### 基础
在transformer层之间插入小模块(adapter)。adapter层一般采用$W_{down}\in R^{d \times r}$的**向下投影**，将输入h投影到瓶颈维度r(bottleneck dimension)指定的低维空间，后接非线性激活函数f (·)，和一个$W_{up}\in R^{r \times d}$的向上投影。这些adapter被一个残差连接包围，导致最终的形式：
$$h\leftarrow h+f(hW_{down})W_{up}$$

#### 2种变体
1. 将两个适配器依次放置在transformer的一个层内，一个在多头注意力之后，一个在FFN子层之后。
2. 一种更高效的变体，仅在FFN的add & layer norm子层后插入

### Prefix tuning
受文本提示方法的成功启发，前缀调优预先将l个可调的前缀向量放在每层多传感头注意力的键和值中。
具体来说，它将两组前缀向量$l\times d$维$P_k,P_v$和原始的keyK、valueV进行链接。然后堆芯的前缀key、value进行多头注意力计算，公式为：
$$
head_i=atten(xW_q^{(i)},concat(P_k^{(i)},CW_k^{(i)}),concat(P_v^{(i)},CW_v^{(i)}))
$$

> 就是以前的多头注意力公式上加了Pk和pv

### LoRA
LoRA将可训练的低秩矩阵注入到transformer层来近似权重更新。
对一个预训练的权重矩阵$W\in R^{d\times k}$来说，LoRA用低秩分解过程$W+\Delta W=W+W_{down}W_{up}$来更新，这里的$W_{down}\in R^{d\times r},W_{up}\in R^{r\times k}$都是可调节的参数。
LoRA将这一步更新应用在多头注意力子层中的query、value投影矩阵(也就是$W_q,W_v$)中。
对多头注意力中线性投影的特定输入x，lora将投影输出h修改为
$$
h\leftarrow h+s\cdot xW_{down}W_{up}
$$
这里$s\geq1$，是一个可调的标量超参数。

### 其它方案
- 仅对预训练模型中的偏置向量进行微调的BitFit
- 学习稀疏参数更新向量的diff - pruning 

# 统一视图
我们首先推导出前缀调优的等效形式，以建立其与适配器的连接。
然后，我们提出了一个统一的参数高效调优框架，该框架包括几个最先进的方法作为实例化。

## Prefix tuning细节
之前prefix tuning那一节提到的公式(atten)描述了前缀调优的机制，通过将l个可学习的向量（就是pk和pv，它们都是lxd维度的）前置到原始的注意力query和value来改变注意力模块。
对刚刚那个公式等价转换，把i去掉：
![前缀调优公式推导](https://ooo.0x0.ooo/2024/03/06/OycyFU.png)

有几个问题：

1. concat是怎样连接的？
   分头之后的维度：
   - $P_v:l\times d/N_h$
   - $C:m\times d$
   - $W_v:d\times d_h$
   - d是模型的维度，MHA中一般设置$d_h=d/N_h$，$N_h$是头的个数。m是key-value对的个数。所以最后Concat连接的两个向量的列数相同。
2. softmax求的是什么
   
   参考{% post_link softmax 'softmax' %}

   公式里面softmax(一个行向量)指的是求出每个项对应的softmax值，最后输出还是一个同样大小的行向量
3. 它没有除以根号d
   
公式推导出来之后，前缀调优的本质就是通过线性插值对原始头注意力输出h进行位置上的修改，也就是
$$
h\leftarrow (1-\lambda(x))h+\lambda(x)\Delta h\\
\Delta h:=softmax(xW_qP_k^T)P_v
$$

### vs Adapters联系
改写刚刚的h，定义：
$$
W_1=W_qP_k^T,W_2=P_v,f=softmax\\
h\leftarrow(1-\lambda(x)h+\lambda(x)f(xW_1))W_2
$$
整个公式和adpters中的特别相似
$$
h\leftarrow h+f(hW_{down})W_{up}
$$
除了prefix tuning是进行加权相加的，但是adapter没有加权

适配器中的h和前缀调优通常是不同的，下面将详细介绍。然而，这里我们主要讨论适配器原则上可以插入任意位置的功能形式。

因此可以将prefix-tuning抽象为和adapters一样的一种插件。

除此以外，当l比较小时$W_1\in R^{d_h\times l},W_2\in R^{l\times d_h}$都是low-rank矩阵（低秩），因此也可以近似到adapters里的$W_{up}W_{down}$
  
同样可以得到：前缀向量的个数l与适配器中的瓶颈维度r有类似的作用：
都表示计算修改向量$\Delta h$的秩限制。因此，我们也可以将l叫做瓶颈维度。

直觉上，秩限制意味着对于任意x，∆h是相同l (或≤l)个基向量的线性组合。

### vs adapters的区别
除了门控变量$\lambda$外，还有3个区别：
1. prefix tuning用x、PLM层的输入计算$\Delta h$；adapters用h和plm层的输出。
因此，前缀调优可以被认为是对PLM层的"并行"计算，而典型的适配器则是"顺序"计算。
2. adapters在插入位置上比前缀调优更灵活：适配器通常修改注意力或FFN输出，而前缀调优只修改每个头的注意力输出。
3. 刚刚推导的公式$h\leftarrow(1-\lambda(x)h+\lambda(x)f(xW_1))W_2$适用于每个注意力头，but适配器总是单头的，so前缀调优更有表现力：
   
- 注意力头是$d/N_h$维度
- 当l≥d / Nh时，我们基本上对每个注意力头都有满秩更新
- 但当r≥d时，我们只对使用适配器的整个注意力输出进行满秩更新。
- 值得注意的是，当l = r时，前缀调整并不比适配器增加更多的参数。

> lora是不是被作者吃了

## 统一框架
因为prefix tuning和adapters之间的联系，我们也提出一种统一框架。

具体：学习一个修改向量$\Delta h$，并将其应用于各种隐藏表示。

在形式上，将要直接修改的隐表示记为h；将计算h的【PLM子模块的直接输入】表示为x (例如, h和x可以分别作为注意的输出和输入)。为了表征这种修改过程，我们定义了一组设计维度，不同的方法可以通过这些维度上的不同值来实例化。
![示意图](https://ooo.0x0.ooo/2024/03/11/OyuQwF.png)

> 以下只把4个维度的大概说了一下，对每种算法的具体实现还在后面的章节
 
### Functional form
是用来计算$\Delta h$的特定函数。

所有这些方法的functional form都类似这样的架构：
proj_down -> nonlinear -> proj_up

而其中的nonlinear在LoRA中就退化为恒等函数。

### Modified Representation
指示哪些隐藏表示被直接修改

### Insertion Form
表示添加的模块怎样插入到网络中的。

传统上，apapters以顺序的方式插入再一个位置，其中输入和输出都是h，prefix tuning和lora虽然最初不是以这种方式描述的，最终也相当于一个并行插入，其中x是输入。

### Composition Function
关于如何将修改后的向量$\Delta h$与原始的隐藏表示h进行组合的，从而形成新的隐藏表示的。例如：

- adapters执行简单的加法组合；
- prefix-tuning使用门控加法组合，也就是这个公式：$h\leftarrow(1-\lambda(x)h+\lambda(x)f(xW_1))W_2$；
- Lora通过一个常数因子对h进行缩放，并将其添加到原始的隐藏表示中，也就是$h\leftarrow h+s\cdot xW_{down}W_{up}$

此外其他方法也可以用这个框架代入，关键的是，该统一框架允许我们沿着这些设计维度研究参数有效的调优方法，确定关键的设计选择，并可能在不同方法之间转移设计元素。

## 迁移设计元素
仅仅描述了几种新颖的方案，什么样的方案：可以通过迁移设计元素从上述的统一框架推导出来的。
1. Parallel Adapter 并行的适配器
   是将prefix tuning的并行插入转换为adapters的变体。

   有趣的是，我们在激励并行适配器的同时，由于其与前缀调优的相似性，协同工作独立地提出了这一变体并进行了实证研究；(题外话)
2. Multi-head Parallel Adapter 多头平行适配器
   是使适配器更类似于前缀调优的进一步步骤
   我们使用并行适配器来修改头部注意力输出作为前缀调优。


   这样，变体利用3.1中讨论的多头投影，提高了空闲容量。
3. Scaled Parallel Adapter
   通过将LoRA的组成和插入形式转移到适配器中

到目前为止，我们的讨论和表述提出了几个问题：

1. 改变上述设计元素的方法是否表现出不同的性质?
2. 哪些设计维度尤为重要?
3. 上述新方法是否产生更好的性能?
   
我们接下来回答这些问题。

# 实验
## 通用设置
### 数据集
研究了四个下游任务

1. XSum 是一个英文摘要数据集，其中模型预测给定新闻文章的摘要
2. 使用WMT 2016 en - ro数据集进行英译罗马尼亚语翻译；
3. MNLI 是一个英文自然语言推理数据集，其中模型预测一个句子是否包含、矛盾或中立于另一个句子。
4.  SST2是一个英语情感分类基准，模型预测句子的情感是积极的还是消极的。

### 设置
使用BART LARGE和它的多语言(multilingual)版本mBARTLARGE 分别作为XSum和en - ro翻译的底层预训练模型，RoBERTa BASE用于MNLI和SST2。
我们根据需要在{ 1，30，200，512，1024 }内改变瓶颈维度。

我们主要研究适配器、前缀调优( prefix )和LoRA，它们在我们的实验中大大优于bitfit和prompt tuning。

在分析部分中，我们在注意力层或FFN层插入Adapters以方便分析，但在最后的比较中包括了在这两处插入的结果。
我们基于各自的公开代码重新实现了这些方法，使用huggingface 的transformers库来实现

### 评估
- 我们在XSum测试集上报告了ROUGE 1/2/L得分
- 在en - ro测试集上报告了BLEU得分
- 在MNLI和SST2 dev集上报告了准确率。
对于MNLI和SST2，我们取5次随机试验的中位数，还报告了相对于完全微调(#params)的参数的数目。
 
### 可调参数的数量
BART和mBART具有编码器-解码器结构，具有3种类型的注意力：编码器自注意力、解码器自注意力和解码器交叉注意力。
RoBERTa只有编码器自注意力。
对于每个注意力子层，每种方法使用的参数个数是：

1. 前缀调优，将l个向量前置到键和值，使用2 × l × d个参数；
2. 适配器具有Wdown和Wup，因此使用2 × r × d参数；
3. LoRA使用一对Wdown和Wup进行查询和值投影，因此使用4 × r × d参数。对于ffn处的适配器修改，使用了2 × r × d的参数，与注意处的适配器相同。

因此，对于特定的r或l值，前缀调优使用与适配器相同数量的参数，而LoRA使用更多的参数。更多细节见附录B。

## 现有方法的结果
我们首先概述了现有方法在四个任务上的结果。
（四个任务：指数据集章节中提到的四个下游任务：XSum、en-ro翻译、MNLI、SST2）
![实验结果](https://ooo.0x0.ooo/2024/03/12/Oy3Tds.png)
虽然现有方法在MNLI和SST2（表2）上通过调节不到1%的参数就可以获得有竞争力的性能
但是如果在XSum和en - ro（图4）中增加5 %的参数，仍然存在很大的差距。即使我们将相对参数的大小提高到> 10 %，这种差距依然显著。其他实验表示高资源MT任务上差距更大。

许多生成只要用编码器模型就能在GLUW基准测试集上，或者用编码器-解码器模型在相对简单的生成基准测试集上达到和全参数微调相近的方法，它们可能无法很好地推广到其他标准基准上。

影响因素可能是复杂的，包括训练样本数量、任务复杂度或模型架构等。因此，我们提倡今后对这一领域的研究，以报告在更多样化的基准上的结果，以展示其性能概况的更完整的画面。

下面，我们的分析将主要集中在**XSum和en - ro**数据集上，以更好地区分不同的设计选择。我们注意到这两个基准是使用编码器-解码器模型(BART)执行的较高资源，而我们将在§4.6中讨论仅使用编码器模型(RoBERTa)在MNLI和SST2上的结果。

## 插入形式
我们首先研究了**插入形式(Insertion Form)**的设计维度，将提出的并行适配器(PA)变体与传统的顺序适配器(SA)在注意力(att)和FFN修改进行了比较。我们将prefix tuning作为参考点。

> 为什么用这个作为参考点？
> PA将prefix tuning的并行插入转换为adapters；SA是lora变成adapters的变体，干嘛非得prefix，不能用adapters吗
 
如表3所示，使用并行插入(parallel insertion)的前缀调优优于注意力顺序适配器(attention sequential adapters)。
此外，在所有情况下，并行适配器(parallel adapter)都能战胜顺序适配器(sequential adapters)，在XSum上PA(ffn)比SA(ffn)高出1.7个R-2点，在en-ro中则是0.8个BLEU点。
考虑到并行适配器比顺序适配器具有更好的结果，我们在下面的部分中关注**并行适配器**的结果。

## 修改表示
### 设置
我们现在研究修改不同表征的影响。我们主要对**注意力和FFN**修饰进行比较。
为了便于分析，我们将修改注意力子层(例如,头部输出、查询等)中任何隐藏表示的*方法*归类为修改注意力模块。

我们比较了并行适配器(parallel adapters)在注意力和FFN和prefix tuning（因为上一节刚说过并行适配器好）。

我们还将FFN的修改转移到LoRA上，得到LoRA ( ffn )的变体，以便进行完整的比较。
具体来说，我们使用LoRA来近似FFN权重$W_1\in R^{d\times d_m}$和$W_2\in R^{R^{d_m\times d}}$的参数更新。在这种情况下，LoRA中的Wup定义为W1(类似于Wdown到 W2)，维数为r × dm，其中dm = 4d。

因此，在后面的实验中，我们通常使用比其他方法更小的r来匹配LoRA (ffn)的总体参数大小。

### 结果
![结果](https://ooo.0x0.ooo/2024/03/12/OyCekb.png)
如图5所示，任何**有FFN修正**的方法在所有情况下都优于所有有注意力修正的方法(红色标记普遍高于蓝色标记,唯一的例外是ffn - PA ,为2.4 %参数)，往往参数更少。
其次，在FFN中应用的相同方法总是比其注意力对应物要更好。例如，LoRA ( ffn )在XSum上将LoRA ( attn )提高了1个R - 2点。
我们还强调，当我们进一步增加容量时，前缀调优并不会继续改善，这在Li & Liang ( 2021 )中也观察到了这一点。
这些结果表明，不管是功能形式functional form还是组成函数composition function，FFN修饰都能比注意力更有效地利用所增加的参数。
我们假设这是因为FFN学习的是任务特定的文本模式( Geva等, 2021)，而注意力学习的是两两位置的交互，不需要较大的容量来适应新的任务。

### 用0.1%参数的时候有什么不同
在§3.1中，我们认为前缀调优比适配器(attn)更具有表达力，但这一点在图5中没有体现。我们猜想这是因为多头注意力只有在参数预算较小时才具有优越性。
为了验证这个假设，我们将前缀调优与并行适配器进行了比较，给它们添加了0.1 %的预训练参数。

为了消除合成函数composition function的影响，我们还报告了在前缀调优去掉门控的结果，记为h+∆h。
我们包括多头并行适配器变体(multi-head parallel adapter variant， MH PA )的结果。
如表4所示，当使用0.1%的参数时，多头方法--前缀调优和MH PA ( attn ) --至少比其他方法高出1.6个BLEU点。令人惊讶的是，将l从200减小到30，前缀调优仅造成0.4 BLEU的损失，而PA ( attn )损失了1.9个点。

前缀调优中的门控合成函数对结果略有帮助，提高了0.3个点。值得注意的是，MH并行适配器将单头版本提高了1.6个点，再次验证了多头方案的有效性。

结合图5和表4中的结果，我们得出结论，**当参数预算很小时，修改头注意力能给出最好的结果，而FFN可以在更大的容量下更好地利用修改**。这表明，将更大的参数预算分配给FFN修改可能是有效的，而不是像Houlsby等( 2019 )那样将注意力和FFN同等对待。

## 起作用的成分是
之前3.2节提到了3种composition functions：simple addition (adapter), gated addition (prefix tuning) and scaled addition (LoRA).
由于在不适用softmax函数形式的方法中加入精确的门控加法是不自然的，我们通过LoRA上进行的消融实验考察了另外两种方，并与提出的缩放并行适配器( Scaled PA )进行比较。
由于4.4节种说到FFN更有效，使用ffn。

我们将r设置为512 (适配器)和102 ( LoRA )，使它们的调节参数大小相同。(?为什么,r是什么)

![table5](https://ooo.0x0.ooo/2024/03/18/OgbIXX.png)

根据开发集(dev set)上的R-2分数来选择s。我们观察到LoRA ( s = 4 )的性能优于并行适配器。然而，如果我们通过设置s = 1来消除缩放，这种优势就会消失。
通过将LoRA的组合功能(composition function)插入到并行适配器中，得到的Scaled PA将vanilla（普通）并行适配器提高了0.56 ROUGE - 2点。

我们还用学习到的标量(scalar)进行了实验，但没有得到更好的结果。因此，我们得出结论，缩放组合功能优于普通相加功能，同时易于应用(the scaling composition function is better than the vanilla additive one while being easily applicable.)

## 通过传递有利的设计元素进行有效整合
我们首先强调前文的三个发现：

1. 缩放的并行适配器是修改FFN的最佳变体；
2. FFN可以在更大的容量下更好地利用改性；
3. 修改头注意力，如前缀调优，只需要0.1 %的参数就可以达到很强的性能。

受他们的启发，我们混合匹配了这些发现背后的有利设计：具体来说，我们在注意力子层使用瓶颈维度较小的前缀调优( l = 30)，并分配更多的参数预算来修改FFN表示，使用缩放的并行适配器( r = 512)。
由于前缀调优在我们的统一框架中可以看作是适配器的一种形式，我们将这种变体命名为Mix - And - Match适配器( MAM Adapter )。

![table 6](https://ooo.0x0.ooo/2024/03/18/OgkBOv.png)

在表6中，我们比较了MAM适配器与各种参数有效的调优方法。为了完整起见，我们还在表6中给出了其他组合版本的结果：在注意力层和FFN层使用并行适配器，并将前缀调优( attn )与LoRA ( ffn )相结合- -这两个组合版本都可以在各自的原型上进行改进。然而，MAM Adapter在两个任务上都取得了最好的性能，并且只需更新6.7 %的预训练参数就能匹配我们完全微调的结果。在表2中，我们还展示了MAM Adapter在MNLI和SST2上的结果，其中MAM Adapter通过只添加0.5 %的预训练参数达到了与完全微调相当的结果。

# 讨论
我们为几种性能参数调优方法提供了一个统一的框架，使我们能够通过跨方法的迁移技术实例化出与完全调优方法性能相匹配的更有效的模型。我们希望我们的工作能够为以后的参数高效调节研究提供见解和指导。

# 个人总结
看到现在有点懵逼，写个总结

1. adapters|prefix tuning|lora的结构和放的位置(2.2)
   1. Adapters
      1. 结构：$h\leftarrow h+f(hW_{down})W_{up}$，中间的过度维度为r
      2. 位置：有两种
   2. prefix tuning
      1. 结构：2个前缀向量$P_k,P_v\in R^{l\times d}$
         每个头的计算变成$head_i=atten(xW_q^{(i)},concat(P_k^{(i)},CW_k^{(i)}),concat(P_v^{(i)},CW_v^{(i)}))$
      2. 位置：放在attention的K和V旁边
   3. lora
      1. 结构：$h\leftarrow h+s\cdot xW_{down}W_{up}$
      2. 位置：attention $W_q$和$W_k$旁边
2. prefix tuning和adapters的差别与一致(3.1)
   1. 先对prefix的公式转化，变成一个和adapters的公式相近的形式
         $$
         h\leftarrow (1-\lambda(x))h+\lambda(x)\Delta h\\
         \Delta h:=softmax(xW_qP_k^T)P_v
         $$ 
   2. 相似
      对prefix tuning进一步修改
      $$
      W_1=W_qP_k^T,W_2=P_v,f=softmax\\
      h\leftarrow(1-\lambda(x)h+\lambda(x)f(xW_1))W_2
      $$ 
      这样和adapters的公式就是$\lambda$的区别
      也可以得出l相当于adapters里面的r
   3. 不同
      不是很重要
3. 统一框架(3.2)
   从adapters与prefix funting的联系得到启发，诞生了一个统一框架
   学习一个修改向量$\Delta h$，并将其应用于各种隐藏表示。
   在形式上，将要直接修改的隐表示记为h；
   将计算h的【PLM子模块的直接输入】表示为x (例如, h和x可以分别作为注意的输出和输入)。
   设计了四种维度：
   1. functional form ：就是那三个h
   2. modified representation：隐表示怎么直接修改的(?)
   3. insertion form：模块插入到网络中的方式
   4. Composition Function：$\Delta h$咋算的

![4维](https://ooo.0x0.ooo/2024/03/19/OghBgG.png) 

4. 迁移设计(3.3)
   另外设计了3种变体
   1. parallel adapter：将prefix tuning的并行插入转换为adapters
   2. multi-head parallel adapter：adapters更加接近prefix tuning
   3. scaled paraellel adapter：LoRA的组成和插入形式转移到适配器中
5. 实验设计(4.1)
   1. 数据集：4个下游任务
      1. XSum：总结文章
      2. Eng-Ro翻译
      3. MNLI：判断
      4. SST2：分类积极或者消极
   2. 设置
      用$BERT_{LARGE}$和多语言版本的作为前两个任务的预训练模型
      $RoBERTa_{BASE}$给后两个任务
      在{1, 30, 200, 512, 1024}范围内变动瓶颈维度(也就是r)
   3. 评估
      ROUGE 1/2/L 分数：XSum
      BLEU分数：en-ro翻译
      准确率：MNLI SST2
      另外还有和全参相比的参数相对值 
   4. 可调参数
      BART和mBART具有编码器-解码器结构，具有3种类型的注意力：编码器自注意力、解码器自注意力和解码器交叉注意力。
      RoBERTa只有编码器自注意力。
      对于每个注意力子层，每种方法使用的参数个数是：
      1. 前缀调优，将l个向量预置到键和值，使用2 × l × d个参数；
      2. 适配器具有Wdown和Wup，因此使用2 × r × d参数；
      3. LoRA使用一对Wdown和Wup进行查询和值投影，因此使用4 × r × d参数。对于ffn处的适配器修改，使用了2 × r × d的参数，与注意处的适配器相同。
      因此，对于特定的r或l值，前缀调优使用与适配器相同数量的参数，而LoRA使用更多的参数 
      不是很重要的样子orz
6. 结果分析(4.2-6)
   1. 刚刚提到了4个维度：functional form|modified represenation|insertion form|composition function
   2. 确认选择哪个任务：集中在**XSum和en - ro**数据集，
      因为如果在XSum和en - ro（图4）中增加5 %的参数，仍然存在很大的差距 
   3. Insertion form：
      1. 可选：sequential parallel，paralle就是新的PA，sequential是传统的
      2. 结果：选parallel adapter
   4. modified presentation:
      1. 可选：attention FFN（modify at attention layer的叫attention，所以是修改那个模块的）
      2. 结果：FFN修饰都能比注意力更有效地利用所增加的参数，我们假设这是因为FFN学习的是任务特定的文本模式( Geva等, 2021)，而注意力学习的是两两位置的交互，不需要较大的容量来适应新的任务。
      3. 0.1%参数：当参数预算很小时，修改头注意力能给出最好的结果
   5. composition function:
      1. 可选：simple addition (adapter), gated addition (prefix tuning) and scaled addition (LoRA)
      2. 结果：the scaling composition function is better than the vanilla additive one while being easily applicable
   6. functional form 没有比较
   7. 综合以上3个维度，设计了一个集大成的：
      1. 3个维度选最优
         1. 缩放的并行适配器Scaled parallel adapter是修改FFN的最佳变体；
         2. FFN可以在更大的容量下更好地利用修改；
         3. 修改头注意力，如前缀调优，只需要0.1%的参数就可以达到很强的性能。
      2. MAM adapter：在**注意力**子层使用瓶颈**维度较小**的**前缀调优**( l = 30)，并分配**更多的参数预算来修改FFN表示**，使用**缩放的并行适配器**
      3. 结果：在两个任务上都取得了最好的性能，并且只需更新6.7 %的预训练参数就能匹配我们完全微调的结果

