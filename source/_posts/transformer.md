---
title: transformer
tags:
  - transformer
categories: ai
date: 2024-03-05 21:17:53
mathjax: true
---

Transformer是一个利用注意力机制来提高模型训练速度的神经网络架构。

#文本生成的架构 这一部分确定完成了，其它部分未完全完成，以后有空再补
<!-- more -->
唯一的问题：那个贼大的框架图每一步怎么走的？
# 前提
1. [编码器+解码器]()
2. [自注意力]()
# 解析
image 一个盒子，以文本翻译为例，输入中文句子，输出英文句子。
## 解码器和编码器
这个盒子有两部分：解码器和编码器。
编码器之间：每一个的小编码器的输入是前一个小编码器的输出，而每一个小解码器的输入不光是它的前一个解码器的输出，还包括了整个编码部分的输出。
解析编码器：自注意力+前馈神经网络
解析解码器：同样的，在decoder中使用的也是同样的结构。也是首先对输出（machine learning）计算自注意力得分，不同的地方在于，进行过自注意力机制后，将self-attention的输出再与Decoders模块的输出计算一遍注意力机制得分
## 残差神经网络

为了解决梯度消失的问题，在Encoders和Decoder中都是用了残差神经网络的结构

## softmax
回归最初的问题，将“机器学习”翻译成“machine learing”，解码器输出本来是一个浮点型的向量，怎么转化成“machine learing”这两个词呢？

最后的线性层接上一个softmax，其中线性层是一个简单的全连接神经网络，它将解码器产生的向量投影到一个更高维度的向量（logits）上，假设我们模型的词汇表是10000个词，那么logits就有10000个维度，每个维度对应一个惟一的词的得分。之后的softmax层将这些分数转换为概率。选择概率最大的维度，并对应地生成与之关联的单词作为此时间步的输出就是最终的输出

## 位置编码
把词向量输入变成携带位置信息的输入：

我们可以给每个词向量加上一个有顺序特征的向量，发现sin和cos函数能够很好的表达这种特征，所以通常位置向量用以下公式来表示：
$$
PE (pos,2i)= sin(\frac{pos}{10000^{2i/d_{model}}}) \\
PE (pos,2i+ 1)= cos(\frac{pos}{10000^{2i/d_{model}}}) 
$$
# 结构
![transformer结构图](https://ooo.0x0.ooo/2024/03/05/Oy0VHp.png)

Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。
红色圈中的部分为 Multi-Head Attention，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。

Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。


- [ ] 模型参数量分析
- [ ] transformer里面有隐藏层吗



[transformer-explainer](https://poloclub.github.io/transformer-explainer/)
[过程可视化](https://bbycroft.net/llm)



# 文本生成的架构
上面的主要是把transformer block说了下，这一章是以文本生成举例，是整个训练流程。以GPT-2为例，参考[transformer explainer的demo](https://poloclub.github.io/transformer-explainer/)。

整个流程是用于生成文本的


包含以下3个最基础的部分：处理输入、transformer、输出概率

1. embedding：文本输入被划分为更小的单元，称为标记tokens，可以是单词或子单词。这些标记被转换成称为嵌入的数字向量 embeddings，这些向量捕获单词的语义含义。
2. Transformer Block：是处理和转换输入数据的模型的基本构建块。每个区块包括：
   1. 注意力机制，Transformer块的核心组件。它允许令牌与其他令牌进行通信，捕获上下文信息和单词之间的关系。
   2. MLP（多层感知器）层，一个独立操作每个令牌的前馈网络。注意层的目标是在标记之间路由信息，而MLP的目标是细化每个标记的表示。
3. 输出概率：最后的线性和softmax层将处理后的嵌入转换为概率，使模型能够预测序列中的下一个令牌。

## embedding
embedding将文本转换为模型可以使用的数字表示。要将提示转换为嵌入，需要以下4个步骤

### tokenize the input
令牌化是将输入文本分解为更小、更易于管理的片段（称为令牌）的过程。这些标记可以是单词或子单词。单词“Data”和“visualization”对应于唯一的标记，而单词“empowers”被分成两个标记。

令牌的完整词汇表是在训练模型**之前**确定的，比如GPT-2的词汇表有50，257个唯一令牌。现在我们将输入文本分割成具有不同ID的标记，我们可以从嵌入中获得它们的向量表示。

### obtain token embeddings
GPT-2 Small将词汇表中的每个标记表示为768维向量;(768维是人为设定的)向量的维度取决于模型。

这些嵌入向量存储在形状矩阵（50257，768）中，包含大约3900万个参数！（50257*768=38,597,376）这个广泛的矩阵允许模型为每个标记分配语义含义。

### add positional information
Embedding层还对每个标记在输入提示符中的位置信息进行编码。不同的模型使用不同的方法进行位置编码。GPT-2从头开始训练自己的位置编码矩阵，将其直接集成到训练过程中。

> 我记得transformer里面也有位置编码啊，后面再确认吧

### add up token and position encodings to get the final embedding
最后，我们对标记和位置编码求和以获得最终的嵌入表示。这种组合表示捕获了标记的语义含义及其在输入序列中的位置。


token embedding这一块最后得到的每个输入token是768大小的vector

## Transformer Block
Transformer处理的核心在于Transformer块，该块包括**多头自注意**和**多层感知器层**。大多数模型由多个这样的块组成，这些块一个接一个地顺序堆叠。令牌表示通过这些层进行演变，从第一个块到第12个块，使模型能够建立对每个令牌的复杂理解。这种分层方法导致输入的高阶表示。

为什么是第12个块？因为在小型的 GPT-2 中，有 12 个注意力头

### 多头自注意力
自注意机制使模型能够专注于输入序列的相关部分，从而捕获数据中的复杂关系和依赖关系。

#### Query, Key, and Value Matrices

输入序列长度l，在经过embedding后得到lx768的matrix，再点乘QKV权重（都是768*(768*3)的matrix，这个是一起做的，有的教程会让qkv分开来计算），加上一个m大小的vector，得到最终lxm的matrix结果，也就是需要的QKV

每个令牌的嵌入向量被转换为三个向量：Query（Q）、Key（K）和Value（V）。这些向量是通过将输入嵌入矩阵与学习到的Q、K和V的权重矩阵相乘得到的。这里有一个网络搜索类比，可以帮助我们在这些矩阵背后建立一些直觉：

- 查询（Q）是您在搜索引擎栏中键入的搜索文本。这是您要“查找有关的更多信息”的令牌。
- Key（K）是搜索结果窗口中每个网页的标题。它表示查询可以关注的可能令牌。
- 值（V）是显示的网页的实际内容。一旦我们将适当的搜索词（查询）与相关的结果（关键字）匹配，我们希望获得最相关页面的内容（值）。

通过使用这些QKV值，该模型可以计算注意力分数，该分数确定每个令牌在生成预测时应该获得多少关注。

####  Masked Self-Attention
> 为什么要mask？

Masked Self-Attention允许模型通过关注输入的相关部分来生成序列，同时防止访问未来的token。

- 注意力评分：Query和Key矩阵的点积确定每个查询与每个键的对齐，产生一个反映所有输入标记之间关系的square matrix。
- mask：将掩码应用于注意力矩阵的上三角形，以防止模型访问未来的令牌，将这些值设置为负无穷大。模型需要学习如何在不“窥视”未来的情况下预测下一个令牌。
- Softmax：掩蔽后，注意力分数通过softmax运算转换为概率，该运算取每个注意力分数的指数。矩阵的每一行总和为1，并指示其左侧的每一个其他标记的相关性。

#### Output
该模型使用掩蔽的自我注意力分数，并将其与值矩阵相乘，以获得自我注意力机制的最终输出。GPT-2有12个自我注意头，每个头捕捉标记之间的不同关系。这些头的输出被连接并通过线性投影。

### MLP
使用MLP层将自我注意表示投射到更高的维度，以增强模型的表示能力。

在多个自我注意力头捕获输入标记之间的不同关系之后，级联输出通过多层感知器（MLP）层以增强模型的表示能力。

MLP块由两个线性变换组成，这两个变换中间间有一个GELU激活函数。

- 第一个线性变换将输入的维数从768增加到3072。
- 第二个线性变换将维度降低到原始大小768，确保后续层接收一致维度的输入。

与自注意机制不同，MLP独立地处理令牌，并简单地将它们从一种表示映射到另一种表示。

## Output Probabilities
在输入通过所有Transformer块处理之后，输出通过最后的线性层以准备令牌预测。这一层将最终表示投射到50，257维空间中，其中词汇表中的每个标记都有一个称为logit的对应值。任何标记都可以是下一个单词，所以这个过程允许我们简单地根据它们成为下一个单词的可能性对这些标记进行排名。然后，我们应用softmax函数将logits转换为总和为1的概率分布。这将允许我们根据其可能性对下一个令牌进行采样。

最后一步是通过从这个分布中采样来生成下一个令牌。温度超参数在这个过程中起着关键作用。从数学上讲，这是一个非常简单的操作：模型输出logit简单地除以温度temperature：

- temperature = 1：logit除以1对softmax输出没有影响。
- temperature< 1：较低的温度通过锐化概率分布使模型更有信心和确定性，从而产生更可预测的输出。
- temperature> 1：更高的温度会产生更柔和的概率分布，允许生成的文本具有更多的随机性-有些人称之为模型“创造力”。

> 这个除以的计算怎么搞的……把softmax章节补一下

## Advanced Architectural Features
有几个高级的体系结构特性可以增强Transformer模型的性能。虽然它们对于模型的整体性能很重要，但是对于理解架构的核心概念来说并不那么重要。层规范化、丢弃和残余连接是Transformer模型中的关键组件，尤其是在训练阶段。层规范化稳定训练并帮助模型更快地收敛。Dropout通过随机停用神经元来防止过拟合。残差连接允许梯度直接流过网络，并有助于防止梯度消失问题。

### Layer Normalization
有助于稳定训练过程并提高收敛性。它的工作原理是对特征之间的输入进行归一化，确保激活的均值和方差是一致的。

这种归一化有助于缓解与内部协变量偏移相关的问题，使模型能够更有效地学习，并降低对初始权重的敏感性。

层规范化在每个Transformer块中应用两次，一次在自注意机制之前，一次在MLP层之前。

### Dropout
Dropout是一种正则化技术，用于防止神经网络中的过拟合，方法是在训练过程中将一部分模型权重随机设置为零。

这鼓励模型学习更强大的特征，并减少对特定神经元的依赖，帮助网络更好地泛化到新的、看不见的数据。

在模型推理期间，dropout被禁用。这本质上意味着我们使用的是训练好的子网络的集合，这会带来更好的模型性能。

### Residual Connections
从本质上讲，残余连接是绕过一个或多个层的快捷方式，将层的输入添加到其输出。这有助于缓解梯度消失的问题，使训练具有多个相互堆叠的Transformer块的深度网络变得更容易。

在GPT-2中，剩余连接在每个Transformer块中使用两次：一次在MLP之前，一次在MLP之后，确保梯度更容易流动，并且较早的层在反向传播期间接收足够的更新。

## 总结
1. 输入长度l，首先embedding，根据选择的模型有词汇表，比如GPT-2就是768维的，加上位置后得到最终结果（和positional encoding的区别在哪？）
   > 通常，embedding是指学习出来的encoding，是将位置信息“嵌入”到某个空间的意思。例如，bert的位置编码是学出来的，所以称为position embedding。而transformer的位置编码是用三角函数直接算出来的（当然，论文中说也可以学出来，效果差不多，所以最后还是采用了直接编码），不涉及嵌入的思想，所以叫position encoding。
2. dropout
3. Residual Connection->
4. Layer Normalization
5. transformer块
   1. 多头自注意力计算
      1. 根据输入算出QKV，需要用的权重矩阵是要训练的
      2. masked self-attention：Q·K^T，再mask
      3. softmax、dropout
   2. MLP
      1. dropout
      2. <-residual connection
      3. layer normalization
      4. residual connection
         1. GeLU Activation
6. layer normalization
7. 输出概率结果

这部分的代码参考{% post_link 'TRANSFORMER EXPLAINER: Interactive Learning of Text-Generative Models' 'transformer explainer' %} 