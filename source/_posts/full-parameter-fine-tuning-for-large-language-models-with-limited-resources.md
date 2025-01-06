---
title: FULL PARAMETER FINE-TUNING FOR LARGE LANGUAGE MODELS WITH LIMITED RESOURCES
date: 2023-11-24 17:10:38
tags: 
  - fine tune
  - llm
  - full parameterwww
categories: paper
mathjax: true
---
百篇paper计划(1/100)，推荐阅读的，没有选择趋向。内容是：用有限的资源（主要是内存）进行全参数大模型微调，研究重点集中在微调部分，做的是内存优化的工作。

<!-- more -->
- 论文标题：Full Parameter Fine-tuning for Large Language Models with Limited Resources
- arxiv地址：[链接](https://arxiv.org/abs/2306.09782)
- code：[github](https://github.com/OpenLMLab/LOMO)

# abstract
- 背景：LLMs给NLP问题带来了革命性变化，but需要大量GPU资源
- 已有：高效参数微调（PEFT），没有彻底解决
  [PEFT参考文档](高效参数微调.md)
  > parameter-efficient fine-tuning vs full-parameter fine-tune 

- 创新：提出了新的优化器 LOw-Memory-Optimization
  内容：在一个步骤融合了梯度计算和参数更新
  > 怎么融合(fuse)？

- 效果：和标准方法（DeepSpeed解决方案）相比，我们将内存使用量减少到10.8%。
  因此，我们的方法可以在一台拥有8×RTX 3090的机器上对65B模型进行全参数微调，每台机器都有24GB内存

# introduction
> 感觉就是把摘要扩写了一遍

- 背景：LLMs 厉害，但是要钱。
- 目前：有效参数微调方法
  比如：LoRA,Prefix-tuning
  但是：没有全参数的
- 解决问题：我们要找个办法实现全参数微调
- 干了啥
  - 分析LLMs里面内存使用的四个主要方面：
    激活，优化器状态，梯度张量和参数
  - 从三个方面优化了训练过程：
    - SGD-优化器：由于SGD不存储任何中间状态，可以删除整个优化器状态部分[SGD是什么](SGD.md)
    - LOMO把梯度tensor的内存使用降到了O(1)
    - 为了稳定使用混合精度训练：集成梯度归一化和损失缩放，并将某些计算过渡到全精度
- 结果
  内存使用量等于参数使用量加上激活量和最大梯度张量。

  分析了：

  1. 内存和吞吐量
  2. 下游表现
- 贡献
  1. SGD
  2. LOMO
  3. 验证
# related work
都是节省内存的技术

## Activation Checkpointing
计算图中策略性地选择的检查点节点的激活变量在正向阶段后保留在内存中，而其余节点的激活变量在梯度计算时最多重新计算一次。激活内存可以降低原始内存使用量的平方根，代价是需要多一次正向传递。
在前向传播过程中，有选择性的丢弃一些产生的中间激活值，当反向传播需要使用这些激活值时，再依据上一个保存的激活值来计算出这些需要的激活值来帮助完成反向传播。这样一来，便可以大大减少训练过程中所需要的存储空间。

## Mixed-Precision Training
混合精度，32位浮点数和16位浮点数混合使用

## Heterogeneous Training System
通过利用异构内存（如CPU和NVMe内存）来减少GPU内存消耗

> 但是怎么做？？

# method
## 优化器
adam和SGD比起来哪里不好了！
参考https://blog.csdn.net/S20144144/article/details/103417502

### 使用SGD
SGD遭人诟病的有三个：
- 大曲率损失面（什么叫曲率？）
- 局部最优
- 鞍点
作者对LLMs问题上用SGD时这些可能的问题逐一分析

#### 更平滑的损失面
验证了更大的模型有更加平滑的损失面
所以LLMs上面这一点可以忽略

#### 局部最优就够了
微调的目标是让模型能够解决新任务，基础模型没变。
而且有限的训练数据（和预训练时期相比）也不可能实现全局最优。

#### 鞍点很遥远
对于传统的NLP任务，LLM 参数的初始点通常是在谷底，如果模型是通过指令微调，当出现新任务时，模型在某种程度上对这些任务都是似曾相识，那么saddle points通常出现在边缘且距离valley有一定距离，因此如果在改变参数值不大的情况下是也许不会碰到saddle points在finetune过程中。
关于散点详细介绍可以看[局部最优和鞍点](局部最优和鞍点.md)

从结论上来看，SGD是可以胜任微调工作。
### batch size
在一个batch中含有2个样本的参数更新公式如下：
$$
\theta ^{'} = \theta - \alpha[\nabla \mathcal{L} (d_i, f(d_i, \theta)) + \nabla \mathcal{L}(d_j, f(d_j, \theta)) ] 
$$
模型参数在这两个样本上的更新过程：
$$
\theta_1 = \theta - \alpha \nabla \mathcal{L}(d_i, f(d_i, \theta)) \\ \theta_2 = \theta_1 - \alpha \nabla \mathcal{L}(d_j, f(d_j, \theta_1)) \\ \theta_2 = \theta - \alpha \nabla \mathcal{L}(d_i, f(d_i, \theta)) - \alpha \nabla \mathcal{L}(d_j, f(d_j, \theta_1)) 
$$

利用微分中值定理

$$
\mathcal{L'}(d_j,\xi) = \frac{\mathcal{L(d_j, f(d_j,\theta_1))} - \mathcal{L}(d_j, f(d_j, \theta)}{f(d_j, \theta_1) - f(d_j, \theta)} \\ \mathcal{L}(d_j, f(d_j, \theta_1)) = \mathcal{L}(d_j, f(d_j, \theta)) + \mathcal{L'}(d_j, \xi)(f(d_j, \theta_1) - f(d_j, \theta)) 
$$

代入后得到
$$
\theta_2 = \theta - \alpha \nabla \mathcal{L}(d_i,f(d_i,\theta)) - \alpha \nabla [\mathcal{L(d_j, f(d_j, \theta))} + \mathcal{L'}(d_j, \xi)(f(d_j, \theta_1) - f(d_j, \theta))] \\ \theta_2 = \theta - \alpha [\nabla \mathcal{L}(d_i,f(d_i,\theta)) +  \nabla \mathcal{L(d_j, f(d_j, \theta))}] + \alpha \nabla \mathcal{L'}(d_j, \xi)(f(d_j, \theta_1) - f(d_j, \theta)) 
$$

化简之后
$$
\theta_2 - \theta^{'} = \alpha \nabla \mathcal{L'}(d_j, \xi)(f(d_j, \theta_1) - f(d_j, \theta))
$$
按照假设loss surface足够平滑，等号后边的式子可以忽略不计。当使用SGD作为optimizer时，可以采用较大的batch size 提高训练稳定性；这也从另外一个方面解释了SGD在小模型上效果不佳但在大模型上效果较好。

## lomo
梯度张量的作用：计算优化器状态和梯度正则化，因为用SGD，第一个可以省掉了，接下来的问题就是梯度正则化。

LOMO：在一步内同时完成梯度计算和参数更新，以避免存储梯度张量。

两步计算过程是先计算梯度，再更新参数：而所提出的方法是计算完梯度之后，直接对参数进行更新。

Pytorch中提供类似 injecting hook function但无法实现参数的即时更新；作者所采用方法是随着backward，progagation的进行只存储一个参数的梯度，这种方案将之前存储所有参数的梯度，降低到只存储一个参数，从而减少了内存的占用

## 训练过程的稳定性
### 梯度正则化和裁剪的替代方案
梯度正则化和裁剪用来干什么的：防止梯度爆炸和消失。提出了两个方法

- Clip gradient by values，而不是采用gradient norm。Clip gradient by values是缓解梯度爆炸有效的方案，但这种方式唯一需要考虑的是，通过gradient value来进行裁剪会改变方向。采用这种方法时需要考虑学习率的设置，根据经验学习率应低于1e-3
- 用额外的过程来计算gradient norm。计算梯度的norm，由于所提出的算法并不会记录所有参数值，作者提出了一个可以探索的方案就是计算相邻层参数的norm来替代全局；当然这种方法仍值得广泛讨论，因为对于不同的parameters采用了不同的update step size
  
矛盾点：目前的训练框架是根据所有参数来计算梯度准则的，因此需要两次后向传递（backward）。

解决：一种解决方案是用一组参数（例如相邻层）来近似梯度张量的规范。然而，这种方法确实存在偏差，因为它会导致不同参数的更新步长不同。更新时，参数会根据梯度准则乘以一个比例系数。由于不同参数组的梯度准则不同，这种近似方法会导致比例系数的差异。尽管存在这种局限性，这种分组梯度削波方法仍可视为根据不同参数组的梯度准则对其采用动态学习率。

真解决：动态learning rate

### 保证精度
为了保证训练速度和精度，采用dynamic loss scaling 和将确定的计算转化为全精度计算。作者提出将一个dynamic loss scaler 和LOMO进行集成，它在整个训练过程中条件scaling factor。

当特定数量的后向传播为发生溢出时，scale factor加倍，反正则减半。这里出现个疑惑点即怎么去判断是否有溢出，实际过程中只有当后向传递结束时我们才能知道是否有溢出发生。为了解决这个问题，作者提出二次后传方案，即第一次后向传播时判断是否有溢出，如果没有溢出在第二次传递时更新参数。

# experiment
评估标准：内存占用、吞吐率、下游任务表现
模型：LLaMA(7B-65B)
硬件：RTX 3090 GPU

## 内存占用
比较的是是否用激活检查点、三个优化器的内存占用

结论：

1. LOMO不存储优化器状态，保存一点的梯度
2. 使用ac的明显消耗更少
3. 内存使用量的大幅减少主要归功于梯度和优化器状态对内存需求的降低。

我提的问题：

1. 为什么没做bacth-size？
2. 为什么SGD还保存优化器状态，不是说不保存吗？

### 优化器状态
用AdamW,sgd和LOMO实验时优化器状态、梯度、参数和激活的内存使用占比。

混合精度训练方法，在这种方法中，权重、动量和方差的全精度副本被保留在优化器状态中，用于权重更新。用 SGD 优化器替换 AdamW 优化器可以有效降低内存中优化器状态的百分比，从而减少 GPU 内存使用量。这是因为 SGD 优化器不需要存储全精度动量和方差。对 LOMO 而言，参数更新和后退融合为一个步骤，进一步消除了对优化器状态内存的需求。

### 梯度
在使用 LOMO 的训练过程中，一旦接收到梯度，就会立即更新参数，然后将梯度从内存中丢弃。因此，梯度内存消耗的上限取决于与最大参数矩阵相关的梯度。这种方法大大减少了内存使用量，几乎与参数的大小相当。

### 激活
LOMO 与激活检查点等减少激活内存的技术兼容。

## 吞吐量
吞吐量以每 GPU 每秒处理的令牌数（TGS）来衡量，参数分区使用 ZeRO-3 实现。

- 对于 7B 模型，LOMO 的吞吐量非常可观，比 AdamW 和 SGD 高出约 11 倍。因为LOMO可以仅在一个CPU上就能run 7B 模型，从而减少了 GPU 之间的通信开销。与 AdamW 相比，SGD 的吞吐量略高，这是因为 SGD 排除了动量和方差计算。
- 至于 13B 模型，由于内存限制，无法在现有的 8 个 RTX 3090 GPU 上使用 AdamW 进行训练。在这种需要 LOMO 进行模型并行化的情况下，LOMO 的吞吐量仍然优于 SGD。这一优势归功于 LOMO 的内存效率特性，以及只需要两个 GPU 就能在相同设置下训练模型，从而降低了通信成本，提高了吞吐量。
- 在训练 30B 模型时，SGD 在使用 8 个 RTX 3090 GPU 时遇到了内存不足（OOM）问题，而 LOMO 仅在使用 4 个 GPU 时表现良好。
- 最后，我们使用 8 个 RTX 3090 GPU 成功训练了 65B 模型，吞吐量达到 4.93 TGS。利用这样的服务器配置和 LOMO，对 1000 个样本（每个样本包含 512 个 token）的训练过程大约需要 3.6 个小时。

## 下游表现
> downstream performance是什么啊？就是真正要做的fine-tune任务

- 目的：为了评估 LOMO 在微调大型语言模型方面的有效性
- 干活：将LOMO 与其他两种方法进行了比较，一种是不需要微调的 Zero-shot，另一种是目前最流行的参数高效微调技术之一 LoRA。LoRA优点： 对密集层重新参数化，只更新低等级矩阵，同时在推理过程中不引入延迟。
- 数据集： SuperGLUE 
- 样本数据构成：鉴于运行大型语言模型的计算成本较高，从训练集中随机抽样 1000 个训练数据，从验证集中随机抽样 1000 个测试数据，并报告使用相同随机种子获得的最佳结果。
- 评价指标： 准确度 

### 主要结果
LOMO 的性能明显优于 Zero-shot
在大多数实验中，LOMO 的性能普遍优于 LoRA
LOMO可以扩展到65B

### 比较lora
无论 LoRA 取得了多高的结果，LOMO 始终能增强 LoRA 的性能。这表明，LOMO 和 LoRA 采用的不同微调方法是互补的。
具体来说，LOMO 侧重于微调预训练模型的权重，而 LoRA 则对其他模块进行微调。因此，LOMO 不会影响 LoRA 的性能，反而有助于为下游任务进行更好的模型调整。
实验结果表明，在某些场景下，LOMO的效果是比LoRA差，原因是LOMO是全参数微调，微调数据有限训练不充分，另一方面LOMO和LoRA采用两种不同的架构，后者只是采用了一个shortcut，在某些场景下会更加有利。LOMO关注的是微调pre-trained 模型权重，而LoRA微调附加的模块。

# 总结
贡献：

- LOMO优化器 在利用有限的资源促进大型语言模型的全参数微调
- 演示了在配备 RTX 3090 GPU 的服务器上对 65B 模型进行微调的可行性
- 通过分析 LOMO 的内存使用情况、进行吞吐量测试以及在 SuperGLUE 上进行实验，我们展示了其有效性和潜在影响

未来：

- 进一步降低训练大型语言模型所需的资源门槛，从而使这些模型得到更广泛的访问和采用
- 参数占用多，探索参数量化技术，这可以大大减少内存的使用
- LOMO 的更多应用场景，并深入研究优化大型语言模型的理论分析

# 附
## 代码解读
- [ ] 代码以后再看，也不知道还要不要看了

```python
# 继承Optimizer类，从torch.optim中import的Optimizer类中已经有hook相关
class LOMO(Optimizer):
```
