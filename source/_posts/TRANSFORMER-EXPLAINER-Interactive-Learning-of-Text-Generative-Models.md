---
title: 'TRANSFORMER EXPLAINER: Interactive Learning of Text-Generative Models'
tags:
  - transformer
mathjax: false
categories: paper
date: 2024-09-13 10:12:29
---


百篇paper计划(7/100)，很短，才2页，重点在code和其project，理解transformer。论文是没啥创新点了，写得也比较拉，看看项目吧。

这一篇写完啦！^^
<!--more-->

- 论文标题：TRANSFORMER EXPLAINER: Interactive Learning of Text-Generative Models
- code：[github](https://github.com/poloclub/transformer-explainer)
- rank: none
- 打标签：transformer可视化
- 时间：2024年8月8日

# abstract
TRANSFORMER EXPLAINER是交互式可视化工具，通过GPT-2模型让普通人看懂transformer怎么运行的。

> 很适合用来科普呢……

# introduction
要可视化，作者提了3个贡献

## 开源的、基于Web的、面向非专业人士的交互式可视化工具
- 学习transformer的高层模型结构和低层数学运算
- Sankey diagram 可视化设计，强调输入数据如何通过模型的组件"流动"

## 开源的、基于Web的实时推理的实现
GPT-2和GPT-3/4有相似的结构

# 系统设计与实现
前端使用Svelte和D3进行交互可视化，后端使用ONNX runtime和HuggedFace的Transformers库在浏览器中运行GPT - 2模型。

难点在于管理底层架构的复杂性，不能显示太多细节。

## 通过多层次抽象降低复杂度
为了展现不同的抽象层次，将工具结构化。

- 工具的完整pipeline：从将用户提供的文本作为输入，将其嵌入，通过多个Transformer块进行处理，到使用转换后的数据对最可能的下一个标记预测进行排序。
- 中间操作，如注意力矩阵的计算，默认为折叠，以可视化计算结果的显著性，并可扩展到通过动画序列检查其推导。
- 我们使用了一致的可视化语言，例如堆叠注意力头(stacking Attention Heads)和折叠重复的Transformer Block，帮助用户识别架构中的重复模式，同时保持数据的端到端流动。

## 通过互动增进理解和参与度
温度(temperature)参数在控制transformer输出概率分布方面至关重要，影响下一阶段的预测是更确定的(在低温下)还是随机的(高温)。也就是控制the prediction determinism



# 正在进行的工作
我们正在增强该工具的交互式解释（例如，图层标准化），以改善学习体验。我们还使用 WebGPU 提高推理速度，并通过压缩技术（例如量化quantization、淡化palettization）减小模型大小。

# code
需要Node.js v20 or higher+NPM v10 or higher的环境，可惜代码我跑不起来，提交issue了，不知道能不能解决

代码主要看\\explainer\\transformer-explainer\\src\\utils\\model\\model.py

好奇temperature到底在哪里

## model.py
一共包含6个class

### layernorm
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        # 初始化，weight都是1，bias都是0/None
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.epsilon = 1e-5
        # self.dict = {}

    def forward(self, input):
        layer_norm = F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.epsilon)
    
        # Compute mean and variance
        # dim=-1表示按行求，keepdim
        # keepdim（bool）– 保持输出的维度
        # 当keepdim=False时，输出比输入少一个维度（就是指定的dim求范数的维度）
        # keepdim=True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1。
        input_mean = input.mean(dim=-1, keepdim=True)
        # torch.std默认设置了unbiased=True。此时计算标准差的公式则使用贝塞尔校正 的方法
        # 贝塞尔校正就是前面会除以(n-1)
        input_var = input.var(dim=-1, unbiased=False, keepdim=True)
        input_normalized = (input - input_mean) / torch.sqrt(input_var + self.epsilon)
        # Store values
        # self.dict.update({
        #     'input_mean': input_mean,
        #     'input_var': input_var,
        #     'input_normalized': input_normalized,
        #     'weight': self.weight,
        #     'bias': self.bias,
        #     'output': layer_norm
        # })

        return layer_norm
```


### CausalSelfAttention
```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # 加速的一个方法，居然不设置没有加速的嘛
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        """
        torch.save(model.state_dict())，model.state_dict()是一个字典，里边存着我们模型各个部分的参数。
        在model中，我们需要更新其中的参数，训练结束将参数保存下来。
        但在某些时候，我们可能希望模型中的某些参数参数不更新（从开始到结束均保持不变），但又希望参数保存下来（model.state_dict() ）
        这时我们就会用到 register_buffer() 。

        举例：self.register_buffer(‘my_buffer’, self.tensor)
        my_buffer是名字，str类型；self.tensor是需要进行register登记的张量。
        这样我们就得到了一个新的张量，这个张量会保存在model.state_dict()中，也就可以随着模型一起通过.cuda()复制到gpu上。

        torch.tril主要用于返回一个矩阵主对角线以下的下三角矩阵，其它元素全部为0。
        当输入是一个多维张量时，返回的是同等维度的张量并且最后两个维度的下三角矩阵的。

        PyTorch 中的view( )函数相当于numpy中的resize( )函数，都是用来重构(或者调整)张量维度的
        """
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        self.dict = {}

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # get weights
        q_weights, k_weights, v_weights = self.c_attn.weight.split(self.n_embd, dim=0)
        # split into multihead weights
        q_weights = q_weights.view(self.n_head, C // self.n_head, C) # (nh, hs, C)
        k_weights = k_weights.view(self.n_head, C // self.n_head, C) # (nh, hs, C)
        v_weights = v_weights.view(self.n_head, C // self.n_head, C) # (nh, hs, C)

        # get biases
        q_bias, k_bias, v_bias = self.c_attn.bias.split(self.n_embd, dim=0)
        # split biases
        q_bias = q_bias.view(self.n_head, -1) # (nh, hs)
        k_bias = k_bias.view(self.n_head, -1) # (nh, hs)
        v_bias = v_bias.view(self.n_head, -1) # (nh, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # @是矩阵乘法
        attn = q @ k.transpose(-2, -1) 
        attn_scaled = attn * (1.0 / math.sqrt(k.size(-1)))
        # mask
        attn_masked = attn_scaled.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn_softmax = F.softmax(attn_masked, dim=-1)
        attn_dropout = self.attn_dropout(attn_softmax)
        y = attn_dropout @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 分头行动
        for i in range(self.n_head):
            if (i == 0):
                self.dict[f"head_{i}"] = {}
                # self.dict[f"head_{i}"]["q_weights"], self.dict[f"head_{i}"]["k_weights"], self.dict[f"head_{i}"]["v_weights"] = q_weights[0], k_weights[0], v_weights[0],
                # self.dict[f"head_{i}"]["q_bias"], self.dict[f"head_{i}"]["k_bias"], self.dict[f"head_{i}"]["v_bias"] = q_bias[0], k_bias[0], v_bias[0],
                # self.dict[f"head_{i}"]["q"], self.dict[f"head_{i}"]["k"], self.dict[f"head_{i}"]["v"] = q[:, 0], k[:, 0], v[:, 0]
                q_transposed, k_transposed, v_transposed = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
                # self.dict[f"head_{i}"]["q_transposed"], self.dict[f"head_{i}"]["k_transposed"], self.dict[f"head_{i}"]["v_transposed"] = q_transposed[:, 0], k_transposed[:, 0], v_transposed[:, 0]
                self.dict[f"head_{i}"]["attn"], self.dict[f"head_{i}"]["attn_scaled"], self.dict[f"head_{i}"]["attn_masked"], self.dict[f"head_{i}"]["attn_softmax"], self.dict[f"head_{i}"]["attn_dropout"] = attn[:, 0], attn_scaled[:, 0], attn_masked[:, 0], attn_softmax[:, 0], attn_dropout[:, 0]
                # self.dict[f"head_{i}"]["v_output"] = y[:, 0]

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # self.dict["v_output_combined"] = y

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        # self.dict["proj_weights"] = self.c_proj.weight
        # self.dict["proj_bias"] = self.c_proj.bias
        # self.dict["attn_output"] = y

        return y
```

### MLP
```python
class MLP(nn.Module):

    # gelu激活函数Gaussian Error Linear Unit，用其它的也可以吧
    # 为什么MLP
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # self.dict = {}

    def forward(self, x):
        x = self.c_fc(x)
        # self.dict["linear_1_weight"] = self.c_fc.weight 
        # self.dict["linear_1_bias"] = self.c_fc.bias 
        # self.dict["linear_1_output"] = x 
        x = self.gelu(x)
        # self.dict["gelu_output"] = x 
        x = self.c_proj(x)
        # self.dict["linear_2_weight"] = self.c_proj.weight
        # self.dict["linear_2_bias"] = self.c_proj.bias 
        # self.dict["linear_2_output"] = x 
        x = self.dropout(x)
        # self.dict["output_after_dropout"] = x 
        return x
```

### transformer blcok
```python
class Block(nn.Module):

    # transformer block

    def __init__(self, config):
        super().__init__()
        # 用之前的4个类
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        self.dict = {}

    def forward(self, x):
        # self.dict["ln_1"] = self.ln_1.dict 
        self.dict["attn"] = self.attn.dict 
        # 第一步，计算attn
        x = x + self.attn(self.ln_1(x))
        # self.dict["res_1"] = x 
        # self.dict["ln_2"] = self.ln_2.dict
        # self.dict["mlp"] = self.mlp.dict 
        # 第二步，mlp
        x = x + self.mlp(self.ln_2(x))
        # self.dict["res_2"] = x 
 
        return x
```

### GPTConfig
```python
# @dataclass 装饰器的主要目标是简化类的创建。
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

### GPT
```python
class GPT(nn.Module):
    # 初始化
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        # 这个config是从GPTConfig函数得到的
        self.config = config
        # 这个是运行步骤嘛？
        self.transformer = nn.ModuleDict(dict(
            # embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # transformer块
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # lm_head=language modeling head(hugging face里面查到的通用方法，不确定)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.dictionary = {}

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        input_emb = tok_emb + pos_emb
        input_emb_dropout = self.transformer.drop(input_emb)
  
        self.dictionary["block"] = {}

        x = input_emb_dropout
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if (i == 0):
                self.dictionary["block"][f"block_{i}"] = block.dict
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
  
        # self.dictionary["embedding"] = {
        #     "tok_emb": tok_emb,
        #     "transformer.wpe.weight": self.transformer.wpe.weight,
        #     "pos_emb": pos_emb,
        #     "input_emb": input_emb,
        #     "input_emb_dropout": input_emb_dropout
        # }

        # self.dictionary["ln_f"] = self.transformer.ln_f.dict
        
        self.dictionary["linear"] = {
            # "weight": self.lm_head.weight,
            "output": logits
        }

        return self.dictionary

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    # # 类方法（不需要实例化类就可以被类本身调用）
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # 好像也没用到，但我看不懂这个是什么
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # 不是重点
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # MFU=算力利用率
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
```

很多看不懂……而且这部分代码太多了，啃不动

## 其它代码
py文件里没找到temperature相关的内容，前端显示里面有，但不知道在哪调整的

temperature：介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定性。我们通常建议更改此设置或top_p但不能同时更改两者。
temperature 越高，文章内容随机性越强，创造力越好。