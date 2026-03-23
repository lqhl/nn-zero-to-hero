# Lecture 7：从零构建 GPT

> 对应视频：[YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
> 对应代码：[ng-video-lecture GitHub Repo](https://github.com/karpathy/ng-video-lecture)（含 `bigram.py` 和 `gpt.py`）
> 对应 Colab：[Google Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
> 参考论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
> nanoGPT：[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

---

## 目录

1. 本讲目标与背景
2. 数据准备：Tiny Shakespeare 与字符级 Tokenization
3. 数据加载：Block Size 与 Batch
4. Bigram 基线模型
5. 自注意力的数学基础：矩阵乘法加权聚合技巧
6. 单头自注意力（Single Head Self-Attention）
7. 多头注意力（Multi-Head Attention）
8. 前馈网络（Feed-Forward Network）
9. Transformer Block 与残差连接
10. Layer Normalization
11. Dropout 正则化
12. 规模化训练与最终结果
13. Decoder-only vs Encoder-Decoder 架构
14. nanoGPT 代码导览
15. ChatGPT 的训练流程：预训练与微调
16. 本讲总结

---

## 1. 本讲目标与背景

### 现象：ChatGPT 震惊世界

ChatGPT 是一个基于文本的语言模型，给定提示词，它能从左到右逐 token 生成文字，且是概率性系统——同一个 prompt 每次输出略有不同。它的核心底层架构是 **Transformer**，来自 2017 年那篇影响深远的论文：

> **"Attention Is All You Need"** — Vaswani et al., 2017

GPT 全称 **Generatively Pre-trained Transformer**。Transformer 是做所有繁重工作的神经网络，它于机器翻译场景诞生，但作者并未充分预见它对整个 AI 领域的冲击。在此后五年里，这个架构几乎被复制粘贴进了 AI 的每一个角落。

### 本讲的目标

- **不能**复现 ChatGPT（需要大量数据和多阶段训练）
- **可以**从零训练一个 **字符级 Transformer 语言模型**
- 数据集：Tiny Shakespeare（~1MB，全部莎士比亚作品）
- 学完本讲后，你将理解 ChatGPT 底层发生了什么

### 前置要求

- Python 熟练
- 高中微积分与统计学基础
- 建议先看前几讲 makemore 系列（语言建模框架、PyTorch 张量）

---

## 2. 数据准备：Tiny Shakespeare 与字符级 Tokenization

### 数据集

```python
# 下载 Tiny Shakespeare 数据集
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# 约 1,000,000 个字符
```

### 字符级 Tokenizer

**Tokenization** = 把原始字符串转为整数序列。

我们使用最简单的字符级方案：每个字符 → 一个整数。

```python
chars = sorted(list(set(text)))  # 所有出现的字符，排序
vocab_size = len(chars)           # 65

stoi = { ch:i for i,ch in enumerate(chars) }  # 字符 → 整数
itos = { i:ch for i,ch in enumerate(chars) }  # 整数 → 字符

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 示例
encode("hi there")  # → [46, 47, ...]
decode([46, 47, ...])  # → "hi there"
```

词汇表共 **65 个字符**：一个空格、各种标点、大小写字母。

### 不同 Tokenizer 的对比

| Tokenizer | 词汇表大小 | "hi there" 编码长度 |
|-----------|-----------|-------------------|
| 字符级（本讲） | 65 | 8 个整数 |
| Google SentencePiece（子词） | ~32,000 | 更短 |
| OpenAI tiktoken (GPT-2 BPE) | 50,257 | 3 个整数 |

> 词汇表越大 → 序列越短；词汇表越小 → 序列越长。GPT 实际用的是子词 BPE tokenizer，但本讲为简单起见采用字符级。

### 训练/验证集划分

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]   # 前 90%
val_data   = data[n:]   # 后 10%，藏起来评估泛化能力
```

---

## 3. 数据加载：Block Size 与 Batch

### Block Size（上下文长度）

我们不把整个文本一次喂给 Transformer（计算代价极高），而是每次随机采样固定长度的 **chunk**，这个最大长度叫 **block size**（也叫 context length）。

**关键洞察**：一个长度为 `block_size + 1` 的 chunk，实际上打包了 `block_size` 个独立训练样本：

```python
block_size = 8
x = train_data[:block_size]      # [18, 47, 56, 57, 58, 1, 15, 47] ← 输入
y = train_data[1:block_size+1]   # [47, 56, 57, 58, 1, 15, 47, 58] ← 目标

for t in range(block_size):
    context = x[:t+1]   # 上下文从长度 1 到 block_size
    target  = y[t]      # 下一个字符
    # 例：context=[18], target=47
    #     context=[18,47], target=56
    #     ...以此类推，共 8 个样本
```

> **为什么要训练所有上下文长度（1 到 block_size）？**
> 不只是效率原因，更重要的是：在推理时我们可能从只有 1 个字符开始生成，Transformer 必须学会在极短上下文下也能预测。

### Batch Dimension

为了充分利用 GPU 并行性，每次同时处理多个 chunk：

```python
batch_size = 32  # 并行处理 32 个独立序列

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]   for i in ix])  # (B, T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # (B, T)
    x, y = x.to(device), y.to(device)
    return x, y
```

- `x` shape：`(B, T)` = `(32, 8)` → 4×8 = 32 个完全独立的样本
- 不同 batch 中的序列彼此**完全独立**，不交叉通信

---

## 4. Bigram 基线模型

### 模型定义

最简单的语言模型：每个 token 只根据**自身**预测下一个 token，完全不看上下文。

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C=vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)    # PyTorch cross_entropy 要求 C 在第二维
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]          # 只取最后一步的预测 (B, C)
            probs  = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
```

> **为什么 `view(B*T, C)`？**
> PyTorch 的 `F.cross_entropy` 对多维输入要求 channel 在第二维（`(N, C)`格式），而我们的 logits 是 `(B, T, C)`，所以需要 reshape。

### 训练循环

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### 损失估计（减少噪声）

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

两个要点：
1. `model.eval()` / `model.train()`：BatchNorm、Dropout 等层在训练/推理模式下行为不同
2. `@torch.no_grad()`：告诉 PyTorch 不保存中间变量，节省内存

### Bigram 结果

| 迭代次数 | 验证损失 |
|---------|---------|
| 初始（随机） | ~4.87（预期 -ln(1/65) ≈ 4.17） |
| 10,000 步训练后 | ~2.5 |

> **初始损失为什么是 4.87 而不是 4.17？**
> 因为权重随机初始化，预测并非均匀分布，存在一定偏差，所以比均匀分布的理论值稍大。

Bigram 的局限：token 之间完全不交流，只看自己就做预测，显然不够好。下面开始引入 self-attention。

---

## 5. 自注意力的数学基础：矩阵乘法加权聚合技巧

在实现 self-attention 之前，Karpathy 先展示了一个关键的数学技巧。

### 目标：让每个 token 能看到它之前所有 token 的信息

最简单的方案：**bag of words**——对过去所有 token 做平均。

```python
# 低效的 for 循环版本
xbow = torch.zeros_like(x)  # (B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # (t+1, C)：从第0个到第t个
        xbow[b, t] = xprev.mean(0)  # 平均成一个 C 维向量
```

### 版本 2：用矩阵乘法高效实现

关键洞察：**下三角权重矩阵 × 数据矩阵 = 加权平均**。

```python
# 构造下三角均匀权重矩阵
wei = torch.tril(torch.ones(T, T))          # 下三角全 1
wei = wei / wei.sum(dim=1, keepdim=True)    # 每行归一化

# (T, T) @ (B, T, C) → PyTorch 广播：(B, T, C)
xbow2 = wei @ x
```

直觉：
- 第 1 行 = `[1, 0, 0, ...]`：只看 token 0
- 第 2 行 = `[0.5, 0.5, 0, ...]`：均匀平均 token 0 和 1
- 第 T 行 = `[1/T, 1/T, ..., 1/T]`：均匀平均所有 token

### 版本 3：用 Softmax 实现（Self-Attention 的雏形）

```python
tril = torch.tril(torch.ones(T, T))
wei  = torch.zeros(T, T)
wei  = wei.masked_fill(tril == 0, float('-inf'))  # 未来位置 → -inf
wei  = F.softmax(wei, dim=-1)                      # 归一化 → 均匀分布

xbow3 = wei @ x
```

> **为什么这个版本更重要？**
> `wei` 目前初始化为零（对应均匀聚合），但**马上就要变成数据依赖的**！不同 token 之间的"亲和度"将由 key-query 点积计算，而不是常数零。这就是 self-attention 的核心思想。

---

## 6. 单头自注意力（Single Head Self-Attention）

### 核心思想

简单平均是极弱的交互方式——每个 token 被等权处理。我们希望：

> 每个 token 能根据**内容**决定自己对哪些过去 token 感兴趣，并从那些 token 收集更多信息。

例如，如果当前 token 是一个元音，它可能想找之前的辅音；如果是某个介词，可能想找名词。这种"寻找"行为应该是**数据依赖**的。

### Query、Key、Value

每个 token 发出三个向量：

| 向量 | 含义 | 类比 |
|------|------|------|
| **Query (Q)** | "我在寻找什么？" | 搜索关键词 |
| **Key (K)** | "我包含什么？" | 文档描述 |
| **Value (V)** | "如果你觉得我有用，我给你这个" | 实际内容 |

亲和度 = Q 与 K 的点积：如果 query 和 key 对齐，点积大，说明这两个 token 相互感兴趣。

### 实现：单头注意力

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # 计算注意力分数（"亲和度"）
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)

        # 因果掩码：未来 token 不能给过去发送信息
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # 加权聚合 value
        v   = self.value(x)  # (B, T, head_size)
        out = wei @ v        # (B, T, head_size)
        return out
```

### 为什么要除以 √head_size？—— Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

> **问题**：如果 Q 和 K 是单位高斯分布（均值 0，方差 1），那么 `Q @ K.T` 的方差 ≈ head_size。
>
> **后果**：head_size 越大，`wei` 的数值越极端 → softmax 趋向 one-hot → 每个 token 只关注一个 token，完全失去多样性，尤其在初始化时非常糟糕。
>
> **修复**：除以 `√head_size` 将方差压回 1，使 softmax 输出在初始化时比较"弥散"（diffuse）。

```python
# Karpathy 的演示
t1 = torch.zeros(3)
t2 = t1 * 8
print(F.softmax(t1, dim=-1))  # 均匀分布
print(F.softmax(t2, dim=-1))  # 接近 one-hot
```

### 关于自注意力的 5 个重要笔记

**笔记 1：注意力是一个通信机制**

可以把它想象成一个**有向图**：节点是 token，边表示信息流向。节点通过数据依赖的加权求和从它指向的节点收集信息。在语言建模场景下，图的结构是固定的下三角形：token $t$ 可以从 token $0, 1, ..., t$ 收集信息，不能从未来收集。

**笔记 2：注意力没有空间概念**

不像卷积有内置的位置归纳偏置，注意力就是作用在一组向量上的运算——它不知道这些向量在序列中的位置。所以**必须手动加入位置编码**，告诉每个 token "你在哪里"。

**笔记 3：Batch 维度的 token 彼此独立**

不同 batch 样本之间永远不通信，batch 矩阵乘法只是在所有 batch 元素上并行执行相同操作。

**笔记 4：Encoder vs Decoder**

| 类型 | 是否有因果掩码 | 用途 |
|------|--------------|------|
| **Decoder block**（本讲） | 有（下三角掩码） | 语言建模、文本生成 |
| **Encoder block** | 无（所有 token 互看） | 情感分析、BERT 等理解任务 |

删掉 `masked_fill` 那行，decoder 就变成了 encoder。

**笔记 5：Self-attention vs Cross-attention**

- **Self-attention**：Q、K、V 来自同一个源 X（token 们彼此交流）
- **Cross-attention**：Q 来自 X，但 K 和 V 来自**另一个外部源**（如 encoder 的输出）——用于条件生成，如机器翻译

### 单头效果

添加单头 self-attention 后：
- 验证损失从 **2.5 → 2.4**（有提升，但还不够）

---

## 7. 多头注意力（Multi-Head Attention）

### 为什么需要多头？

不同的 token 关注不同类型的信息（辅音、元音、位置、语法角色……），单个注意力头只能同时关注一个"角度"。多头注意力就是**并行运行多个注意力头**，每个头独立学习不同的交流模式，最后拼接结果。

类比：就像组卷积（group convolution）——不是一个大卷积，而是多个小组卷积。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)  # 投影回残差路径
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 沿 channel 维度拼接
        out = self.dropout(self.proj(out))
        return out
```

**尺寸关系**：如果 `n_embd = 32`，用 4 个头，则每个头的 `head_size = 8`，4 个头输出拼接后还是 32 维，通道数不变。

### 多头效果

| 配置 | 验证损失 |
|------|---------|
| 单头（n_embd=32） | ~2.4 |
| 4头（head_size=8）| ~2.28 |

---

## 8. 前馈网络（Feed-Forward Network）

### 问题

多头注意力完成了"通信"——token 们彼此交换信息。但刚交换完就立刻预测，没有给 token 时间"思考"刚收到的信息。

> **Karpathy 的比喻**：token 们开会交流了（self-attention），但还没有时间个别消化会议内容（feed-forward）。

### 实现

Feed-forward 是**逐 token 独立**运行的 MLP：

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展 4 倍（来自原始论文）
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 压缩回来
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # 对每个 token 独立应用，token 之间不交流
```

> **为什么内层是 4 × n_embd？**
> 直接来自原始论文："The inner-layer has dimensionality $d_{ff} = 2048$"（是 $d_{model} = 512$ 的 4 倍）。这个 4× 扩展被证明在实践中效果很好。

### 效果

- 加入 feed-forward 后，验证损失：**2.28 → 2.24**

---

## 9. Transformer Block 与残差连接

### 完整的 Transformer Block

**Transformer 的核心模式**：通信（self-attention） + 计算（feed-forward），交替堆叠多次。

```python
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 通信 + 残差连接
        x = x + self.ffwd(self.ln2(x))  # 计算 + 残差连接
        return x
```

### 残差连接（Residual Connections / Skip Connections）

来自 **ResNet（2015）** 论文。核心公式：

$$x \leftarrow x + \text{SubLayer}(x)$$

**为什么残差连接对深层网络至关重要？**

从反向传播的角度理解：加法操作会把梯度**等分**地传递给两个分支。所以存在一条从损失直达输入的"**梯度超级公路**"（gradient superhighway），梯度可以不受阻碍地流回去。残差块就像从主干路上"叉出去"做计算，再"汇合回来"。

关键细节：残差块在**初始化时贡献接近零**（通过合理的权重初始化），所以训练开始时模型几乎就是一条直通路径，优化非常容易。随着训练进行，这些块逐渐"上线"，贡献越来越多。

> **Karpathy**："在我们 micrograd 视频里讲过，加法节点把梯度等量分给两个输入分支——所以监督信号（梯度）可以通过每个加法节点一路跳回输入，同时也叉到残差块里，但至少有这条不受阻的通路。"

### Projection 层

在多头注意力中，各头的输出拼接后需要一个**线性投影层**回到残差路径：

```python
self.proj = nn.Linear(head_size * num_heads, n_embd)
```

同样，feed-forward 的最后一层也是投影回残差路径的操作（`4*n_embd → n_embd`）。

### 加入残差连接后的效果

- 验证损失：**2.24 → 2.08**（显著提升！）
- 开始观察到训练损失低于验证损失 → 模型足够大，开始有轻微过拟合

---

## 10. Layer Normalization

### BatchNorm vs LayerNorm

在 makemore 第三讲里，我们实现了 **BatchNorm**：在 batch 维度上计算均值和方差，保证每个神经元的输出是零均值单位方差。

**LayerNorm** 几乎一样，只是**在特征维度（而非 batch 维度）上归一化**：

```python
# BatchNorm：归一化每一列（跨 batch 看同一神经元）
# LayerNorm：归一化每一行（每个样本的所有特征）
```

转变极其简单——把 `dim=0` 改成 `dim=1` 即可。

| | BatchNorm | LayerNorm |
|-|-----------|-----------|
| 归一化维度 | Batch（列） | Feature（行） |
| 需要 running buffer | 是 | 否 |
| 训练/推理行为差异 | 是 | 否 |
| 适用场景 | 固定 batch size | 序列建模（NLP） |

LayerNorm 更适合 Transformer，因为它在推理时的行为与训练时完全相同，且不需要维护 running statistics。

### Pre-Norm vs Post-Norm

原始论文中 LayerNorm 在子层**之后**（Post-Norm）：
```
x → SubLayer → x + LayerNorm(SubLayer(x))
```

现代实践（本讲采用）是 **Pre-Norm**，在子层**之前**：
```python
x = x + self.sa(self.ln1(x))    # LayerNorm 先作用于 x，再进 attention
x = x + self.ffwd(self.ln2(x))  # LayerNorm 先作用于 x，再进 FFN
```

Pre-Norm 在实践中训练更稳定。

### 最终 LayerNorm

在 Transformer 最后、解码线性层之前，还要加一个额外的 LayerNorm：

```python
self.ln_f = nn.LayerNorm(n_embd)  # 最后的 layer norm

def forward(self, idx, targets=None):
    ...
    x = self.blocks(x)   # 经过所有 Block
    x = self.ln_f(x)     # 最后的归一化
    logits = self.lm_head(x)
```

### LayerNorm 的效果

- 验证损失：**2.08 → 2.06**（轻微改善）
- 预期在更大、更深的网络上帮助更明显

---

## 11. Dropout 正则化

### 什么是 Dropout？

来自 **Srivastava et al., 2014**。在每次前向/后向传播时，**随机关闭一部分神经元**（置为 0），强迫网络学习冗余的表达。

```python
self.dropout = nn.Dropout(dropout)  # dropout = 0.2 → 随机关闭 20% 的激活
```

在本讲中，Dropout 加在三个地方：
1. Feed-forward 网络末尾（投影回残差路径前）
2. 多头注意力的投影层
3. 注意力权重 `wei` 经过 softmax 之后

**理解**：随机关闭的掩码每次前向传播都变，相当于在训练一个**子网络的集成**（ensemble）；测试时所有神经元全开，相当于所有子网络取平均。

> 这是一种正则化技术，在即将大幅扩大模型规模、担心过拟合时加入。

---

## 12. 规模化训练与最终结果

### 最终超参数

```python
# 小模型（教学用，CPU 可跑）
batch_size = 32
block_size = 8
n_embd     = 32
n_head     = 4
n_layer    = 3
dropout    = 0.0
max_iters  = 5000
learning_rate = 1e-3

# 大模型（需要 GPU）
batch_size = 64
block_size = 256   # 上下文扩展：8 → 256
max_iters  = 5000
learning_rate = 3e-4
n_embd     = 384   # 嵌入维度：32 → 384
n_head     = 6     # 每头 384/6 = 64 维
n_layer    = 6     # 6 层 Block
dropout    = 0.2
```

### 完整 GPT 模型结构

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                          # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb   # 广播：(B, T, C) + (T, C)
        x = self.blocks(x)      # 经过 n_layer 个 Block
        x = self.ln_f(x)        # 最终 LayerNorm
        logits = self.lm_head(x)  # (B, T, vocab_size)
        ...
```

**位置编码**：`torch.arange(T)` 生成 `[0, 1, ..., T-1]`，经嵌入表得到位置向量，与 token 嵌入**相加**。这给每个 token 注入了它在序列中的位置信息。

**权重初始化**：Linear 层用标准差 0.02 的正态分布，偏置为零；Embedding 也用正态分布。

### 训练进程总结

| 模型版本 | 验证损失 | 关键变化 |
|---------|---------|---------|
| Bigram | ~2.5 | 无 attention |
| + 单头注意力 | ~2.4 | head_size=32 |
| + 多头注意力（4头） | ~2.28 | 并行通信 |
| + 前馈网络 | ~2.24 | 通信+计算 |
| + 残差连接 | ~2.08 | 深层网络可优化 |
| + Layer Norm | ~2.06 | 训练稳定性 |
| 规模化（大模型） | **~1.48** | A100 GPU，15分钟 |

> 大模型（10M 参数）在 A100 GPU 上训练约 **15 分钟**，验证损失从 2.07 降至 **1.48**。

### 规模化生成样本

```
is here this now grief syn like
this starts to almost look like English
```

虽然语义荒谬，但格式上已经非常像莎士比亚（人物对话、场景切换等）。对于一个只在 100 万字符上训练的字符级模型，这已相当不错。

---

## 13. Decoder-only vs Encoder-Decoder 架构

### 我们实现的是 Decoder-only Transformer

我们的架构**没有**：
- Encoder 模块
- Cross-attention 模块

只有 decoder block（带因果掩码的 self-attention + feed-forward），直接用于语言建模。

**为什么这就够了？**

我们的任务是无条件文本生成——只需要模仿一个数据集（莎士比亚），不需要条件于任何外部输入。

### 原始论文为什么有 Encoder-Decoder？

原始论文是**机器翻译**论文（法语 → 英语）。它需要：

1. **Encoder**：读入法语句子（无因果掩码，所有 token 互看），编码成向量表示
2. **Decoder**：生成英语翻译，通过 cross-attention 条件于 encoder 的输出

```
[法语输入] → Encoder → 键和值（K, V）
                              ↓
[英语生成] → Decoder（Q） → Cross-Attention → 每一层的解码 block
```

Cross-attention 中：
- **Q** 来自 decoder 的当前状态 X
- **K, V** 来自 encoder 的输出

这让 decoder 在生成英语时能"读取"法语上下文。

### 现代 GPT 都是 Decoder-only

GPT-2、GPT-3、GPT-4 都采用 decoder-only 架构（只有下三角掩码的自注意力）。它们通过超大规模预训练学会了理解和生成，无需 encoder。

---

## 14. nanoGPT 代码导览

nanoGPT（[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)）是 Karpathy 的完整 GPT 复现，两个文件各约 300 行：

- **`model.py`**：GPT 模型定义
- **`train.py`**：训练脚本（支持多 GPU、学习率衰减、checkpoint 等）

与本讲代码的主要差异：

| 方面 | 本讲 | nanoGPT |
|------|------|---------|
| 多头实现 | 分离的 `Head` + `MultiHeadAttention` | 全部在一个 `CausalSelfAttention` 里，heads 作为 batch 维度 |
| 激活函数 | ReLU | GELU（为了能加载 OpenAI 权重） |
| 训练特性 | 简单训练循环 | 学习率衰减、梯度裁剪、weight decay、分布式训练 |
| 权重加载 | 无 | 支持加载 GPT-2 预训练权重 |

**更高效的多头实现**（nanoGPT 风格）：把 `num_heads` 当作额外的 batch 维度，一次矩阵乘法处理所有头，无需 for 循环。数学上完全等价，但更高效。

Karpathy 已在 nanoGPT 上验证能复现 GPT-2（124M 参数）的性能。

---

## 15. ChatGPT 的训练流程：预训练与微调

### 阶段一：预训练

和本讲做的事情完全一样，只是规模大得多：

| 指标 | 本讲 | GPT-3 |
|------|------|-------|
| 参数量 | ~10M | 175B（175,000M） |
| 训练 token 数 | ~30 万（莎士比亚） | 3000 亿 |
| 数据 | Tiny Shakespeare | 大部分互联网 |
| 训练时间 | 15 分钟（A100） | 数千 GPU 数月 |

架构几乎相同，只是规模差了 **10,000 到 1,000,000 倍**。

预训练结果：一个**文档补全器**（document completer）。给它一个句子，它会续写。但它不会回答问题——给它一个问题，它可能继续问更多问题，或者补全成某个新闻稿。行为是**未对齐**的。

### 阶段二：微调（SFT + RLHF）

来自 OpenAI 的 ChatGPT 博客，大致三步：

**Step 1：监督微调（SFT）**
- 收集"问题 + 高质量回答"的配对数据（可能只有数千条）
- 在这些数据上继续微调
- 模型开始学会"看到问题，生成答案"的格式
- 大模型样本效率极高，数千条数据就有显著效果

**Step 2：训练奖励模型（Reward Model）**
- 让模型对同一个问题生成多个回答
- 人类评级员对这些回答排序
- 用这些排序数据训练一个奖励模型，能预测"哪个回答更好"

**Step 3：PPO 强化学习**
- 用奖励模型作为信号
- 用 PPO（策略梯度强化学习）优化语言模型
- 使生成的回答能获得更高的奖励模型分数

最终结果：从文档补全器 → 有用的问答助手（ChatGPT）。

> nanoGPT 只覆盖预训练阶段。微调阶段的数据（人类偏好标注）大多是 OpenAI 内部的，难以复现。

---

## 16. 本讲总结

### 核心知识点速查

| 组件 | 作用 | 关键细节 |
|------|------|---------|
| **Token Embedding** | 将 token ID 映射到向量 | shape: `(vocab_size, n_embd)` |
| **Position Embedding** | 注入位置信息 | shape: `(block_size, n_embd)`，与 token emb 相加 |
| **Self-Attention** | Token 之间的数据依赖通信 | Q·K 点积 / √head_size，下三角掩码 |
| **Multi-Head Attention** | 并行多个注意力头 | 每头独立学习，拼接后投影 |
| **Feed-Forward** | 每个 token 独立"思考" | 4× 扩展，ReLU，再压缩 |
| **Residual Connection** | 让梯度流通，使深层网络可训练 | `x = x + SubLayer(x)` |
| **Layer Norm** | 稳定每层的激活分布 | Pre-Norm：在子层前归一化 |
| **Dropout** | 防止过拟合 | 训练时随机置零 |

### 损失改善进程

```
Bigram (2.5)
  → + Single Head Attention (2.4)
  → + Multi-Head Attention (2.28)
  → + Feed-Forward (2.24)
  → + Residual Connections (2.08)
  → + Layer Norm (2.06)
  → Scale Up (1.48) ← A100 GPU, 15 min
```

### 关键直觉

1. **Self-attention = 有向图上的数据依赖通信**：每个节点根据内容决定从哪些过去节点收集多少信息
2. **因果掩码** 是 decoder（生成模型）的核心，防止未来信息泄露
3. **残差连接** 是深层网络可训练的关键，提供"梯度超级公路"
4. **Decoder-only Transformer** ≈ GPT 系列的核心架构
5. **ChatGPT = 大规模预训练 + 监督微调 + RLHF**

### 课后练习（来自视频描述）

- **EX1**：将 `Head` 和 `MultiHeadAttention` 合并为一个类，把 heads 当作 batch 维度并行处理（参考 nanoGPT 的实现）
- **EX2**：在你自己的数据集上训练 GPT。进阶：训练 GPT 做加减乘除，用 `y=-1` 掩掉输入位置的 loss（`CrossEntropyLoss ignore_index`）
- **EX3**：找一个很大的数据集让 train/val loss 不发散，预训练后再在 Tiny Shakespeare 上 finetune——预训练能降低验证损失吗？
- **EX4**：读 Transformer 相关论文，实现一个新特性，验证是否改善结果

---

## 附：关键代码速查

### 完整 GPT 超参数（大模型版）

```python
batch_size    = 64
block_size    = 256   # context length
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.2
```

### 单头自注意力

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v     # (B, T, hs)
```

### Transformer Block

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 通信 + 残差
        x = x + self.ffwd(self.ln2(x))  # 计算 + 残差
        return x
```

### 文本生成

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # 裁剪到 block_size（位置编码的范围）
        idx_cond = idx[:, -block_size:]
        logits, loss = self(idx_cond)
        logits   = logits[:, -1, :]          # 只取最后时间步 (B, C)
        probs    = F.softmax(logits, dim=-1)  # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx      = torch.cat((idx, idx_next), dim=1)        # (B, T+1)
    return idx

# 启动生成（从换行符开始）
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

### 损失的无偏估计

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```
