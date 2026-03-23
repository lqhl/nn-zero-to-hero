# Lecture 6：构建 WaveNet —— 层次化卷积语言模型

> 对应视频：[YouTube](https://youtu.be/t3YJ5hKiMQ0)
> 对应 Notebook：[makemore_part5_cnn1.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)

---

## 目录

1. 本讲目标与背景
2. 回顾：从 Part 3 继承的代码框架
3. 代码整理：将 Embedding 和 Flatten 模块化
4. 引入 Sequential 容器
5. 扁平架构的瓶颈：为什么需要层次化
6. WaveNet 的层次化融合思想
7. FlattenConsecutive 的实现
8. 线性层对高维张量的处理（意外的惊喜）
9. 构建层次化网络
10. BatchNorm1d 的 Bug 修复（3D 输入）
11. 规模扩大与性能突破
12. 卷积简介与预告
13. 深度学习开发流程心得
14. 本讲总结

---

## 1. 本讲目标与背景

### 从哪里来，要去哪里

前几讲我们一步步搭建了一个多层感知机（MLP）字符级语言模型，验证损失已经到达 2.10。这不差，但还有很多改进空间。

本讲的目标是**让模型更深、上下文更长**，同时以一种更聪明的方式来利用这些上下文——而不是粗暴地把所有字符一次性压扁到单层里。

我们最终会实现一个架构，它和 DeepMind 2016 年发表的 **WaveNet** 论文中的架构在结构上非常相似。

### WaveNet 是什么

WaveNet 本质上也是一个自回归语言模型——只不过它建模的是**音频序列**而非字符序列，但建模框架完全相同：给定前面的内容，预测下一个元素。

它的核心创新是用一种**树状层次结构**来处理上下文，而不是把所有信息一股脑地压进单层网络。

---

## 2. 回顾：从 Part 3 继承的代码框架

### 起点代码

本讲的起点代码几乎是从 Part 3 直接复制过来的（Part 4 是反向传播练习，是一个"支线任务"）。已有的模块：

```python
class Linear:       # 矩阵乘法层，Kaiming 初始化
class BatchNorm1d:  # 批归一化层（有 running_mean / running_var）
class Tanh:         # 逐元素激活层
```

### 当前性能基准

| 配置 | 参数量 | train loss | val loss |
|------|--------|-----------|---------|
| 原始（3 字符上下文，200 隐藏单元）| ~12K | 2.058 | 2.105 |

### BatchNorm 的注意事项（复习）

BatchNorm 有几个"坑"值得重温：

1. **训练/推断状态**：必须手动设置 `layer.training = False` 才会用 running stats
2. **样本间耦合**：batch 内样本的激活值会互相影响，这是设计的一部分
3. **单样本推断**：若 BatchNorm 处于训练模式，传入单个样本会导致方差为 NaN

> **Karpathy 的话：** "BatchNorm 是个非常奇怪的层，引入了很多 bug 的可能……因为你要维护训练和评估两种状态，一旦忘记切换，结果就会悄悄出错，不报错但是算的是错的。"

---

## 3. 代码整理：将 Embedding 和 Flatten 模块化

### 问题：前向传播太"裸"

原来的前向传播里有两处特殊处理游离在 `layers` 列表之外：

```python
# 旧的前向传播（有点乱）
emb = C[Xb]                     # Embedding，单独处理
x = emb.view(emb.shape[0], -1)  # Flatten，单独处理
for layer in layers:
    x = layer(x)
```

### 解决方案：为它们创建专属模块

```python
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []
```

**对应的 PyTorch 原版：**
- `Embedding` → `torch.nn.Embedding(num_embeddings, embedding_dim, ...)`
- `Flatten` → `torch.nn.Flatten(...)`

两者都存在于 PyTorch 中，只是 PyTorch 版本接受更多关键字参数。

### 整理后的前向传播

```python
# 整洁的前向传播
for layer in layers:
    x = layer(x)  # 输入直接是整数 tensor Xb
```

原来特殊处理的部分都被收进了 `layers` 列表，代码干净多了。

---

## 4. 引入 Sequential 容器

### 什么是 Sequential

`torch.nn.Sequential` 是 PyTorch 中的一个**容器模块**，按顺序将输入传过所有子层。我们来实现自己的版本：

```python
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### 使用方式

```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

# 前向传播极度简化
logits = model(Xb)
loss = F.cross_entropy(logits, Yb)
```

### 调试引入的 Bug

整理代码时 Karpathy 犯了一个经典错误：修改了单元格代码但**忘记重新运行**，导致：

- BatchNorm 仍处于训练模式
- 评估时传入单样本 → 方差 = NaN → 结果污染

> **教训：** 修改了关键单元格一定要重新执行，PyTorch 不会报错，但结果会悄悄出错。

---

## 5. 扁平架构的瓶颈：为什么需要层次化

### 现有方式：一次性压扁

```
[c1, c2, c3, c4, c5, c6, c7, c8]  → Flatten → Linear(80 → 200) → ...
```

把 8 个字符（每个 10 维）拼成一个 80 维向量，直接送入单个隐藏层。

**问题：** 信息一下子被压扁太厉害了。第一层线性层要同时学会"理解"所有 8 个字符的关系，负担很重。

> **Karpathy 的话：** "就算把这个层做得很大、加更多神经元，在单步里把所有信息压扁这件事本身就很蠢。"

### 做大上下文长度的初步尝试

先把 `block_size` 从 3 改为 8，其他不变：

| 配置 | 参数量 | val loss |
|------|--------|---------|
| block_size=3 | ~12K | 2.105 |
| block_size=8（扁平） | ~22K | 2.027 |

仅仅扩大上下文长度就从 2.105 → 2.027，改善明显。但架构本身还是"暴力压扁"的，能做得更好。

---

## 6. WaveNet 的层次化融合思想

### 核心思想：渐进融合

WaveNet 的灵感在于，不要把所有字符一次性压扁，而是**像一棵树一样，两两配对，逐层融合**：

```
层 1：字符对 → bigrams
         [c1,c2] [c3,c4] [c5,c6] [c7,c8]
            ↓       ↓       ↓       ↓
层 2：bigram 对 → 4-gram 块
         [b1,b2]         [b3,b4]
            ↓               ↓
层 3：4-gram 对 → 8-gram（整个上下文）
                [g1,g2]
                   ↓
              最终预测
```

每一层只融合**相邻的两个**元素，信息是**渐进地**从底层向上融合的，而不是在第一层就全部压扁。

### 与 WaveNet 论文的关系

WaveNet 使用的是**因果扩张卷积**（causal dilated convolutions），这是一种高效实现上述层次结构的方式。我们现在先实现其**朴素版本**，理解结构本身，卷积的高效实现留到后面。

---

## 7. FlattenConsecutive 的实现

### 需求分析

原来的 `Flatten` 把 `(B, T, C)` → `(B, T*C)`，把所有时间步全部展开。

新的 `FlattenConsecutive(n)` 只把**相邻 n 个**时间步拼在一起：

```
输入 (B, T, C)  →  输出 (B, T//n, C*n)
```

例如，`n=2`：
```
(4, 8, 10)  →  (4, 4, 20)
```

### 关键洞察：view 就够了，不需要 cat

你可能想用 `torch.cat` 来显式拼接奇偶位置：

```python
# 显式拼接（可行但繁琐）
even = e[:, 0::2, :]  # (4, 4, 10)
odd  = e[:, 1::2, :]  # (4, 4, 10)
explicit = torch.cat([even, odd], dim=2)  # (4, 4, 20)
```

但实际上，直接 `view` 就能得到完全相同的结果：

```python
e.view(4, 4, 20) == explicit  # 全部为 True
```

**原因：** PyTorch 的内存布局是行优先的（C-contiguous），相邻时间步的嵌入向量在内存中本来就是连续存放的，`view` 只是换一种解读方式，不做任何数据复制。

### 完整实现

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)  # 去掉多余的长度维度
        self.out = x
        return self.out

    def parameters(self):
        return []
```

**squeeze 的原因：** 最后一层 `FlattenConsecutive` 可能会把长度压到 1，产生 `(B, 1, C*n)` 的形状。我们不想要这个多余的维度 1，用 `squeeze(1)` 把它去掉，得到 `(B, C*n)`，让后续线性层可以正常处理。

> **为什么显式指定维度：** Karpathy 说他喜欢尽量显式，而不是用 `squeeze()` 不加参数（会把所有维度为1的维度都去掉），这样出了问题更容易发现。

---

## 8. 线性层对高维张量的处理

### 令人惊喜的特性

PyTorch 的矩阵乘法（`@` 运算符和 `torch.nn.Linear`）支持**任意批维度**。

```python
# 原来：2D 输入
(4, 80)  @ (80, 200)  =  (4, 200)

# 现在：3D 输入（多出一个批维度）
(4, 4, 20)  @ (20, 200)  =  (4, 4, 200)

# 甚至更多维
(4, 5, 20)  @ (20, 200)  =  (4, 5, 200)
```

**规则：** 矩阵乘法只作用于**最后一个维度**，前面所有维度都被视为批维度，并行处理。

### 对我们的意义

这意味着：我们可以直接把 `(B, T//n, C*n)` 形状的张量送进 `Linear` 层，不需要做任何额外的 reshape。Linear 层会在所有 `T//n` 个位置上**并行地**做矩阵乘法。

```python
# 第一层：把每对字符 (每个20维) 融合成 128 维
x = FlattenConsecutive(2)(x)  # (B, 4, 20)
x = Linear(20, 128, bias=False)(x)  # (B, 4, 128)

# 第二层：把每对 bigram 表示融合
x = FlattenConsecutive(2)(x)  # (B, 2, 256)
x = Linear(256, 128, bias=False)(x)  # (B, 2, 128)

# 第三层：最终融合
x = FlattenConsecutive(2)(x)  # (B, 256) — squeeze 去掉维度1
x = Linear(256, 128, bias=False)(x)  # (B, 128)
```

---

## 9. 构建层次化网络

### 完整的模型定义

```python
n_embd = 24    # 字符嵌入维度
n_hidden = 128 # 隐藏层宽度

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
```

### 张量形状追踪（以 B=4, block_size=8, n_embd=10 为例）

| 层 | 输出形状 | 说明 |
|----|---------|------|
| `Embedding` | `(4, 8, 10)` | 每个字符 → 10 维向量 |
| `FlattenConsecutive(2)` | `(4, 4, 20)` | 每对字符拼成 20 维 |
| `Linear(20→128)` | `(4, 4, 128)` | 并行处理 4 个位置 |
| `BatchNorm1d` | `(4, 4, 128)` | 批归一化 |
| `Tanh` | `(4, 4, 128)` | 逐元素激活 |
| `FlattenConsecutive(2)` | `(4, 2, 256)` | bigram 对拼成 256 维 |
| `Linear(256→128)` | `(4, 2, 128)` | 并行处理 2 个位置 |
| `BatchNorm1d` | `(4, 2, 128)` | |
| `Tanh` | `(4, 2, 128)` | |
| `FlattenConsecutive(2)` | `(4, 256)` | **squeeze 生效**，去掉维度1 |
| `Linear(256→128)` | `(4, 128)` | |
| `BatchNorm1d` + `Tanh` | `(4, 128)` | |
| `Linear(128→27)` | `(4, 27)` | 输出 logits |

### 参数量对比

| 架构 | n_embd | n_hidden | 参数量 | val loss |
|------|--------|----------|--------|---------|
| 扁平（block=3） | 10 | 200 | ~12K | 2.105 |
| 扁平（block=8） | 10 | 68 | ~22K | 2.027 |
| 层次化（block=8） | 10 | 68 | ~22K | 2.029 |
| 层次化 + bug修复 | 10 | 68 | ~22K | 2.022 |
| 层次化 + 规模扩大 | 24 | 128 | ~76K | **1.993** |

> **注意：** 初始切换到层次化架构时，性能和扁平架构几乎一样（2.029 vs 2.027）。Karpathy 指出这只是"我的第一次猜测"，真正精调超参数之后层次化架构应该能更明显地超过扁平架构。

---

## 10. BatchNorm1d 的 Bug 修复（重要！）

### 问题发现

当 `BatchNorm1d` 接收到 3D 输入 `(B, T, C)` 时，原来的实现只在第 0 维求均值：

```python
# 旧实现（有 bug）
xmean = x.mean(0, keepdim=True)  # 只在 batch 维度求均值
```

**结果：** 对于形状 `(32, 4, 68)` 的输入：
- 实际计算：`xmean.shape = (1, 4, 68)`
- 含义：对每个位置（4个位置）分别维护独立的均值/方差
- running_mean 变成了 `(1, 4, 68)` 而不是 `(68,)`

**后果：** BatchNorm 实际上在对 `4 × 68 = 272` 个"通道"分别做归一化，每个通道只用 32 个样本来估计统计量，而不是用 `32 × 4 = 128` 个样本来估计 68 个通道的统计量。

### 根本原因

BatchNorm 的语义是：对**特征维度**（最后一维）做归一化，前面所有维度都是"样本"。所以对于 3D 输入，应该把维度 0 和维度 1 都视为批维度：

$$\mu_c = \frac{1}{B \times T} \sum_{b=1}^{B} \sum_{t=1}^{T} x_{b,t,c}$$

### 修复方案

```python
class BatchNorm1d:
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0        # 2D 输入：(B, C)
            elif x.ndim == 3:
                dim = (0, 1)   # 3D 输入：(B, T, C)，对前两维求均值
            xmean = x.mean(dim, keepdim=True)
            xvar  = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        # ... 其余不变
```

修复后 `running_mean.shape` 从 `(1, 4, 68)` 变为 `(1, 1, 68)`，正确地只维护 68 个通道的统计量。

### 与 PyTorch 官方 BatchNorm1d 的差异

PyTorch 的 `BatchNorm1d` 对 3D 输入期望的格式是 `(N, C, L)` —— 通道 C 在中间。
我们的实现期望 `(N, L, C)` —— 通道 C 在最后。

这是两种合理的约定，Karpathy 表示更喜欢"通道在最后"的风格，所以我们保留自己的实现。

> **Bug 修复的效果：** val loss 从 2.029 → 2.022，提升不大但说明确实有影响。因为修复后每个统计量是用更多样本（`32×4=128` 个）来估计的，统计更稳定。

---

## 11. 规模扩大与性能突破

### 最终模型配置

```python
n_embd = 24    # 嵌入维度（从 10 增加到 24）
n_hidden = 128 # 隐藏层（适当增大）
# 总参数量：76,579
```

### 训练设置（与之前相同）

```python
max_steps = 200_000
batch_size = 32
# 学习率：前 150K 步用 0.1，之后用 0.01
lr = 0.1 if i < 150_000 else 0.01
```

### 训练过程

```
      0/ 200000: 3.3167
  10000/ 200000: 2.0576
  50000/ 200000: 1.7836
 100000/ 200000: 1.7736
 150000/ 200000: 1.7466   # 学习率衰减后大幅下降
 190000/ 200000: 1.8555
```

### 最终性能

```
train 1.7690
val   1.9937
```

**val loss < 2.0！** 这是本讲的重要里程碑。

### 样本生成（定性评估）

```
arlij.    chetta.   heago.   rocklei.
hendrix.  jamylie.  broxin.  denish.
anslibt.  marianah. astavia. annayve.
aniah.    jayce.    nodiel.  remita.
niyelle.  jaylene.  aiyan.   aubreana.
```

名字已经相当像样了，虽然还有一些奇怪的组合，但整体质量明显提升。

### 平滑 Loss 曲线的技巧

```python
# 原来：直接 plot 每步 loss（噪声极大）
plt.plot(lossi)

# 改进：每 1000 步取均值
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
```

这利用了 PyTorch `view` 的技巧：把 200,000 个值 view 成 200×1000 的矩阵，然后沿行求均值，得到 200 个平均值，曲线就平滑多了。

---

## 12. 卷积简介与预告

### 当前的"朴素"实现

以名字 "diondre" 为例，它有 8 个独立的预测任务（每个位置预测下一个字符）。我们现在的做法是：

```python
# 8 次独立调用
for i in range(8):
    logits[i] = model(Xtr[[7+i]])  # 8 次单独的前向传播
```

这是 8 次**独立的**前向计算，效率低下。

### 卷积的本质

> **Karpathy 的话：** "卷积本质上就是一个 for 循环——把一个小线性滤波器沿着输入序列滑动。"

卷积允许我们把这个滑动的 for 循环从 Python 移到 CUDA 内核里，极大地提升效率。

### 参数复用的优势

在层次化树结构中，某些中间节点被多个上层节点共享：

```
位置的 bigram 表示
    ↑           ↑
[c1,c2]→b1  [c2,c3]→b2   ← c2 的嵌入被用了两次！
```

在朴素实现中，`c2` 的嵌入要被重复计算；卷积实现可以自然地复用这些中间结果。

> **预告：** 下一讲将实现真正的因果扩张卷积（causal dilated convolution），那才是 WaveNet 中高效实现这个层次结构的方式。

---

## 13. 深度学习开发流程心得

Karpathy 在收尾时分享了他的实际工作方式，值得认真记录：

### 工具使用

1. **大量阅读文档**：反复查 PyTorch 文档，关注层的输入输出形状、参数含义
2. **文档不可全信**：

> **Karpathy 的话：** "PyTorch 的文档非常糟糕……它会骗你，会出错，会不完整，会不清晰。我们只能凑合着用，尽力而为。"

### 张量形状是核心难点

- 大量时间花在"跑通各维度的形状"上
- NCL 还是 NLC？什么层接受什么形状？Broadcasting 正确了吗？
- 一旦形状出错，可能静默地给出错误结果（就像 BatchNorm 的 bug 一样）

### 开发工作流

```
Jupyter Notebook（原型）
    ↓ 验证形状正确、功能符合预期
VS Code（代码库）
    ↓ 粘贴进去，清理代码
训练脚本（正式跑实验）
```

> Karpathy 通常同时开着 Jupyter 和 VS Code：在 Jupyter 里开发，确认无误后粘贴到 VS Code 里，然后从代码仓库跑完整训练。

### 当前的局限

- **没有实验框架**：现在只是"猜一个，跑一下，看损失"
- 正规的深度学习工作流需要：大量超参数搜索、同时看训练和验证曲线、有结构地比较不同配置

---

## 14. 本讲总结

### 性能进展全貌

| 里程碑 | 关键改动 | val loss |
|--------|---------|---------|
| 基准 | block=3, 单层 MLP, 12K 参数 | 2.105 |
| 扩大上下文 | block_size: 3 → 8 | 2.027 |
| 引入层次化架构 | FlattenConsecutive(2) × 3 层 | 2.029 |
| 修复 BatchNorm bug | 3D 输入时 reduce over (0,1) | 2.022 |
| 规模扩大 | n_embd=24, n_hidden=128, 76K 参数 | **1.993** |

### 关键技术点回顾

| 概念 | 要点 |
|------|------|
| **FlattenConsecutive** | `view(B, T//n, C*n)` 实现相邻 n 个时间步的拼接 |
| **3D 线性层** | PyTorch 矩阵乘法自动广播批维度，`(B, T, C) @ (C, D) = (B, T, D)` |
| **BatchNorm 3D fix** | 输入 `(B, T, C)` 时，reduce 维度应为 `(0, 1)` 而非 `0` |
| **层次化融合** | 像树一样，逐层两两配对，信息渐进融合 |
| **卷积预告** | 卷积 = 在序列上滑动的线性滤波器，for loop 移入 CUDA 核心 |

### 本讲"解锁"的事情

1. **torch.nn 基本摸透了**：我们已经从头实现了 `Linear`、`BatchNorm1d`、`Tanh`、`Embedding`、`Flatten`、`Sequential`，理解了 PyTorch 模块系统的核心工作方式
2. **层次化架构思路**：不要一次性压扁信息，让网络渐进地融合
3. **开发流程**：Jupyter → VS Code → 训练脚本

### 后续方向（Karpathy 预告）

1. 卷积神经网络：用因果扩张卷积高效实现这个层次结构
2. 残差连接和跳跃连接（Skip connections）
3. 实验框架搭建（超参数搜索、评估流水线）
4. RNN、LSTM、GRU
5. Transformer

---

## 附：关键代码速查

### FlattenConsecutive

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []
```

### BatchNorm1d（支持 2D 和 3D 输入）

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)   # ← 关键修复
            xmean = x.mean(dim, keepdim=True)
            xvar  = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

### 完整层次化模型

```python
n_embd = 24
n_hidden = 128

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])

# 最后一层权重缩小，避免初始时过于自信
with torch.no_grad():
    model.layers[-1].weight *= 0.1
```

### 训练循环

```python
max_steps = 200_000
batch_size = 32

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 150_000 else 0.01
    for p in model.parameters():
        p.data += -lr * p.grad
```

### 评估与采样

```python
# 切换为评估模式（BatchNorm 使用 running stats）
for layer in model.layers:
    layer.training = False

# 评估损失
@torch.no_grad()
def split_loss(split):
    x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev)}[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

# 生成样本
context = [0] * block_size
while True:
    logits = model(torch.tensor([context]))
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1).item()
    context = context[1:] + [ix]
    if ix == 0:
        break
```

### 平滑 Loss 曲线

```python
# 每 1000 步取均值，去掉噪声
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
```
