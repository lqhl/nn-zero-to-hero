# Lecture 3：用 MLP 构建字符级语言模型（makemore Part 2）

> 对应视频：[YouTube](https://youtu.be/TCH_1BHY58I)
> 对应 Notebook：[makemore_part2_mlp.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)

---

## 目录

1. 本讲目标与背景
2. 从二元模型到 MLP：为什么要升级？
3. Bengio 2003 论文的核心思路
4. 构建数据集：上下文窗口（block_size）
5. 嵌入查找表（Embedding Lookup Table）
6. One-Hot 编码 vs. 直接索引的等价性
7. `torch.Tensor.view()` 的高效重塑
8. 构建隐藏层和输出层
9. 为什么用 `F.cross_entropy` 而非手写
10. 训练循环：小批量（Mini-batch）梯度下降
11. 如何确定学习率：指数搜索法
12. 学习率衰减（Learning Rate Decay）
13. 数据集划分：训练集 / 验证集 / 测试集
14. 欠拟合 vs. 过拟合：如何诊断
15. 扩大模型规模：突破性能瓶颈
16. 嵌入向量的可视化
17. 从模型中采样
18. 本讲总结

---

## 1. 本讲目标与背景

本讲在第 2 讲（二元字符语言模型）的基础上，引入了 **多层感知机（MLP）** 来预测下一个字符。核心动机是：二元模型只看一个字符的上下文，效果很差；而 MLP 可以利用更长的上下文，并且参数量不会指数爆炸。

**本讲核心技能：**
- 理解嵌入查找表的原理与 PyTorch 实现
- 掌握 mini-batch 梯度下降、学习率搜索、学习率衰减
- 学会划分训练 / 验证 / 测试集，并用验证集调超参数
- 理解欠拟合与过拟合，知道如何通过扩大模型规模改善

---

## 2. 从二元模型到 MLP：为什么要升级？

### 现象 / 问题

二元模型（Bigram）只能利用 **1 个** 前置字符来预测下一个字符。如果想利用更多上下文，统计计数表会指数级膨胀：

| 上下文长度 | 可能的上下文数量 |
|-----------|----------------|
| 1 个字符   | 27             |
| 2 个字符   | 27² = 729      |
| 3 个字符   | 27³ ≈ 20,000   |

每种上下文对应的样本数太少，计数极度稀疏，整个方法就崩溃了。

### 根本原因

统计方法的参数数量随上下文长度**指数增长**，不可扩展。

### 解决方案

用神经网络（MLP）来处理更长的上下文，参数量仅线性增长。

> **Karpathy 的话：** "things quickly blow up and this table grows exponentially with the length of the context… that's why today we're going to implement a multi-layer perceptron model."

---

## 3. Bengio 2003 论文的核心思路

本讲的 MLP 架构直接来自论文：**Bengio et al. 2003, "A Neural Probabilistic Language Model"**。

### 论文的核心洞察

不要把每个词/字符当成独立的离散符号，而是把它们**嵌入到一个低维连续空间**中。

- 原始词汇量：17,000 个词（论文中是词级别，我们是字符级别 27 个）
- 每个词映射到一个 **30 维** 的嵌入向量
- 这些嵌入向量通过反向传播训练，**语义相近的词会在空间中靠近**

### 为什么嵌入有泛化能力？

论文中给出的直觉例子：

> 假设训练集里没有 "a dog was running in a ___"，但有 "the dog was running in a ___"。
> 如果模型学会了 "a" 和 "the" 的嵌入很接近，就可以把在 "the" 上学到的知识**迁移**到 "a" 上，从而泛化到未见过的句子。

同理，"cat" 和 "dog" 在嵌入空间中会靠近，因为它们在相似的上下文中出现。

### 论文的网络架构

```
输入层：前 3 个词/字符的索引
  ↓ 查找嵌入矩阵 C
嵌入层：3 × 30 = 90 维输入
  ↓ 全连接 + tanh
隐藏层：100 个神经元
  ↓ 全连接
输出层：27 个 logits（字符级）
  ↓ softmax
概率分布
```

**嵌入矩阵 C 在所有输入位置上共享**，这是一个关键设计。

---

## 4. 构建数据集：上下文窗口（block_size）

### 代码实现

```python
block_size = 3  # 用前 3 个字符预测下一个

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size  # 用 '.' (索引0) 填充初始上下文
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # 滑动窗口：丢掉最老的，加入最新的
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
```

### 数据集示例（以单词 "emma" 为例）

| 上下文（X）   | 标签（Y） |
|-------------|---------|
| `[., ., .]` | `e`     |
| `[., ., e]` | `m`     |
| `[., e, m]` | `m`     |
| `[e, m, m]` | `a`     |
| `[m, m, a]` | `.`     |

### 数据集规模

- 完整数据集（所有 32,000 个名字）：约 **228,000** 个训练样本
- X shape：`[228000, 3]`，Y shape：`[228000]`

---

## 5. 嵌入查找表（Embedding Lookup Table）

### 核心概念

嵌入矩阵 `C` 是一个形状为 `[27, 2]`（或更高维）的可训练参数矩阵。每个字符对应矩阵中的一行，即它的嵌入向量。

```python
C = torch.randn((27, 2))  # 27 个字符，每个嵌入到 2 维空间
```

### PyTorch 的强大索引

PyTorch 支持用多维整数张量直接索引：

```python
# 同时嵌入所有 228000 × 3 个整数！
emb = C[X]       # X shape: [228000, 3] → emb shape: [228000, 3, 2]
```

**这一行代码完成了所有字符的批量嵌入查找。**

---

## 6. One-Hot 编码 vs. 直接索引的等价性

### 两种方式完全等价

**方式一：直接索引**
```python
C[5]  # 直接取第 5 行
```

**方式二：One-Hot 乘法**
```python
x_onehot = F.one_hot(torch.tensor(5), num_classes=27).float()
x_onehot @ C  # 矩阵乘法，结果与 C[5] 相同
```

### 为什么等价？

One-Hot 向量只有第 5 位是 1，其余都是 0。矩阵乘法时，0 把所有行都屏蔽掉，只有第 5 行被选出。

### 重要含义

嵌入层可以**理解为神经网络的第一层**：
- 权重矩阵 = C
- 输入 = one-hot 编码
- 激活函数 = 无（线性层）

但直接索引比 one-hot 乘法**快得多**，所以实践中总是直接用索引。

---

## 7. `torch.Tensor.view()` 的高效重塑

### 问题

嵌入后的张量形状是 `[32, 3, 2]`，但隐藏层的权重矩阵 W1 期望输入是 `[32, 6]`（把 3 个 2 维嵌入拼接成 6 维向量）。

### 低效方案：`torch.cat`

```python
# 把三个嵌入分别取出再拼接
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)
# 问题：会创建新的内存，效率低
```

### 高效方案：`.view()`

```python
emb.view(-1, 6)  # [32, 3, 2] → [32, 6]，零拷贝！
```

### `.view()` 的内部原理

PyTorch tensor 的底层存储始终是**一维的连续内存**。`.view()` 只是修改了 tensor 的元数据（shape、stride、offset），**没有移动任何数据**，因此极其高效。

```python
a = torch.arange(18)  # shape: [18]
a.view(2, 9)    # → [2, 9]，无数据复制
a.view(3, 3, 2) # → [3, 3, 2]，无数据复制
a.view(9, 2)    # → [9, 2]，无数据复制
# 只要元素总数相同就能 view
```

> **Karpathy 的话：** "no memory is being changed, copied, moved or created when we call that view. The storage is identical."

### 使用 `-1` 让 PyTorch 自动推断维度

```python
emb.view(-1, 30)  # PyTorch 自动算出第一维应该是多少
```

---

## 8. 构建隐藏层和输出层

### 完整前向传播

```python
# 参数初始化
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10), generator=g)   # 嵌入矩阵：27字符 × 10维
W1 = torch.randn((30, 200), generator=g)  # 隐藏层权重：30输入 × 200神经元
b1 = torch.randn(200, generator=g)        # 隐藏层偏置
W2 = torch.randn((200, 27), generator=g)  # 输出层权重：200 × 27字符
b2 = torch.randn(27, generator=g)         # 输出层偏置
parameters = [C, W1, b1, W2, b2]

# 前向传播
emb = C[X]                        # [N, 3, 10] — 嵌入
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # [N, 200] — 隐藏层激活
logits = h @ W2 + b2              # [N, 27] — 输出 logits
loss = F.cross_entropy(logits, Y) # 标量损失
```

### 广播注意事项

`emb.view(-1, 30) @ W1` 的形状是 `[N, 200]`，加上 `b1` 的形状 `[200]`，PyTorch 会自动广播为 `[1, 200]` 再扩展到每一行。这是正确的：同一个偏置向量加到每个样本上。

> **养成好习惯：** 每次写广播代码，检查一下形状是否符合预期，避免踩坑。

---

## 9. 为什么用 `F.cross_entropy` 而非手写

手写的交叉熵：

```python
# 手写版（仅用于教学）
counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
loss = -prob[torch.arange(N), Y].log().mean()
```

使用 PyTorch 内置版：

```python
loss = F.cross_entropy(logits, Y)  # 推荐！
```

### 三大优势

#### 1. 前向传播更高效
PyTorch 会把多个操作融合成一个 **fused kernel**，减少中间张量的内存分配。

#### 2. 反向传播更高效
数学上可以推导出更简洁的梯度表达式，避免逐步反传。（就像 `tanh` 的梯度 `1 - t²` 比展开计算简单得多。）

#### 3. 数值稳定性

当 logits 很大（如 100）时，`exp(100)` 会溢出为 `inf`，导致 NaN：

```python
# 危险！
logits = torch.tensor([100.0, 2.0, -3.0])
counts = logits.exp()  # [inf, ...]  → 出现 NaN！
```

`F.cross_entropy` 内部会先减去最大值：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - \max}}{\sum_j e^{x_j - \max}}$$

减去任何常数不影响结果，但能确保指数值不溢出：

```python
# PyTorch 内部做法（伪代码）
max_val = logits.max()
adjusted = logits - max_val  # 最大值变为 0，其余都是负数
probs = adjusted.exp() / adjusted.exp().sum()  # 安全！
```

---

## 10. 训练循环：小批量（Mini-batch）梯度下降

### 为什么要用 Mini-batch

用全部 228,000 个样本做一次前向传播很慢。实际上，用 **32 个样本的小批量** 估计的梯度方向已经足够好，可以更快迭代。

### 完整训练循环

```python
for i in range(200000):
    # 1. 随机采样 mini-batch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # 2. 前向传播
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # 3. 反向传播
    for p in parameters:
        p.grad = None  # 清零梯度（等价于 zero_grad）
    loss.backward()

    # 4. 参数更新
    lr = 0.1 if i < 100000 else 0.01  # 后半段学习率衰减
    for p in parameters:
        p.data += -lr * p.grad
```

### 参数总量

```python
sum(p.nelement() for p in parameters)  # 约 11,897 个参数
```

> **Karpathy 的话：** "the quality of our gradient is lower, the direction is not as reliable… but the gradient direction is good enough even when estimating on only 32 examples."

---

## 11. 如何确定学习率：指数搜索法

### 方法

不要猜学习率。用指数扫描法系统地找到合适范围：

```python
# 准备候选学习率（对数均匀分布，从 0.001 到 1）
lre = torch.linspace(-3, 0, 1000)  # 指数从 -3 到 0
lrs = 10 ** lre                     # 实际学习率：0.001 → 1

lri = []
lossi = []

for i in range(1000):
    # ... 前向、反向 ...
    lr = lrs[i]
    # ... 参数更新 ...

    lri.append(lre[i])   # 记录指数（用于 x 轴）
    lossi.append(loss.item())

# 画图：x 轴是学习率指数，y 轴是损失
plt.plot(lri, lossi)
```

### 如何解读图像

- **左侧（学习率很小）**：损失几乎不下降 → 学习率太小
- **中间某个区间**：损失快速下降且稳定 → 好的学习率区间（通常在指数 -1 附近，即 lr ≈ 0.1）
- **右侧（学习率太大）**：损失不降反升或剧烈震荡 → 学习率太大

**结论：本例中 lr = 0.1 是合适的初始学习率。**

---

## 12. 学习率衰减（Learning Rate Decay）

### 做法

训练初期用较大学习率快速收敛，后期用小学习率精细调整：

```python
# 前 100,000 步
lr = 0.1

# 后 100,000 步（学习率衰减 10 倍）
lr = 0.01
```

### 效果

| 阶段 | 学习率 | 验证集损失 |
|------|--------|-----------|
| 初始随机 | — | ~17 |
| 训练中（lr=0.1） | 0.1 | ~2.3 |
| 学习率衰减后（lr=0.01） | 0.01 | ~2.17 |
| 二元模型基线（上讲）| — | 2.45 |

学习率衰减后，损失从 2.3 降到 **2.17**，显著优于二元模型的 2.45。

---

## 13. 数据集划分：训练集 / 验证集 / 测试集

### 为什么要划分

如果只用训练集评估，你不知道模型是真的学会了规律，还是只是在死记硬背（过拟合）。

### 标准划分方案

```python
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))  # 80%
n2 = int(0.9 * len(words))  # 90%

Xtr,  Ytr  = build_dataset(words[:n1])    # 训练集：182,441 样本
Xdev, Ydev = build_dataset(words[n1:n2])  # 验证集：22,902 样本
Xte,  Yte  = build_dataset(words[n2:])   # 测试集：22,803 样本
```

### 三个集合的用途

| 数据集 | 用途 | 使用频率 |
|--------|------|---------|
| **训练集（80%）** | 优化模型参数（梯度下降） | 每次迭代都用 |
| **验证集（10%）** | 调超参数（网络大小、学习率、嵌入维度等） | 频繁 |
| **测试集（10%）** | 最终评估，报告论文结果 | **极少，最多几次** |

> **Karpathy 的话：** "every single time you evaluate your test loss and you learn something from it, you are basically starting to also train on the test split."
> 每次看测试集损失并据此做决定，就相当于在测试集上训练了，所以要极度节制。

---

## 14. 欠拟合 vs. 过拟合：如何诊断

### 关键指标：训练损失 vs. 验证损失的差距

| 情况 | 训练损失 | 验证损失 | 结论 |
|------|---------|---------|------|
| **欠拟合（Underfitting）** | 高 | 高（≈ 训练） | 模型太小，增加容量 |
| **正常** | 低 | 低（略高于训练） | 良好状态 |
| **过拟合（Overfitting）** | 很低 | 明显高于训练 | 模型太大，需要正则化 |

### 本讲的诊断

初始配置（100 个神经元，2 维嵌入）：
- 训练损失 ≈ 2.3，验证损失 ≈ 2.3 → **训练损失 ≈ 验证损失 → 欠拟合**
- 说明模型容量不够，应该**增大模型**

扩大后（200 个神经元，10 维嵌入）：
- 训练损失 ≈ 2.16，验证损失 ≈ 2.19 → 差距略微增大，说明开始有轻微过拟合

> **Karpathy 的话：** "we are what's called underfitting because the training loss and the dev losses are roughly equal… we expect to make performance improvements by scaling up the size of this neural net."

---

## 15. 扩大模型规模：突破性能瓶颈

### 识别瓶颈

初始模型有两个可能的瓶颈：

1. **隐藏层太小**（100 个神经元）
2. **嵌入维度太小**（2 维 → 只能学习非常粗糙的表示）

### 实验对比

| 配置 | 参数量 | 训练损失 | 验证损失 |
|------|--------|---------|---------|
| 2 维嵌入 + 100 神经元 | ~3,400 | 2.23 | 2.24 |
| 2 维嵌入 + 300 神经元 | ~10,000 | 2.23 | 2.24 |
| **10 维嵌入 + 200 神经元** | **~11,897** | **2.16** | **2.17** |

**结论：** 扩大嵌入维度（从 2 到 10）是更关键的改进。2 维嵌入是真正的瓶颈，把 27 个字符压进 2 维空间太拥挤了。

### 代码变化

```python
# 之前
C  = torch.randn((27,  2))   # 2 维嵌入
W1 = torch.randn(( 6, 100))  # 3×2=6 输入，100 神经元

# 之后
C  = torch.randn((27, 10))   # 10 维嵌入
W1 = torch.randn((30, 200))  # 3×10=30 输入，200 神经元
```

---

## 16. 嵌入向量的可视化

在扩大嵌入维度之前（还是 2 维时），可以直接把所有字符的嵌入绘制出来：

```python
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color='white')
plt.grid('minor')
```

### 观察到的结构

- **元音聚类**：`a, e, i, o, u` 被聚在一起，说明神经网络发现它们在预测下一个字符时有相似的作用
- **特殊字符离群**：`q` 和 `.`（句点）被放在远离其他字符的位置，因为它们的上下文分布非常独特
- **初始化是随机的**，但训练之后这些结构自然涌现

这说明嵌入向量**自动学到了字符之间的语言学相似性**，而没有任何人工标注。

---

## 17. 从模型中采样

训练完成后，用模型生成名字：

```python
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size  # 初始上下文：全是 '.'
    while True:
        emb = C[torch.tensor([context])]  # [1, block_size, 10]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]  # 滑动窗口
        out.append(ix)
        if ix == 0:  # 遇到 '.' 结束
            break
    print(''.join(itos[i] for i in out))
```

### 生成样本（训练后）

```
carmahela.    jhovi.     kimrin.
thil.         halanna.   jazhien.
amerynci.     aqui.      nellara.
```

和二元模型相比，这些名字明显更像真实名字，更有语言感。

---

## 18. 本讲总结

### 知识点汇总

| 主题 | 核心要点 |
|------|---------|
| **为什么用 MLP** | 统计方法上下文长度增加时参数指数爆炸；MLP 参数线性增长 |
| **嵌入查找表** | 用 `C[X]` 批量嵌入，等价于 one-hot × C 但快得多 |
| **view() 重塑** | 零拷贝操作，把 `[N, 3, 10]` 变成 `[N, 30]` |
| **F.cross_entropy** | 更高效、数值更稳定（自动减最大值防溢出） |
| **Mini-batch 训练** | 用 32 个样本的近似梯度，比全量梯度迭代更快 |
| **学习率搜索** | 用指数均匀扫描 0.001~1，找损失下降最快的区间 |
| **学习率衰减** | 后期降低 10 倍，精细调整 |
| **三路数据集划分** | 80% 训练 / 10% 验证 / 10% 测试；测试集极少使用 |
| **欠拟合诊断** | 训练损失 ≈ 验证损失 → 扩大模型 |
| **瓶颈识别** | 嵌入维度（从 2 到 10）比隐藏层大小更关键 |
| **嵌入可视化** | 2 维时可以画散点图，元音自动聚类 |

### 最终性能

| 模型 | 验证集损失 |
|------|-----------|
| 二元模型（第 2 讲） | 2.45 |
| MLP（2 维嵌入，100 神经元） | 2.24 |
| MLP（10 维嵌入，200 神经元，学习率衰减） | **2.17** |

---

## 附：关键代码速查

### 构建数据集

```python
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr, Ytr   = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte   = build_dataset(words[n2:])
```

### 参数初始化

```python
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
```

### 训练循环

```python
for i in range(200000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

### 指数学习率搜索

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri, lossi = [], []
for i in range(1000):
    # ... 前向/反向 ...
    lr = lrs[i]
    # ... 更新 ...
    lri.append(lre[i])
    lossi.append(loss.item())

plt.plot(lri, lossi)  # 看哪个指数区间损失最小
```

### 评估损失

```python
# 验证集损失
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
print(F.cross_entropy(logits, Ydev))
```

### 从模型采样

```python
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
```
