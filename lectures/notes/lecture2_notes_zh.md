# Lecture 2：逐步拆解语言建模——构建 makemore

> 对应视频：[YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo)
> 对应 Notebook：[makemore_part1_bigrams.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)

---

## 目录

1. 本讲目标与背景
2. makemore 是什么？
3. 数据集：names.txt
4. 二元语言模型（Bigram Language Model）——直觉理解
5. 用 Python 字典统计 Bigram
6. 用 PyTorch Tensor 存储计数：N 矩阵
7. 构建字符映射表：stoi / itos
8. 可视化计数矩阵
9. 从模型中采样（Sampling）
10. 广播语义（Broadcasting）——一个高危陷阱
11. 评估模型质量：负对数似然损失（NLL）
12. 模型平滑（Model Smoothing）
13. 神经网络方法：从计数到梯度下降
14. One-Hot 编码
15. 神经网络前向传播：Logits → Softmax → 概率
16. 反向传播与参数更新
17. 完整训练循环
18. 神经网络方法与计数方法的等价性
19. 正则化（Regularization）≈ 模型平滑
20. 从神经网络采样
21. 本讲总结

---

## 1. 本讲目标与背景

Lecture 1 用 micrograd 从零构建了反向传播引擎，展示了神经网络的基本原理。
本讲迈出下一步：**构建 makemore**——一个字符级语言模型。

学习路线：
```
Bigram (本讲) → MLP → RNN → Transformer (≈GPT-2)
```

Karpathy 的核心思路是**先用最简单的统计模型把框架搭起来**，然后用神经网络替换其中的核心部件，一步步复杂化，但 loss 计算和梯度下降框架始终不变。

---

## 2. makemore 是什么？

makemore 的含义正如其名：**make more**（产生更多）。

- **输入**：任意文本数据集（本讲用人名）
- **输出**：与输入同类但全新的样本（听起来像人名但实际上是杜撰的名字）
- **实现**：字符级语言模型（character-level language model）

示例生成（训练后）：
```
dontel
irot
zhendi
```

**字符级语言模型**的含义：模型以字符为单位处理序列，学习"当前字符后面最可能跟什么字符"。

---

## 3. 数据集：names.txt

```python
words = open('names.txt', 'r').read().splitlines()
words[:10]
# ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia', 'harper', 'evelyn']

len(words)       # 32033
min(len(w) for w in words)   # 2
max(len(w) for w in words)   # 15
```

- 共 32,033 个人名，按频率排序
- 最短 2 个字符，最长 15 个字符

**每个单词包含多个训练样本**。以 `isabella` 为例，它隐含以下统计信息：
- `i` 很可能是首字符
- `s` 很可能跟在 `i` 后面
- `a` 很可能跟在 `is` 后面
- ……
- 在 `isabella` 结束后，单词结束的概率很高

> Karpathy：一个单词里其实 packed 了大量关于字符序列统计结构的信息，我们有 32,000 个这样的单词，所以有大量结构可以学习。

---

## 4. 二元语言模型（Bigram Language Model）——直觉理解

**Bigram 模型**：每次只看**一个前导字符**，预测下一个字符。

- 优点：极简，容易理解
- 缺点：忘记了所有更早的上下文，质量很差
- 作用：绝佳的起点，让我们先把整个框架建立起来

**特殊 token**：我们需要标记单词的开始和结束。

早期设计（最终弃用）：使用 `<S>`（start）和 `<E>`（end）两个不同 token。
最终设计：使用单一特殊字符 `.`，既表示起始也表示终止，索引为 0。

```python
# 对单词 "emma" 的所有 bigram：
# ('.', 'e'), ('e', 'm'), ('m', 'm'), ('m', 'a'), ('a', '.')
```

遍历单词中所有 bigram 的 Pythonic 写法：
```python
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)
```

`zip(w, w[1:])` 的妙处：将字符串与其偏移一位的自身配对，得到所有相邻字符对。

---

## 5. 用 Python 字典统计 Bigram

```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
```

查看最常见的 bigram（按频率排序）：
```python
sorted(b.items(), key=lambda kv: -kv[1])
# [('n', '<E>'), 6763), ('a', '<E>'), 6640), ('a', 'n'), 5438), ...]
```

关键发现：
- `n` 作为词尾字符出现了 6,763 次——`n` 是最常见的结尾
- `a` 后面跟 `n` 出现了 5,438 次
- 以 `a` 开头的单词有 4,410 个

---

## 6. 用 PyTorch Tensor 存储计数：N 矩阵

字典方式不够高效，也不方便之后的数学运算。改用 2D 张量：

**为什么用 Tensor？**
- 支持高效的多维数组操作
- 支持批量运算（broadcasting）
- 是后续神经网络操作的基础

```python
import torch

# 27 个字符：'.' (index 0) + 'a'-'z' (index 1-26)
N = torch.zeros((27, 27), dtype=torch.int32)
```

**N[i, j]** = 字符 `i` 后面跟字符 `j` 的次数

---

## 7. 构建字符映射表：stoi / itos

```python
chars = sorted(list(set(''.join(words))))  # 26个小写字母
stoi = {s: i+1 for i, s in enumerate(chars)}  # a→1, b→2, ..., z→26
stoi['.'] = 0                                   # '.'→0（特殊起止符）
itos = {i: s for s, i in stoi.items()}          # 反向映射：0→'.', 1→'a', ...
```

填充 N 矩阵：
```python
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

**设计决策**：把 `.` 放在 index 0，字母从 index 1 开始，这样更整洁。

---

## 8. 可视化计数矩阵

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
```

注意 `.item()`：从 Tensor 中提取单个 Python 整数（索引操作返回的是张量，不是整数）。

可视化矩阵的直觉：
- 第 0 行（`.` 开头）：各字母作为首字符的频率
- 最后一列（`.` 结尾）：各字母作为末字符的频率
- 中间区域：字符间的转移频率

---

## 9. 从模型中采样（Sampling）

### 9.1 概率矩阵 P

将计数归一化得到概率：

```python
# 方法一：先处理单行
p = N[0].float()
p = p / p.sum()
```

但更高效的是一次性处理整个矩阵：

```python
P = N.float()
P /= P.sum(1, keepdim=True)  # 注意：必须用 keepdim=True！见第10节
```

### 9.2 torch.multinomial 采样

```python
g = torch.Generator().manual_seed(2147483647)  # 固定随机种子，保证可复现
```

`torch.multinomial`：给定概率分布，按概率采样整数索引。

```python
# 示例：3个元素的概率分布
p = torch.tensor([0.6064, 0.3033, 0.0903])
torch.multinomial(p, num_samples=100, replacement=True, generator=g)
# 结果中约60%是0，30%是1，9%是2
```

注意：`replacement=True` 必须显式指定（默认为 False）。

### 9.3 采样循环

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0  # 从 '.' 开始
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # 采到 '.' 表示结束
            break
    print(''.join(out))
```

输出（加了平滑后）：
```
mor.
axx.
minaymoryles.
kondlaisah.
anchshizarie.
```

> Karpathy：这些结果看起来很糟糕，但这就是 bigram 模型的真实水平。它不记得任何上下文，只知道"上一个字符是什么"，所以它不知道当前是否是刚开始，可能就直接输出一个结束符，生成单字母的"名字"。

**对比实验**：用均匀分布（完全未训练）采样：
```python
p = torch.ones(27) / 27  # 均匀分布
```
结果是完全随机的字母序列——比 bigram 还差。这说明 bigram 模型确实学到了一些东西。

---

## 10. 广播语义（Broadcasting）——一个高危陷阱

### 10.1 keepdim 的重要性

```python
P = N.float()
# 正确写法：
P /= P.sum(1, keepdim=True)   # ✅ P.sum(1, keepdim=True).shape = (27, 1)

# 错误写法（有 bug！）：
P /= P.sum(1)                  # ❌ P.sum(1).shape = (27,)
```

**为什么错误写法有 bug？**

| | 正确（keepdim=True） | 错误（keepdim=False） |
|---|---|---|
| sum 的 shape | (27, 1) 列向量 | (27,) 一维向量 |
| broadcast 方向 | 沿列方向复制（每行除以自己的和）✅ | 被解释为行向量 (1, 27)，沿行方向复制（每列除以一个数）❌ |
| 结果 | 每行归一化为概率分布 | 每**列**归一化（方向完全相反！） |

**Broadcasting 规则**（从右往左对齐维度）：
- `(27, 27)` 除以 `(27, 1)`：维度从右看，`27==27`（行数）✅，`27` 遇到 `1`（复制列）✅ → 每行除以自己的行和 ✅
- `(27, 27)` 除以 `(27,)` → 内部变为 `(1, 27)` → 被当作行向量复制 → 每**列**除以一个数 ❌

> Karpathy：我强烈建议大家真正理解 broadcasting，多看文档，多练习，把它当作需要尊重的东西。不小心的话会引入非常隐蔽的 bug，很难发现。

**验证方法**：
```python
P[0].sum()  # 正确时应该接近 1.0
```

### 10.2 原地操作

```python
P /= P.sum(1, keepdim=True)   # in-place 操作，比 P = P / P.sum(...) 更高效
```

---

## 11. 评估模型质量：负对数似然损失（NLL）

### 11.1 似然（Likelihood）

**目标**：量化模型对训练数据的拟合程度。

对于整个训练集，我们希望模型给每个 bigram 赋予的概率**尽可能高**。

$$\text{Likelihood} = \prod_{i} P(c_{i+1} | c_i)$$

问题：概率的连乘会得到极小的数（数值不稳定）。

### 11.2 对数似然（Log-Likelihood）

取对数将连乘变成连加：

$$\log P(a \cdot b \cdot c) = \log P(a) + \log P(b) + \log P(c)$$

性质：
- 概率 = 1 → log 概率 = 0（完美预测）
- 概率 → 0 → log 概率 → −∞（极差预测）
- log 是单调函数，最大化 log-likelihood 等价于最大化 likelihood

### 11.3 负对数似然（NLL）——损失函数

$$\text{Loss} = \text{NLL} = -\frac{1}{N}\sum_{i} \log P(c_{i+1} | c_i)$$

性质：
- 最小值为 0（完美预测）
- 越大越差
- **低 = 好**（符合损失函数语义）

```python
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

print(f'{log_likelihood=}')       # tensor(-564996.8125)
nll = -log_likelihood
print(f'{nll=}')                  # tensor(564996.8125)
print(f'{nll/n}')                 # 2.476（平均 NLL）
```

Bigram 模型的损失：**约 2.45**（直接计数 + 归一化，无平滑）。

**量化感知**：27 个字符，若完全随机（均匀分布），每个字符的概率是 1/27 ≈ 4%，对应 NLL = -log(1/27) ≈ 3.3。而我们的 bigram 模型达到了 2.45，说明模型确实学到了有用的统计规律。

### 11.4 一个特殊案例

```python
# 测试 "andrejq"——其中 'jq' 在训练集中从未出现
# P[stoi['j'], stoi['q']] = 0
# log(0) = -inf → NLL = +inf
```

模型给 `jq` 赋予了零概率，loss 是无穷大——这不好。解决方法见下节。

---

## 12. 模型平滑（Model Smoothing）

**问题**：某些 bigram 在训练集中从未出现 → 概率为 0 → log(0) = -∞。

**解决**：给所有计数加上假计数（fake counts）：

```python
P = (N + 1).float()               # 每个格子加 1（最小平滑）
P /= P.sum(1, keepdim=True)
# 也可以加更多：N + 5，N + 100 等
```

效果：
- 加得越多 → 分布越均匀（接近均匀分布）
- 加得越少 → 分布越尖锐（更接近原始统计）
- 加 1：合理的默认选择，消除零概率同时不过度扭曲

```python
# 加平滑后，"jq" 的概率不再是0，而是一个很小的正数
# loss 从 +∞ 变为某个有限的大数
```

---

## 13. 神经网络方法：从计数到梯度下降

### 13.1 为什么需要神经网络？

计数方法的局限性：
- 只能处理 bigram（一个前导字符）
- 如果要考虑最近 10 个字符，需要的表格大小 = 27^10 ≈ 20 万亿个格子，完全不可行

神经网络方法：
- 可以接受任意多个前导字符
- 通过参数共享学习有效表示
- 可以扩展到 MLP → RNN → Transformer

> Karpathy：神经网络框架显著更灵活。我们现在就要构建最简单的版本，但这套框架会一直延伸到 transformer，框架本身不会改变。

### 13.2 神经网络概述

```
输入字符（整数索引）
    → One-Hot 编码（27维向量）
    → 线性层（27×27 权重矩阵 W）
    → Logits（27维）
    → Softmax
    → 概率分布（27维，和为1）
    → 对比真实标签
    → NLL Loss
    → 反向传播
    → 更新 W
```

---

## 14. One-Hot 编码

整数索引不能直接输入神经网络（神经元做的是 `w·x + b`，整数没有方向意义）。

**One-Hot 编码**：将整数 k 转化为一个只有第 k 位为 1 的向量。

```python
import torch.nn.functional as F

xs = torch.tensor([0, 5, 13, 13, 1])   # 输入字符索引（emma的5个bigram）
xenc = F.one_hot(xs, num_classes=27).float()
xenc.shape  # torch.Size([5, 27])
```

**数据类型注意**：
```python
xenc.dtype  # torch.int64（int64 不能用于神经网络！）
# 必须加 .float() 转为 float32
```

**`torch.tensor` vs `torch.Tensor`**：

| | `torch.tensor(data)` | `torch.Tensor(data)` |
|---|---|---|
| 推断 dtype | 自动推断（整数→int64）✅ | 总是 float32 ❌ |
| 推荐 | ✅ 使用这个 | ❌ 避免 |

> Karpathy：这两者差别很微妙，文档也不够清晰。总之用小写的 `torch.tensor`。

---

## 15. 神经网络前向传播：Logits → Softmax → 概率

### 15.1 单个神经元 → 27 个神经元

```python
W = torch.randn((27, 27), generator=g)   # 27个输入 × 27个输出神经元
logits = xenc @ W                         # (5, 27) @ (27, 27) = (5, 27)
```

**矩阵乘法的含义**：
- `xenc` 是 (5, 27)，每行是一个样本的 one-hot 向量
- `W` 是 (27, 27)，每列是一个神经元的权重
- `xenc @ W` 一次性计算了 5 个样本在 27 个神经元上的激活值

> Karpathy：注意 one-hot 向量乘以 W 矩阵，其实就是**直接取出 W 的对应行**。因为 one-hot 向量只有一位是 1，矩阵乘法结果就是那一行。这和之前直接用索引查 N 矩阵本质上是一回事！

### 15.2 Softmax

神经网络输出的 logits 可以是正数、负数——不能直接作为概率。

**Softmax** 将任意实数向量转化为概率分布：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```python
counts = logits.exp()                          # 等价于 N（伪计数）
probs = counts / counts.sum(1, keepdim=True)   # 归一化 → 概率
# 以上两行合称 Softmax
```

**为什么用指数？**

| logits 值 | exp 结果 |
|---|---|
| 负数 | 0 到 1 之间（小概率）|
| 0 | 1 |
| 正数 | 大于 1（大概率）|
| 任意实数 | 结果恒正 ✅ |

指数函数将全实数轴映射到正数，再归一化就得到概率。

**logits 的含义**：对数计数（log-counts）。神经网络输出 logits，exp 之后等价于计数，normalize 之后是概率。

### 15.3 前向传播完整代码

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 前向传播
xenc = F.one_hot(xs, num_classes=27).float()   # one-hot 编码
logits = xenc @ W                               # 预测 log-counts
counts = logits.exp()                           # 伪计数
probs = counts / counts.sum(1, keepdims=True)   # 概率（softmax）

# 计算损失
loss = -probs[torch.arange(num), ys].log().mean()
print(loss.item())   # 初始 loss ≈ 3.69（随机初始化，很高）
```

**高效的 loss 计算**：
```python
# probs[torch.arange(5), ys]
# 等价于：对每个样本 i，取 probs[i, ys[i]]（正确字符的概率）
probs[torch.arange(5), ys]   # tensor([0.0607, ..., ...])
```

---

## 16. 反向传播与参数更新

```python
# 反向传播
W.grad = None           # 清零梯度（比 zero_() 更高效）
loss.backward()         # 自动求导，填充 W.grad

# 参数更新（梯度下降）
W.data += -0.1 * W.grad   # learning rate = 0.1
```

**`requires_grad=True`**：告诉 PyTorch 需要为 W 计算梯度（默认 False）。

**`loss.backward()`**：
- PyTorch 在前向传播时自动建立计算图
- `backward()` 从 loss 出发，反向遍历计算图，填充每个参数的 `.grad`
- 和 micrograd 原理完全相同，只是规模更大

**`W.grad = None` vs `W.grad.zero_()`**：两者等价，设为 None 更高效（PyTorch 会把 None 当 zeros 处理）。

每次更新后，loss 会略微下降：
```
初始：loss ≈ 3.76
更新一次：loss ≈ 3.74
更新两次：loss ≈ 3.72
...
```

---

## 17. 完整训练循环

### 17.1 构建完整数据集

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)   # 228146
```

### 17.2 梯度下降训练

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
    # 前向传播
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())

    # 反向传播
    W.grad = None
    loss.backward()

    # 参数更新
    W.data += -50 * W.grad   # learning rate = 50（可以用较大的学习率）
```

训练结果：loss 从约 3.7 降到约 **2.46**。

> Karpathy：可以用很大的学习率（50）因为这个问题非常简单。在这里 100 次迭代就足够了。

---

## 18. 神经网络方法与计数方法的等价性

这是本讲最精彩的洞见之一：

**两种方法最终得到相同的结果**（约 2.45 的 NLL loss），生成的样本也相同：
```
mor.
axx.
minaymoryles.
kondlaisah.
anchthizarie.
```

**为什么等价？**

1. **One-hot 乘以 W = 直接查表**：one-hot 向量只有一位为 1，`xenc @ W` 的结果就是 W 的对应行。这与之前直接索引 N 矩阵 `N[ix1]` 是相同的操作。

2. **W 的物理意义**：W 是 log-counts（对数计数），而 N 是 counts（计数）。`W.exp()` 就等价于 N。

3. **最优解一致**：对 bigram 问题，计数方法直接给出了 NLL 最优解；梯度下降也收敛到同一个最优解。

```
计数方法：       直接估计 P(c2|c1) = count(c1,c2) / count(c1)
梯度下降方法：   通过最小化 NLL，学出 W，其中 exp(W) ≈ 计数表
```

> Karpathy：我们用两种截然不同的方式得到了同一个结果。计数方法对 bigram 来说直接可以精确最优化 loss，而梯度框架更灵活，可以扩展到更复杂的模型。

---

## 19. 正则化（Regularization）≈ 模型平滑

### 19.1 W=0 时的均匀分布

```python
# 如果所有 W 都是 0：
logits = xenc @ 0  # = 全 0 向量
counts = 0 .exp() # = 全 1
probs = 1 / 27    # 均匀分布！
```

**W 的值越接近 0，概率分布越均匀**——这和平滑中"加的假计数越多，分布越均匀"完全对应！

### 19.2 L2 正则化

```python
loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
#                          ↑ NLL 损失                    ↑ 正则化项
```

正则化损失 `(W**2).mean()` 的效果：
- W = 0 → 正则化 loss = 0（最小）
- W 越大 → 正则化 loss 越大（惩罚）

**双重压力**：
1. NLL 项：推动 W 使预测正确
2. 正则化项：推动 W 趋向 0（即趋向均匀分布）

两个力量相互竞争，找到平衡点。正则化强度 0.01 控制了这个权衡，类比于平滑中假计数的大小。

> Karpathy：你可以把正则化想象成一个弹簧力或重力，把 W 往 0 方向拉。W 想要是 0，概率想要是均匀的，但同时 W 也想让概率分布和数据相符。

| 对应关系 | 计数方法 | 梯度方法 |
|---|---|---|
| 模型参数 | N 矩阵（计数） | W 矩阵（log 计数）|
| 平滑 | N + k（假计数）| 正则化 λ·(W²).mean() |
| k 越大 | 分布越均匀 | λ 越大，W 越趋向 0，分布越均匀 |
| 效果 | 消除零概率 | 防止过拟合 |

---

## 20. 从神经网络采样

训练完 W 之后，用神经网络采样：

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        # 用神经网络计算概率
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W           # pluck out row of W（等价于查表）
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```

输出与计数方法完全相同——因为这是完全等价的模型。

---

## 21. 本讲总结

| 主题 | 关键点 |
|---|---|
| makemore | 字符级语言模型，生成名字；从 bigram 到 Transformer |
| 数据集 | 32,033 个名字，字符最短 2，最长 15 |
| Bigram 表示 | 每个单词分解为字符对；特殊 token `.` 表示起止 |
| 计数方法 | 27×27 矩阵 N 存储计数；归一化得到概率矩阵 P |
| Broadcasting | `keepdim=True` 至关重要；方向错误会引入隐蔽 bug |
| NLL Loss | 标准分类损失；bigram 模型约 2.45 |
| 模型平滑 | N+k 消除零概率；k 越大分布越均匀 |
| 神经网络 | one-hot → 线性层 → softmax → 概率 |
| 等价性 | 两种方法收敛到相同结果（2.46）|
| 正则化 | λ·(W²).mean() ≈ 模型平滑；W→0 对应均匀分布 |
| 下一步 | MLP：输入多个前导字符，更强的模型 |

**核心框架（永久不变）**：
```
输入字符 → 神经网络（越来越复杂）→ Logits → Softmax → 概率 → NLL Loss → 梯度下降
```

> Karpathy：从 bigram 到 Transformer，整个框架不会改变。唯一会变的是"神经网络"这一部分——它会越来越复杂，但 loss 函数、梯度下降、softmax 全部保持不变。这一讲建立的框架是整个课程的基础。

---

## 附：关键代码速查

### 字符映射表
```python
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
```

### 计数方法（完整）
```python
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

P = (N + 1).float()               # +1 平滑
P /= P.sum(1, keepdim=True)       # keepdim=True！
```

### 采样（计数方法）
```python
g = torch.Generator().manual_seed(2147483647)
ix = 0
out = []
while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0: break
print(''.join(out))
```

### NLL Loss 计算
```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1
nll = -log_likelihood / n
print(f'NLL = {nll:.4f}')  # ≈ 2.45
```

### 神经网络训练（完整）
```python
import torch.nn.functional as F

# 构建数据集
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1]); ys.append(stoi[ch2])
xs = torch.tensor(xs); ys = torch.tensor(ys)
num = xs.nelement()

# 初始化
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 训练循环
for k in range(200):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

print(f'final loss: {loss.item():.4f}')  # ≈ 2.46
```

### Broadcasting 安全检查
```python
# 归一化后验证
assert abs(P[0].sum().item() - 1.0) < 1e-5, "行归一化失败！"
```
