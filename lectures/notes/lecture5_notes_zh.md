# Lecture 5：手动反向传播——成为反向传播忍者

> 对应视频：[YouTube](https://youtu.be/q8SA3rM6ckI)
> 对应 Notebook：[makemore_part4_backprop.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)

---

## 目录

1. 本讲目标与背景
2. 为什么要手动写反向传播？
3. 前向传播：分解为细粒度中间变量
4. 验证工具：`cmp` 函数
5. 练习一：逐步手动反向传播
   - 5.1 从 `logprobs` 开始
   - 5.2 `probs`（log 反向）
   - 5.3 `counts_sum_inv` 与 `counts`（带广播的乘法）
   - 5.4 `counts_sum`（求和的反向）
   - 5.5 `norm_logits` 与 `logit_maxes`（减法+广播）
   - 5.6 第二条支路：`logit_maxes → logits`（max 的反向）
   - 5.7 线性层 2：`h`, `W2`, `b2`（矩阵乘法反向）
   - 5.8 `tanh` 的反向传播
   - 5.9 BatchNorm 参数：`bngain`, `bnraw`, `bnbias`
   - 5.10 BatchNorm 内部：逐步展开
   - 5.11 线性层 1：`embcat`, `W1`, `b1`
   - 5.12 Embedding 的反向传播
6. 练习二：交叉熵损失的高效反向传播
7. 练习三：BatchNorm 的解析反向传播
8. 练习四：完整训练——不用 `loss.backward()`
9. 本讲总结

---

## 1. 本讲目标与背景

本讲在 Lecture 4（激活、梯度与 BatchNorm）的基础上，保持完全相同的网络结构——两层 MLP + BatchNorm——但将整个反向传播过程**从 `loss.backward()` 替换为完全手写的梯度计算**。

**网络结构回顾**：

```
字符嵌入 → Linear1 → BatchNorm → tanh → Linear2 → Softmax → Loss
```

**本讲四个练习**：

| 练习 | 内容 |
|------|------|
| 练习一 | 逐变量逐操作手动反向传播（原子级别） |
| 练习二 | 将交叉熵损失一步解析求导，跳过中间变量 |
| 练习三 | BatchNorm 层的单公式解析反向传播 |
| 练习四 | 将所有推导整合，实现完整的无 autograd 训练 |

---

## 2. 为什么要手动写反向传播？

> "反向传播是一个**漏抽象（leaky abstraction）**。你不能只是堆叠任意的可微分模块，然后闭上眼睛期待它自动工作。"
> —— Karpathy

### 现象 / 问题

PyTorch 的 `loss.backward()` 隐藏了所有细节，当网络行为异常时，很难找到根本原因。

### 具体例子

Karpathy 展示了网络上真实存在的 bug：有人想要"裁剪梯度的最大值"，却写成了"裁剪损失的最大值"。裁剪损失实际上会把某些离群样本的梯度置零，使它们完全被忽略，这是一个极其隐蔽的错误。

如果你不理解反向传播的内部机制，就无法发现这类问题。

### 历史背景

- **2010 年前后**：所有深度学习研究者都手写反向传播，Karpathy 本人在 2014 年的论文（图像文本对齐，类似 CLIP 的工作）中仍然手动实现完整的 backward pass
- **今天**：使用 autograd 是标准做法，但理解其内部仍然极为重要

### 练习的价值

- 彻底理解梯度如何在计算图中流动
- 从标量级别（micrograd）升级到**张量级别**
- 为调试现代神经网络打下基础

---

## 3. 前向传播：分解为细粒度中间变量

本讲将之前的简洁前向传播展开为**完全原子化**的步骤，每一步都保存中间变量，因为我们要对每个中间变量求导。

```python
# 嵌入
emb = C[Xb]                              # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)      # (32, 30)  ← 拼接

# 线性层 1
hprebn = embcat @ W1 + b1                # (32, 64)

# BatchNorm（手动展开）
bnmeani = 1/n * hprebn.sum(0, keepdim=True)          # (1, 64) 均值
bndiff  = hprebn - bnmeani                            # (32, 64)
bndiff2 = bndiff**2                                   # (32, 64)
bnvar   = 1/(n-1) * bndiff2.sum(0, keepdim=True)     # (1, 64) 方差（Bessel校正）
bnvar_inv = (bnvar + 1e-5)**-0.5                      # (1, 64)
bnraw   = bndiff * bnvar_inv                          # (32, 64) X-hat
hpreact = bngain * bnraw + bnbias                     # (32, 64)

# 激活
h = torch.tanh(hpreact)                  # (32, 64)

# 线性层 2
logits = h @ W2 + b2                     # (32, 27)

# Loss（手动展开）
logit_maxes = logits.max(1, keepdim=True).values   # (32, 1) 数值稳定
norm_logits = logits - logit_maxes                 # (32, 27)
counts = norm_logits.exp()                         # (32, 27)
counts_sum = counts.sum(1, keepdims=True)          # (32, 1)
counts_sum_inv = counts_sum**-1                    # (32, 1) ← 注意不用除法
probs = counts * counts_sum_inv                    # (32, 27)
logprobs = probs.log()                             # (32, 27)
loss = -logprobs[range(n), Yb].mean()              # 标量
```

**为什么用 `counts_sum**-1` 而非 `1/counts_sum`？**

PyTorch 对除法的反向传播有一个实现问题，会给出真实结果但不符合预期。用 `**-1` 可以绕过这个问题。

**关键形状汇总**：

| 变量 | 形状 |
|------|------|
| `emb` | (32, 3, 10) |
| `embcat` | (32, 30) |
| `hprebn` / `bnraw` / `hpreact` / `h` | (32, 64) |
| `logits` | (32, 27) |
| `bnmeani`, `bnvar`, `bnvar_inv` | (1, 64) |
| `counts_sum` | (32, 1) |
| `bngain`, `bnbias` | (1, 64) |
| `W1` | (30, 64) |
| `W2` | (64, 27) |

---

## 4. 验证工具：`cmp` 函数

```python
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

- `dt`：我们手动计算的梯度
- `t.grad`：PyTorch autograd 计算的梯度
- 验证**精确相等**和**近似相等**（浮点误差），以及**最大差值**

> **技巧**：初始化时故意用小随机数而非零来初始化偏置。如果变量是零，一些梯度的错误计算式会"退化"成正确结果，掩盖 bug。

---

## 5. 练习一：逐步手动反向传播

> 反向传播的核心规则：**链式法则**。从损失开始，由后往前，对每个中间变量，局部导数乘以来自上游的梯度。

**通用模式**：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

其中 $\frac{\partial L}{\partial y}$ 是下游已有梯度，$\frac{\partial y}{\partial x}$ 是当前操作的局部导数。

---

### 5.1 从 `logprobs` 开始

**前向**：`loss = -logprobs[range(n), Yb].mean()`

**分析**：`logprobs` 是 (32, 27) 的矩阵，但只有每行 `Yb[i]` 对应的位置参与了损失计算：

$$\text{loss} = -\frac{1}{n}(a_0 + a_1 + \cdots + a_{n-1})$$

所以 $\frac{\partial L}{\partial a_i} = -\frac{1}{n}$，其余位置不影响损失，梯度为 0。

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
```

✅ 验证：`exact: True`

---

### 5.2 `probs`（log 反向）

**前向**：`logprobs = probs.log()`

$$\frac{d}{dx}\log(x) = \frac{1}{x}$$

```python
dprobs = (1.0 / probs) * dlogprobs
```

**直觉**：如果网络对正确字符预测概率很低（接近 0），则 `1/probs` 会很大，梯度被放大——这正是网络需要强力修正的情况。

✅ 验证：`exact: True`

---

### 5.3 `counts_sum_inv` 与 `counts`（乘法 + 广播）

**前向**：`probs = counts * counts_sum_inv`
- `counts`：(32, 27)
- `counts_sum_inv`：(32, 1) ← 被广播到 (32, 27)

**核心规则**：对于 $c = a \cdot b$，$\frac{\partial c}{\partial a} = b$，$\frac{\partial c}{\partial b} = a$

**处理广播**：`counts_sum_inv` 在广播中被"重用"了 27 次，等价于 micrograd 中一个节点输出流向多个分支——反向传播时要**沿广播维度求和**。

```python
# dcounts 的一部分（稍后还有第二条支路）
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)  # (32,27)→(32,1)
dcounts = counts_sum_inv * dprobs                          # 广播乘法，形状OK
```

**关键对偶性**：

| 前向 | 反向 |
|------|------|
| 广播（重用） | 求和（累积梯度） |
| 求和（压缩） | 广播（分发梯度） |

---

### 5.4 `counts_sum`（求和的反向）

**前向**：`counts_sum = counts.sum(1, keepdims=True)`

$$\text{counts\_sum\_inv} = \text{counts\_sum}^{-1}$$

$$\frac{d}{dx}x^{-1} = -x^{-2}$$

```python
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
```

求和操作的反向：梯度广播回原来的每一列。每个元素的局部导数是 1，所以梯度"复制"到所有列：

```python
dcounts += torch.ones_like(counts) * dcounts_sum  # plus equals！第二条支路
```

✅ `counts` 的梯度现在完整了

---

### 5.5 `norm_logits` 与 `logit_maxes`（减法 + 广播）

**前向**：`norm_logits = logits - logit_maxes`
- `logits`, `norm_logits`：(32, 27)
- `logit_maxes`：(32, 1) ← 广播

**`norm_logits`**：

$$\frac{\partial c}{\partial a} = 1 \Rightarrow \text{d\_norm\_logits} = \text{d\_counts} \cdot \text{counts} \quad (\text{exp的反向：} \frac{d}{dx}e^x = e^x)$$

```python
dnorm_logits = counts * dcounts   # exp 的反向
```

**`logit_maxes`**（负号 + 广播 → 沿列求和）：

```python
# 第一条支路进 dlogits
dlogits = dnorm_logits.clone()

# logit_maxes 是列向量，广播被用了 27 次，反向要求和
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
```

✅ 验证 `logit_maxes` 的梯度极小（约 1e-9）——因为 logit_maxes 只影响数值稳定性，不影响 softmax 的值！

---

### 5.6 第二条支路：`logit_maxes → logits`（max 的反向）

`logit_maxes` 是从 `logits` 每行取最大值得到的，最大值的位置索引由 PyTorch 的 `argmax` 给出。

**max 的反向**：梯度只流向最大值所在的位置（其他位置梯度为 0）。

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=27) * dlogit_maxes
```

`F.one_hot` 生成每行只有一个 1 的指示矩阵，再乘以列向量 `dlogit_maxes`（广播），将梯度路由到正确位置。

✅ `logits` 梯度完整，`exact: True`

---

### 5.7 线性层 2：`h`, `W2`, `b2`（矩阵乘法反向）

**前向**：`logits = h @ W2 + b2`

**矩阵乘法反向传播推导**（Karpathy 在纸上推导的版本）：

设 $D = A \cdot B + C$，写出每个元素的表达式，对每个变量求导，结果总可以表示为矩阵乘法：

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial D} \cdot B^T$$
$$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial D}$$
$$\frac{\partial L}{\partial C} = \frac{\partial L}{\partial D}.\text{sum}(0)$$

> **记忆技巧**：不需要背公式！只需要**根据形状倒推**。`dh` 必须是 (32, 64)，只有 `dlogits @ W2.T` 能把 (32,27) 变成 (32,64)。

```python
dh      = dlogits @ W2.T          # (32,27) @ (27,64) → (32,64)
dW2     = h.T @ dlogits            # (64,32) @ (32,27) → (64,27)
db2     = dlogits.sum(0)           # (32,27) → (27,)
```

✅ `h`, `W2`, `b2` 全部 `exact: True`

---

### 5.8 `tanh` 的反向传播

**前向**：`h = torch.tanh(hpreact)`

tanh 的导数公式（令 $a = \tanh(z)$）：

$$\frac{da}{dz} = 1 - a^2 = 1 - \tanh^2(z)$$

注意：公式用的是**输出** $a$（即 `h`），而不是输入 $z$。

```python
dhpreact = (1.0 - h**2) * dh
```

✅ `exact: True`

---

### 5.9 BatchNorm 参数：`bngain`, `bnraw`, `bnbias`

**前向**：`hpreact = bngain * bnraw + bnbias`

- `bngain`, `bnbias`：(1, 64) ← 会广播
- `bnraw`, `hpreact`：(32, 64)

**`bngain`**（元素乘法 + 广播 → 沿 0 轴求和）：

```python
dbngain  = (bnraw * dhpreact).sum(0, keepdim=True)   # (1,64)
```

**`bnraw`**（元素乘法，形状已对齐）：

```python
dbnraw   = bngain * dhpreact                          # (32,64)
```

**`bnbias`**（偏置，广播 → 求和）：

```python
dbnbias  = dhpreact.sum(0, keepdim=True)              # (1,64)
```

✅ 三者全部 `exact: True`

---

### 5.10 BatchNorm 内部：逐步展开

现在我们有了 `dbnraw`，需要继续往前传播。BatchNorm 内部的计算图如下：

```
hprebn → bnmeani
hprebn → bndiff → bndiff2 → bnvar → bnvar_inv → bnraw
                  bndiff ─────────────────────→ bnraw（两条支路！）
```

**步骤 1：`bnraw = bndiff * bnvar_inv`**

```python
dbndiff    = bnvar_inv * dbnraw        # 第一条支路（稍后 += 第二条）
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

**步骤 2：`bnvar_inv = (bnvar + 1e-5)**-0.5`**

$$\frac{d}{dx}x^{-0.5} = -0.5 \cdot x^{-1.5}$$

```python
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv
```

**Bessel 校正的说明**

> Karpathy 的吐槽：PyTorch 的 `BatchNorm1d` 训练时用有偏估计（除以 $n$），推断时却切换到无偏估计（除以 $n-1$），这是一个**训练/推断不一致的 bug**，在 batch 较大时影响很小，但在原则上是错的。本讲统一使用 $n-1$（Bessel 校正）。

**步骤 3：`bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)`**

求和的反向是广播，局部导数为 $\frac{1}{n-1}$：

```python
dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
```

**步骤 4：`bndiff2 = bndiff**2`**

$$\frac{d}{dx}x^2 = 2x$$

```python
dbndiff += 2 * bndiff * dbndiff2   # plus equals！第二条支路
```

✅ `bndiff` 梯度现在完整

**步骤 5：`bndiff = hprebn - bnmeani`**

```python
dhprebn  = dbndiff.clone()                     # 第一条支路（稍后 +=）
dbnmeani = (-dbndiff).sum(0, keepdim=True)
```

**步骤 6：`bnmeani = 1/n * hprebn.sum(0, keepdim=True)`**

均值的反向：将梯度均分给所有行：

```python
dhprebn += 1.0/n * torch.ones_like(hprebn) * dbnmeani
```

✅ `hprebn` 梯度完整

---

### 5.11 线性层 1：`embcat`, `W1`, `b1`

**前向**：`hprebn = embcat @ W1 + b1`

形状：`embcat` (32, 30)，`W1` (30, 64)，`b1` (64,)

与线性层 2 完全相同的推导，用形状倒推：

```python
dembcat = dhprebn @ W1.T          # (32,64) @ (64,30) → (32,30)
dW1     = embcat.T @ dhprebn       # (30,32) @ (32,64) → (30,64)
db1     = dhprebn.sum(0)           # (32,64) → (64,)
```

✅ 全部 `exact: True`

---

### 5.12 Embedding 的反向传播

**前向**：

```python
emb    = C[Xb]                                  # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)             # (32, 30)
```

**`emb` ← `embcat`**（view 是零拷贝的重新解释，反向传播只是换个 shape）：

```python
demb = dembcat.view(emb.shape)    # (32, 30) → (32, 3, 10)
```

**`C` ← `emb`**（索引操作，需要把梯度路由回正确的行，同一行可能被用多次）：

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]    # plus equals 累积梯度
```

✅ `exact: True`

**完整练习一的所有变量验证结果**：

```
logprobs        | exact: True  | approximate: True  | maxdiff: 0.0
probs           | exact: True  | approximate: True  | maxdiff: 0.0
counts_sum_inv  | exact: True  | approximate: True  | maxdiff: 0.0
counts_sum      | exact: True  | approximate: True  | maxdiff: 0.0
counts          | exact: True  | approximate: True  | maxdiff: 0.0
norm_logits     | exact: True  | approximate: True  | maxdiff: 0.0
logit_maxes     | exact: True  | approximate: True  | maxdiff: 0.0
logits          | exact: True  | approximate: True  | maxdiff: 0.0
h               | exact: True  | approximate: True  | maxdiff: 0.0
W2              | exact: True  | approximate: True  | maxdiff: 0.0
b2              | exact: True  | approximate: True  | maxdiff: 0.0
hpreact         | exact: True  | approximate: True  | maxdiff: 0.0
bngain          | exact: True  | approximate: True  | maxdiff: 0.0
bnbias          | exact: True  | approximate: True  | maxdiff: 0.0
bnraw           | exact: True  | approximate: True  | maxdiff: 0.0
...（所有 BatchNorm 内部变量）
hprebn          | exact: True  | approximate: True  | maxdiff: 0.0
embcat          | exact: True  | approximate: True  | maxdiff: 0.0
W1              | exact: True  | approximate: True  | maxdiff: 0.0
b1              | exact: True  | approximate: True  | maxdiff: 0.0
emb             | exact: True  | approximate: True  | maxdiff: 0.0
C               | exact: True  | approximate: True  | maxdiff: 0.0
```

---

## 6. 练习二：交叉熵损失的高效反向传播

### 问题

练习一中，从 `logits` 到 `loss` 经过了约 7 个中间变量，每一步都单独反向传播。这既繁琐又低效。

### 解法：解析求导

损失函数的数学形式（对单个样本）：

$$L = -\log p_y = -\log \frac{e^{l_y}}{\sum_j e^{l_j}}$$

对 $l_i$ 求偏导（分两种情况）：

$$\frac{\partial L}{\partial l_i} = \begin{cases} p_i - 1 & \text{if } i = y \\ p_i & \text{if } i \neq y \end{cases}$$

其中 $p_i = \text{softmax}(l)_i$。

对整个 batch（损失是均值），还需要除以 $n$：

$$\frac{\partial L_{\text{batch}}}{\partial l_i} = \frac{1}{n}\left(p_i - \mathbf{1}[i=y]\right)$$

### 实现

```python
dlogits = F.softmax(logits, 1)      # (32, 27) — 先算 softmax 概率
dlogits[range(n), Yb] -= 1          # 正确类别位置减 1
dlogits /= n                         # 除以 batch size
```

验证：

```
logits | exact: False | approximate: True | maxdiff: 5.12e-09
```

不能精确相等是因为我们用了不同的数值计算路径（浮点误差），但差距在 5e-9 量级，完全正确。

### 直觉解释

> 把梯度想象成**力**：我们同时在对正确字符的 logit 施加"上拉力"，对错误字符的 logit 施加"下压力"。拉力和压力大小相等（每行梯度之和为零），施力大小正比于当前预测的概率——预测越自信的错误，受到的修正力越大。

```python
dlogits[0].sum()  # ≈ 0（精确守恒）
```

可视化：`dlogits` 的热力图中每行几乎全灰，只有正确类别处有一个明显的黑色方块（负值）。

---

## 7. 练习三：BatchNorm 的解析反向传播

### 问题

BatchNorm 有 6+ 个中间变量，逐步反向传播虽然正确，但效率低。能否推导出一个直接从 `dbnraw` 到 `dhprebn` 的公式？

### 数学推导（Karpathy 纸上推导）

设批次大小为 $n$，BatchNorm 的计算：

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

其中 $\mu = \frac{1}{n}\sum_j x_j$，$\sigma^2 = \frac{1}{n-1}\sum_j(x_j - \mu)^2$（Bessel 校正版本）

**反向传播路径**（拓扑顺序）：

1. $\frac{\partial L}{\partial \hat{x}_i} = \gamma \cdot \frac{\partial L}{\partial y_i}$（通过 `bngain`）

2. $\frac{\partial L}{\partial \sigma^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot (-\frac{1}{2})(\sigma^2 + \epsilon)^{-3/2}$

3. $\frac{\partial L}{\partial \mu} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2+\epsilon}}$（第二项因 $\mu$ 是均值而消失）

4. **最终结果**（经过代入化简）：

$$\frac{\partial L}{\partial x_i} = \frac{1}{n} \cdot (\sigma^2+\epsilon)^{-1/2} \left[ n \cdot \frac{\partial L}{\partial \hat{x}_i} - \sum_j \frac{\partial L}{\partial \hat{x}_j} - \frac{n}{n-1}\hat{x}_i \sum_j \frac{\partial L}{\partial \hat{x}_j}\hat{x}_j \right]$$

### 实现

```python
dhprebn = bngain/n * bnvar_inv * (
    n * dbnraw
    - dbnraw.sum(0)
    - n/(n-1) * bnraw * (dbnraw * bnraw).sum(0)
)
```

验证：

```
hprebn | exact: False | approximate: True | maxdiff: 9.31e-10
```

> **为什么不精确相等？** 同样是浮点误差，差距在 1e-9 量级，完全可以接受。

> 这一个公式等价于之前 10 行代码的逐步反向传播，且同时并行处理 64 个神经元（沿 `sum(0)` 操作在所有列独立进行）。

---

## 8. 练习四：完整训练——不用 `loss.backward()`

### 目标

将练习一到三推导出的梯度整合到完整训练循环中，完全替换 `loss.backward()`。

### 完整反向传播（约 20 行代码）

```python
# Exercise 4 backward pass
# ---- 交叉熵（解析公式）----
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
# ---- 线性层 2 ----
dh     = dlogits @ W2.T
dW2    = h.T @ dlogits
db2    = dlogits.sum(0)
# ---- tanh ----
dhpreact = (1.0 - h**2) * dh
# ---- BatchNorm 参数 ----
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw  = bngain * dhpreact
# ---- BatchNorm（解析公式）----
dhprebn = bngain/n * bnvar_inv * (
    n*dbnraw - dbnraw.sum(0) - n/(n-1)*bnraw*(dbnraw*bnraw).sum(0)
)
# ---- 线性层 1 ----
dembcat = dhprebn @ W1.T
dW1     = embcat.T @ dhprebn
db1     = dhprebn.sum(0)
# ---- Embedding view ----
demb = dembcat.view(emb.shape)
# ---- Embedding lookup ----
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
```

### 参数更新

```python
grads = [dC, dW1, db1, dbngain, dbnbias, dW2, db2]
for p, grad in zip(parameters, grads):
    p.data += -lr * grad
```

注意：不再使用 `p.grad`，而是直接用我们手算的 `grad`。

### 与 PyTorch autograd 的梯度对比

```
logits | exact: False | approximate: True | maxdiff: ~1e-9
```

所有梯度均与 PyTorch 近似相等，差异在浮点误差范围内。

### 训练结果（n_hidden=200）

```
train loss: 2.0705
val   loss: 2.1099
```

与之前使用 `loss.backward()` 的结果几乎完全一致：

```
# 对比（使用 loss.backward()）
train: 2.0718
val:   2.1162
```

**结论**：我们的手动反向传播是完全正确的！

### 采样示例

```
carmahzamille.
khi.
mreigeet.
khalaysie.
mahnen.
...
```

生成的名字与之前相当，证明整个训练流程正常运行。

---

## 9. 本讲总结

### 核心知识点汇总

| 知识点 | 关键规则 |
|--------|---------|
| 链式法则 | 局部导数 × 上游梯度 |
| 广播的反向 | 沿广播维度**求和**（因为同一个变量被"重用"了） |
| 求和的反向 | 沿求和维度**广播**（梯度被"分发"到每个元素） |
| 节点被多次使用 | 梯度**相加**（多条支路的贡献累积） |
| 矩阵乘法 $C=AB$ | $dA = dC \cdot B^T$，$dB = A^T \cdot dC$ |
| tanh 反向 | $(1 - \tanh^2(z)) \cdot \text{上游梯度}$，用**输出**而非输入 |
| 交叉熵反向 | `softmax(logits) - one_hot(y)`，除以 $n$ |
| BatchNorm 反向 | 单公式（含求和修正项），效率远高于逐步推导 |

### Karpathy 金句

> "反向传播是一个**漏抽象**。你必须理解它的内部，不然它会在你不知情的情况下悄悄伤害你。"

> "我永远记不住矩阵乘法反向传播的公式，但我不需要记——**让形状来告诉你答案**。"

> "每当我看到这个反向传播代码，我都会想：其实也就 20 行，不复杂嘛。"

### 本讲的意义

练习完本讲后，你应该能够：

1. 对任意张量操作，推导其反向传播公式
2. 理解广播在反向传播中的对应关系
3. 对常见层（线性、tanh、softmax、BatchNorm）直接写出高效的梯度公式
4. 不依赖 `loss.backward()` 训练一个完整的神经网络

---

## 附：关键代码速查

### 反向传播通用模式

```python
# 元素乘法 c = a * b
da += b * dc   # a的梯度
db += a * dc   # b的梯度（注意广播时需要sum）

# 广播乘法（b被广播）
da += b * dc
db += (a * dc).sum(dim, keepdim=True)  # 沿广播轴求和

# 矩阵乘法 C = A @ B
dA += dC @ B.T
dB += A.T @ dC

# 求和 b = a.sum(dim, keepdim=True)
da += torch.ones_like(a) * db  # 或让广播自动处理

# 取幂 b = a ** k
da += k * a**(k-1) * db
```

### 交叉熵一步反向

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

### BatchNorm 解析反向

```python
dhprebn = bngain/n * bnvar_inv * (
    n * dbnraw
    - dbnraw.sum(0)
    - n/(n-1) * bnraw * (dbnraw * bnraw).sum(0)
)
```

### Embedding 反向

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        dC[Xb[k,j]] += demb[k,j]
```
