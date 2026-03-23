# Lecture 4：激活值、梯度与批归一化（Activations, Gradients, BatchNorm）

> 对应视频：[YouTube](https://youtu.be/P6sfmUTpUmc)
> 对应 Notebook：[makemore_part3_bn.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb)

---

## 目录

1. [本讲目标与背景](#1-本讲目标与背景)
2. [初始化问题一：输出层过于自信](#2-初始化问题一输出层过于自信)
3. [初始化问题二：tanh 隐藏层过度饱和](#3-初始化问题二tanh-隐藏层过度饱和)
4. [死神经元（Dead Neurons）](#4-死神经元dead-neurons)
5. [Kaiming 初始化（He 初始化）](#5-kaiming-初始化he-初始化)
6. [Batch Normalization](#6-batch-normalization)
7. [推理阶段的处理：Running Statistics](#7-推理阶段的处理running-statistics)
8. [BatchNorm 的奇特副作用：正则化](#8-batchnorm-的奇特副作用正则化)
9. [BatchNorm 的注意事项](#9-batchnorm-的注意事项)
10. [代码 PyTorch 化：自定义 Module](#10-代码-pytorch-化自定义-module)
11. [诊断工具：如何看出网络是否健康](#11-诊断工具如何看出网络是否健康)
12. [本讲总结](#12-本讲总结)

---

## 1. 本讲目标与背景

Karpathy 在课程开头说明：在进入 RNN、LSTM、GRU 等更复杂的架构之前，有必要先深入理解 MLP 的**激活值（activations）**和**梯度（gradients）**的行为。

**为什么要理解这些？**

> "理解激活值和梯度的行为，是理解为什么 RNN 难以用一阶梯度方法优化的关键。"

历史上，RNN 虽然理论上是通用近似器（universal approximator），但实践中极难优化。理解梯度如何流动，以及它们为什么会消失或爆炸，才能理解后续所有的架构改进（如 LSTM、残差连接、归一化层等）。

**本讲的起点代码**是上一讲的 MLP，做了如下整理：
- 提取了超参数：`n_embd = 10`（嵌入维度），`n_hidden = 200`（隐藏层神经元数）
- 数据集划分：80% 训练，10% 验证，10% 测试
- 参数量约 11,000

---

## 2. 初始化问题一：输出层过于自信

### 2.1 现象

训练的第 0 步（初始化时），loss 高达 **27**，然后迅速下降。这个 "曲棍球形"（hockey stick）的 loss 曲线是一个**危险信号**。

### 2.2 理论期望

初始化时，网络对任何字符都不应该有偏好，因此输出应该接近**均匀分布**。27 个字符的均匀分布对应的 loss 是：

```python
-torch.log(torch.tensor(1/27))  # ≈ 3.29
```

初始 loss 应该在 **3.29** 左右，而不是 27！

### 2.3 根本原因

```python
logits = h @ W2 + b2
```

`logits` 的值过大，导致 softmax 输出极端值——某些字符的概率接近 1，其余接近 0。网络"非常自信地犯了错"（confidently wrong），记录了极高的 loss。

**演示：**
```python
# 只有 4 个字符的简化例子
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = F.softmax(logits, dim=0)  # 均匀分布 [0.25, 0.25, 0.25, 0.25]
# loss = -log(0.25) ≈ 1.38  ← 正常

logits = torch.tensor([5.0, -2.0, 3.0, 1.0])  # 极端值
probs = F.softmax(logits, dim=0)  # 集中在第一个
# loss 急剧增大 ← 不正常
```

### 2.4 修复方法

让输出层的权重初始化得非常小，使 logits 接近 0：

```python
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01  # 乘以 0.01
b2 = torch.randn(vocab_size,             generator=g) * 0     # 偏置初始化为 0
```

**效果：**
- 初始 loss ≈ 3.32，接近理论值 3.29 ✓
- loss 曲线没有曲棍球形状，优化从一开始就在做"真正的工作"
- 最终验证集 loss：2.17 → **2.13**（有提升）

> **Karpathy 的忠告：** 不要把权重初始化为恰好 0，因为这会破坏对称性（symmetry breaking）。应该是"小的随机数"，而非"精确的零"。

---

## 3. 初始化问题二：tanh 隐藏层过度饱和

### 3.1 现象

即使 logits 的问题修复了，隐藏层还有一个更深层的问题。可以通过查看 `h`（tanh 的输出）发现：

```python
# 查看 h 的直方图
plt.hist(h.view(-1).tolist(), 50)
```

**结果**：大量的值聚集在 -1 和 +1 附近，而不是分散的。

再看进入 tanh 之前的 pre-activation `hpreact`：它的值分布在 -15 到 +15 之间，非常极端。

### 3.2 为什么这是个问题？

回忆 tanh 的反向传播（在 micrograd 中实现过）：

```python
# tanh 的反向传播
# t = tanh(x)，则 dt/dx = 1 - t²
out.grad += (1 - t**2) * t.grad
```

- 当 `t` 接近 **+1 或 -1**（即 tanh 饱和时），`1 - t²` ≈ **0**
- 梯度被**乘以接近 0 的数**，梯度消失！
- 无论 `out.grad` 有多大，经过饱和的 tanh 后梯度都变成 0

**形象理解：** 当神经元处于 tanh 的"平坦区域"（尾部），无论输入怎么变化，输出几乎不变，因此 loss 对这个神经元的参数几乎没有梯度，它无法学习。

### 3.3 可视化饱和情况

```python
# 查看有多少 tanh 输出值的绝对值超过 0.99（即进入饱和区）
(h.abs() > 0.99).float().mean()  # 大量神经元处于饱和区
```

可视化为二值图：白色 = 饱和，黑色 = 正常。如果某一**列全是白色**，说明这是一个**死神经元**（后面详述）。

### 3.4 修复方法

让 pre-activation `hpreact` 的值更小、更接近 0：

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.1  # 乘以 0.1
b1 = torch.randn(n_hidden,                        generator=g) * 0.01 # 小的随机偏置
```

**效果：**
- tanh 饱和率大幅降低
- 最终验证集 loss：2.13 → **2.10**

---

## 4. 死神经元（Dead Neurons）

### 4.1 什么是死神经元

**定义**：对于数据集中的**任意**输入，某个神经元都处于非线性激活函数的平坦区域，从不产生梯度，因此永远无法更新。

### 4.2 tanh 的死神经元

对于 tanh 神经元：如果不管输入什么数据，该神经元输出总是接近 +1 或 -1，则该神经元是死的。

### 4.3 ReLU 的死神经元（更常见）

ReLU 的定义：`f(x) = max(0, x)`

- 当 pre-activation < 0 时，输出恒为 0（完全平坦区域）
- 在此区域梯度**精确为 0**（tanh 是"接近 0"，ReLU 是"精确 0"）

**死 ReLU 的成因：**
1. **初始化时就死了**：初始权重/偏置恰好让某神经元对所有数据都有负的 pre-activation
2. **训练中被"打死"**：学习率过高，某次梯度更新幅度太大，把神经元推出了数据流形，之后没有任何数据能激活它

> Karpathy 把这种情况比作**"永久脑损伤"**（permanent brain damage）。

### 4.4 各种非线性的对比

| 激活函数 | 死亡风险 | 原因 |
|---------|--------|------|
| tanh | 中等 | 尾部接近 0，但不是精确 0 |
| sigmoid | 中等 | 类似 tanh，尾部接近 0 |
| ReLU | 高 | 负区间梯度**精确为 0** |
| Leaky ReLU | 低 | 负区间有小斜率，不会完全截断 |
| ELU | 低~中 | 负区间有平滑过渡 |

---

## 5. Kaiming 初始化（He 初始化）

### 5.1 问题：魔法数字从哪来？

前面我们手动调整了 `0.1`、`0.2` 等系数。这些数字是怎么来的？对于更大的网络怎么设置？

### 5.2 方差分析的直觉

Karpathy 用代码演示：假设输入 `x`（1000个样本，每个10维，均值0方差1的高斯），权重 `W`（高斯分布），做矩阵乘法 `y = x @ W`（200个神经元）：

```python
x = torch.randn(1000, 10)   # 输入：均值0，标准差1
W = torch.randn(10, 200)    # 权重

y = x @ W
print(y.std())  # ≈ 3.16，不是 1！
```

**标准差从 1 扩大到了约 3.16（≈ √10）**。这是因为：

> 对于 fan_in 个独立的均值为 0、方差为 1 的随机变量的求和，结果的标准差是 √(fan_in)。

### 5.3 保持标准差的正确初始化

为了让输出的标准差保持为 1，**权重需要除以 √(fan_in)**：

```python
W = torch.randn(fan_in, fan_out) / fan_in**0.5  # 或等价地 * (1/√fan_in)
y = x @ W
print(y.std())  # ≈ 1.0 ✓
```

### 5.4 Kaiming/He 初始化论文

论文：**"Delving deep into rectifiers"**（He et al., 2015）

- 专门研究 ReLU 网络的初始化问题
- 因为 ReLU 会截断负数（丢弃一半分布），需要额外乘以 **√2**（称为 gain）来补偿
- 公式：标准差 = `gain / √(fan_in)`

### 5.5 不同非线性对应的 gain

| 非线性 | gain |
|-------|------|
| 线性/无 | 1 |
| tanh | 5/3 ≈ 1.667 |
| ReLU | √2 ≈ 1.414 |

**为什么 tanh 需要 gain = 5/3？**

tanh 是一个**收缩变换**（contractive transformation），它会把分布的尾部压缩进去。乘以 5/3 是为了抵消这种压缩，使输出分布保持标准差为 1。

（Karpathy 坦承他不知道 5/3 的确切数学推导，但实验表明这个值让 Linear+tanh 的堆叠能保持稳定的标准差。）

### 5.6 PyTorch 中的 Kaiming 初始化

```python
# PyTorch 的 kaiming_normal_ 实现
torch.nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='tanh')

# 等价于手动：
W1 = torch.randn((fan_in, fan_out), generator=g) * (5/3) / (fan_in**0.5)
```

**注意**：PyTorch 的 `nn.Linear` 默认使用均匀分布而不是高斯，但思路相同：

$$
W \sim \text{Uniform}\left(-\frac{1}{\sqrt{k}},\ \frac{1}{\sqrt{k}}\right), \quad k = \text{fan\_in}
$$

### 5.7 现代为何不再那么重要

> Karpathy：大约 7 年前（即 2015 年前后），精确设置初始化非常关键，网络非常脆弱。现代有了以下创新，使初始化变得不那么敏感：
> 1. **残差连接（Residual connections）**
> 2. **归一化层（BatchNorm、LayerNorm 等）**
> 3. **更好的优化器（Adam、RMSProp 等）**

---

## 6. Batch Normalization

### 6.1 核心思想

Batch Normalization 出自 2015 年 Google 的论文（Ioffe & Szegedy）。

**洞见（Insight）**：既然我们希望隐藏层的激活值大致服从高斯分布（均值 0，标准差 1），那为什么不**直接把它们归一化为高斯分布**呢？

这个归一化操作是**完全可微分的**，可以接入反向传播。

### 6.2 BatchNorm 的数学

对于一个 batch 内的 pre-activation `hpreact`（形状：`[batch_size, n_hidden]`）：

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i \quad \text{（batch 均值）}
$$

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2 \quad \text{（batch 方差）}
$$

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \text{（归一化）}
$$

$$
y_i = \gamma \hat{x}_i + \beta \quad \text{（scale and shift）}
$$

代码实现：

```python
# 计算 batch 统计量（沿 dim=0，即跨样本方向）
bnmeani = hpreact.mean(0, keepdim=True)   # shape: [1, n_hidden]
bnstdi  = hpreact.std(0, keepdim=True)    # shape: [1, n_hidden]

# 归一化
hpreact_norm = (hpreact - bnmeani) / bnstdi

# scale & shift（可学习参数）
bngain = torch.ones((1, n_hidden))   # γ，初始化为 1
bnbias = torch.zeros((1, n_hidden))  # β，初始化为 0
hpreact = bngain * hpreact_norm + bnbias
```

### 6.3 为什么需要 γ 和 β？

如果只有归一化，网络**被强制要求**每一层的激活值都是标准高斯分布，但这并不总是最优的。我们只希望在**初始化时**是高斯分布，随着训练进行，网络应该能够自由调整分布的形状（更宽、更窄、更偏）。

- `γ`（gain）：可学习，初始化为 1
- `β`（bias）：可学习，初始化为 0
- 初始化时，`γ=1, β=0`，所以输出就是标准化后的结果
- 训练后，`γ` 和 `β` 会根据任务需要调整分布

### 6.4 BatchNorm 在网络中的位置

```
Linear → BatchNorm → Tanh → Linear → BatchNorm → Tanh → ...
```

**标准做法（来自 ResNet 等）**：BatchNorm 放在线性层/卷积层**之后**，非线性激活**之前**。

```python
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    # ...
]
```

---

## 7. 推理阶段的处理：Running Statistics

### 7.1 问题

BatchNorm 在训练时用**当前 batch 的均值和方差**来归一化。但推理（inference）时如果只有单个样本怎么办？

一个样本无法计算均值和方差！

### 7.2 方法一：事后校准（post-hoc calibration）

训练完成后，把**整个训练集**过一遍，计算全局均值和标准差：

```python
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    bnmean = hpreact.mean(0, keepdim=True)  # 全局均值
    bnstd  = hpreact.std(0, keepdim=True)   # 全局标准差
```

推理时用这两个固定值代替动态计算的 batch 统计量。

**问题**：没人想要这个"第二阶段"，太麻烦了。

### 7.3 方法二：训练时维护 Running Statistics（✓推荐）

在训练过程中，**用指数移动平均（EMA）同步更新**全局估计：

```python
# 初始化 running stats
bnmean_running = torch.zeros((1, n_hidden))  # 初始均值估计：0
bnstd_running  = torch.ones((1, n_hidden))   # 初始标准差估计：1

# 训练循环中（每个 iteration）
with torch.no_grad():  # 这个更新不参与反向传播！
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

- 动量（momentum）= 0.001（每次更新权重很小）
- 原因：batch size = 32，统计量每次波动较大，需要小动量使估计稳定

**PyTorch 默认 momentum = 0.1**，适用于 batch size 较大的情况。对于小 batch size，应减小 momentum。

推理时：
```python
# 不再用 batch 统计量，用 running 统计量
hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
```

**验证**：训练结束后，`bnmean_running ≈ bnmean`（两种方法估计结果相近）。

---

## 8. BatchNorm 的奇特副作用：正则化

### 8.1 coupling 现象

BatchNorm 引入了一个奇怪的性质：**同一个 batch 内的样本通过均值和方差相互耦合**。

对于某个输入样本 `x_i`，它的隐藏状态 `h_i` 不只取决于 `x_i`，还取决于**同 batch 中其他随机采样的样本**——因为 `μ_B` 和 `σ_B` 是整个 batch 共同计算的。

### 8.2 意外的正则化效果

这种 coupling 带来了一个**意外的正则化效果**：

- 每次 forward pass，同一个输入样本因为 batch 组成不同，得到的 `h` 会略有不同（"jitter"）
- 相当于对输入做了轻微的**数据增强**
- 使网络更难对具体训练样本过拟合
- 类似于 dropout 的效果

> Karpathy："这在某种奇怪的方式上实际上是有益的……它实际上是一种正则化器。"

### 8.3 为什么人们想摆脱 BatchNorm

- 样本耦合导致**各种奇怪的 Bug**
- 训练和推理行为不一致（training mode vs eval mode）
- Karpathy 自称"在这个层上多次踩坑"

**替代方案：**
- **Layer Normalization**（沿特征维度归一化，不耦合样本）—— 在 Transformer 中广泛使用
- **Group Normalization**
- **Instance Normalization**

> "人们一直试图淘汰 BatchNorm，转向这些不耦合样本的归一化技术，但很难，因为 BatchNorm 确实很好用。"

---

## 9. BatchNorm 的注意事项

### 9.1 ε（epsilon）的作用

```python
xhat = (x - xmean) / torch.sqrt(xvar + 1e-5)
```

`ε`（默认 `1e-5`）防止除以 0——当某个 batch 的方差恰好为 0 时避免数值崩溃。

### 9.2 前面的线性层不需要 bias！

**重要细节**：如果线性层后面紧跟 BatchNorm，线性层的 bias 是**无效的**！

原因：
```
h_preact = W @ x + b         # 线性层输出，加了 bias b
h_norm = (h_preact - mean(h_preact)) / std(h_preact)  # BatchNorm 减去均值
```

`b` 是常数，加到每个样本上，然后立即在 BatchNorm 的均值减法中被消掉了！`b` 对结果毫无影响，但浪费了参数和计算。

**正确做法：**
```python
# ✗ 错误：浪费参数
Linear(fan_in, fan_out, bias=True), BatchNorm1d(fan_out)

# ✓ 正确：禁用 bias，让 BatchNorm 的 β 承担偏置职责
Linear(fan_in, fan_out, bias=False), BatchNorm1d(fan_out)
```

这也是 PyTorch ResNet 实现中 `Conv2d(..., bias=False)` 的原因。

---

## 10. 代码 PyTorch 化：自定义 Module

### 10.1 构建 Module 系统

Karpathy 将代码重构为类似 PyTorch `nn.Module` 的风格，便于组合和复用。

#### Linear 层

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias   = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

#### BatchNorm1d 层

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps      = eps
        self.momentum = momentum
        self.training = True          # 训练/推理模式切换
        # 可学习参数（通过反向传播训练）
        self.gamma = torch.ones(dim)  # γ（scale）
        self.beta  = torch.zeros(dim) # β（shift）
        # Buffers（通过 EMA 更新，不参与反向传播）
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)   # batch 均值
            xvar  = x.var(0, keepdim=True)    # batch 方差
        else:
            xmean = self.running_mean
            xvar  = self.running_var

        xhat     = (x - xmean) / torch.sqrt(xvar + self.eps)  # 归一化
        self.out = self.gamma * xhat + self.beta               # scale & shift

        if self.training:
            with torch.no_grad():   # EMA 更新不建图
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]  # 注意：running stats 不在这里
```

#### Tanh 层

```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

### 10.2 构建 6 层深度网络

```python
n_embd  = 10
n_hidden = 100

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),             BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, bias=False),           BatchNorm1d(vocab_size),
]

with torch.no_grad():
    # 最后一层的 BatchNorm γ 乘以 0.1，让 softmax 初始时不那么自信
    layers[-1].gamma *= 0.1

parameters = [C] + [p for layer in layers for p in layer.parameters()]
# 总参数量：~47,000
```

### 10.3 forward pass

```python
emb = C[Xb]                    # 字符嵌入
x   = emb.view(emb.shape[0], -1)  # 拼接
for layer in layers:
    x = layer(x)               # 顺序通过各层
loss = F.cross_entropy(x, Yb)
```

### 10.4 训练后切换到 eval 模式

```python
for layer in layers:
    layer.training = False  # 切换到推理模式，使用 running stats
```

---

## 11. 诊断工具：如何看出网络是否健康

Karpathy 强调：**可视化是神经网络调试的核心工具**。

### 11.1 工具一：前向传播激活值分布

```python
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
            i, layer.__class__.__name__,
            t.mean(), t.std(),
            (t.abs() > 0.97).float().mean() * 100
        ))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')
```

**健康的迹象：**
- 每层的分布形状大致相似（各层均匀）
- 标准差在合理范围（比如约 0.65），不趋近于 0 或爆炸
- 饱和率（saturation）在合理范围，比如 5% 左右
- **不健康**：标准差越来越小（梯度消失前兆）或越来越大（梯度爆炸前兆）

**gain 的影响演示：**

| gain | 现象 |
|------|------|
| 1.0（太小） | 标准差层层收缩 → 0，饱和率 → 0% |
| 5/3（正确） | 标准差稳定在约 0.65，饱和率约 5% ✓ |
| 3.0（太大） | 饱和率过高，神经元大量处于 tanh 尾部 |

### 11.2 工具二：反向传播梯度分布

```python
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad  # 需要 retain_grad()
        print('layer %d (%10s): mean %+f, std %e' % (
            i, layer.__class__.__name__, t.mean(), t.std()
        ))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
plt.title('gradient distribution')
```

**健康的迹象：**
- 各层梯度的标准差大致相等（梯度均匀流动）
- 没有从深层到浅层的单调递减（梯度消失）
- 没有从深层到浅层的单调递增（梯度爆炸）

### 11.3 工具三：权重梯度分布

```python
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:  # 只看权重矩阵，跳过 1D 参数
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (
            tuple(p.shape), t.mean(), t.std(), t.std() / p.std()
        ))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {tuple(p.shape)}')
plt.title('weights gradient distribution')
```

**观察点：**
- 各权重矩阵的梯度标准差是否大致相等
- 输出层（最后一层）是否是"异类"（通常初始梯度偏大）

### 11.4 工具四：更新幅度比（Update-to-Data Ratio）⭐ 最重要

这是 Karpathy 认为**最重要的诊断工具**之一。

```python
# 训练循环中记录
ud = []
with torch.no_grad():
    ud.append([
        ((lr * p.grad).std() / p.data.std()).log10().item()
        for p in parameters
    ])
```

**含义：** 每次更新中，参数变化量的标准差 / 参数当前值的标准差，取 log10。

```python
# 绘图
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append(f'param {i}')
plt.plot([0, len(ud)], [-3, -3], 'k')  # 参考线：-3（即 1e-3）
plt.legend(legends)
```

**参考标准：**

| 比值（log10） | 含义 |
|-------------|------|
| ≈ -3 | 理想状态：更新量约为参数量的 0.1% |
| 远低于 -3 | 学习率太小，训练太慢 |
| 高于 -3（如 -1） | 学习率太大，更新幅度过大，不稳定 |

**实际演示：**
- 学习率 `0.001`（太小）：所有参数的比值在 -4 ~ -5，远低于 -3 → 训练太慢
- 学习率正常：参数比值大致在 -3 附近 ✓
- 没有 fan_in 归一化：各层比值差异巨大，有些高达 -1 ~ -1.5 → 严重失衡

> **重要观察**：没有 fan_in 归一化时，激活值图会显示神经元过度饱和，梯度分布混乱，而 update ratio 图也会揭示各层学习速率差异巨大。**多种工具交叉验证**，才能全面诊断问题。

---

## 12. 本讲总结

### 三条主线

| # | 主题 | 核心结论 |
|---|------|---------|
| 1 | 初始化 | 好的初始化减少"浪费的训练"，提升最终性能 |
| 2 | Batch Normalization | 从根本上稳定激活值分布，使深层网络可训练 |
| 3 | 诊断工具 | 可视化激活值、梯度、参数更新比，是调试神经网络的必备技能 |

### 问题 → 解法 → 效果

```
初始 val loss: 2.17
↓ 修复 softmax 过度自信（输出层权重 × 0.01）
val loss: 2.13
↓ 修复 tanh 过度饱和（隐藏层权重 × 0.2）
val loss: 2.10
↓ 使用 Kaiming 初始化（(5/3) / √fan_in）
val loss: 2.10（等效，但无魔法数字）
↓ 添加 BatchNorm
val loss: 2.10（在此数据集上无提升，因为瓶颈在上下文长度）
```

> Karpathy 指出：这个数据集的瓶颈已经不再是优化，而是**上下文长度**（只用 3 个字符预测下一个太短了）。进一步提升需要更强大的架构（RNN、Transformer）。

### BatchNorm 的取舍

```
优点：✓ 极大稳定训练
     ✓ 减少对精确初始化的依赖
     ✓ 有正则化副作用

缺点：✗ 耦合 batch 内样本
     ✗ 训练/推理行为不一致
     ✗ 导致各种难查的 Bug
     ✗ 不适合小 batch 或单样本推理
```

### 现代深度学习的 "配方"

来自 ResNet 等实际代码的模式：

```
卷积/线性层（bias=False）
   ↓
BatchNorm（包含自己的 bias=β）
   ↓
非线性激活（ReLU 或 tanh）
   ↓
重复以上结构
```

---

## 附：关键代码速查

### 初始化最佳实践（无 BatchNorm）

```python
W = torch.randn(fan_in, fan_out, generator=g) * (5/3) / (fan_in**0.5)  # tanh 网络
W_out = torch.randn(n_hidden, vocab_size, generator=g) * 0.01            # 输出层
b_out = torch.zeros(vocab_size)                                           # 输出层偏置
```

### BatchNorm 完整实现

```python
# 前向传播（训练时）
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi  = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

# 同步更新 running stats（不参与梯度）
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

### @torch.no_grad() 装饰器

```python
@torch.no_grad()
def split_loss(split):
    # 评估时不需要建立计算图，更高效
    ...
```

等价于在函数体中使用 `with torch.no_grad():` 上下文管理器。
