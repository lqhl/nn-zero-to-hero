# Lecture 1：神经网络与反向传播入门——从零构建 micrograd

> 对应视频：[YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
> 对应 Notebook：[micrograd_lecture_first_half_roughly.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb) 和 [micrograd_lecture_second_half_roughly.ipynb](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb)

---

## 目录

1. 本讲目标与背景
2. 什么是 micrograd？为什么重要？
3. 直观理解导数
4. 构建 Value 类——表达式图的核心
5. 可视化计算图
6. 手动反向传播：从简单例子出发
7. 链式法则——反向传播的数学核心
8. 通过神经元手动反向传播
9. 自动化反向传播：实现 `_backward`
10. 拓扑排序与 `backward()` 方法
11. 重要 Bug：梯度应累加（`+=`）而非覆盖（`=`）
12. tanh 的两种实现方式：整体 vs 原子分解
13. 与 PyTorch 对比验证
14. 构建神经网络：Neuron、Layer、MLP
15. 损失函数与训练循环
16. 重要 Bug：忘记 zero_grad
17. 本讲总结

---

## 1. 本讲目标与背景

### 为什么要从头构建 autograd？

现代深度学习库（PyTorch、JAX）的核心是**自动微分引擎**（autograd engine），它实现了**反向传播**（backpropagation）算法。反向传播让我们可以高效地计算损失函数对神经网络所有参数的梯度，进而用梯度下降来优化网络。

Karpathy 的 micrograd 只有约 **150 行纯 Python**，却完整实现了这套机制：
- `engine.py`：约 100 行，实现 autograd 引擎，**完全不知道神经网络是什么**
- `nn.py`：约 50 行，在 autograd 之上搭建神经网络库（Neuron、Layer、MLP）

> "micrograd 就是你训练网络所需要的全部，其余都只是效率问题。"

**标量 vs 张量**：micrograd 工作在单个标量级别，真实库使用张量（多维数组）是为了并行计算效率，数学本质完全一样。

---

## 2. 什么是 micrograd？为什么重要？

micrograd 是一个**标量值自动微分引擎**，支持建立数学表达式并对其求导。

### 示例：一个简单的计算图

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10)
d = a * b + c

d.backward()
print(d)			# 4.0
print(a.grad)	# -3.0
print(b.grad)	# 2.0
print(c.grad)	# 1.0
```

`a.grad = -3.0` 的含义：如果把 `a` 的值微微增大，`g` 会以 -3 的斜率减小（负梯度表示方向相反）。

**关键洞察**：神经网络本质上就是这样的数学表达式，只不过输入是数据和权重，输出是预测值或损失函数。

---

## 3. 直观理解导数

### 导数的数值估计

对于单变量函数 $f(x) = 3x^2 - 4x + 5$：

$$\frac{df}{dx} \approx \frac{f(x+h) - f(x)}{h}, \quad h \to 0$$

```python
h = 0.0001
x = 3.0
print((f(x + h) - f(x)) / h)  # ≈ 14.0
```

解析验证：$f'(x) = 6x - 4$，代入 $x=3$ 得 $14$，吻合。

### 多变量情形

对于 $d = a \cdot b + c$（$a=2, b=-3, c=10$），可以分别估计各偏导：

| 变量 | 偏导数（数值估计） | 解析值 | 原因 |
|------|------------------|--------|------|
| $a$  | $-3.0$           | $b = -3$ | $\partial d/\partial a = b$ |
| $b$  | $2.0$            | $a = 2$  | $\partial d/\partial b = a$ |
| $c$  | $1.0$            | $1$      | $\partial d/\partial c = 1$ |

---

## 4. 构建 Value 类——表达式图的核心

### 核心思路

`Value` 对象包装一个标量数值，并记录：
- `data`：当前值
- `grad`：该节点对最终输出的梯度（初始为 0）
- `_prev`：产生本节点的子节点集合（记录图结构）
- `_op`：产生本节点的操作（`'+'`、`'*'` 等）
- `_backward`：反向传播时该节点执行的函数（闭包）

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0          # 初始梯度为 0
        self._backward = lambda: None  # 默认什么都不做
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

### 构建表达式图

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b;  e.label = 'e'   # e = -6
d = e + c;  d.label = 'd'   # d = 4
f = Value(-2.0, label='f')
L = d * f;  L.label = 'L'   # L = -8
```

通过 `_prev`，我们可以从 `L` 追溯整棵计算树：`L → (d, f) → (e, c) → (a, b)`。

---

## 5. 可视化计算图

使用 `graphviz` 可以将计算图渲染出来，每个节点显示 `label | data | grad`：

```python
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)),
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                 shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot
```

**可视化的意义**：可以直观地看到前向传播中每个中间值，以及反向传播后每个节点的梯度。

---

## 6. 手动反向传播：从简单例子出发

目标：计算 $\frac{\partial L}{\partial x}$，对所有中间节点 $x$。

### 计算图：$L = d \cdot f$，$d = e + c$，$e = a \cdot b$

**步骤 1：基础情形**

$$\frac{\partial L}{\partial L} = 1$$

```python
L.grad = 1.0
```

**步骤 2：$L = d \cdot f$ 的节点**

对乘法求导：

$$\frac{\partial L}{\partial d} = f = -2, \quad \frac{\partial L}{\partial f} = d = 4$$

```python
d.grad = f.data  # = -2.0
f.grad = d.data  # = 4.0
```

**步骤 3：$d = e + c$（加法节点）**

加法的本地导数为 1，所以梯度直接"路由"到两个子节点：

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} = (-2) \cdot 1 = -2$$

$$\frac{\partial L}{\partial e} = (-2) \cdot 1 = -2$$

> **直觉**：加法节点就像一个"分叉路口"，把梯度平等地分发给所有子节点。

**步骤 4：$e = a \cdot b$（乘法节点）**

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial a} = (-2) \cdot b = (-2) \cdot (-3) = 6$$

$$\frac{\partial L}{\partial b} = (-2) \cdot a = (-2) \cdot 2 = -4$$

**数值验证（梯度检验）**：

```python
# 验证 a.grad = 6
h = 0.001
a.data += h
L2 = (a*b + c) * f
print((L2.data - L.data) / h)  # ≈ 6.0 ✓
```

### 利用梯度"推动" L 上升

```python
a.data += 0.01 * a.grad  # a 沿梯度方向调整
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad
# 重新计算 L，L 从 -8 变为约 -7 → 确实上升了
```

---

## 7. 链式法则——反向传播的数学核心

### 链式法则

若 $z$ 依赖 $y$，$y$ 依赖 $x$，则：

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

> **Karpathy 的比喻**：如果汽车速度是自行车的 2 倍，自行车速度是步行的 4 倍，那么汽车速度是步行的 $2 \times 4 = 8$ 倍。把中间速率乘起来就是链式法则。

### 在计算图中的应用

对于图中任意一个节点，我们知道：
1. **全局梯度**：$\frac{\partial L}{\partial \text{out}}$（从后续节点传来的）
2. **本地梯度**：$\frac{\partial \text{out}}{\partial \text{input}}$（只需看本节点的操作）

将二者相乘，就得到对输入节点的梯度：

$$\frac{\partial L}{\partial \text{input}} = \frac{\partial L}{\partial \text{out}} \cdot \frac{\partial \text{out}}{\partial \text{input}}$$

反向传播本质上就是**递归地在计算图上应用链式法则**。

---

## 8. 通过神经元手动反向传播

### 神经元的数学模型

一个神经元计算：

$$o = \tanh\!\left(\sum_i w_i x_i + b\right)$$

其中 $w_i$ 是权重（突触强度），$x_i$ 是输入，$b$ 是偏置（控制神经元的"触发倾向"）。

```python
# 两个输入的神经元示例
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)

x1w1 = x1 * w1
x2w2 = x2 * w2
n = x1w1 + x2w2 + b   # n ≈ 0.8814
o = n.tanh()           # o ≈ 0.7071
```

### tanh 的导数

$$\frac{d}{dn}\tanh(n) = 1 - \tanh^2(n) = 1 - o^2$$

代入数值：$1 - 0.7071^2 \approx 0.5$。

### 手动反向传播各节点

从 $o$ 向 $n$ 再向各输入反向传播：

| 节点 | 梯度 | 推导 |
|------|------|------|
| `o` | 1.0 | 基础情形 |
| `n` | 0.5 | $1 - o^2 = 1 - 0.5$ |
| `x1w1 + x2w2` | 0.5 | 加法路由 |
| `b` | 0.5 | 加法路由 |
| `x2w2` | 0.5 | 加法路由 |
| `x1w1` | 0.5 | 加法路由 |
| `w2` | 0.0 | $x_2 \cdot 0.5 = 0 \cdot 0.5 = 0$（因为 $x_2=0$！） |
| `x2` | 0.5 | $w_2 \cdot 0.5 = 1 \cdot 0.5 = 0.5$ |
| `w1` | 1.0 | $x_1 \cdot 0.5 = 2 \cdot 0.5 = 1.0$ |
| `x1` | -1.5 | $w_1 \cdot 0.5 = -3 \cdot 0.5 = -1.5$ |

> **直觉**：`w2.grad = 0` 是因为 $x_2 = 0$，所以不管 $w_2$ 怎么变，乘以 0 之后对输出没有任何影响。

---

## 9. 自动化反向传播：实现 `_backward`

### 核心思路

在每次运算（`+`、`*`、`tanh`）时，创建一个**闭包函数** `_backward`，这个函数知道如何将 `out.grad` 传回 `self.grad` 和 `other.grad`。

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad  += 1.0 * out.grad   # 加法本地导数 = 1
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad  += other.data * out.grad  # 乘法本地导数 = 另一方的值
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad  # tanh 导数 = 1 - tanh^2
    out._backward = _backward
    return out
```

**注意**：这里用 `+=` 而不是 `=`，原因见第 11 节。

### 手动调用顺序

```python
o.grad = 1.0          # 初始化根节点梯度
o._backward()         # 传播到 n
n._backward()         # 传播到 x1w1+x2w2 和 b
# ... 依次调用，直到所有叶节点
```

---

## 10. 拓扑排序与 `backward()` 方法

### 问题

我们需要按正确顺序调用各节点的 `_backward`：必须先处理"后面"的节点，再处理"前面"的节点。这就是**拓扑排序**（Topological Sort）。

### 拓扑排序算法

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)  # 所有子节点处理完才加入自己
```

**关键性质**：节点只在其所有子节点都已入列后才加入 `topo`，因此 `reversed(topo)` 给出从输出到输入的正确顺序。

### 完整 `backward()` 方法

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0          # 根节点梯度 = 1
    for node in reversed(topo):
        node._backward()     # 按拓扑序反向调用
```

现在可以一行搞定：

```python
o.backward()  # 自动完成所有反向传播！
```

---

## 11. 重要 Bug：梯度应累加（`+=`）而非覆盖（`=`）

### 问题复现

```python
a = Value(3.0, label='a')
b = a + a   # 同一个变量用了两次
b.backward()
print(a.grad)  # 错误！应该是 2.0，却得到 1.0
```

### 原因分析

当 `self` 和 `other` 是**同一个对象**时（即一个变量用了两次），`__add__` 的 `_backward` 先设 `self.grad = 1.0`，然后再设 `other.grad = 1.0`，但这两行都在操作同一个对象，第二行把第一行的结果覆盖了！

类似地，当一个变量在多条路径上被使用时（如既参与加法又参与乘法），反向传播会从多条路径传回梯度，必须将它们**累加**。

### 修复方案

将所有 `_backward` 中的赋值改为累加：

```python
self.grad  += other.data * out.grad  # ✓ += 而非 =
other.grad += self.data * out.grad
```

由于 `grad` 初始化为 0，累加是安全的。

> **数学依据**：多元链式法则中，当一个变量影响多条路径时，各路径的梯度贡献应该**相加**。

---

## 12. tanh 的两种实现方式

### 方式一：整体实现（作为一个操作）

```python
def tanh(self):
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
```

计算图中只有一个 `tanh` 节点。

### 方式二：分解为原子操作

```python
# exp, *, +, / 等操作
e = (2 * n).exp()
o = (e - 1) / (e + 1)
```

计算图展开为更多节点，但结果**完全等价**。

### 关键洞察

> **操作粒度完全由你决定**：只要你知道该操作的前向传播公式，以及对应的本地导数，就可以把它实现为 micrograd 中的一个操作。可以是一个原子加法，也可以是一整个 tanh，甚至是更复杂的函数块。

这也是为什么 PyTorch 允许用户注册自定义函数（只需提供 `forward` 和 `backward`）。

### 实验验证

分解后的反向传播结果：
- `w2.grad = 0`，`x2.grad = 0.5`
- `w1.grad = 1.0`，`x1.grad = -1.5`

与整体 tanh 完全一致 ✓

---

## 13. 与 PyTorch 对比验证

```python
import torch

x1 = torch.Tensor([2.0]).double();  x1.requires_grad = True
x2 = torch.Tensor([0.0]).double();  x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double();  w2.requires_grad = True
b  = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
o.backward()

print(x1.grad.item())  # -1.5000 ✓
print(w1.grad.item())  #  1.0000 ✓
print(x2.grad.item())  #  0.5000 ✓
print(w2.grad.item())  #  0.0000 ✓
```

micrograd 和 PyTorch 结果完全一致。

**差异点**：
- PyTorch 基于张量（多维数组），micrograd 基于标量
- `requires_grad` 默认为 `False`（出于效率考虑，通常叶节点不需要梯度）
- 数据类型需要注意：PyTorch 默认 `float32`，需要 `.double()` 匹配 Python 的 `float64`

---

## 14. 构建神经网络：Neuron、Layer、MLP

### Neuron（单个神经元）

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w · x + b，然后过 tanh
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]
```

`nin` 个权重 + 1 个偏置 = `nin + 1` 个参数。

### Layer（神经元层）

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

### MLP（多层感知机）

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### 使用示例

```python
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])  # 3 输入，两个隐藏层各 4 神经元，1 输出
output = n(x)          # Value(data=0.165...)
print(len(n.parameters()))  # 41 个参数
```

**参数量计算**：
- 第 1 层：$3 \times 4 + 4 = 16$ 个参数
- 第 2 层：$4 \times 4 + 4 = 20$ 个参数
- 第 3 层：$4 \times 1 + 1 = 5$ 个参数
- 合计：$16 + 20 + 5 = 41$ 个参数

---

## 15. 损失函数与训练循环

### 数据集与均方误差损失

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # 期望目标

# 均方误差（MSE）损失
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

**MSE 损失的性质**：
- 当预测 = 目标时，损失 = 0（最优）
- 偏差越大，损失越高
- 平方操作确保符号不影响损失大小

### 梯度下降训练循环（正确版本）

```python
for k in range(20):
    # 1. 前向传播
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2. 清零梯度（关键！）
    for p in n.parameters():
        p.grad = 0.0

    # 3. 反向传播
    loss.backward()

    # 4. 更新参数（梯度下降）
    for p in n.parameters():
        p.data += -0.05 * p.grad  # 负号：沿梯度反方向最小化损失

    print(k, loss.data)
```

**训练过程中的损失变化**（学习率 0.05，20 步）：

```
0  4.84
1  4.36
...
19 0.0021  ← 接近 0，训练成功
```

最终预测：

```python
ypred  # [≈0.982, ≈-0.986, ≈-0.977, ≈0.973] 非常接近 [1, -1, -1, 1]
```

---

## 16. 重要 Bug：忘记 zero_grad

### 现象

```python
# 错误的训练循环（缺少 zero_grad）
for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum(...)
    loss.backward()        # ← 没有先清零梯度！
    for p in n.parameters():
        p.data += -0.1 * p.grad
```

> "这是我人生中第 20 次犯这个 bug，还是在镜头前……这是最常见的神经网络 bug。"

### 根本原因

`_backward` 使用 `+=` 累加梯度。没有清零的话，第 2 步的梯度会加在第 1 步的梯度上，第 3 步的梯度会加在前两步之和上……梯度会**无限累积**，等效于使用了一个越来越大的学习率。

### 为什么这次"侥幸"起了作用？

这个特定的 4 样本问题过于简单，即使梯度爆炸也能收敛。对于更复杂的问题，这个 bug 会导致训练失败。

### 修复

```python
# 每次反向传播前清零
for p in n.parameters():
    p.grad = 0.0
loss.backward()
```

PyTorch 中对应：`optimizer.zero_grad()`

---

## 17. 本讲总结

### 核心概念总结

| 概念 | 含义 |
|------|------|
| **前向传播** | 从输入经过表达式图计算输出（损失） |
| **反向传播** | 从输出出发，递归应用链式法则计算所有梯度 |
| **梯度** | 参数对损失的影响方向和强度 |
| **梯度下降** | 沿梯度反方向微小移动参数，使损失下降 |
| **学习率** | 每步更新的步长，太小收敛慢，太大发散 |
| **zero_grad** | 每次反向传播前必须清零梯度 |

### 神经网络训练的完整流程

```
1. 前向传播：输入数据 → 网络 → 预测值 → 损失
2. 清零梯度：所有参数的 .grad = 0
3. 反向传播：损失.backward() → 计算所有参数的梯度
4. 参数更新：param.data -= lr * param.grad
5. 重复 1-4
```

### 关键洞察

- 神经网络就是数学表达式，反向传播就是链式法则
- 操作粒度由你决定（整体 tanh vs 分解为原子操作，数学等价）
- 真实的 PyTorch 与 micrograd 的 API 高度相似，原理完全相同；区别仅在于张量化以实现并行效率
- 参数有几十亿时，基本逻辑和这里 41 个参数时**完全一样**

---

## 附：关键代码速查

### Value 类（完整版）

```python
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        # other 必须是 int 或 float
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  return self * other
    def __radd__(self, other):  return self + other
    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other)
    def __truediv__(self, other): return self * other**-1

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad  # d/dx(e^x) = e^x
        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad  # d/dx(tanh x) = 1 - tanh²x
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

### 完整训练循环

```python
# 初始化网络
n = MLP(3, [4, 4, 1])

# 数据
xs = [[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(20):
    # 前向传播
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 清零梯度（必须在 backward 之前！）
    for p in n.parameters():
        p.grad = 0.0

    # 反向传播
    loss.backward()

    # 梯度下降
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```

### 数值梯度检验

```python
def numerical_grad(f, x, h=1e-5):
    """估计 f 关于 x.data 的导数"""
    x.data += h
    y2 = f()
    x.data -= 2*h
    y1 = f()
    x.data += h  # 恢复原值
    return (y2 - y1) / (2 * h)
```
