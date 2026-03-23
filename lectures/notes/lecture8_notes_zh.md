# Lecture 8：从零构建 GPT Tokenizer

> 对应视频：[YouTube](https://www.youtube.com/watch?v=zduSFxRajkE)
> 对应代码：[minBPE GitHub](https://github.com/karpathy/minbpe)

---

## 目录

1. 本讲目标与背景
2. 什么是 Tokenization（分词）
3. Unicode 与 UTF-8 编码
4. BPE 算法原理
5. 从零实现 BPE：get_stats 与 merge
6. 训练 Tokenizer：完整循环
7. Decode（解码）实现
8. Encode（编码）实现
9. GPT-2 的正则表达式预分割
10. Special Tokens（特殊标记）
11. tiktoken 库与 GPT-4 Tokenizer
12. SentencePiece vs tiktoken
13. 词表大小（vocab_size）的选择
14. Tokenization 导致的 LLM 怪异行为
15. minBPE 代码库总览
16. 本讲总结

---

## 1. 本讲目标与背景

### 为什么要学 Tokenization

Karpathy 在视频开头就承认：

> "I have a set face and that's because tokenization is my least favorite part of working with large language models."

Tokenization（分词）是 LLM pipeline 中一个**必要但令人头疼**的预处理阶段。它是 LLM 很多奇怪行为的根本原因，但往往被初学者忽视。

### 与前讲的联系

- **Lecture 7（GPT from scratch）** 中我们使用了字符级别（character-level）的 tokenizer，词表只有 65 个字符
- 本讲构建真正的 **工业级 BPE Tokenizer**，词表大小约 50k–100k
- Tokenizer 是独立于语言模型的**单独的预处理阶段**，有自己的训练集和训练算法

### Tokenizer 的两个核心功能

```
字符串 (raw text)
    ↓  encode()
token 序列 (list of integers)
    ↓  decode()
字符串 (raw text)
```

---

## 2. 什么是 Tokenization（分词）

### 字符级 Tokenizer（naive 版）

在 GPT from scratch 中：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

- 词表 = 所有可能字符，约 65 个
- 每个字符对应一个整数 token
- 1000 个字符 → 1000 个 token（无压缩）

### 现代 LLM 的 Tokenizer

**不是字符级，而是字符块（chunk）级别**：

| 模型 | 词表大小 | 最大上下文 |
|------|---------|-----------|
| GPT-2 | 50,257 | 1,024 tokens |
| GPT-4 | ~100,278 | 128k tokens |
| Llama 2 | 32,000 | 4,096 tokens |

GPT-4 词表约是 GPT-2 的 2 倍，同样的文本压缩到约一半的 token 数：

```
GPT-2 tokenizer: 同一段文本 → 300 tokens
GPT-4 tokenizer: 同一段文本 → 185 tokens
```

> Karpathy 的直觉：词表越大 → 序列越短 → Transformer 能"看到"的文本越多（相当于注意力机制覆盖了更多内容），但也有上限

---

## 3. Unicode 与 UTF-8 编码

### Python 字符串 = Unicode 码点序列

```python
# 获取字符的 Unicode 码点
ord('H')      # → 104
ord('안')     # → 50000
ord('🌍')    # → 128000
```

Unicode 标准目前定义了约 **150,000 个字符**，跨越 161 种文字系统。

### 为什么不直接用 Unicode 码点？

- 词表会有 150,000+ 个元素，太大
- Unicode 标准还在持续更新，不稳定

### 三种 UTF 编码方式

| 编码 | 特点 | 字节长度 |
|------|------|---------|
| UTF-8 | 变长，兼容 ASCII | 1~4 字节/码点 |
| UTF-16 | 变长，对 ASCII 有冗余（含 0 字节） | 2~4 字节/码点 |
| UTF-32 | 定长，但非常浪费 | 固定 4 字节/码点 |

**结论：使用 UTF-8**
- 兼容 ASCII（历史遗留系统友好）
- 英文字符 = 1 字节
- 中文/表情等复杂字符 = 2~4 字节

```python
text = "hello 안녕 🌍"
list(text.encode("utf-8"))
# → [104, 101, 108, 108, 111, 32, 236, 149, 136, 235, 133, 149, 32, 240, 159, 140, 141]
```

### 为什么不直接用 UTF-8 字节流？

- 词表只有 256（0~255），**太小**
- 文本序列变得非常长
- Transformer 上下文窗口是有限的，超长序列根本不够用

**解决方案 → BPE（Byte Pair Encoding）**：在 256 字节的基础上，通过合并常见字节对来扩大词表、压缩序列。

---

## 4. BPE 算法原理

### 算法来源

BPE（Byte Pair Encoding）将 UTF-8 字节级别的 BPE 用于 LLM tokenization 的方式来自 **GPT-2 论文（2019）**，OpenAI 在其 "Input Representation" 章节中详细描述。

### 核心思路（Wikipedia 示例）

初始序列（词表大小 4，序列长 11）：
```
a a b d a a b c a b c d
```

**迭代步骤**：
1. 找最频繁的字节对 `aa` → 新 token `Z = aa` → 替换，序列长度减少
2. 再找最频繁的字节对 `ab` → 新 token `Y = ab` → 替换
3. 再找 `zy` → 新 token `X = zy` → 替换

最终：序列从 11 缩短到 5，词表从 4 扩展到 7。

### 应用到字节流

```
起点：256 个原始字节 token（词表大小 = 256）
目标：扩大到 vocab_size（如 GPT-4 的 100,256）
过程：做 vocab_size - 256 次合并
```

每次合并：
1. 统计所有相邻字节对的频率
2. 找出频率最高的字节对
3. 为该字节对分配新 ID（从 256 开始递增）
4. 将序列中所有该字节对替换为新 ID
5. 记录合并规则到 `merges` 字典

---

## 5. 从零实现 BPE：get_stats 与 merge

### 准备训练文本

```python
# 取一段博客文章文本
text = "..."  # 较长的文本
tokens = list(text.encode("utf-8"))  # raw bytes as integers
# 示例：len(tokens) = 616 for a 533-character paragraph
```

### get_stats：统计相邻字节对频率

```python
def get_stats(ids, counts=None):
    """
    统计 ids 列表中所有相邻对出现的次数
    例：[1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # 相邻元素迭代
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
# 找出最高频率对
top_pair = max(stats, key=stats.get)
# 示例：(101, 32) 出现 20 次，即 'e' + ' '（空格）
```

**用 chr() 验证**：
```python
chr(101), chr(32)  # → ('e', ' ')
# 大量单词以 e 结尾，后面跟空格，所以 "e " 最常见
```

### merge：替换字节对为新 token

```python
def merge(ids, pair, idx):
    """
    在 ids 中将所有连续出现的 pair 替换为新整数 idx
    例：ids=[1,2,3,1,2], pair=(1,2), idx=4 → [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 注意：检查边界条件，避免越界
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# 小测试
merge([5, 6, 6, 7, 9, 1], (6, 7), 99)
# → [5, 6, 99, 9, 1]

# 实际使用：合并最常见对 (101, 32) → 新 ID 256
tokens2 = merge(tokens, (101, 32), 256)
# 原长 616 → 新长 596（减少了 20，即出现了 20 次）
```

---

## 6. 训练 Tokenizer：完整循环

```python
vocab_size = 276  # 目标词表大小（256 + 20 次合并）
num_merges = vocab_size - 256

ids = list(tokens)  # 复制一份，不修改原始数据
merges = {}  # (int, int) -> int，记录合并规则

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)  # 找最频繁对
    idx = 256 + i                      # 新 token ID
    ids = merge(ids, pair, idx)        # 替换
    merges[pair] = idx                 # 记录合并规则
    print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")
```

**输出示例**：
```
merge 1/20: (101, 32) -> 256 had 20 occurrences
merge 2/20: (32, 116) -> 257 had 15 occurrences
...
merge 20/20: (258, 259) -> 275 had 12 occurrences
```

**重要特性**：
- 新 token（如 256）本身也能参与后续合并
- 所以 `merges` 实际上是一个**合并森林**（binary forest），而不是单棵树
- 叶节点 = 原始 256 个字节；新节点 = 合并产生的更长字节序列

### 压缩率验证

```python
# 原始：24,000 字节
# 20 次合并后：19,000 tokens
compression_ratio = 24000 / 19000  # ≈ 1.27
```

20 次合并就获得了 1.27 的压缩率；生产级词表（100k tokens）会有高得多的压缩率。

### Tokenizer 是独立的预处理阶段

```
训练数据（raw text）
    ↓  tokenizer 的训练集（可以不同）
BPE 算法 → merges 字典 + vocab 词表
    ↓  一次性预处理
token 序列文件（存磁盘）
    ↓  语言模型的训练集
Transformer 训练
```

> **关键点**：Tokenizer 有**自己的训练集**，与语言模型的训练集可以不同。训练集的语言分布决定了哪些语言有更多的合并，从而决定了对哪种语言"更友好"。

---

## 7. Decode（解码）实现

解码 = 给定 token ID 序列，返回字符串

### 构建 vocab 字典

```python
# vocab: int -> bytes
vocab = {idx: bytes([idx]) for idx in range(256)}  # 初始化 256 个原始字节

# 按照合并顺序（必须有序！）构建更大的 token 的字节表示
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]  # 字节拼接
```

> **注意**：Python 3.7+ 保证字典按插入顺序迭代，所以这里的 `merges.items()` 迭代顺序是正确的。更早的 Python 版本可能出问题。

### decode 函数

```python
def decode(ids):
    # 1. 查 vocab 得到每个 token 的 bytes
    text_bytes = b"".join(vocab[idx] for idx in ids)
    # 2. 将 bytes 解码为 Python 字符串
    text = text_bytes.decode("utf-8", errors="replace")
    return text
```

### 陷阱：不是所有字节序列都是有效 UTF-8

```python
decode([128])
# 错误！ValueError: 'utf-8' codec can't decode byte 0x80...
```

**原因**：UTF-8 有严格的多字节格式规定，单独的字节 128（二进制 `10000000`）不构成有效的 UTF-8 序列起始字节。

**解决方案**：使用 `errors="replace"`，将无效字节替换为 `\ufffd`（替换字符 `？`）

```python
text_bytes.decode("utf-8", errors="replace")
```

这也是 OpenAI 源码中的处理方式。如果你看到输出中出现 `？`，说明模型生成了无效的 token 序列。

---

## 8. Encode（编码）实现

编码 = 给定字符串，返回 token ID 序列

```python
def encode(text):
    # 1. 转为字节 → 整数列表
    ids = list(text.encode("utf-8"))

    while len(ids) >= 2:
        # 2. 找出当前序列中所有相邻对
        stats = get_stats(ids)

        # 3. 找出 merges 字典中索引最小的那个对
        #    （即最早发生的合并，优先级最高）
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        # 4. 如果没有任何对在 merges 中，说明无法继续合并
        if pair not in merges:
            break

        # 5. 执行合并
        idx = merges[pair]
        ids = merge(ids, pair, idx)

    return ids
```

**核心逻辑**：按照合并的**优先级顺序**（越早的合并优先级越高）来决定编码顺序，因为晚期合并依赖于早期合并的结果。

### 正确性验证

```python
# 编码后再解码，应该得到原始字符串
assert decode(encode(text)) == text  # True

# 注意：反方向不成立
# decode 的结果不一定能 encode 回去（因为不是所有字节序列都是有效 UTF-8）
```

---

## 9. GPT-2 的正则表达式预分割

### 问题：朴素 BPE 的缺陷

如果直接对整个文本运行 BPE，会出现问题：

> 例：单词 `dog` 后面跟着各种标点（`dog.`、`dog!`、`dog?`），BPE 可能将它们全部合并为不同的 token，把语义与标点混在一起。

### GPT-2 的解决方案：先用正则切分，再做 BPE

```python
import regex as re  # 注意：不是 import re，而是 pip install regex

# GPT-2 的正则分割模式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

### 逐段解释正则模式

| 子模式 | 匹配内容 | 示例 |
|--------|---------|------|
| `'(?:[sdmt]\|ll\|ve\|re)` | 英语缩写（'s, 'd, 'm, 't, 'll, 've, 're） | `'s`, `'ll` |
| `?\p{L}+` | 可选空格 + 一个或多个字母（任意语言） | `hello`, ` world` |
| `?\p{N}+` | 可选空格 + 一个或多个数字 | ` 123` |
| `?[^\s\p{L}\p{N}]+` | 可选空格 + 标点/符号 | `.`, `!`, `?` |
| `\s+(?!\S)` | 空白字符，但不包含最后一个（防止空格被吸收到下一个词） | 多余空格 |
| `\s+` | 剩余空白字符 | |

```python
import regex as re
pattern = re.compile(GPT2_SPLIT_PATTERN)
result = re.findall(pattern, "Hello world how are you")
# → ['Hello', ' world', ' how', ' are', ' you']

result = re.findall(pattern, "Hello world 123 how are you")
# → ['Hello', ' world', ' 123', ' how', ' are', ' you']
```

### 分割的作用

文本先被分成若干**独立的块（chunk）**，然后每个块**独立进行 BPE 合并**，最后结果拼接。

**效果**：
- 跨块的字节对**永远不会被合并**
- 字母与数字、数字与标点之间的合并被禁止
- 语义完整的词不会与相邻标点合并

### GPT-2 正则的已知缺陷

1. **大写缩写问题**：模式仅匹配小写缩写（`'s`），不匹配大写（`'S`）。`HOUSE'S` 的 `'S` 会被单独处理，与 `house's` 的处理不一致
2. **Unicode 撇号问题**：只匹配 ASCII 撇号（`'`），不匹配 Unicode 撇号（`'`）

> 在 GPT-2 的 GitHub 评论中，开发者自己承认应该加 `re.IGNORECASE`，这是一个已知的 bug。

### GPT-4 的改进

```python
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

主要改进：
1. `(?i:...)` → 大小写不敏感，修复了缩写问题
2. `\p{N}{1,3}` → 数字最多合并 3 位，防止长数字串合并为单 token（有助于算术）
3. 空白字符处理更精细

---

## 10. Special Tokens（特殊标记）

### 什么是特殊 Token

除了通过 BPE 训练得到的 token，还可以手动插入**任意特殊 token**，用于标注数据结构。

### GPT-2 的特殊 Token

GPT-2 词表大小 = **50,257**：
- 256 个原始字节 token
- 50,000 次合并 = 50,000 个新 token
- **1 个特殊 token：`<|endoftext|>`（ID = 50256）**

`<|endoftext|>` 的作用：
- 在训练数据中，多个文档之间插入该 token
- 告诉语言模型：文档已结束，接下来的内容与之前无关
- 语言模型通过数据学习到：见到这个 token 就"清空记忆"

```python
# 在 tiktoken 中使用
import tiktoken
enc = tiktoken.get_encoding("gpt2")
enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
# → [50256]
```

### GPT-4 的特殊 Token（更多）

```
<|endoftext|>     # 文档结束
<|fim_prefix|>    # Fill-in-the-Middle（FIM）：前缀
<|fim_middle|>    # FIM：中间
<|fim_suffix|>    # FIM：后缀
<|endofprompt|>   # 提示结束
```

### Chat 模型的特殊 Token（更复杂）

以 GPT-3.5-turbo 为例，多轮对话使用了：
```
<|im_start|>  # 消息开始（imaginary machine）
<|im_end|>    # 消息结束
```

这些特殊 token 划定了 system/user/assistant 消息的边界。

### 添加特殊 Token 需要做"模型手术"

当为预训练模型添加新的特殊 token 时，需要：
1. **扩展 embedding 表**：为新 token 添加新行（随机初始化或小随机数）
2. **扩展 LM Head（输出层）**：增加对应的输出维度
3. 通常冻结原始参数，只训练新增参数

```python
# 注册特殊 token（minBPE 示例）
tokenizer.register_special_tokens({"<|endoftext|>": 100257})
```

---

## 11. tiktoken 库与 GPT-4 Tokenizer

### tiktoken 简介

OpenAI 官方的分词库（Rust 实现，速度极快）：

```bash
pip install tiktoken
```

```python
import tiktoken

# 获取 GPT-2 tokenizer（inference only，无训练功能）
enc2 = tiktoken.get_encoding("gpt2")
enc2.encode("hello world")  # → [hello=31373, ' world'=995]

# 获取 GPT-4 tokenizer
enc4 = tiktoken.get_encoding("cl100k_base")
enc4.encode("hello world")
```

### 与手工实现的对应关系

OpenAI 在 `encoder.json` 和 `vocab.bpe` 中保存了 tokenizer，其对应关系为：

| OpenAI 命名 | 我们的命名 | 作用 |
|------------|-----------|------|
| `encoder` | `vocab` | int → bytes 映射 |
| `vocab_bpe` | `merges` | 合并规则 |

### tiktoken 的特点

- **仅有推理功能（inference only）**，无训练代码
- 效率极高（Rust 实现）
- 支持特殊 token 的注册与特殊处理
- 推荐在生产中使用

---

## 12. SentencePiece vs tiktoken

### 主要区别

| 特性 | tiktoken（OpenAI 方式） | SentencePiece（Meta/Google 方式） |
|------|------------------------|-----------------------------------|
| BPE 运行层次 | **字节级**（UTF-8 字节） | **Unicode 码点级** |
| 稀有字符处理 | 自然降级到单字节 | 需要 `byte_fallback` 选项 |
| 支持 | 仅推理 | 训练 + 推理 |
| 使用者 | GPT 系列 | LLaMA、Mistral 等 |
| 代码复杂度 | 相对简洁 | 历史包袱较重 |

### SentencePiece 训练示例

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="toy.txt",
    model_prefix="tok400",
    model_type="bpe",
    vocab_size=400,
    # 关闭不必要的规范化（LLM 更喜欢保留原始数据）
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    # 字节回退：将罕见 Unicode 码点转为 UTF-8 字节 token
    byte_fallback=True,
    # 其他
    unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
    add_dummy_prefix=True,   # 重要：见下
)
```

### SentencePiece 的特殊行为

**1. byte_fallback**

```python
# byte_fallback=True（推荐，LLaMA 2 使用）
# 遇到词表外的字符 → 用 UTF-8 字节 token 表示
# 效果：Korean chars → byte tokens

# byte_fallback=False（不推荐）
# 遇到词表外的字符 → <unk>（所有未知内容合并为一个 token）
```

**2. add_dummy_prefix**

```python
# 解决的问题："world" vs " world" 是不同 token
# 方案：在文本最前面加一个虚拟空格
# "hello" → " hello"（预处理后）
# 效果：句首单词和句中单词的 token 保持一致
```

**3. 词表结构**（SentencePiece 特有顺序）
```
[0] <unk>
[1] <s>  (BOS)
[2] </s> (EOS)
[3..258] 256 个字节 token（如果 byte_fallback=True）
[259..N-M] 合并 token（BPE 合并结果）
[N-M..N] 单独的 Unicode 码点 token
```

### Karpathy 对 SentencePiece 的评价

> "I personally find the tiktoken approach significantly cleaner... sentence piece has a lot of historical baggage... I'm not 100% sure why it switches whitespace into these bold underscore characters... the documentation unfortunately is not super amazing."

**建议**：如果要复现 LLaMA 2 的 tokenizer，需要仔细对照 Meta 发布的 proto 文件，逐个参数复制。

---

## 13. 词表大小（vocab_size）的选择

### vocab_size 影响模型的两个关键位置

```python
class GPT(nn.Module):
    def __init__(self, config):
        # 1. Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 2. LM Head（输出层）
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

### 增大 vocab_size 的收益

- 更长的字节序列被压缩为单个 token → **序列更短**
- 同等上下文长度下能"看到"更多文本
- GPT-4 词表是 GPT-2 的 2 倍，同样文本的 token 数减半

### 增大 vocab_size 的代价

| 代价 | 原因 |
|------|------|
| 计算量增加 | embedding 表更大，LM head 矩阵乘法更大 |
| 参数量增加 | 可能导致部分 token 训练不足 |
| 信息密度过高 | 单个 token 携带太多信息，Transformer 来不及处理 |

### 实际经验值

当前工业界的最佳实践：

```
词表大小 ≈ 高几万 到 约 10 万
GPT-2: ~50k
GPT-4: ~100k
LLaMA 2: 32k
```

这是一个**经验超参数**，需要通过实验权衡。

### 扩展词表（Fine-tuning 时的操作）

在基础模型上添加新 token（如 chat 模型的结构化 token）：

```python
# 步骤 1：扩展 embedding 表（添加新行）
model.token_embedding_table.weight = nn.Parameter(
    torch.cat([old_weights, new_random_rows])
)

# 步骤 2：扩展 LM head（添加新列）
model.lm_head.weight = nn.Parameter(
    torch.cat([old_lm_weights, new_lm_rows])
)

# 通常冻结原始参数，只训练新增参数
```

### 扩展应用：Gist Tokens

一篇论文提出了"Gist Token"技术：
- 将超长 prompt 压缩为几个特殊 token
- 冻结整个模型，只训练这几个 token 的 embedding
- 测试时用这些 token 替换原始 prompt，性能几乎不变
- 这是**参数高效微调（PEFT）**思路的一种体现

---

## 14. Tokenization 导致的 LLM 怪异行为

Karpathy 在讲座开头列举了一系列"LLM 怪异行为"，最后统一揭晓：**都是 Tokenization 的锅**。

### 14.1 为什么 LLM 不能拼写单词？

**原因**：某些单词是单个 token，模型看不到内部的字母结构。

**例子**：`defaultCellStyle` 在 GPT-4 中是**单个 token**

```
问：这个单词里有几个字母 'l'？
GPT-4 答：3 个（实际上是 4 个）
```

模型"看到"的是一个原子 token，无法分解为字母来计数。

### 14.2 为什么 LLM 不能反转字符串？

**例子**：直接要求反转 `defaultCellStyle`，GPT-4 给出错误结果。

**变通方法**：先让模型逐字符输出（强制分解），再反转

```
步骤 1：d e f a u l t C e l l S t y l e（✓ 能做对）
步骤 2：将上述字符反转（✓ 能做对）
```

原因：强制分解后，每个字符成为独立的 token，模型才能逐个处理。

### 14.3 为什么 LLM 对非英语语言更差？

**原因**：Tokenizer 训练集中英语数据远多于其他语言，导致：
- 英语单词有更多合并 → token 更长 → 序列更短
- 其他语言合并更少 → token 更碎 → 序列更长

```
"Hello, how are you?" (英语) → 5 tokens
"안녕하세요, 어떻게 지내세요?" (韩语) → 15+ tokens
```

**结果**：同等上下文窗口中，模型能"看到"的韩语文本量远少于英语，相当于被截断。

### 14.4 为什么 LLM 做简单算术也会出错？

**原因**：数字的分词方式完全随机，没有规律。

```
127   → 单个 token
677   → 两个 token（'6' + '77'）
1275  → 两个 token（'12' + '75'）
6773  → 两个 token（' 6' + '773'）
```

人类做加法是从个位开始，逐位进位，需要访问**固定位置**的数字。但 token 化后，数字的位置是任意的，模型很难学到正确的对位规则。

**LLaMA 2 的应对**：在 SentencePiece 中设置 `split_by_digits=True`，确保每个数字单独成 token，至少保证对齐。

### 14.5 为什么 GPT-2 写 Python 代码特别差？

```python
# Python 缩进示例
def foo():
    if x:
        return y
```

在 GPT-2 tokenizer 中，**每个空格是独立的 token（token ID = 220）**：

```
4个空格 → [220, 220, 220, 220]（4 个 token）
```

这导致：
- Python 代码中大量缩进空格占满了上下文窗口
- 模型能看到的代码行数急剧减少

**GPT-4 的改进**：将多个连续空格合并为单个 token：
```
4个空格 → 单个 token
```

GPT-4 写 Python 的能力提升，很大一部分来自 Tokenizer 的改进，而非模型本身。

### 14.6 为什么输入 `<|endoftext|>` 会让 LLM 行为异常？

**原因**：特殊 token 绕过了正常的 BPE 编码路径，直接插入对应的 token ID。

如果前端代码错误地将用户输入中的 `<|endoftext|>` 当作真正的特殊 token 来处理（而不是普通文本），模型会认为文档结束，产生奇怪行为。

> 这是一个**安全漏洞**！用户可以通过特殊 token 字符串影响模型行为。

### 14.7 为什么多余的尾随空格会导致性能下降？

**场景**：在 OpenAI Playground 中，在提示末尾加一个空格，系统会警告：

> "Your text ends in a trailing space, which causes worse performance due to how the API splits text into tokens."

**原因分析**：

```
正常：... ice cream shop [next token = ' Oh...']
```

GPT-2 中，空格是**下一个 token 的前缀**，例如：
- ` O` 是 token 8840（空格 + O 合并为一个 token）

如果用户在提示末尾加了空格：
```
... ice cream shop [空格 token 220] [下一个 token ???]
```

现在空格被单独抽出，作为提示的最后一个 token，而模型训练时从未见过这种模式（训练数据中空格总是与后面的词合并）。这是一个**分布外（out-of-distribution）**的输入，会导致不可预测的输出。

### 14.8 SolidGoldMagikarp 事件

这是最著名的 Tokenization 怪异行为案例。

**现象**：问 GPT：`请重复以下字符串：SolidGoldMagikarp`，得到了：
- 回避（"我不能回答"）
- 幻觉（谈论完全无关的话题）
- 侮辱用户（！）

**根本原因**：

```
分词训练集         ↔          LLM 训练集
包含 Reddit 数据    ≠          不包含同一份 Reddit 数据
```

1. `SolidGoldMagikarp` 是一个 Reddit 用户，在分词训练集中出现频率极高
2. BPE 将其合并为**单个 token**，占据了词表中的一个位置
3. 但 LLM 训练集中并不包含这个 Reddit 用户的相关数据
4. 因此该 token 的 embedding 从未在 LLM 训练中更新，停留在随机初始化状态
5. 使用该 token = **访问未初始化的内存** → 完全不可预测的行为

> Karpathy：这类 token 就像 C 程序中的"未分配内存"，读取时会有未定义行为（undefined behavior）。

### 14.9 为什么 YAML 比 JSON 更适合 LLM？

**原因**：相同的结构化数据，YAML 使用的 token 更少

```
同一份数据：
JSON 格式 → 116 tokens
YAML 格式 → 99 tokens
```

在"token 经济"中（按 token 付费 + 上下文长度有限），应优先使用 token 更少的格式。

---

## 15. minBPE 代码库总览

Karpathy 为本讲配套发布了 [minBPE](https://github.com/karpathy/minbpe) 代码库，实现了完整的 BPE tokenizer。

### 代码结构

```
minbpe/
├── base.py          # 基类：Tokenizer, get_stats, merge
├── basic.py         # BasicTokenizer（无正则分割）
├── regex.py         # RegexTokenizer（GPT-4 风格）
train.py             # 训练脚本
exercise.md          # 练习题（4个步骤）
```

### BasicTokenizer（最简版）

```python
class BasicTokenizer(Tokenizer):
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
```

### RegexTokenizer（GPT-4 风格）

额外特性：
1. **正则预分割**：先用 GPT-4 pattern 分割文本，再分别对每块做 BPE
2. **特殊 token 支持**：`register_special_tokens()` + `encode(text, allowed_special="all")`

```python
class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        # 1. 用正则切分
        text_chunks = re.findall(self.compiled_pattern, text)
        # 2. 每块独立编码为字节列表
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        # 3. 跨块统计（但不允许跨块合并）
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)  # 累积统计
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            ...
```

### 保存和加载

```python
# 保存两个文件
tokenizer.save("my_tokenizer")
# → my_tokenizer.model（用于 load()，包含 pattern + merges）
# → my_tokenizer.vocab（人类可读，用于检查）

# 加载
tokenizer.load("my_tokenizer.model")
```

### 复现 GPT-4 Tokenizer

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# 验证：两者应该输出相同结果
ids = enc.encode("hello world")
text = enc.decode(ids)
assert text == "hello world"
```

---

## 16. 本讲总结

### 核心概念速查表

| 概念 | 说明 |
|------|------|
| Token | LLM 处理的基本单元，从原始字节通过 BPE 得到 |
| BPE | 迭代合并最频繁字节对，逐步扩大词表、压缩序列 |
| `merges` 字典 | BPE 的"参数"，记录所有合并规则 |
| `vocab` 字典 | token ID → bytes 的映射，用于 decode |
| 正则预分割 | 防止跨语义边界的不当合并 |
| 特殊 token | 人工插入，标注文档/对话结构 |
| byte_fallback | SentencePiece 处理罕见字符的方式 |
| add_dummy_prefix | 统一句首/句中相同词的 token 表示 |

### 三种 Tokenizer 对比

| | BasicTokenizer | RegexTokenizer（tiktoken 风格） | SentencePiece |
|--|----------------|--------------------------------|---------------|
| BPE 层次 | 字节 | 字节 | Unicode 码点 |
| 预分割 | 无 | 有（正则） | 有（内置规则） |
| 特殊 token | 无 | 有 | 有（内置 unk/bos/eos） |
| 训练功能 | 有 | 有 | 有 |
| 推理功能 | 有 | 有 | 有 |
| 推荐场景 | 学习理解 | 生产部署 | LLaMA/Mistral 复现 |

### Karpathy 的最终建议

> - **如果可以复用 GPT-4 的词表**：直接用 `tiktoken`（高效、无训练开销）
> - **如果需要训练自己的词表**：可以用 `sentencepiece` 或等待 `minBPE` 的 Rust 实现
> - **终极目标**：有一天彻底去掉 Tokenization 这一阶段（"eternal glory goes to anyone who can get rid of it"）

### 关键 Takeaways

1. **Tokenization 是独立的预处理阶段**，有自己的训练集和训练流程
2. **词表大小是超参数**，约 30k–100k 是当前最佳实践
3. **很多 LLM "玄学问题" 其实根源是 Tokenization**：拼写、算术、多语言、代码等
4. **正则预分割很重要**：防止跨类别的不当合并
5. **特殊 token 需要"模型手术"**：扩展 embedding 表和 LM head
6. **SolidGoldMagikarp 警示**：tokenizer 训练集和 LLM 训练集若不一致，会产生"幽灵 token"

---

## 附：关键代码速查

### BPE 核心函数

```python
def get_stats(ids, counts=None):
    """统计相邻对出现频率"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """将 ids 中所有 pair 替换为 idx"""
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

### 训练循环

```python
vocab_size = 1024
num_merges = vocab_size - 256

ids = list(text.encode("utf-8"))
merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
```

### Decode

```python
def decode(ids):
    text_bytes = b"".join(vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")
```

### Encode

```python
def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, merges[pair], merges[pair])  # 等价写法
    return ids
```

### 使用 tiktoken（推理）

```python
import tiktoken

# GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
text = enc.decode(tokens)
```

### 使用 minBPE（训练 + 推理）

```python
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train("your training text here", vocab_size=512)
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)
tokenizer.save("my_tokenizer")
```
