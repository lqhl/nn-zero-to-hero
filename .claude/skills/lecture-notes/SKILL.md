---
name: lecture-notes
description: Generate a detailed Chinese markdown lecture handout for a nn-zero-to-hero lecture. Use when the user asks to create notes, handout, or 讲义 for a specific lecture. Combines the YouTube transcript and Jupyter notebook to produce comprehensive study material.
---

# Lecture Notes Generator

Generate a detailed Chinese markdown lecture handout (讲义) for a nn-zero-to-hero lecture by synthesizing the YouTube video transcript and the corresponding Jupyter notebook.

## Trigger

Use this skill when the user says things like:
- "为第 N 讲写讲义"
- "根据 lecture N 写中文讲义"
- "make notes for lecture N"

## Inputs

The user provides a lecture number or name. Everything else is derived from `README.md`.

## Workflow

### Step 1 — Look up the lecture in README.md

Read `README.md` to find:
1. The **YouTube URL** for the requested lecture
2. The **Jupyter notebook path** under `lectures/`

Do both reads in parallel.

### Step 2 — Fetch sources in parallel

Run these two operations **simultaneously**:

1. **Get the YouTube transcript** using the `youtube-transcript` skill:
   ```
   /youtube-transcript <YouTube URL>
   ```
   Save the full transcript to `/tmp/<video_id>_transcript.txt` for reference.

2. **Read the Jupyter notebook** using the `Read` tool on the `.ipynb` file path found in Step 1.

### Step 3 — Read the full transcript

The transcript may be long (2000–3000+ lines). Read it in multiple chunks:
- First chunk: lines 1–500 (to understand the lecture arc)
- Continue in 700–800 line batches until the end

Key things to extract from the transcript:
- The main topics and their order
- Karpathy's intuitive explanations and analogies
- Specific numbers, formulas, and thresholds he mentions
- "Gotchas", bugs, and practical advice he emphasizes
- Any live experiments or comparisons he does on screen

### Step 4 — Write the Chinese handout

Write the handout to `lectures/notes/lecture<N>_notes_zh.md`.

#### Structure requirements

The handout MUST follow this structure:

```markdown
# Lecture N：<中文标题>

> 对应视频：[YouTube](<url>)
> 对应 Notebook：[<notebook_filename>.ipynb](../makemore/<notebook_filename>.ipynb)

---

## 目录
[numbered list of all sections]

---

## 1. 本讲目标与背景
[Why this lecture exists, what problem it solves, connection to previous/next lectures]

## 2–N. [One section per major topic]
[Detailed explanations — see Content Requirements below]

## 最后一节. 本讲总结
[Summary table + key takeaways]

---

## 附：关键代码速查
[Short, copy-pasteable code snippets for the most important patterns]
```

#### Content requirements

For **each major topic**, include:

1. **现象 / 问题**：What problem or observation motivates this topic
2. **根本原因**：The underlying reason (with math where Karpathy explains it)
3. **修复方法 / 解决方案**：The concrete fix, with code
4. **效果**：Quantitative results (loss values, percentages) when Karpathy shows them
5. **Karpathy 的话**：Include at least one direct insight or warning as a blockquote when he says something memorable

Use **tables** to compare options (e.g., different activation functions, different gains).

Use **inline code** for tensor shapes, variable names, and short expressions.

Use **math blocks** ($$...$$) for key formulas.

#### Tone and style

- Written for a Chinese-reading learner who is following along with the course
- Explain *why*, not just *what*
- Include Karpathy's intuitive analogies (even the colorful ones like "永久脑损伤")
- Preserve the progression: show how each fix builds on the previous one
- Do NOT paraphrase away important technical detail from the transcript

#### Length

A complete handout should be **800–1500 lines** of markdown. If it's shorter, you have likely missed important content from the transcript.

## Output location

`lectures/notes/lecture<N>_notes_zh.md`

The notebook link in the header must use markdown link format with a relative path from `lectures/notes/`:
- makemore notebooks: `[<filename>.ipynb](../makemore/<filename>.ipynb)`
- micrograd notebooks: `[<filename>.ipynb](../micrograd/<filename>.ipynb)`

## Notes

- The transcript contains all of Karpathy's explanations, including things not visible in the notebook (e.g., his reasoning, dead ends, comparisons)
- The notebook contains the final cleaned-up code with output values
- Both sources are needed — the transcript for depth, the notebook for accuracy of code and numbers
- If the transcript is hard to parse (auto-generated captions have no punctuation), focus on the semantic content rather than exact wording
