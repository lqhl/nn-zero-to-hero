#!/bin/bash

# 脚本：将 lectures/notes/ 中的中文笔记合并成一本 epub 书
# 输出：nn-zero-to-hero-notes.epub（放在根目录）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTES_DIR="$PROJECT_ROOT/lectures/notes"
METADATA_FILE="$NOTES_DIR/book_metadata.md"
OUTPUT_FILE="$PROJECT_ROOT/nn-zero-to-hero-notes.epub"

# 检查 pandoc 是否安装
if ! command -v pandoc &> /dev/null; then
    echo "错误：pandoc 未安装，请先安装 pandoc"
    echo "  macOS: brew install pandoc"
    echo "  Ubuntu/Debian: sudo apt-get install pandoc"
    exit 1
fi

# 检查元数据文件是否存在
if [ ! -f "$METADATA_FILE" ]; then
    echo "错误：元数据文件不存在: $METADATA_FILE"
    exit 1
fi

# 创建临时目录
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# 创建合并后的 markdown 文件
MERGED_FILE="$TMP_DIR/merged.md"

# 首先添加元数据文件（包含 YAML 前置数据和前言）
cat "$METADATA_FILE" > "$MERGED_FILE"
echo -e "\n\n---\n" >> "$MERGED_FILE"

# 按顺序合并所有笔记文件
for i in {1..8}; do
    NOTE_FILE="$NOTES_DIR/lecture${i}_notes_zh.md"
    if [ -f "$NOTE_FILE" ]; then
        echo "正在处理: lecture${i}_notes_zh.md"
        cat "$NOTE_FILE" >> "$MERGED_FILE"
        echo -e "\n\n---\n" >> "$MERGED_FILE"
    else
        echo "警告: $NOTE_FILE 不存在，跳过"
    fi
done

# 使用 pandoc 生成 epub
echo "正在生成 epub 文件..."
pandoc "$MERGED_FILE" \
    -f markdown+tex_math_dollars \
    -t epub \
    -o "$OUTPUT_FILE" \
    --toc \
    --toc-depth=2 \
    --mathml 2>/dev/null || pandoc "$MERGED_FILE" -f markdown -t epub -o "$OUTPUT_FILE" --toc --toc-depth=2

echo "✅ 生成成功: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
