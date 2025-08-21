#!/bin/bash

# 目录路径
INPUT_DIR="dclm-jsonl"
OUTPUT_DIR="dclm-tokenize"
MEGATRON_DIR="Megatron-LM"
WORKERS=4

TOKENIZER_MODEL="EleutherAI/pythia-12b"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.jsonl; do
    filename=$(basename "$file" .jsonl)
    output_prefix="$OUTPUT_DIR/$filename"

    if [[ -f "${output_prefix}_text_document.bin" ]]; then
        echo "已处理过: $file，跳过"
        continue
    fi

    echo "正在切词: $file"

    for i in {1..3}; do
        python "$MEGATRON_DIR/tools/preprocess_data.py" \
            --input "$file" \
            --output-prefix "$output_prefix" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            --append-eod \
            --workers "$WORKERS" && break

        echo "第 $i 次切词失败，重试中..."
        sleep 5
        if [ $i -eq 3 ]; then
            echo "❗ 三次失败，跳过 $file" >&2
        fi
    done
done

echo "所有 Pythia 分词任务完成"
