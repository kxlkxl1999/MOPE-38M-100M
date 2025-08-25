#!/bin/bash

# 建议先登录并同意各模型的 License
huggingface-cli login

# 例1：Llama-3.1-8B vs Qwen3-8B-Base
python mdir_haar_falsification.py \
  --pairs meta-llama/Llama-3.1-8B,Qwen/Qwen3-8B-Base \
  --max-common 20000 --subset 12000 --permutes 50 \
  --cache-dir ~/.cache/hf_mdir --hf-token $HF_TOKEN

# 例2：DeepSeek-V3-Base vs Kimi-K2-Instruct
python mdir_haar_falsification.py \
  --pairs deepseek-ai/DeepSeek-V3-Base,moonshotai/Kimi-K2-Instruct \
  --max-common 20000 --subset 12000 --permutes 50 \
  --cache-dir ~/.cache/hf_mdir --hf-token $HF_TOKEN
