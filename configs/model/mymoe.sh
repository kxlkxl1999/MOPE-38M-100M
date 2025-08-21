#!/bin/bash
# Model configuration for local single-GPU training with MoE.

MODEL_ARGS=(
    # 网络规模
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-layers $NUM_LAYERS
    --num-attention-heads 16  # 单卡训练建议调小
    --swiglu
    --max-position-embeddings 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear

    # MoE 配置
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
    --num-experts 64                    # 单卡建议设置较小
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size $((2 * MOE_FFN_HIDDEN_SIZE))
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff 0.01
    --moe-z-loss-coeff 0.001

    # 正则
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # 初始化
    --init-method-std 0.02

    # tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b
)
