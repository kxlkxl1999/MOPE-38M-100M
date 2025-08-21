#!/bin/bash
# Training configuration for local single-GPU run with FLAME MoE.

# ⚠️ 已完全移除分布式/Slurm/Torchrun 参数
TORCH_ARGS=()  # 保留空列表供主脚本兼容

INFRA_ARGS=(
    # 禁用分布式优化器/通信
    # 注释掉以下参数，因为本地训练用不到：
    # --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    # --moe-token-dispatcher-type alltoall
    # --distributed-timeout-minutes 30

    --bf16  # 或改为 --fp16，如果你的 GPU 不支持 BF16
)

TRAIN_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 4         # 根据内存和 batch size 调整
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style WSD
    --lr-warmup-fraction 0.01
    --lr-wsd-decay-iters $((TRAIN_ITERS / 10))
    --train-iters $TRAIN_ITERS
)
