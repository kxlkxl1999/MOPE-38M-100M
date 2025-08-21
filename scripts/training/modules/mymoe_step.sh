#!/bin/bash

# 设置本地数据路径和权重保存路径
export SSD_DATASET="/home/kangxinlai/PycharmProjects/MOPE/dclm-tokenize"
export SSD_WEIGHTS="weight"

# 创建权重保存目录
mkdir -p "$SSD_WEIGHTS"

# 设置基础环境变量（可选）
export OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# 禁用 WandB
export WANDB_MODE=disabled

# 加载模型和训练配置参数（你可以在这两个脚本里定义 MODEL_ARGS, TRAIN_ARGS 等）
source configs/model/mymoe.sh
source configs/train/mymoe.sh


#DATA_PATH=$(find "$SSD_DATASET" -type f -name '*.bin' \
#  | sort -V \
#  | sed 's/\.bin$//' \
#  | awk '{printf "1.0 %s ", $0}')
#
## 不要加双引号，允许展开多个参数
#DATA_ARGS=(
#  --seq-length 2048
#  --data-path ${DATA_PATH}
#  --split 90,5,5
#)
DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 90,5,5
)

# 设置保存相关参数（去除 wandb 参数）
SAVE_ARGS=(
    --log-interval 5
    --log-throughput
    --save "$SSD_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    --load "$SSD_WEIGHTS"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_WEIGHTS"
)

export MASTER_ADDR=localhost
export MASTER_PORT=6000
export RANK=0
export WORLD_SIZE=1

# 启动训练（单卡，无 torchrun）
cd Megatron-LM
python pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"
