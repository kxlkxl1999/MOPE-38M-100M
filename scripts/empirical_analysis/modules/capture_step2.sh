#!/bin/bash
# Launch the evaluation, which traces the top-k router.

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

source configs/model/flame-moe.sh
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,0,5
)

SAVE_ARGS=(
    --test-mode
    --skip-train
    --load $SSD_WEIGHTS
    --eval-iters 25
)

cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"
