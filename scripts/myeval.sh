#!/bin/bash

# 本地配置
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# 模型路径
SSD_WEIGHTS=weight
JOBID=localtest
ITER=2121  # 可选，写当前checkpoint标识符

export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212

export USE_LOCAL_DATA=1
export LOCAL_DATA_DIR="$(pwd)/../data"
unset HF_DATASETS_TRUST_REMOTE_CODE

# 创建日志目录
mkdir -p logs/evaluate/$JOBID
echo $ITER > $SSD_WEIGHTS/latest_checkpointed_iteration.txt

# 切换到 lm-evaluation-harness 根目录
cd lm-evaluation-harness


 few-shot 设置与任务定义
num_fewshots=(0 10)
fewshot_tasks=(
    "openbookqa,winogrande"
    "hellaswag"
)

#num_fewshots=(10)
#fewshot_tasks=(
#    "hellaswag"
#)

# 模型参数定义（根据你使用的 Megatron 模型结构配置）
source /home/kangxinlai/PycharmProjects/MOPE/configs/model/mymoe.sh

#python - <<EOF
#from datasets import load_dataset
#tasks = [
#  ("openbookqa", None),
#  ("winogrande", "winogrande_xl"),
##  ("allenai/ai2_arc", "ARC-Easy"),
#  ("allenai/ai2_arc", "ARC-Challenge"),
#  ("hellaswag", None)
#]
#for task, cfg in tasks:
#    ds = load_dataset(task, cfg) if cfg else load_dataset(task)
#    ds.save_to_disk(f"data/{task}_local")
#    print(f"✅ Cached {task}_local")
#EOF

# 逐个任务评估
for i in "${!num_fewshots[@]}"; do
    echo "Evaluating ${num_fewshots[$i]} shot for ${fewshot_tasks[$i]}"

    CKPT=$ITER PYTHONPATH=$PWD/../Megatron-LM CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((12345 + i)) -m lm_eval \
        "${MODEL_ARGS[@]}" \
        --bf16 \
        --seq-length 2048 \
        --micro-batch-size 4 \
        --num_fewshot ${num_fewshots[$i]} \
        --batch_size 2 \
        --max-tokens-to-oom 10000000 \
        --seed 42 \
        --load ../weight \
        --model megatron_lm \
        --tasks "${fewshot_tasks[$i]}" \
        --output_path ../results/flame-moe/$JOBID \
        > ../logs/evaluate/$JOBID/${num_fewshots[$i]}shot-${i}.log 2>&1

    # 等待前一个任务完成（串行执行）
    wait
done