export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212

export WANDB_MODE=disabled
export TRAIN_DATASET="dclm-tokenize"


export SSD_DATASET="/home/kangxinlai/PycharmProjects/MOPE/dclm-tokenize"
export SSD_WEIGHTS="weight"


# 创建权重保存目录
mkdir -p "$SSD_WEIGHTS"

# 设置基础环境变量（可选）
export OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

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
      # 设置otep
    --moe-router-load-balancing-type otep
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-otep-step 5
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

    # froze 配置
    --last-trainable-layers 2
)

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
    --micro-batch-size 4                # 保持不变
    --global-batch-size 16               # 增加全局 batch

    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style WSD
    --lr-warmup-fraction 0.01

    --lr-wsd-decay-iters $((TRAIN_ITERS / 10))
    --train-iters $((TRAIN_ITERS))
)

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
#    --load "weight_initial"
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
