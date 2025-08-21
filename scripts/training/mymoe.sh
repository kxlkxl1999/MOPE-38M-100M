
export WANDB_MODE=disabled
export TRAIN_DATASET="dclm-tokenize"
export TRAIN_WEIGHTS="weight"

bash scripts/training/modules/mymoe_step.sh
