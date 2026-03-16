#!/bin/bash

NUM_NODES=${1:-1}
NUM_GPUS=${2:-1}
ALLOW_BF16=${3:-true}

MULTI_GPU_FLAG=""
if [ "$NUM_GPUS" -gt 1 ]; then
    MULTI_GPU_FLAG="--multi_gpu"
fi

HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines "$NUM_NODES" \
    $MULTI_GPU_FLAG \
    --num_processes "$NUM_GPUS" \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.allow_bf16="$ALLOW_BF16"
