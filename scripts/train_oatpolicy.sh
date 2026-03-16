#!/bin/bash

set -euo pipefail

NUM_NODES=${1:-1}
NUM_GPUS=${2:-1}
TOKENIZER_CKPT=${3:-}
LAZY_EVAL=${4:-true}
ALLOW_BF16=${5:-false}

if [ -z "$TOKENIZER_CKPT" ]; then
    echo "Usage: $0 [num_nodes=1] [num_gpus=1] <path/to/oattok.ckpt> [lazy_eval=true] [allow_bf16=false]"
    exit 1
fi

MULTI_GPU_FLAG=""
if [ "$NUM_GPUS" -gt 1 ]; then
    MULTI_GPU_FLAG="--multi_gpu"
fi

HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines "$NUM_NODES" \
    $MULTI_GPU_FLAG \
    --num_processes "$NUM_GPUS" \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    "task.policy.lazy_eval=$LAZY_EVAL" \
    "policy.action_tokenizer.checkpoint=$TOKENIZER_CKPT" \
    "training.allow_bf16=$ALLOW_BF16"