#!/bin/bash

set -euo pipefail

CODE_PATH="/mlp_vepfs/share/junbo/oat"
DEBUG=false
TASK="train"
TASK_TOKENIZER="libero/libero10"
NUM_DEMO=500
NUM_GPUS=8
ALLOW_BF16=true

# 网络配置
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://proxy-hs.dexmal-inc.com:3128
export https_proxy=http://proxy-hs.dexmal-inc.com:3128
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,100.0.0.0/8,ivolces.com"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task_tokenizer) TASK_TOKENIZER="$2"; shift 2 ;;
        --num_demo) NUM_DEMO="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --allow_bf16) ALLOW_BF16="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --code_path) CODE_PATH="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

cd "$CODE_PATH"

if [ "$DEBUG" = true ]; then
    echo "Debug mode: sleeping for 10 days"
    sleep 10d
    exit 0
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python not found in PATH."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, trying to install uv with pip..."
    "$PYTHON_BIN" -m pip install --user -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple || true
    export PATH="$HOME/.local/bin:$PATH"
fi

if command -v uv >/dev/null 2>&1; then
    uv sync
    uv pip install -e .
else
    echo "uv is still unavailable, falling back to pip/python."
    "$PYTHON_BIN" -m pip install -e .
fi

TEMP_DIR="temp_$(date +%s)_$(openssl rand -hex 4)" && \
mkdir "$TEMP_DIR" && \
cd "$TEMP_DIR" && \
wget https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/python_sdk_installer/volcengine_ml_platform-1.1.13-py3-none-any.whl && \
pip install volcengine_ml_platform-1.1.13-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple && \
cd .. && \
rm -rf "$TEMP_DIR"

if [ "$TASK" = "train" ]; then
    echo "Running train_oattok"
    export WANDB_API_KEY="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"

    NNODES="${MLP_WORKER_NUM:-1}"
    NODE_RANK="${MLP_ROLE_INDEX:-0}"
    MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
    PORT="${MLP_WORKER_0_PORT:-29500}"

    ACCEL_ARGS=(
        --num_machines "$NNODES"
        --machine_rank "$NODE_RANK"
        --main_process_ip "$MASTER_ADDR"
        --main_process_port "$PORT"
        --num_processes "$NUM_GPUS"
    )

    if [ "$NUM_GPUS" -gt 1 ]; then
        ACCEL_ARGS+=(--multi_gpu)
    fi

    TRAIN_ARGS=(
        scripts/run_workspace.py
        --config-name=train_oattok
        "task/tokenizer=$TASK_TOKENIZER"
        "training.num_demo=$NUM_DEMO"
        "training.allow_bf16=$ALLOW_BF16"
    )

    if command -v uv >/dev/null 2>&1; then
        HYDRA_FULL_ERROR=1 uv run accelerate launch "${ACCEL_ARGS[@]}" "${TRAIN_ARGS[@]}"
    else
        HYDRA_FULL_ERROR=1 "$PYTHON_BIN" -m accelerate.commands.launch "${ACCEL_ARGS[@]}" "${TRAIN_ARGS[@]}"
    fi
else
    echo "Unsupported task: $TASK"
    echo "Only --task train is supported in this script."
    exit 1
fi