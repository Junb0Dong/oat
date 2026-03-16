#!/bin/bash

set -euo pipefail

CODE_PATH="/mlp_vepfs/share/junbo/oat"
DEBUG=false
TASK="eval"
CHECKPOINT=""
OUTPUT_DIR="output/eval/libero10"
NUM_EXP=5

# 网络配置
# huggingface 镜像
export HF_ENDPOINT=https://hf-mirror.com
# 科学上网配置
export http_proxy=http://proxy-hs.dexmal-inc.com:3128
export https_proxy=http://proxy-hs.dexmal-inc.com:3128
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,100.0.0.0/8,ivolces.com"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint|--ckp) CHECKPOINT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_exp) NUM_EXP="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --code_path) CODE_PATH="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ "$TASK" = "eval" ] && [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 --task eval --checkpoint <path/to/oatpolicy.ckpt> [--output_dir output/eval/libero10] [--num_exp 5] [--code_path /mlp_vepfs/share/junbo/oat] [--debug]"
    exit 1
fi

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

if [ "$TASK" = "eval" ]; then
    echo "Running eval_policy_sim.py"
    export WANDB_API_KEY="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"
    # Always create a minute-stamped eval output directory to avoid overwrite prompt.
    OUTPUT_DIR="${OUTPUT_DIR%/}_$(date +%Y%m%d_%H%M)"
    echo "Resolved output_dir: $OUTPUT_DIR"
    if command -v uv >/dev/null 2>&1; then
        uv run scripts/eval_policy_sim.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$OUTPUT_DIR" \
            --num_exp "$NUM_EXP"
    else
        "$PYTHON_BIN" scripts/eval_policy_sim.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$OUTPUT_DIR" \
            --num_exp "$NUM_EXP"
    fi
else
    echo "Unsupported task: $TASK"
    echo "Only --task eval is supported in this script."
    exit 1
fi