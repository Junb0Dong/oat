#!/bin/bash

set -euo pipefail

CODE_PATH="/mlp_vepfs/share/junbo/oat"
DEBUG=false
NUM_NODES=1
NUM_GPUS=4
QUEUE='模型算法部-算法预研-H20'
TOKENIZER_CKPT=""
LAZY_EVAL=true
ALLOW_BF16=false
TASK_POLICY="libero/libero10"
CONFIG_PATH="./scripts/oat-volc/config_train_policy.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tokenizer_ckpt|--tokenizer-ckpt) TOKENIZER_CKPT="$2"; shift 2 ;;
        --lazy_eval) LAZY_EVAL="$2"; shift 2 ;;
        --allow_bf16) ALLOW_BF16="$2"; shift 2 ;;
        --task_policy) TASK_POLICY="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --num_nodes) NUM_NODES="$2"; shift 2 ;;
        --code_path) CODE_PATH="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -z "$TOKENIZER_CKPT" ]; then
    echo "Usage: $0 --tokenizer_ckpt <path/to/oattok.ckpt> [--task_policy libero/libero10] [--num_nodes 1] [--num_gpus 8] [--lazy_eval true] [--allow_bf16 false] [--code_path /mlp_vepfs/share/junbo/oat] [--queue 模型算法部-算法预研-H20] [--config ./scripts/oat-volc/config_train_policy.yaml] [--debug]"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -e "$TOKENIZER_CKPT" ]; then
    echo "Warning: tokenizer checkpoint path does not exist locally: $TOKENIZER_CKPT"
fi

set -x

cd "$CODE_PATH"

# 根据GPU数量选择对应的实例规格
if [ "$NUM_GPUS" -eq 4 ]; then
    FLAVOR="ml.pni3ln.17xlarge"
elif [ "$NUM_GPUS" -eq 1 ]; then
    FLAVOR="ml.pni3ln.4xlarge"
else
    FLAVOR="ml.pni3ln.45xlarge"  # 默认8卡规格
    echo "警告：GPU数量$NUM_GPUS不匹配4或1，使用默认规格$FLAVOR"
fi


TASK_TAG="$(basename "$TASK_POLICY")"
TOK_TAG="$(basename "${TOKENIZER_CKPT%.*}")"
JOB_NAME="train_oatpolicy_${TASK_TAG}_${TOK_TAG}_$(date +%m%d_%H%M)"

ENTRYPOINT_CMD=$(printf 'cd %q && bash %q --task train --tokenizer_ckpt %q --task_policy %q --num_gpus %q --lazy_eval %q --allow_bf16 %q --code_path %q' \
    "$CODE_PATH" \
    "scripts/oat-volc/entropy_policy.sh" \
    "$TOKENIZER_CKPT" \
    "$TASK_POLICY" \
    "$NUM_GPUS" \
    "$LAZY_EVAL" \
    "$ALLOW_BF16" \
    "$CODE_PATH")

if [ "$DEBUG" = true ]; then
    volc ml_task submit \
        --conf "$CONFIG_PATH" \
        -n "debug_train_oatpolicy_$(date +%m%d)" \
        -e "$ENTRYPOINT_CMD --debug" \
        --set TaskRoleSpecs[0].RoleReplicas=$NUM_NODES \
        --set TaskRoleSpecs[0].Flavor="$FLAVOR" \
        --set TaskRoleSpecs[0].RoleName="worker" \
        --resource_queue_name "$QUEUE"
else
    volc ml_task submit \
        --conf "$CONFIG_PATH" \
        -n "$JOB_NAME" \
        -e "$ENTRYPOINT_CMD" \
        --set TaskRoleSpecs[0].RoleReplicas=$NUM_NODES \
        --set TaskRoleSpecs[0].Flavor="$FLAVOR" \
        --set TaskRoleSpecs[0].RoleName="worker" \
        --resource_queue_name "$QUEUE"
fi