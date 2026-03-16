#!/bin/bash

set -euo pipefail

CODE_PATH="/mlp_vepfs/share/junbo/oat"
DEBUG=false
NUM_NODES=1
NUM_GPUS=8
QUEUE='模型算法部-算法预研-H20'
ALLOW_BF16=true
TASK_TOKENIZER="libero/libero10"
NUM_DEMO=500
CONFIG_PATH="./scripts/oat-volc/config_train_tokenizer.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --allow_bf16) ALLOW_BF16="$2"; shift 2 ;;
        --task_tokenizer) TASK_TOKENIZER="$2"; shift 2 ;;
        --num_demo) NUM_DEMO="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --num_nodes) NUM_NODES="$2"; shift 2 ;;
        --code_path) CODE_PATH="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
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


TASK_TAG="$(basename "$TASK_TOKENIZER")"
JOB_NAME="train_oattok_${TASK_TAG}_N${NUM_DEMO}_$(date +%m%d_%H%M)"

ENTRYPOINT_CMD=$(printf 'cd %q && bash %q --task train --task_tokenizer %q --num_demo %q --num_gpus %q --allow_bf16 %q --code_path %q' \
    "$CODE_PATH" \
    "scripts/oat-volc/entropy_tokenizer.sh" \
    "$TASK_TOKENIZER" \
    "$NUM_DEMO" \
    "$NUM_GPUS" \
    "$ALLOW_BF16" \
    "$CODE_PATH")

if [ "$DEBUG" = true ]; then
    volc ml_task submit \
        --conf "$CONFIG_PATH" \
        -n "debug_train_oattok_$(date +%m%d)" \
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