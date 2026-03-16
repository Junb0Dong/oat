#!/bin/bash

set -euo pipefail

CODE_PATH="/mlp_vepfs/share/junbo/oat"
DEBUG=false
NUM_NODES=1
QUEUE='模型算法部-算法预研-H20'
CHECKPOINT=""
OUTPUT_DIR="/mlp_vepfs/share/junbo/oat/output/eval/libero10"
NUM_EXP=5
CONFIG_PATH="./scripts/oat-volc/config.yaml"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint|--ckp) CHECKPOINT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_exp) NUM_EXP="$2"; shift 2 ;;
        --code_path) CODE_PATH="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        --num_nodes) NUM_NODES="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 --checkpoint <path/to/oatpolicy.ckpt> [--output_dir output/eval/libero10] [--num_exp 5] [--code_path /mlp_vepfs/share/junbo/oat] [--num_nodes 1] [--queue 模型算法部-DB-H20] [--config ./scripts/oat-volc/config.yaml] [--debug]"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -e "$CHECKPOINT" ]; then
    echo "Warning: checkpoint path does not exist locally: $CHECKPOINT"
fi

set -x

cd "$CODE_PATH"

if [ "$NUM_NODES" -gt 1 ]; then
    FLAVOR="ml.hpcpni3ln.45xlarge"
else
    FLAVOR="ml.pni3ln.4xlarge"
fi

CKPT_TAG="$(basename "${CHECKPOINT%.*}")"
JOB_NAME="eval_${CKPT_TAG}_$(date +%m%d)_exp${NUM_EXP}"

ENTRYPOINT_CMD=$(printf 'cd %q && bash %q --task eval --checkpoint %q --output_dir %q --num_exp %q --code_path %q' \
    "$CODE_PATH" \
    "scripts/oat-volc/entropy.sh" \
    "$CHECKPOINT" \
    "$OUTPUT_DIR" \
    "$NUM_EXP" \
    "$CODE_PATH")

if [ "$DEBUG" = true ]; then
    volc ml_task submit \
        --conf "$CONFIG_PATH" \
        -n "debug_eval_$(date +%m%d)" \
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