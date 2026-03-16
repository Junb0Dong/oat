#!/bin/bash

set -euo pipefail

CHECKPOINT=${1:-}
OUTPUT_DIR=${2:-output/eval/libero10}
NUM_EXP=${3:-5}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <path/to/oatpolicy.ckpt> [output_dir=output/eval/libero10] [num_exp=5]"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv run scripts/eval_policy_sim.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --num_exp "$NUM_EXP"
else
    python scripts/eval_policy_sim.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --num_exp "$NUM_EXP"
fi