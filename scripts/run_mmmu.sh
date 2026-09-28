#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Respect an existing CUDA selection; otherwise use the previously requested GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "${PYTHON:-python}" "$SCRIPT_DIR/eval_mmmu.py" \
    --model "$SCRIPT_DIR/Qwen3-VL-4B-Instruct" \
    --dataset-root "$SCRIPT_DIR/datasets/MMMU" \
    --backend transformers \
    --gpu-memory-utilization 0.90 \
    --max-model-len 128000 \
    --seed 42 \
    --max-new-tokens 32768 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5 \
    "$@"
