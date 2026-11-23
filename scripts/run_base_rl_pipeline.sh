#!/usr/bin/env bash
# base Qwen3-VL-4B -> hard-104 selection -> GRPO -> merge -> MMMU eval (serial, one GPU)
# Usage: GPU=1 bash scripts/run_base_rl_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-1}"
PY=/home/hyk/miniconda3/envs/mmdl/bin/python
RL_PY=/home/hyk/.venvs/mmdl-rlvr/bin/python
PICK_DIR=datasets/rl_pick_base
RUN=outputs/base_m3cot_rlvr_hard104
EVAL=results/base_m3cot_rlvr_hard104_mmmu_full900
LOG=logs/base_rl_pipeline
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES="$GPU"

# Triton needs libcuda.so (same workaround as run_mmmu_eval.sh)
cuda_driver_path="$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" {print $NF; exit}')"
cuda_link_dir="$(mktemp -d -t mmdl-libcuda-XXXXXX)"
trap 'rm -f "${cuda_link_dir}/libcuda.so"; rmdir "${cuda_link_dir}" 2>/dev/null || true' EXIT
ln -s "${cuda_driver_path}" "${cuda_link_dir}/libcuda.so"
export LIBRARY_PATH="${cuda_link_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

FILTER=(--domain science,mathematics --min-choices 4 --max-candidates 600 --rl-prompt --output-dir "$PICK_DIR")

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

stamp "1/4 generate (4 samples x 300 questions, 1024 tokens)"
"$PY" scripts/gen_self_distill.py --stage generate "${FILTER[@]}" \
  --max-new-tokens 1024 --max-model-len 8192 --gpu-memory-utilization 0.30 2>&1 | tee "$LOG/1_generate.log"
"$PY" scripts/gen_self_distill.py --stage pick-rl "${FILTER[@]}" --pick 104 2>&1 | tee "$LOG/1_pick.log"

stamp "2/4 GRPO from base"
"$RL_PY" scripts/train_m3cot_rlvr.py --from-base \
  --dataset datasets/m3cot_1600/train.json --ids-file "$PICK_DIR/rl_hard_104_ids.json" \
  --output-dir "$RUN" --samples 104 --steps 26 --seed 42 \
  --learning-rate 1e-6 --max-completion-length 1024 --mmmu-reward --execute 2>&1 | tee "$LOG/2_grpo.log"

stamp "3/4 merge (CPU)"
CUDA_VISIBLE_DEVICES="" "$PY" scripts/merge_lora_simple.py \
  --adapter "$RUN/adapter" --output "$RUN/merged_model" 2>&1 | tee "$LOG/3_merge.log"

stamp "4/4 MMMU eval"
"$PY" scripts/eval_mmmu.py --model-path "$RUN/merged_model" \
  --data-root /home/hyk/.cache/huggingface/datasets --output-dir "$EVAL" \
  --prompt-style official --max-new-tokens 16384 --max-model-len 32768 \
  --chunk-size 128 --seed 42 --physical-gpu-index "$GPU" --gpu-memory-utilization 0.50 2>&1 | tee "$LOG/4_eval.log"

stamp "done"
"$PY" -c "import json;s=json.load(open('$EVAL/summary.json'));print('MMMU overall:', s['overall_macro_accuracy'])"
