#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${POINT_GQA_GPU:-0}"
py="${POINT_GQA_PYTHON:-$HOME/miniconda3/envs/mmdl/bin/python}"
data="datasets/point_gqa_4k"
mkdir -p reports

"$py" - "$data" <<'PYTHON'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
plain = json.loads((root / 'plain/train.json').read_text())
grounded = json.loads((root / 'grounded/train.json').read_text())
valid = json.loads((root / 'validation.json').read_text())
assert len(plain) == len(grounded) == 4000 and len(valid) == 300
assert plain[:2000] == grounded[:2000]
for a, b in zip(plain[2000:], grounded[2000:], strict=True):
    assert a['id'] == b['id'] and a['source'] == b['source'] == 'gqa'
    assert a['qwen3vl']['image'] == b['qwen3vl']['image']
    assert a['qwen3vl']['conversations'][0] == b['qwen3vl']['conversations'][0]
    answer = a['qwen3vl']['conversations'][1]['value']
    target = b['qwen3vl']['conversations'][1]['value']
    assert target.endswith(answer) and '[point:' in target
train_images = {Path(r['qwen3vl']['image']).stem for r in plain[2000:]}
assert len(train_images) == 2000
assert len({r['image_id'] for r in valid}) == 300
assert train_images.isdisjoint(r['image_id'] for r in valid)
print('Verified paired training inputs and image-disjoint validation')
PYTHON

for mode in plain grounded; do
    dataset="$data/$mode/train.json"
    "$py" scripts/data_preparation/validate_mixed_inputs.py \
        --dataset "$dataset" --output-dir "$data/$mode/input_checks" \
        --point-image-root datasets/point_cot/source_images --mmvu-image-root "$HOME/mmvu_48k_images" \
        >"reports/point_gqa_4k_${mode}_validation.log" 2>&1
    "$py" scripts/train_point_gqa_lora.py --experiment point-gqa-4k \
        --dataset "$dataset" --output-dir "outputs/point_gqa_4k_$mode" \
        --save-steps 50 >"reports/point_gqa_4k_${mode}_dry_run.log" 2>&1
done
for mode in plain grounded; do
    "$py" scripts/train_point_gqa_lora.py --experiment point-gqa-4k \
        --dataset "$data/$mode/train.json" --output-dir "outputs/point_gqa_4k_$mode" \
        --save-steps 50 --execute >"reports/point_gqa_4k_${mode}_training.log" 2>&1
done
base="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
"$py" scripts/eval_grounded_gqa.py --model "$base" --validation "$data/validation.json" \
    --output results/point_gqa_4k_baseline_gqa >reports/point_gqa_4k_baseline_gqa.log 2>&1
for mode in plain grounded; do
    run="outputs/point_gqa_4k_$mode"
    "$py" scripts/merge_point_gqa_adapter.py "$run/adapter" "$run/merged_model" \
        >"reports/point_gqa_4k_${mode}_merge.log" 2>&1
    "$py" scripts/eval_grounded_gqa.py --model "$run/merged_model" \
        --validation "$data/validation.json" --output "results/point_gqa_4k_${mode}_gqa" \
        >"reports/point_gqa_4k_${mode}_gqa.log" 2>&1
done
