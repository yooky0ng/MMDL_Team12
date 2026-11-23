#!/usr/bin/env bash
# Evaluate an ailab LoRA using super's local base and a temporary RAM-backed merge.
set -euo pipefail
cd "$(dirname "$0")/.."

gpu="$1"
name="$2"
case "$name" in
  mmvu_only_4k_local_base|point_gqa_4k_no_gqa_coordinates_local_base) ;;
  *) echo "Unknown ailab run: $name" >&2; exit 2 ;;
esac

checkpoint="outputs/$name/checkpoints/checkpoint-250"
base_revision=ebb281ec70b05090aa6165b016eac8ec08e71b17
base="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$base_revision"
results="results/${name}_mmmu_full900"
if [[ -s "$results/summary.json" ]]; then
  echo "Summary already exists: $results/summary.json"
  exit 0
fi

python - "$checkpoint" "$base_revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
config = json.loads((checkpoint / "run_config.json").read_text())
assert config["revision"] == sys.argv[2], "Unexpected base revision"
assert config["total_updates"] == 250, "Unexpected update count"
checksums = json.loads((checkpoint / "checksums.json").read_text())
assert {"adapter_model.safetensors", "adapter_config.json", "run_config.json"} <= checksums.keys()
for filename, expected in checksums.items():
    if filename == "training_state.pt":
        continue
    with (checkpoint / filename).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
print(f"Verified {checkpoint}")
PY

scratch=$(mktemp -d /dev/shm/mmdl-ailab-eval.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
merged="$scratch/merged_model"
mkdir -p reports
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu"

python scripts/merge_point_gqa_adapter.py "$checkpoint" "$merged" --base-model "$base" \
  2>&1 | tee -a "reports/${name}_merge.log"
python -u scripts/eval_mmmu.py \
  --model-path "$merged" \
  --data-root "$HOME/.cache/huggingface/datasets" \
  --output-dir "$results" \
  --prompt-style official \
  --max-new-tokens 16384 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.50 \
  --physical-gpu-index "$gpu" \
  2>&1 | tee -a "reports/${name}_mmmu_full900.log"
