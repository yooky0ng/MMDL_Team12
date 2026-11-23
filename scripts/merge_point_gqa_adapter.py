import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--base-model", type=Path, help="Local copy of the checkpoint's base model")
args = parser.parse_args()
checkpoint, output = args.checkpoint, args.output
pending = output.with_name(output.name + ".pending")
if output.exists() or pending.exists():
    raise FileExistsError(output)
config = json.loads((checkpoint / "run_config.json").read_text())
if args.base_model is not None and not (args.base_model / "config.json").is_file():
    raise FileNotFoundError(args.base_model / "config.json")
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    str(args.base_model or config["model"]), revision=config["revision"], local_files_only=True,
    dtype=torch.bfloat16, device_map={"": 0},
)
model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval()
assert sum(hasattr(module, "lora_A") for module in model.modules()) == 144
merged = model.merge_and_unload(safe_merge=True).eval()
merged.save_pretrained(pending, safe_serialization=True)
processor.save_pretrained(pending)
pending.rename(output)
print(f"Saved {output}", flush=True)
