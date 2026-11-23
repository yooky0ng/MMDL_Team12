#!/usr/bin/env python3
"""Merge a LoRA adapter into the pinned Qwen3-VL-4B base and save it for vLLM evaluation.

Usage: python scripts/merge_lora_simple.py --adapter outputs/<run>/adapter --output outputs/<run>/merged_model
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scale", type=float, default=1.0, help="Multiply the LoRA delta before merging")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, a.adapter)
    n_lora = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))
    print(f"LoRA modules injected: {n_lora}")
    if a.scale != 1.0:
        # W = W_base + scale * (alpha/r) * B A
        for _, m in model.named_modules():
            if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                for k in m.scaling:
                    m.scaling[k] *= a.scale
        print(f"LoRA delta scaled by {a.scale}")
    if n_lora == 0:
        raise ValueError("No LoRA modules found in adapter")
    merged = model.merge_and_unload()
    a.output.mkdir(parents=True)
    merged.save_pretrained(a.output, safe_serialization=True)
    AutoProcessor.from_pretrained(MODEL, revision=REVISION).save_pretrained(a.output)
    print(f"Saved merged model to {a.output}")


if __name__ == "__main__":
    main()
