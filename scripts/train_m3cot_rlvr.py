#!/usr/bin/env python3
"""Prepare or run single-GPU GRPO on M3CoT or ChartQA from an existing LoRA."""
import argparse
from decimal import Decimal
from hashlib import file_digest
import json
from pathlib import Path
from random import Random
import re

from datasets import Dataset, Image as DatasetImage
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import prepare_multimodal_messages


MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ROLLOUTS = 4
PROMPTS_PER_UPDATE = 4
FINAL_CHOICE = re.compile(r"(?:(?i:final answer|answer)\s*:\s*)?(?:\(([A-E])\)|([A-E]))[.!]?")
FINAL_NUMBER = re.compile(
    r"(?:(?i:final answer|answer)\s*:\s*)?([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)[.!]?"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-format", choices=("m3cot", "chartqa"), default="m3cot")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ids-file", type=Path, help="Ordered training IDs to use instead of random sampling")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/mmvu_only_4k_lr1e5/adapter"))
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty to the starting model (0 = off, previous runs)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Generate rollouts with vLLM colocated on the training GPU instead of HF generate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--vllm-max-model-length", type=int, default=8192)
    parser.add_argument("--from-base", action="store_true",
                        help="Ignore --adapter and start from the base Qwen with a fresh LoRA "
                             "(r=32, alpha=64, dropout=0.05, LLM self-attn q/k/v/o; same scope as the MMVU adapter)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--mmmu-reward", action="store_true", help="Score M3CoT completions with MMMU parser v2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = Path("datasets/chartqa_rlvr/train_16.json" if args.dataset_format == "chartqa"
                            else "datasets/m3cot_1600/train.json")
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/mmvu_{args.dataset_format}_rlvr_smoke")
    if args.dataset_format == "chartqa" and args.ids_file:
        parser.error("--ids-file applies to M3CoT only. ChartQA uses the order in --dataset.")
    if args.dataset_format == "chartqa" and args.mmmu_reward:
        parser.error("--mmmu-reward is for M3CoT multiple-choice prompts")
    if args.samples < args.steps * PROMPTS_PER_UPDATE or args.steps < 1:
        parser.error(f"--samples must cover {PROMPTS_PER_UPDATE} prompts per step")
    if args.max_completion_length < 1 or not 0 < args.learning_rate < 1:
        parser.error("completion length and learning rate must be positive")
    return args


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"] if completion else ""


def last_line(completion):
    lines = completion_text(completion).strip().splitlines()
    return lines[-1].replace("**", "").strip() if lines else ""


def reward_choices(completions, answer, allowed_options, **kwargs):
    rewards = []
    for completion, gold, choices in zip(completions, answer, allowed_options, strict=True):
        match = FINAL_CHOICE.fullmatch(last_line(completion))
        selected = (match[1] or match[2]) if match else None
        rewards.append(float(selected is not None and selected in choices and selected == gold))
    return rewards


def reward_numbers(completions, answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        match = FINAL_NUMBER.fullmatch(last_line(completion).replace("−", "-"))
        earned = match is not None and Decimal(match[1].replace(",", "")) == Decimal(gold)
        rewards.append(float(earned))
    return rewards


def make_mmmu_reward(tokenizer):
    from eval_mmmu import PARSER_VERSION, score_response

    if PARSER_VERSION != 2:
        raise RuntimeError("--mmmu-reward expects MMMU parser v2")
    stop_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id}

    def reward_mmmu_v2(completions, completion_ids, answer, options, **kwargs):
        rewards = []
        for completion, tokens, gold, choices in zip(completions, completion_ids, answer, options, strict=True):
            finish_reason = "stop" if tokens and tokens[-1] in stop_tokens else "length"
            sample = {"question_type": "multiple-choice", "answer": gold}
            _, correct = score_response(sample, completion_text(completion), choices, finish_reason)
            rewards.append(float(correct))
        return rewards

    return reward_mmmu_v2


def chartqa_training_data(args):
    root = args.dataset.resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads(args.dataset.read_text())[:args.samples]
    if (manifest["source"] != "HuggingFaceM4/ChartQA" or manifest["split"] != "train"
            or manifest["authors"] != "human" or manifest["answer_type"] != "numeric"
            or len(rows) != args.samples or len({row["id"] for row in rows}) != len(rows)
            or len({row["image_sha256"] for row in rows}) != len(rows)):
        raise ValueError("Expected distinct, human-authored ChartQA training prompts")
    items = []
    for row in rows:
        image = (root / row["image"]).resolve()
        if (row["source"] != manifest["source"] or row["split"] != "train" or not row["question"].strip()
                or not image.is_relative_to(root / "images")
                or reward_numbers([row["original_answer"]], [row["answer"]]) != [1.0]):
            raise ValueError(f"Invalid ChartQA training sample: {row['id']}")
        with image.open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != row["image_sha256"]:
                raise ValueError(f"Chart image differs from selection: {row['id']}")
        instruction = (
            "On a separate last line, give the final answer as 'Final answer: <number>'. "
            "Use the question's units and omit unit symbols on that line."
        )
        items.append({"prompt": [{"role": "user", "content": f"{row['question']}\n{instruction}"}],
                      "image": str(image), "answer": row["answer"]})
    return Dataset.from_list(items).cast_column("image", DatasetImage())


def training_data(args):
    report = json.loads(args.dataset.with_name("validation.json").read_text())
    with args.dataset.open("rb") as handle:
        sha = file_digest(handle, "sha256").hexdigest()
    rows = json.loads(args.dataset.read_text())
    if (report["source_split"] != "train" or report["train_json_sha256"] != sha
            or report["base_model_revision"] != REVISION or report["n_selected"] != len(rows)
            or report["parser_v2_answers_verified"] != len(rows)):
        raise ValueError("M3CoT training dataset and validation report must match")
    if args.samples > len(rows):
        raise ValueError("--samples exceeds the verified dataset size")
    if args.ids_file:
        ids = json.loads(args.ids_file.read_text())
        if (not isinstance(ids, list) or not all(isinstance(item, str) for item in ids)
                or len(ids) < args.samples or len(set(ids)) != len(ids)):
            raise ValueError("Expected an ordered list of distinct training IDs covering --samples")
        rows_by_id = {row["id"]: row for row in rows}
        if missing := set(ids) - rows_by_id.keys():
            raise ValueError(f"IDs missing from the verified training split: {sorted(missing)}")
        selected = [rows_by_id[sample_id] for sample_id in ids[:args.samples]]
    else:
        selected = Random(args.seed).sample(rows, args.samples)
    items = []
    for row in selected:
        question, target = (message["value"] for message in row["qwen3vl"]["conversations"])
        allowed = "ABCDE"[:len(row["choices"])]
        gold = row["answer"]
        image = Path(row["qwen3vl"]["image"])
        if (row["source"] != "m3cot" or not isinstance(gold, str) or len(gold) != 1
                or gold not in allowed or not image.is_file()
                or reward_choices([target], [gold], [allowed]) != [1.0]):
            raise ValueError(f"Invalid M3CoT training sample: {row['id']}")
        instruction = "End your response with a separate line: Final answer: <option letter>."
        items.append({"prompt": [{"role": "user", "content": f"{question}\n{instruction}"}],
                      "image": str(image), "answer": gold, "allowed_options": allowed, "options": row["choices"]})
    return Dataset.from_list(items).cast_column("image", DatasetImage())


def make_config(args, *, cpu):
    return GRPOConfig(
        output_dir=str(args.output_dir),
        use_cpu=cpu,
        bf16=not cpu,
        use_vllm=args.use_vllm and not cpu,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        # TRL's default sequence-level IS correction multiplies each sequence by exp(sum of per-token
        # logp gaps between vLLM (top-k/top-p processed) and HF; ~0.1/token over hundreds of tokens),
        # which drove grad_norm to ~1e-10 in the smoke test. Off = same objective as the HF-generate path.
        vllm_importance_sampling_correction=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ROLLOUTS * PROMPTS_PER_UPDATE,
        generation_batch_size=ROLLOUTS,
        num_generations=ROLLOUTS,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type="grpo",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
    )


def main():
    args = arguments()
    if not args.from_base:
        config_path = args.adapter / "run_config.json"
        if not config_path.is_file():
            config_path = args.adapter.parent / "run_config.json"
        original_run = json.loads(config_path.read_text())
        if original_run["revision"] != REVISION or not (args.adapter / "adapter_model.safetensors").is_file():
            raise ValueError("Expected a saved LoRA trained from the pinned Qwen base")
    starting_point = "base (fresh LoRA)" if args.from_base else str(args.adapter.resolve())

    if args.dataset_format == "chartqa":
        dataset, reward = chartqa_training_data(args), reward_numbers
    else:
        dataset, reward = training_data(args), reward_choices
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if args.mmmu_reward:
        reward = make_mmmu_reward(processor.tokenizer)
    conf = make_config(args, cpu=not args.execute)
    print(json.dumps({
        "adapter": starting_point, "data": str(args.dataset.resolve()),
        "dataset_format": args.dataset_format, "reward": reward.__name__,
        "ids_file": str(args.ids_file.resolve()) if args.ids_file else None,
        "samples": len(dataset), "rollouts_per_prompt": conf.num_generations,
        "max_completion_length": conf.max_completion_length, "updates": conf.max_steps,
        "learning_rate": conf.learning_rate, "vllm": conf.use_vllm,
    }, ensure_ascii=False, indent=2), flush=True)

    if not args.execute:
        for item in dataset.select(range(min(3, len(dataset)))):
            chat = prepare_multimodal_messages(item["prompt"], [item["image"]])
            tokens = processor.apply_chat_template(chat, tokenize=True, return_dict=True,
                                                  return_tensors="pt", add_generation_prompt=True)
            if "pixel_values" not in tokens or not tokens["input_ids"].numel():
                raise ValueError("Qwen could not encode the multimodal prompt")
        print("CPU preparation passed. GPU training requires --execute.")
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The RLVR smoke run needs a CUDA device with BF16 support")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, local_files_only=True,
    )
    if args.from_base:
        # lora_B starts at zero, so step 0 is exactly the base model
        model = get_peft_model(base, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=r".*\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        ))
    else:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not names or any("lora_" not in n or ".language_model.layers." not in n or ".self_attn." not in n for n in names):
        raise ValueError("Expected trainable LoRA parameters in LLM attention")
    trainer = GRPOTrainer(
        model=model, args=conf, processing_class=processor,
        train_dataset=dataset, reward_funcs=reward,
    )
    trainer.train()
    output_adapter = args.output_dir / "adapter"
    trainer.save_model(str(output_adapter))
    (output_adapter / "run_config.json").write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "method": "grpo",
        "starting_adapter": starting_point, "dataset": str(args.dataset.resolve()),
        "dataset_format": args.dataset_format, "reward": reward.__name__,
        "ids_file": str(args.ids_file.resolve()) if args.ids_file else None, "seed": args.seed,
        "samples": len(dataset), "steps": trainer.state.global_step, "learning_rate": conf.learning_rate,
        "rollouts_per_prompt": conf.num_generations, "max_completion_length": conf.max_completion_length,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
