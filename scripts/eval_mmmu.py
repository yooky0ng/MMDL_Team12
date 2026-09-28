"""Assignment-pinned HF MMMU with Qwen prompts/scoring and selectable inference.

Upstream implementation: official_mmmu/ (commit recorded in OFFICIAL_COMMIT).
The HF adapter, Transformers generation, checks, CLI and output bookkeeping are local.
"""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official_mmmu"))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_ID = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
OFFICIAL_COMMIT = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
SUBJECTS = json.loads((ROOT / "mmmu_subjects.json").read_text())


def verify_download(root, relative, revision):
    """Check HF download provenance; this is not a full file-content hash check."""
    file = root / relative
    metadata = root / ".cache/huggingface/download" / (relative + ".metadata")
    if not file.is_file() or not metadata.is_file():
        raise ValueError(f"Missing pinned file or HF download metadata: {file}")
    if metadata.read_text().splitlines()[0] != revision:
        raise ValueError(f"Wrong downloaded revision: {file}; expected {revision}")


def verify_model(root):
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                 "tokenizer.json", "chat_template.json", "generation_config.json",
                 "model.safetensors.index.json"):
        verify_download(root, name, MODEL_REVISION)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        verify_download(root, name, MODEL_REVISION)


def parse_options(value):
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, list):
        raise ValueError(f"Invalid options: {value!r}")
    return options


def adapt_sample(sample, subject):
    """Convert HF columns to official TSV columns without changing question text."""
    import string

    question_type = sample["question_type"]
    if question_type not in ("multiple-choice", "open"):
        raise ValueError(f"Unexpected question type: {question_type!r}")
    line = {
        "index": sample["id"], "subject": subject, "split": "validation",
        "question": sample["question"], "answer": sample["answer"],
        "question_type": question_type,
    }
    # HF open questions can contain a placeholder options list: ignore it.
    options = parse_options(sample["options"]) if question_type == "multiple-choice" else []
    if question_type == "multiple-choice" and not 2 <= len(options) <= 26:
        raise ValueError(f"Invalid choice count: {sample['id']}")
    line.update(dict(zip(string.ascii_uppercase, options)))
    if sample.get("hint") is not None:
        line["hint"] = sample["hint"]
    return line


def load_annotations(root):
    import pyarrow.parquet as pq

    rows = []
    for subject in SUBJECTS:
        relative = f"{subject}/validation-00000-of-00001.parquet"
        verify_download(root, relative, DATASET_REVISION)
        parquet = pq.ParquetFile(root / relative)
        columns = ["id", "question", "answer", "question_type", "options"]
        if "hint" in parquet.schema_arrow.names:
            columns.append("hint")
        samples = parquet.read(columns=columns).to_pylist()
        if len(samples) != 30:
            raise ValueError(f"{subject}: expected 30 samples, found {len(samples)}")
        rows.extend(adapt_sample(sample, subject) for sample in samples)
    if len(rows) != 900 or len({row["index"] for row in rows}) != 900:
        raise ValueError("Expected exactly 900 unique validation samples")
    return rows


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class GeneratedPresencePenalty:
    """vLLM-style additive penalty: once per token seen in the generated suffix."""

    def __init__(self, prompt_length, penalty):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, self.prompt_length:].unique()
            scores[row, seen] -= self.penalty
        return scores


def prepare_transformers_input(messages, processor):
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
    )
    # qwen_vl_utils already resized images using the official pixel limits.
    inputs = processor(text=[text], images=images, return_tensors="pt", do_resize=False)
    inputs.pop("token_type_ids", None)
    return inputs


def generate_transformers(args, processor, messages_list):
    import torch
    from tqdm import tqdm
    from transformers import Qwen3VLForConditionalGeneration, LogitsProcessorList, set_seed

    set_seed(args.seed)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    outputs = []
    start = time.monotonic()
    with torch.inference_mode():
        for messages in tqdm(messages_list, desc="Transformers inference"):
            inputs = prepare_transformers_input(messages, processor).to(model.device)
            prompt_length = inputs["input_ids"].shape[1]
            if prompt_length + args.max_new_tokens > args.max_model_len:
                raise ValueError(
                    f"Input ({prompt_length}) + max_new_tokens ({args.max_new_tokens}) "
                    f"exceeds max_model_len ({args.max_model_len}); no silent truncation."
                )
            processors = LogitsProcessorList([
                GeneratedPresencePenalty(prompt_length, args.presence_penalty),
            ])
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                logits_processor=processors,
            )
            outputs.append(processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0])
    return outputs, time.monotonic() - start, {
        "dtype": str(model.dtype), "batch_size": 1, "tensor_parallel_size": 1,
        "attention_implementation": "sdpa", "gpu_memory_utilization": None,
    }


def generate_vllm(args, processor, messages_list):
    import torch
    from vllm import LLM, SamplingParams
    from run_mmmu import prepare_inputs_for_vllm

    tp = args.tensor_parallel_size or torch.cuda.device_count()
    llm = LLM(
        model=str(args.model), tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 10}, seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty, stop_token_ids=[],
    )
    inputs = [prepare_inputs_for_vllm(messages, processor) for messages in messages_list]
    start = time.monotonic()
    outputs = llm.generate(inputs, sampling_params=sampling)
    return [output.outputs[0].text for output in outputs], time.monotonic() - start, {
        "dtype": str(llm.llm_engine.model_config.dtype), "tensor_parallel_size": tp,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def infer(args, annotations):
    # Import GPU dependencies only for inference; eval/check work without vLLM.
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor
    from run_mmmu import build_mmmu_prompt

    prediction_file = args.output_dir / "predictions.jsonl"
    if prediction_file.exists():
        raise FileExistsError(f"Predictions already exist: {prediction_file}. Use eval or a new output directory.")
    verify_model(args.model)
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CUDA GPU not detected")
    processor = AutoProcessor.from_pretrained(str(args.model))
    annotation_map = {row["index"]: row for row in annotations}
    all_messages, ordered_rows = [], []
    image_root = args.output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        dataset = load_dataset(
            "parquet", data_files={"validation": str(args.dataset_root / subject / "validation-00000-of-00001.parquet")},
            split="validation",
        )
        for sample in dataset:
            line = annotation_map[sample["id"]]
            paths = []
            for number in range(1, 8):
                image = sample.get(f"image_{number}")
                if image is not None:
                    # Preserve decoded HF image pixels; no JPEG recompression.
                    path = image_root / f"{line['index']}_{number}.png"
                    image.save(path)
                    paths.append(str(path))
            messages = build_mmmu_prompt(line, lambda _: paths, "MMMU_DEV_VAL")
            all_messages.append(messages)
            ordered_rows.append(line)
    if args.backend == "transformers":
        outputs, elapsed, backend_config = generate_transformers(args, processor, all_messages)
    else:
        outputs, elapsed, backend_config = generate_vllm(args, processor, all_messages)
    if len(outputs) != 900:
        raise RuntimeError(f"Expected 900 outputs, found {len(outputs)}")
    records = []
    for line, messages, output in zip(ordered_rows, all_messages, outputs):
        raw = output
        records.append({
            "question_id": line["index"], "annotation": line,
            "task": "MMMU_HF_VALIDATION",
            "result": {"gen": str(raw).split("</think>")[-1].strip(), "gen_raw": raw},
            "messages": messages,
        })
    atomic_jsonl(prediction_file, records)
    atomic_json(args.output_dir / "run_config.json", {
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "split": "validation", "num_subjects": 30, "num_samples": 900,
        "official_commit": OFFICIAL_COMMIT,
        "generation": {"seed": 42, "max_new_tokens": args.max_new_tokens,
                       "temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                       "presence_penalty": args.presence_penalty},
        "max_model_len": args.max_model_len,
        "inference_backend": args.backend,
        **backend_config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "min_pixels": 1003520, "max_pixels": 4014080,
        "inference_seconds": elapsed,
    })
    print(f"Saved {prediction_file} ({elapsed:.1f}s inference)")


def evaluation_items(records, annotations):
    import pandas as pd
    from dataset_utils import MMMU_preproc

    by_id = {row["index"]: row for row in annotations}
    ids = [record["question_id"] for record in records]
    if len(ids) != 900 or len(set(ids)) != 900 or set(ids) != set(by_id):
        raise ValueError("Predictions must cover exactly the pinned 900 validation IDs")
    rows = []
    for record in records:
        row = dict(by_id[record["question_id"]])
        # Ground truth always comes from the pinned dataset, not the prediction file.
        if record["annotation"]["question"] != row["question"]:
            raise ValueError(f"Prediction question does not match dataset: {row['index']}")
        row["prediction"] = str(record["result"]["gen"])
        rows.append(row)
    data = MMMU_preproc(pd.DataFrame(rows).sort_values("index"))
    # HF explicitly identifies open questions. Their literal answer can itself
    # be A/B/C (e.g. a figure label); after MMMU_preproc their correct option is
    # still A. Upstream's answer-is-a-letter heuristic cannot distinguish this.
    data["GT"] = [row["answer"] if row["question_type"] == "multiple-choice" else "A"
                  for _, row in data.iterrows()]
    return [row for _, row in data.iterrows()]


def judge_environment(api_type):
    return (("CHATGPT_DASHSCOPE_API_KEY", "DASHSCOPE_API_BASE") if api_type == "dash"
            else ("MIT_SPIDER_TOKEN", "MIT_SPIDER_URL"))


def evaluate(args, annotations):
    from eval_utils import build_judge, can_infer, build_choices, eval_single_sample

    prediction_file = args.input_file or args.output_dir / "predictions.jsonl"
    with prediction_file.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    items = evaluation_items(records, annotations)
    if args.scoring == "rules":
        from local_scoring import evaluate_local, PARSER_VERSION
        results = [evaluate_local(item) for item in items]
        parser_version = PARSER_VERSION
    else:
        needs_judge = sum(not can_infer(item["prediction"], build_choices(item)) for item in items)
        missing = [name for name in judge_environment(args.api_type) if not os.environ.get(name)]
        if needs_judge and missing:
            raise RuntimeError(
                f"{needs_judge} answers require the official LLM Judge. Set {', '.join(missing)} "
                "and rerun with --stage eval. Predictions are preserved; no partial accuracy is reported."
            )
        model = build_judge(args.eval_model, args.api_type) if needs_judge else None
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            results = list(executor.map(eval_single_sample, ((model, item) for item in items)))
        parser_version = "qwen-judge-hf-adapter"
    subjects = {row["index"]: row["subject"] for row in annotations}
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        subject = subjects[result["index"]]
        result["subject"] = subject
        stats[subject]["correct"] += result["hit"]
        stats[subject]["total"] += 1
    correct = sum(result["hit"] for result in results)
    subject_accuracy = {s: counts["correct"] / counts["total"] * 100 for s, counts in stats.items()}
    suffix = "_rules" if args.scoring == "rules" else "_judge"
    atomic_jsonl(args.output_dir / f"evaluation{suffix}.jsonl", results)
    atomic_json(args.output_dir / f"scores{suffix}.json", {
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "split": "validation", "official_commit": OFFICIAL_COMMIT,
        "prediction_file": str(prediction_file),
        "total_correct": correct, "total_samples": 900,
        "macro_average": sum(subject_accuracy.values()) / len(subject_accuracy),
        "accuracy": correct / 900 * 100, "subject_accuracy": subject_accuracy,
        "scoring": args.scoring, "parser_version": parser_version,
        "eval_model": args.eval_model if args.scoring == "judge" else None,
        "api_type": args.api_type if args.scoring == "judge" else None,
        "truncation_check": "not available: legacy predictions omit finish_reason",
        "extraction_methods": dict(Counter(r["extraction_method"] for r in results)),
        "extraction_failures": sum(not r["extraction_success"] for r in results),
    })
    import csv
    with (args.output_dir / f"scores{suffix}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "correct", "total", "accuracy_percent"])
        for subject in SUBJECTS:
            writer.writerow([subject, stats[subject]["correct"], stats[subject]["total"], subject_accuracy[subject]])
        writer.writerow(["Overall (macro)", correct, 900, sum(subject_accuracy.values()) / 30])
    print(f"Accuracy: {correct}/900 = {correct / 900 * 100:.2f}%")
    print(f"Scoring: {args.scoring}; summary: {args.output_dir / f'scores{suffix}.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "infer", "eval", "check"), default="all")
    parser.add_argument("--model", "--model-path", type=Path, default=ROOT / "Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/MMMU")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=128000)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--scoring", choices=("rules", "judge"), default="rules",
                        help="API-free rules (default), or official rule + external Judge")
    parser.add_argument("--eval-model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--api-type", choices=("dash", "mit"), default="dash")
    parser.add_argument("--nproc", type=int, default=4)
    args = parser.parse_args(argv)
    if args.nproc < 1:
        parser.error("--nproc must be positive")
    if args.input_file and args.stage != "eval":
        parser.error("--input-file is only supported with --stage eval")
    if args.backend == "transformers" and args.tensor_parallel_size not in (None, 1):
        parser.error("Transformers backend uses one GPU; --tensor-parallel-size must be 1")
    if args.max_model_len <= args.max_new_tokens or args.max_new_tokens < 1:
        parser.error("Require 0 < max-new-tokens < max-model-len")
    if args.output_dir is None:
        args.output_dir = ROOT / ("results_transformers" if args.backend == "transformers" else "results_official")
    args.output_dir = args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    annotations = load_annotations(args.dataset_root)
    if args.stage == "check":
        verify_model(args.model)
        print(f"Pinned downloads verified: {len(annotations)} validation samples, {len(SUBJECTS)} subjects")
        print(dict(Counter(row["question_type"] for row in annotations)))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("infer", "all"):
        infer(args, annotations)
    if args.stage in ("eval", "all"):
        evaluate(args, annotations)


if __name__ == "__main__":
    main()
