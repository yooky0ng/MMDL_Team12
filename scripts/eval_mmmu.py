#!/usr/bin/env python3
"""고정된 MMMU validation split에서 Qwen3-VL을 평가한다."""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.metadata
import json
import os
import re
import string
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health",
    "Sociology",
]

# 2: 객관식 명시 답안 패턴의 소문자 오탐 수정 및 굵게(**) 표시 제거
PARSER_VERSION = 2

# Qwen 공식 MMMU 평가 코드의 이미지 해상도
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28

# 터미널 명령에 옵션 추가
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mc-per-subject",
        type=int,
        default=None,
        help="Evaluate the first N multiple-choice samples from every subject.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Evaluate only the listed MMMU IDs for targeted smoke tests.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--physical-gpu-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-style",
        choices=["direct", "official"],
        default="official",
        help="Prompt used for the submitted baseline or Qwen's official MMMU template.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Re-parse saved responses in <output-dir>/predictions.jsonl without generation.",
    )
    return parser.parse_args()


def replace_image_markers(text: str) -> str:
    return re.sub(r"<image\s+(\d+)>", r"[Image \1]", text)


def build_prompt(
    sample: dict[str, Any], prompt_style: str
) -> tuple[list[dict[str, Any]], list[str]]:
    options = ast.literal_eval(sample["options"])
    question = replace_image_markers(sample["question"])

    prompt = f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        for idx, option in enumerate(options):
            label = string.ascii_uppercase[idx]
            prompt += f"{label}. {replace_image_markers(str(option))}\n"
        prompt += "Please select the correct answer from the options above."
        if prompt_style == "direct":
            prompt += (
                " Respond with only the option letter (for example, A). "
                "Do not include an explanation."
            )
    else:
        if prompt_style == "direct":
            prompt += (
                "Please provide only the short final answer to the question. "
                "Do not include an explanation."
            )

    content: list[dict[str, Any]] = []
    for image_idx in range(1, 8):
        image = sample[f"image_{image_idx}"]
        if image is not None:
            content.append(
                {
                    "type": "image",
                    "image": image.convert("RGB"),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], options


def prepare_vllm_input(messages: list[dict[str, Any]], processor: Any) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    multimodal_data: dict[str, Any] = {}
    if image_inputs is not None:
        multimodal_data["image"] = image_inputs
    if video_inputs is not None:
        multimodal_data["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": multimodal_data,
        "mm_processor_kwargs": video_kwargs,
    }


# 객관식 파서
# MMMU eval_utils.py를 참고했으며 파싱 실패는 None으로 처리
def parse_multi_choice_response(
    response: str, choices: list[str], index_to_answer: dict[str, str]
) -> str | None:
    # "**B. ...**" 같은 굵게 표시를 제거하고, 선택지 문자는 대문자만 인정한다
    # ("answer is approximately"의 a를 A로 읽는 오탐 방지)
    response = response.replace("**", "")
    explicit = re.findall(
        rf"(?i:final\s+answer|answer)\s*(?:is|:)?\s*[\(\[]?([{''.join(choices)}])[\)\]]?(?![A-Za-z])",
        response,
    )
    boxed = re.findall(
        rf"\\boxed\{{\s*([{''.join(choices)}])\s*\}}", response, flags=re.IGNORECASE
    )
    if explicit or boxed:
        return (explicit + boxed)[-1].upper()

    cleaned = response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        cleaned = cleaned.strip(char)
    padded = f" {cleaned} "

    candidates: list[str] = []
    positions: list[int] = []
    for choice in choices:
        matches = [f"({choice})", f" {choice} ", f"{choice}."]
        position = max(padded.rfind(match) for match in matches)
        if position >= 0:
            candidates.append(choice)
            positions.append(position)

    if not candidates and len(response.split()) > 5:
        lowered = response.lower()
        for choice, answer_text in index_to_answer.items():
            position = lowered.rfind(str(answer_text).lower())
            if position >= 0:
                candidates.append(choice)
                positions.append(position)

    if not candidates:
        return None
    return candidates[int(np.argmax(positions))]


def is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_open_value(value: str) -> list[str | float]:
    value = value.strip()
    if is_number(value):
        return [round(float(value.replace(",", "")), 2)]
    value = value.lower()
    if len(value) == 1:
        return [f" {value}", f"{value} "]
    return [value]


def extract_numbers(value: str) -> list[str]:
    comma_numbers = re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", value)
    scientific = re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", value)
    simple = re.findall(
        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", value
    )
    return comma_numbers + scientific + simple

# 주관식 파서
def parse_open_response(response: str) -> list[str | float]:
    response = response.strip().strip(".").lower()
    subresponses = re.split(r"\.\s(?=[A-Z])|\n", response)
    indicators = [
        "could be ", "so ", "is ", "thus ", "therefore ", "final ",
        "answer ", "result ",
    ]
    key_responses: list[str] = []
    for idx, subresponse in enumerate(subresponses):
        current_indicators = indicators + (["="] if idx == len(subresponses) - 1 else [])
        tails = [subresponse.split(indicator)[-1].strip() for indicator in current_indicators if indicator in subresponse]
        if tails:
            shortest = min(tails, key=len)
            if shortest not in {":", ",", ".", "!", "?", ";", "'"}:
                key_responses.append(shortest)
    if not key_responses:
        key_responses = [response]

    predictions: list[str | float] = list(key_responses)
    for item in key_responses:
        predictions.extend(extract_numbers(item))
    normalized: list[str | float] = []
    for item in predictions:
        normalized.extend(normalize_open_value(str(item)))
    return list(set(normalized))


def score_open_answer(gold: str, predictions: list[str | float]) -> bool:
    try:
        parsed_gold = ast.literal_eval(gold)
    except (ValueError, SyntaxError):
        parsed_gold = gold
    answers = parsed_gold if isinstance(parsed_gold, list) else [parsed_gold]
    normalized_gold: list[str | float] = []
    for answer in answers:
        normalized_gold.extend(normalize_open_value(str(answer)))

    for prediction in predictions:
        if isinstance(prediction, str):
            if any(isinstance(answer, str) and answer in prediction for answer in normalized_gold):
                return True
        elif prediction in normalized_gold:
            return True
    return False


def parse_and_score(sample: dict[str, Any], response: str, options: list[str]) -> tuple[Any, bool]:
    if sample["question_type"] == "multiple-choice":
        choices = list(string.ascii_uppercase[: len(options)])
        index_to_answer = dict(zip(choices, options))
        prediction = parse_multi_choice_response(response, choices, index_to_answer)
        return prediction, prediction == sample["answer"]
    prediction = parse_open_response(response)
    return prediction, score_open_answer(sample["answer"], prediction)


def score_response(
    sample: dict[str, Any], response: str, options: list[str], finish_reason: str | None
) -> tuple[Any, bool]:
    parsed, correct = parse_and_score(sample, response, options)
    # 잘린 추론 과정에 포함된 선택지 문자는 정답으로 처리하지 않음
    has_explicit_answer = bool(
        re.search(r"(?:final\s+answer|answer)\s*(?:is|:)", response, re.I)
        or re.search(r"\\boxed\{", response)
    )
    if (
        sample["question_type"] == "multiple-choice"
        and finish_reason == "length"
        and not has_explicit_answer
    ):
        return None, False
    return parsed, correct


def load_samples(
    data_root: str,
    limit: int | None,
    sample_ids: list[str] | None,
    mc_per_subject: int | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    requested_ids = set(sample_ids or [])
    for subject in SUBJECTS:
        dataset = load_dataset(
            "MMMU/MMMU",
            subject,
            split="validation",
            revision=DATASET_REVISION,
            cache_dir=data_root,
        )
        if len(dataset) != 30:
            raise RuntimeError(f"{subject}: expected 30 rows, got {len(dataset)}")
        selected_mc = 0
        for row in dataset:
            if requested_ids and row["id"] not in requested_ids:
                continue
            if mc_per_subject is not None and row["question_type"] != "multiple-choice":
                continue
            item = dict(row)
            item["subject"] = subject
            samples.append(item)
            if mc_per_subject is not None:
                selected_mc += 1
            if requested_ids and len(samples) == len(requested_ids):
                return samples
            if limit is not None and len(samples) >= limit:
                return samples
            if mc_per_subject is not None and selected_mc >= mc_per_subject:
                break
        if mc_per_subject is not None and selected_mc != mc_per_subject:
            raise RuntimeError(
                f"{subject}: expected {mc_per_subject} multiple-choice samples, "
                f"got {selected_mc}"
            )
    if requested_ids:
        found_ids = {sample["id"] for sample in samples}
        missing = sorted(requested_ids - found_ids)
        if missing:
            raise RuntimeError(f"Requested sample IDs not found: {missing}")
        return samples
    if mc_per_subject is None and len(samples) != 900:
        raise RuntimeError(f"Expected 900 rows, got {len(samples)}")
    return samples


class GpuMemoryMonitor:
    def __init__(self, gpu_index: int | None) -> None:
        self.gpu_index = gpu_index
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.gpu_index is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi", f"--id={self.gpu_index}",
                        "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.peak_mib = max(self.peak_mib, int(result.stdout.strip().splitlines()[0]))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(1.0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result = json.loads(line)
                completed[result["id"]] = result
    return completed


def write_summary(
    results: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
    expected_count: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["subject"]].append(result)

    subject_scores = []
    for subject in SUBJECTS:
        rows = grouped.get(subject, [])
        correct = sum(bool(row["correct"]) for row in rows)
        accuracy = correct / len(rows) if rows else 0.0
        subject_scores.append(
            {"subject": subject, "data_num": len(rows), "correct": correct, "accuracy": accuracy}
        )

    completed_subject_scores = [row["accuracy"] for row in subject_scores if row["data_num"]]
    overall = float(np.mean(completed_subject_scores)) if completed_subject_scores else 0.0
    parse_failures = sum(
        row["question_type"] == "multiple-choice" and row["parsed_prediction"] is None
        for row in results
    )
    truncated_responses = sum(row.get("finish_reason") == "length" for row in results)
    summary = {
        "metadata": metadata,
        "completed": len(results),
        "expected": expected_count,
        "overall_macro_accuracy": overall,
        "parse_failures": parse_failures,
        "truncated_responses": truncated_responses,
        "subjects": subject_scores,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with (output_dir / "scores.csv").open("w", encoding="utf-8") as handle:
        handle.write("subject,data_num,correct,accuracy\n")
        for row in subject_scores:
            handle.write(
                f"{row['subject']},{row['data_num']},{row['correct']},{row['accuracy']:.6f}\n"
            )
        handle.write(f"Overall,{len(results)},{sum(r['correct'] for r in results)},{overall:.6f}\n")


def rescore(args: argparse.Namespace, output_dir: Path, predictions_path: Path) -> None:
    results = read_completed(predictions_path)
    if not results:
        raise RuntimeError(f"No saved responses found in {predictions_path}")
    samples = {
        sample["id"]: sample for sample in load_samples(args.data_root, None, None, None)
    }

    # 첫 재채점 전의 원본 채점 결과를 보존
    backup_path = output_dir / "predictions.before_rescore.jsonl"
    if not backup_path.exists():
        backup_path.write_bytes(predictions_path.read_bytes())

    for result in results.values():
        sample = samples[result["id"]]
        options = ast.literal_eval(sample["options"])
        parsed, correct = score_response(
            sample, result["response"], options, result.get("finish_reason")
        )
        result["parsed_prediction"] = parsed
        result["correct"] = bool(correct)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results.values():
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata = previous.get("metadata", {})
    metadata["parser_version"] = PARSER_VERSION
    metadata["rescored_without_generation"] = True
    write_summary(
        list(results.values()), output_dir, metadata,
        expected_count=previous.get("expected", len(results)),
    )
    correct_count = sum(result["correct"] for result in results.values())
    print(f"Rescored {len(results)} saved responses: {correct_count} correct")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.limit is not None and args.sample_ids:
        raise ValueError("Use either --limit or --sample-ids, not both")
    selectors = sum(
        value is not None for value in (args.limit, args.sample_ids, args.mc_per_subject)
    )
    if selectors > 1:
        raise ValueError("Use only one sample-selection option")
    if args.mc_per_subject is not None and args.mc_per_subject <= 0:
        raise ValueError("--mc-per-subject must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if args.rescore_only:
        rescore(args, output_dir, predictions_path)
        return
    completed = read_completed(predictions_path)

    start_time = time.time()
    monitor = GpuMemoryMonitor(args.physical_gpu_index)
    monitor.start()
    try:
        samples = load_samples(
            args.data_root, args.limit, args.sample_ids, args.mc_per_subject
        )
        pending = [sample for sample in samples if sample["id"] not in completed]
        print(f"Loaded {len(samples)} samples; {len(completed)} already complete; {len(pending)} pending.")

        revision = None if Path(args.model_path).exists() else args.model_revision
        processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)
        llm = LLM(
            model=args.model_path,
            revision=revision,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": 7},
            seed=args.seed,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            presence_penalty=1.5,
            max_tokens=args.max_new_tokens,
        )

        mode = "a" if predictions_path.exists() else "w"
        with predictions_path.open(mode, encoding="utf-8") as output_file:
            for chunk_start in range(0, len(pending), args.chunk_size):
                chunk = pending[chunk_start : chunk_start + args.chunk_size]
                vllm_inputs = []
                option_lists = []
                prompts = []
                for sample in chunk:
                    messages, options = build_prompt(sample, args.prompt_style)
                    vllm_inputs.append(prepare_vllm_input(messages, processor))
                    option_lists.append(options)
                    prompts.append(messages[0]["content"][-1]["text"])

                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
                for sample, options, prompt, generated in zip(chunk, option_lists, prompts, outputs):
                    completion = generated.outputs[0]
                    response = completion.text.strip()
                    parsed, correct = score_response(
                        sample, response, options, completion.finish_reason
                    )
                    result = {
                        "id": sample["id"],
                        "subject": sample["subject"],
                        "question_type": sample["question_type"],
                        "prompt": prompt,
                        "gold": sample["answer"],
                        "response": response,
                        "parsed_prediction": parsed,
                        "correct": bool(correct),
                        "finish_reason": completion.finish_reason,
                        "output_tokens": len(completion.token_ids),
                    }
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    completed[sample["id"]] = result

                done = min(chunk_start + len(chunk), len(pending))
                print(f"Generated pending samples: {done}/{len(pending)}", flush=True)
                del vllm_inputs, outputs
                gc.collect()
    finally:
        monitor.stop()

    elapsed = time.time() - start_time
    all_results = list(read_completed(predictions_path).values())
    metadata = {
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "dataset": "MMMU/MMMU",
        "dataset_revision": DATASET_REVISION,
        "split": "validation",
        "dtype": "bfloat16",
        "backend": "vLLM",
        "backend_version": importlib.metadata.version("vllm"),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "prompt_style": args.prompt_style,
        "parser_version": PARSER_VERSION,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_gpu_memory_mib_this_invocation": monitor.peak_mib,
    }
    write_summary(all_results, output_dir, metadata, expected_count=len(samples))
    print(f"Results: {predictions_path}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Elapsed: {elapsed:.1f}s; peak GPU memory: {monitor.peak_mib} MiB")


if __name__ == "__main__":
    main()
