#!/usr/bin/env python3
"""Rejection-sampling self-distillation data: generate with the base model, keep correct answers.

Stage 1 (generate): for each M3CoT question, sample N responses from the base Qwen3-VL with the
exact MMMU-eval prompt format, image resolution and sampling settings (only max_tokens differs).
Every raw sample is written to raw.jsonl (resumable).

Stage 2 (select): score each sample with eval_mmmu.score_response (parser v2). A sample is usable
when it is correct, finished with "stop" and states the answer explicitly ("answer is X" / boxed).
Per question the shortest usable sample is kept. Questions the model got only partly right
(1..N-1 of N) are always kept; fully-solved questions are kept with probability --easy-keep.
Output: sft.jsonl with {id, image, prompt, response}.

Only the question, image and gold letter come from the dataset; the trained text is the model's own.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mmmu import MAX_PIXELS, MIN_PIXELS, MODEL_REVISION, prepare_vllm_input, score_response  # noqa: E402

RL_INSTRUCTION = "End your response with a separate line: Final answer: <option letter>."
EXPLICIT = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)|\\boxed\{", re.I)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("generate", "select", "pick-rl", "all"), default="all")
    p.add_argument("--dataset", type=Path, default=Path("datasets/m3cot_1600/train.json"))
    p.add_argument("--domain", default="science",
                   help="Comma-separated M3CoT domains, e.g. 'science,mathematics' ('all' for no filter)")
    p.add_argument("--min-choices", type=int, default=0, help="Keep questions with at least this many options")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="After filtering, keep the K questions with the longest prompts (harder-looking first)")
    p.add_argument("--rl-prompt", action="store_true",
                   help="Append the same final-answer instruction that train_m3cot_rlvr.py uses")
    p.add_argument("--pick", type=int, default=104, help="pick-rl: number of question IDs to write")
    p.add_argument("--ids-out", type=Path, default=None, help="pick-rl: output IDs file")
    p.add_argument("--max-per-category", type=int, default=None,
                   help="pick-rl: cap questions per M3CoT category prefix (e.g. geography)")
    p.add_argument("--limit", type=int, default=None, help="Use only the first K questions (smoke test)")
    p.add_argument("--output-dir", type=Path, default=Path("datasets/self_distill_m3cot_science"))
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model-revision", default=MODEL_REVISION)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--easy-keep", type=float, default=0.3,
                   help="Probability of keeping a question solved in all N samples")
    return p.parse_args()


def load_questions(args):
    rows = json.loads(args.dataset.read_text())
    if args.domain != "all":
        domains = set(args.domain.split(","))
        rows = [r for r in rows if r["domain"] in domains]
    rows = [r for r in rows if len(r["choices"]) >= args.min_choices]
    if args.max_candidates:
        rows = sorted(rows, key=lambda r: (-len(r["qwen3vl"]["conversations"][0]["value"]), r["id"]))
        rows = rows[: args.max_candidates]
    if args.limit:
        rows = rows[: args.limit]
    questions = []
    for r in rows:
        conv = r["qwen3vl"]["conversations"]
        assert conv[0]["from"] == "human"
        if args.rl_prompt:
            # identical to train_m3cot_rlvr.training_data()
            prompt = f"{conv[0]['value']}\n{RL_INSTRUCTION}"
        else:
            prompt = conv[0]["value"].replace("<image>", "").strip()
        questions.append({
            "id": r["id"], "image": r["qwen3vl"]["image"], "prompt": prompt,
            "choices": r["choices"], "answer": r["answer"],
        })
    return questions


def messages_for(q):
    from PIL import Image
    content = [
        {"type": "image", "image": Image.open(q["image"]).convert("RGB"),
         "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": q["prompt"]},
    ]
    return [{"role": "user", "content": content}]


def generate(args, questions, raw_path):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    done = set()
    if raw_path.exists():
        done = {json.loads(l)["id"] for l in raw_path.open()}
    pending = [q for q in questions if q["id"] not in done]
    print(f"{len(questions)} questions; {len(done)} done; {len(pending)} pending", flush=True)
    if not pending:
        return
    revision = None if Path(args.model_path).exists() else args.model_revision
    processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)
    llm = LLM(model=args.model_path, revision=revision, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 1}, seed=args.seed, trust_remote_code=True)
    # Same as eval_mmmu.py except max_tokens and n
    sampling = SamplingParams(n=args.num_samples, temperature=0.7, top_p=0.8, top_k=20,
                              repetition_penalty=1.0, presence_penalty=1.5,
                              max_tokens=args.max_new_tokens)
    start = time.time()
    with raw_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(pending), args.chunk_size):
            chunk = pending[i: i + args.chunk_size]
            inputs = [prepare_vllm_input(messages_for(q), processor) for q in chunk]
            outputs = llm.generate(inputs, sampling_params=sampling)
            for q, out in zip(chunk, outputs):
                samples = [{"text": c.text, "finish_reason": c.finish_reason,
                            "output_tokens": len(c.token_ids)} for c in out.outputs]
                f.write(json.dumps({**q, "samples": samples}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            n = min(i + len(chunk), len(pending))
            print(f"generated {n}/{len(pending)}  elapsed {time.time() - start:.0f}s", flush=True)


def select(args, raw_path, sft_path, stats_path):
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in raw_path.open()]
    kept, hist = [], collections.Counter()
    reasons = collections.Counter()
    for r in rows:
        sample = {"question_type": "multiple-choice", "answer": r["answer"]}
        options = r["choices"]
        assert len(options) <= len(string.ascii_uppercase)
        usable = []
        n_correct = 0
        for s in r["samples"]:
            text = s["text"].strip()
            _, correct = score_response(sample, text, options, s["finish_reason"])
            n_correct += bool(correct)
            if not correct:
                reasons["wrong"] += 1
            elif s["finish_reason"] != "stop":
                reasons["truncated"] += 1
            elif not EXPLICIT.search(text):
                reasons["no_explicit_answer"] += 1
            else:
                usable.append(s)
        hist[n_correct] += 1
        if not usable:
            continue
        if n_correct == len(r["samples"]) and rng.random() >= args.easy_keep:
            continue
        best = min(usable, key=lambda s: s["output_tokens"])
        kept.append({"id": r["id"], "image": r["image"], "prompt": r["prompt"],
                     "response": best["text"].strip(), "output_tokens": best["output_tokens"],
                     "n_correct": n_correct, "n_samples": len(r["samples"])})
    with sft_path.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    n_samples = sum(len(r["samples"]) for r in rows)
    stats = {
        "questions": len(rows), "samples": n_samples,
        "sample_accuracy": sum(c * k for k, c in hist.items()) / max(n_samples, 1),
        "correct_count_histogram": {str(k): hist[k] for k in sorted(hist)},
        "rejected_samples": dict(reasons),
        "kept": len(kept),
        "kept_partial": sum(k["n_correct"] < k["n_samples"] for k in kept),
        "kept_all_correct": sum(k["n_correct"] == k["n_samples"] for k in kept),
        "kept_median_tokens": sorted(k["output_tokens"] for k in kept)[len(kept) // 2] if kept else None,
        "easy_keep": args.easy_keep, "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def pick_rl(args, raw_path):
    """Write IDs of questions the model solves only sometimes (GRPO gets zero signal otherwise).

    Correctness = score_response, the same function the --mmmu-reward GRPO reward uses.
    Questions closest to half-correct come first; ties are shuffled with --seed.
    """
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in raw_path.open()]
    scored = []
    hist = collections.Counter()
    for r in rows:
        sample = {"question_type": "multiple-choice", "answer": r["answer"]}
        n = len(r["samples"])
        k = sum(bool(score_response(sample, s["text"].strip(), r["choices"], s["finish_reason"])[1])
                for s in r["samples"])
        hist[k] += 1
        if 0 < k < n:
            scored.append((abs(k - n / 2), rng.random(), r["id"], k))
    scored.sort()
    if args.max_per_category:
        cat_of = {r["id"]: r["category"].split("-")[0]
                  for r in json.loads(args.dataset.read_text())}
        used, capped = collections.Counter(), []
        for item in scored:
            c = cat_of[item[2]]
            if used[c] < args.max_per_category:
                used[c] += 1
                capped.append(item)
        scored = capped
    picked = scored[: args.pick]
    ids_out = args.ids_out or args.output_dir / f"rl_hard_{args.pick}_ids.json"
    ids_out.write_text(json.dumps([p[2] for p in picked], indent=2) + "\n")
    stats = {
        "questions": len(rows),
        "correct_count_histogram": {str(k): hist[k] for k in sorted(hist)},
        "partial_available": len(scored), "picked": len(picked),
        "picked_correct_counts": dict(collections.Counter(p[3] for p in picked)),
        "ids_out": str(ids_out),
    }
    (args.output_dir / "pick_rl_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if len(picked) < args.pick:
        print(f"WARNING: only {len(picked)} partially-solved questions available (< {args.pick})")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    (args.output_dir / "gen_config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))
    questions = load_questions(args)
    if args.stage in ("generate", "all"):
        generate(args, questions, raw_path)
    if args.stage in ("select", "all"):
        select(args, raw_path, args.output_dir / "sft.jsonl", args.output_dir / "select_stats.json")
    if args.stage == "pick-rl":
        pick_rl(args, raw_path)


if __name__ == "__main__":
    main()
