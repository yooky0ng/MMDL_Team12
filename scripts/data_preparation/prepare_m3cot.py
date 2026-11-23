"""Prepare a stratified M3CoT training set and check it against MMMU validation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import random
import re
import sys
import unicodedata

from PIL import Image, ImageOps
import pyarrow as pa
import pyarrow.parquet as pq


M3COT_REVISION = "48cf35001d595a6b0290c82c897a4b4563390821"
MMMU_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
QUOTAS = {"science": 1152, "mathematics": 512, "commonsense": 384}
MAX_ASSISTANT_TOKENS = 448
MAX_SCIENTIFIC_NAME_QUESTIONS = 256
VISUAL = re.compile(
    r"\b(?:image|picture|photo|photograph|diagram|figure|chart|graph|illustration|"
    r"map|table|schedule|calendar|timetable|tangram|screenshot|pictured|shown|"
    r"depicted|displayed|labeled|labelled|grid)\b", re.I,
)
META_OPTION = re.compile(
    r"\b(?:option|choice|answer)\s*[\[(]?[A-E][\])]?\s+(?:in|from|of)\s+(?:the\s+)?"
    r"(?:(?:given|provided|above)\s+)?(?:image|picture|figure|diagram|text)\b"
    r"|\b(?:options|choices)\s+(?:given|provided|shown)\s+(?:in|within)\s+(?:the\s+)?(?:text|image)\b",
    re.I,
)
FINAL_LETTER = re.compile(
    r"(?i:\b(?:final answer(?:[ \t]+is)?|(?:correct answer|answer)[ \t]*(?:is|:)|"
    r"(?:the[ \t]+)?correct (?:option|choice)[ \t]+is)[ \t]*:?[ \t]*(?:option[ \t]+|"
    r"choice[ \t]+)?[* \t]*)\(?([A-E])\)?\b",
)
TEXT_CHOICE = re.compile(
    r"(?i:\b(?:choice|option)[ \t]+)\(?([A-E])\)?"
    r"(?i:[ \t]+(?:in|from)[ \t]+(?:the[ \t]+)?(?:given[ \t]+|provided[ \t]+)?text"
    r"[ \t]+(?:is[ \t]+(?:the[ \t]+)?(?:correct|right)|should[ \t]+be[ \t]+selected))"
)
IMAGE_CHOICE_SCOPE = re.compile(r"^[ \t]+(?:in|from)[ \t]+(?:the[ \t]+)?(?:given[ \t]+)?(?:image|diagram|figure)\b", re.I)
IMAGE_CHOICE_BLOCK = re.compile(
    r"^Match the image with (?:the )?correct answer[^\n]*:[ \t]*\n"
    r"(?:[ \t]*\([A-E]\)[^\n]*(?:\n|$)){2,}", re.I | re.M,
)
REDUNDANT_LAST_LINE = re.compile(
    r"\s*(?i:(?:therefore,\s*)?(?:the\s+)?(?:final|correct)\s+answer"
    r"\s*(?:is\s*|:\s*))\(?([A-E])\)?[.!]?\s*"
)
CHAT_TAG = re.compile(r"<\|[^|]+\|>|</?(?:think|answer)>", re.I)
STEP_LABEL = re.compile(r"^[ \t]*step[ \t]+\d+[ \t]*[:.)][ \t]*", re.I | re.M)
BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-E])\s*\}", re.I)


@dataclass
class Candidate:
    row: dict
    image: bytes
    image_key: str
    image_format: str
    question_key: str
    options_key: tuple[str, ...]
    rationale: str
    words: int
    quality: float


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmmu-cache", type=Path, default=Path.home() / ".cache/huggingface/datasets")
    parser.add_argument("--base-model", type=Path, default=Path.home() /
                        ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / QWEN_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-ids", type=Path, help="JSON list of source IDs rejected by manual review")
    return parser.parse_args()


def words(text):
    text = re.sub(r"<image\s*\d*>|\[image\s*\d*\]", " ", text, flags=re.I)
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def option_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def shingles(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def image_signature(raw):
    with Image.open(BytesIO(raw)) as opened:
        ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif",
               "BMP": "bmp", "TIFF": "tiff"}.get(opened.format, "img")
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        pixel_hash = sha256()
        pixel_hash.update(f"{width}x{height}:".encode())
        pixel_hash.update(image.tobytes())
        grey = image.convert("L")
        horizontal = list(grey.resize((9, 8), Image.Resampling.LANCZOS).getdata())
        vertical = list(grey.resize((8, 9), Image.Resampling.LANCZOS).getdata())
        dhash = 0
        for j in range(8):
            for i in range(8):
                dhash = (dhash << 1) | (horizontal[j * 9 + i] > horizontal[j * 9 + i + 1])
                dhash = (dhash << 1) | (vertical[j * 8 + i] > vertical[(j + 1) * 8 + i])
        return pixel_hash.hexdigest(), dhash, width / height, min(width, height), ext


def benchmark_index(cache):
    files = sorted((cache / "MMMU___mmmu").glob(f"*/0.0.0/{MMMU_REVISION}/mmmu-validation.arrow"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 MMMU cache files at revision {MMMU_REVISION}, found {len(files)}.")
    exact_text, texts, inverted = set(), [], defaultdict(list)
    pixel_hashes, visual_hashes, ids = set(), set(), set()
    for file in files:
        with pa.memory_map(str(file), "r") as stream:
            table = pa.ipc.open_stream(stream).read_all()
            if table.num_rows != 30:
                raise ValueError(f"{file}: expected 30 questions")
            for row in table.to_pylist():
                ids.add(row["id"])
                question_words = words(row["question"])
                exact_text.add(question_words)
                chunks = shingles(question_words) if len(question_words) >= 8 else set()
                for chunk in chunks:
                    inverted[chunk].append(len(texts))
                texts.append(chunks)
                for i in range(1, 8):
                    reference = row[f"image_{i}"]
                    if reference is None:
                        continue
                    raw = reference["bytes"]
                    if raw is None:
                        raw = Path(reference["path"]).read_bytes()
                    image_key, dhash, aspect, _, _ = image_signature(raw)
                    pixel_hashes.add(image_key)
                    visual_hashes.add((dhash, aspect))
    if len(ids) != 900:
        raise ValueError(f"Expected 900 unique MMMU IDs, found {len(ids)}")
    return exact_text, texts, inverted, pixel_hashes, tuple(visual_hashes)


def text_overlap(question, index):
    exact, texts, inverted, _, _ = index
    tokens = words(question)
    if tokens in exact:
        return True
    if len(tokens) < 8:
        return False
    grams = shingles(tokens)
    counts = Counter(other for gram in grams for other in inverted.get(gram, ()))
    return any(share / (len(grams) + len(texts[other]) - share) >= 0.8
               for other, share in counts.items())


def rationale_answer(rationale):
    tail = rationale[-500:]
    stated_text_choices = TEXT_CHOICE.findall(tail)
    if stated_text_choices:
        return stated_text_choices[-1]
    final_letters = [match[1] for match in FINAL_LETTER.finditer(tail)
                     if not IMAGE_CHOICE_SCOPE.match(tail[match.end():])]
    return final_letters[-1] if final_letters else None


def candidate_rows(files, benchmark, excluded, rng):
    counts = Counter()
    rejected = Counter()
    pools = defaultdict(lambda: defaultdict(list))
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                domain = row["domain"]
                counts[domain] += 1
                reason = None
                choices = row["choices"]
                answer = row["answer"]
                question = (row["question"] or "").strip()
                context = (row["context"] or "").strip()
                rationale = STEP_LABEL.sub("", (row["rationale"] or "").strip())
                stated_answer = rationale_answer(rationale)
                rationale = BOXED_LETTER.sub(r"\1", IMAGE_CHOICE_BLOCK.sub("", rationale)).strip()
                last_line = rationale.rsplit("\n", 1)[-1]
                redundant = REDUNDANT_LAST_LINE.fullmatch(last_line)
                if redundant and redundant[1] == answer and "\n" in rationale:
                    rationale = rationale.rsplit("\n", 1)[0].rstrip()
                length = len(rationale.split())
                option_keys = tuple(option_key(item) if isinstance(item, str) else "" for item in (choices or []))
                if row["id"] in excluded:
                    reason = "manual_rejection"
                elif row["split"] != "train" or domain not in QUOTAS:
                    reason = "unexpected_source"
                elif not choices or not 2 <= len(choices) <= 5 or "" in option_keys or len(set(option_keys)) != len(choices):
                    reason = "invalid_options"
                elif not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCDE"[:len(choices)]:
                    reason = "invalid_answer"
                elif {"true", "false"} <= set(option_keys) or {"yes", "no"} <= set(option_keys):
                    reason = "binary_yes_no"
                elif any(META_OPTION.search(option) for option in choices):
                    reason = "indirect_choice_reference"
                elif not question or not 12 <= length <= 225 or CHAT_TAG.search(question + context + rationale):
                    reason = "unsuitable_text"
                elif stated_answer is not None and stated_answer != answer:
                    reason = "explicit_rationale_conflict"
                elif row["topic"] == "language-science":
                    reason = "language_only_topic"
                elif not VISUAL.search(question + " " + rationale):
                    reason = "no_visual_reference"
                elif context and (FINAL_LETTER.search(context) or option_key(context) == option_keys[ord(answer) - 65]):
                    reason = "answer_in_context"
                elif text_overlap(question, benchmark) or (context and text_overlap(context + " " + question, benchmark)):
                    reason = "mmmu_text_overlap"
                raw = (row.pop("image") or {}).get("bytes")
                if reason is None and raw is None:
                    reason = "missing_image"
                if reason is None:
                    try:
                        image_key, dhash, aspect, smaller_side, ext = image_signature(raw)
                    except (OSError, ValueError, Image.DecompressionBombError):
                        reason = "unreadable_image"
                    else:
                        if smaller_side < 64:
                            reason = "small_image"
                        elif image_key in benchmark[3] or any(
                                abs(math.log(aspect / ratio)) <= 0.05 and (dhash ^ other).bit_count() <= 5
                                for other, ratio in benchmark[4]):
                            reason = "mmmu_image_overlap"
                if reason is not None:
                    rejected[(domain, reason)] += 1
                    continue
                # Prefer short original explanations within each topic and category.
                quality = length - (18 if VISUAL.search(question) else 0) - (10 if len(choices) >= 3 else 0)
                item = Candidate(row, raw, image_key, ext, " ".join(words(question)), option_keys,
                                 rationale, length, quality + rng.random() * 12)
                pools[domain][(row["topic"], row["category"])].append(item)
        print(f"Scanned {file.name}: {sum(counts.values())} source rows", flush=True)
    return counts, rejected, pools


def balanced_candidates(groups, rng):
    topics = defaultdict(list)
    for (topic, _), rows in groups.items():
        items = deque(sorted(rows, key=lambda row: row.quality))
        topics[topic].append(items)
    for groups_in_topic in topics.values():
        rng.shuffle(groups_in_topic)
    active = {topic: deque(groups_in_topic) for topic, groups_in_topic in topics.items()}
    while active:
        # Natural-science diagrams receive two slots per social-science slot.
        for topic in list(active) + (["natural-science"] if "natural-science" in active else []):
            if topic not in active:
                continue
            topic_pools = active[topic]
            group = topic_pools.popleft()
            yield group.popleft()
            if group:
                topic_pools.append(group)
            if not topic_pools:
                del active[topic]


def make_record(item, final_image):
    row = item.row
    choices = row["choices"]
    context = (row["context"] or "").strip()
    question = row["question"].strip()
    prompt = "Question: " + (context + "\n" if context else "") + question + "\nOptions:\n"
    prompt += "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    prompt += "\nPlease select the correct answer from the options above."
    target = f"{item.rationale}\nFinal answer: {row['answer']}"
    return {
        "id": "m3cot_" + row["id"],
        "source": "m3cot",
        "source_dataset": "LightChen2333/M3CoT",
        "original_id": row["id"],
        "image_id": row["image_id"],
        "domain": row["domain"],
        "topic": row["topic"],
        "category": row["category"],
        "question": question,
        "choices": choices,
        "answer": row["answer"],
        "qwen3vl": {"image": str(final_image), "conversations": [
            {"from": "human", "value": prompt}, {"from": "gpt", "value": target}
        ]},
    }


def format_counts(counts):
    grouped = defaultdict(dict)
    for (domain, reason), quantity in sorted(counts.items()):
        grouped[domain][reason] = quantity
    return dict(grouped)


def main():
    args = arguments()
    output = args.output_dir.resolve()
    pending = output.with_name(output.name + ".pending")
    if output.exists() or pending.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {pending}")
    files = sorted(args.parquet_dir.glob("train-*.parquet"))
    if len(files) != 7:
        raise ValueError(f"Expected seven training shards, found {len(files)}")
    excluded = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()
    rng = random.Random(args.seed)
    benchmark = benchmark_index(args.mmmu_cache)
    print("Indexed 900 MMMU validation questions with text and image fingerprints.", flush=True)
    source_counts, rejected, pools = candidate_rows(files, benchmark, excluded, rng)
    print("Eligible candidates", {d: sum(map(len, pools[d].values())) for d in QUOTAS}, flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from transformers import AutoProcessor
    from mixed_sft_inputs import prepare_batch
    from mixed_sft_training import supervised_tokens

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    (pending / "images").mkdir(parents=True)
    safe_queues = {d: balanced_candidates(pools[d], rng) for d in QUOTAS}
    seen_ids, seen_image_ids, seen_pixels, seen_questions = set(), set(), set(), set()
    selected, image_token_records = [], []
    actual, topics, letters = Counter(), Counter(), Counter()
    validation_failures = []
    scientific_name_questions = 0

    def select(domain, count):
        nonlocal scientific_name_questions
        accepted = 0
        for item in safe_queues[domain]:
            source_id, image_id = item.row["id"], item.row["image_id"]
            signature = (item.question_key, tuple(sorted(item.options_key)))
            if source_id in seen_ids or signature in seen_questions or item.image_key in seen_pixels \
                    or image_id and image_id in seen_image_ids:
                rejected[(domain, "duplicate_training_item")] += 1
                continue
            scientific_name_question = "scientific names" in (item.row["category"] or "").lower()
            if scientific_name_question and scientific_name_questions >= MAX_SCIENTIFIC_NAME_QUESTIONS:
                rejected[(domain, "scientific_name_category_cap")] += 1
                continue
            name = sha256(source_id.encode()).hexdigest()[:24] + "." + item.image_format
            candidate_path = pending / "images" / name
            candidate_path.write_bytes(item.image)
            row = make_record(item, candidate_path)
            try:
                batch = prepare_batch(processor, [row], pending, pending, max_length=4096)
                token_count = supervised_tokens(batch["labels"])
                full_tokens = batch["input_ids"].shape[1]
            except (OSError, ValueError) as error:
                candidate_path.unlink()
                rejected[(domain, "qwen_cpu_preflight")] += 1
                validation_failures.append({"id": source_id, "error": str(error)})
                continue
            if token_count > MAX_ASSISTANT_TOKENS:
                candidate_path.unlink()
                rejected[(domain, "long_assistant_target")] += 1
                continue
            row["qwen3vl"]["image"] = str(output / "images" / name)
            seen_ids.add(source_id)
            seen_pixels.add(item.image_key)
            if image_id:
                seen_image_ids.add(image_id)
            seen_questions.add(signature)
            selected.append(row)
            image_token_records.append({"id": row["id"], "domain": domain, "topic": row["topic"],
                                        "rationale_words": item.words, "full_tokens": full_tokens,
                                        "assistant_tokens": token_count, "image_pixel_sha256": item.image_key})
            topics[(domain, row["topic"])] += 1
            letters[(domain, row["answer"])] += 1
            actual[domain] += 1
            scientific_name_questions += scientific_name_question
            accepted += 1
            if len(selected) % 128 == 0:
                print(f"Validated {len(selected)} Qwen inputs: {dict(actual)}", flush=True)
            if accepted == count:
                return
        print(f"{domain}: validated {accepted} of {count} requested from the remaining pool", flush=True)

    for domain, quota in QUOTAS.items():
        select(domain, quota)
    # Spread any shortfall across the remaining visual categories.
    target_count = sum(QUOTAS.values())
    while len(selected) < target_count:
        before = len(selected)
        for domain in ("science", "commonsense"):
            remaining = target_count - len(selected)
            if remaining:
                select(domain, min(32, remaining))
        if len(selected) == before:
            break
    if not selected:
        raise ValueError("No validated training items")

    rng.shuffle(selected)
    with (pending / "train.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    assistant_tokens = Counter()
    for row in image_token_records:
        assistant_tokens[row["domain"]] += row["assistant_tokens"]
    lengths = sorted(row["assistant_tokens"] for row in image_token_records)
    report = {
        "source": "LightChen2333/M3CoT", "source_revision": M3COT_REVISION, "source_split": "train",
        "seed": args.seed, "source_counts": dict(source_counts), "requested_counts": QUOTAS,
        "n_selected": len(selected), "selected_counts": dict(actual), "topics": format_counts(topics),
        "answer_letters": format_counts(letters), "assistant_tokens_by_domain": dict(assistant_tokens),
        "assistant_token_quantiles": {"min": lengths[0], "median": lengths[len(lengths)//2],
                                     "p90": lengths[int(0.90 * (len(lengths)-1))], "max": lengths[-1]},
        "max_full_tokens": max(row["full_tokens"] for row in image_token_records),
        "mmmu_reference_revision": MMMU_REVISION, "mmmu_reference_count": 900,
        "mmmu_overlap_methods": ["normalized exact question", "word-trigram Jaccard >= 0.80",
                                 "exact decoded pixels", "near image difference hashes"],
        "rejections": format_counts(rejected), "processor_failures": validation_failures,
        "base_model_revision": QWEN_REVISION, "max_assistant_tokens_allowed": MAX_ASSISTANT_TOKENS,
        "quality_scope": "Automated screening checks visible references and explicit final-letter conflicts. "
                         "Visual necessity and step-by-step rationale correctness require manual review.",
        "manual_excluded_source_ids": sorted(excluded),
        "max_scientific_name_questions": MAX_SCIENTIFIC_NAME_QUESTIONS,
        "selected_scientific_name_questions": scientific_name_questions,
        "records": image_token_records,
    }
    (pending / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pending.rename(output)
    print(f"Saved {len(selected)} items to {output}; domain counts {dict(actual)}", flush=True)


if __name__ == "__main__":
    main()
