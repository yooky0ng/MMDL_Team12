"""API-free rules-v1: Qwen MC extraction, deterministic fallbacks, open parsing.

Open-answer parsing is adapted from the user-supplied MMMU-style evaluation
script. No extraction rule reads the gold answer; gold is used only for scoring.
"""
import ast
import math
import re

PARSER_VERSION = "qwen-first-local-v1"


def extract_choice(response, choices):
    from eval_utils import can_infer

    text = str(response).split("</think>")[-1].strip()
    if not text:
        return None, "unparsed"
    result = can_infer(text, dict(choices))
    if result:
        return result, "qwen_rule"
    # Remove Markdown emphasis; retain newlines, punctuation and actual order.
    cleaned = text.replace("**", "").replace("__", "")
    labels = re.escape("".join(choices))
    candidates = []
    patterns = [
        rf"(?i:final\s+answer|correct\s+answer|answer)\s*(?i:is)?\s*[:=]?\s*(?i:option\s+)?[\(\[]?([{labels}])(?![A-Za-z0-9])",
        rf"\\boxed\{{\s*([{labels}])\s*\}}",
    ]
    for pattern in patterns:
        candidates.extend((m.start(), m.group(1)) for m in re.finditer(pattern, cleaned))
    if candidates:
        return max(candidates)[1], "explicit_answer"
    # Use a short concluding paragraph before falling back to the full response.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    tail = paragraphs[-1] if paragraphs else cleaned
    result = can_infer(tail, dict(choices))
    if result:
        return result, "conclusion_rule"
    matches = list(re.finditer(rf"(?<!\w)([{labels}])(?!\w)", cleaned))
    if matches:
        return matches[-1].group(1), "last_choice"
    return None, "unparsed"


def normalize_value(value):
    value = str(value).strip()
    try:
        number = float(value.replace(",", ""))
        if math.isfinite(number):
            return round(number, 2)
    except ValueError:
        pass
    return value.casefold()


def extract_open(response):
    text = str(response).split("</think>")[-1].strip().strip(".")
    if not text:
        return []
    # Split before lowercasing: the supplied script's uppercase lookahead could
    # never match after lowercasing. Keep its indicator-based candidate approach.
    segments = re.split(r"\.\s+(?=[A-Z])|\n", text)
    keys = []
    indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
    for index, segment in enumerate(segments):
        segment = segment.casefold()
        markers = indicators + (["="] if index == len(segments) - 1 else [])
        tails = [segment.rsplit(marker, 1)[-1].strip() for marker in markers if marker in segment]
        if tails:
            tail = min(tails, key=len)
            if tail and tail not in {":", ",", ".", "!", "?", ";", "'"}:
                keys.append(tail)
    if not keys:
        keys = [text]
    number_pattern = r"(?<![\w.])[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
    values = []
    for key in keys:
        values.append(normalize_value(key))
        values.extend(normalize_value(m.group()) for m in re.finditer(number_pattern, key))
    return list(dict.fromkeys(values))


def match_open(gold, predictions):
    try:
        gold = ast.literal_eval(str(gold))
    except (ValueError, SyntaxError):
        pass
    answers = gold if isinstance(gold, list) else [gold]
    for answer in map(normalize_value, answers):
        for prediction in predictions:
            if isinstance(answer, float):
                if isinstance(prediction, float) and answer == prediction:
                    return True
            elif isinstance(prediction, str) and answer:
                # Word boundaries avoid matching 'cat' inside 'catalyst', or a
                # one-letter gold inside an unrelated word.
                if re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", prediction):
                    return True
    return False


def evaluate_local(item):
    from eval_utils import build_choices

    text = item["prediction"]
    if item["question_type"] == "multiple-choice":
        prediction, method = extract_choice(text, build_choices(item))
        correct = prediction == item["answer"]
        success = prediction is not None and prediction != "Z"
    else:
        prediction = extract_open(text)
        method = "open_normalized"
        correct = match_open(item["answer"], prediction)
        success = bool(prediction)
    return {
        "index": item["index"], "split": item["split"],
        "question_type": item["question_type"], "question": item["question"],
        "prediction": text, "extracted_answer": prediction,
        "extraction_method": method, "extraction_success": success,
        "gt": item["answer"], "hit": int(correct),
    }
