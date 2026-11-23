"""Compare final answers and grounded evidence on held-out GQA images."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from eval_mmmu import prepare_vllm_input

POINT = re.compile(r'\[point:\s*\[\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*\]\s*\]', re.I)
ANSWER = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.I | re.S)
FEATURE = re.compile(r'(?:\bfeature\s*:|\bits\s+(?:color|material|shape|size)\s+is)\s*([^.<\n]+)', re.I)
INSTRUCTION = (
    '\nLocate the referenced object and give its location as [point: [[x, y]]] '
    'using coordinates from 0 to 100 for each image axis. Write Feature: followed '
    'by the observed attribute, and put the short answer in <answer>...</answer>.'
)


def norm(value):
    value = unicodedata.normalize('NFKC', value).casefold().strip()
    return re.sub(r'[^\w\s-]', '', value).strip()


def score(item, response):
    point = POINT.search(response)
    answer = ANSWER.search(response)
    feature = FEATURE.search(response)
    localized = False
    if point:
        width, height = item['image_size']
        x, y, w, h = item['box']
        px, py = float(point[1]) * width / 100, float(point[2]) * height / 100
        localized = x <= px <= x + w and y <= py <= y + h
    gold = norm(item['answer'])
    attribute_correct = bool(feature and norm(feature[1]) == gold)
    answer_correct = bool(answer and norm(answer[1]) == gold)
    return {'id': item['id'], 'localization_correct': localized,
            'attribute_correct': attribute_correct, 'answer_correct': answer_correct,
            'grounded_correct': localized and attribute_correct and answer_correct,
            'response': response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.validation.read_text())
    images = args.validation.parent / 'images'
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    llm = LLM(model=args.model, dtype='bfloat16', gpu_memory_utilization=.5,
              max_model_len=4096, limit_mm_per_prompt={'image': 1}, seed=42)
    sampling = SamplingParams(temperature=0, max_tokens=256)
    results = []
    with (args.output / 'predictions.jsonl').open('w') as handle:
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            messages = [[{'role': 'user', 'content': [
                {'type': 'image', 'image': str(images / f"{item['image_id']}.jpg")},
                {'type': 'text', 'text': item['question'] + INSTRUCTION},
            ]}] for item in batch]
            inputs = [prepare_vllm_input(message, processor) for message in messages]
            for item, result in zip(batch, llm.generate(inputs, sampling), strict=True):
                record = score(item, result.outputs[0].text)
                results.append(record)
                handle.write(json.dumps(record) + '\n')
            handle.flush()
    summary = {'completed': len(results)}
    for key in ('localization_correct', 'attribute_correct', 'answer_correct', 'grounded_correct'):
        summary[key] = sum(row[key] for row in results)
    localized = summary['localization_correct']
    summary['attribute_given_localization'] = (
        sum(r['attribute_correct'] and r['localization_correct'] for r in results) / localized
        if localized else None
    )
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
