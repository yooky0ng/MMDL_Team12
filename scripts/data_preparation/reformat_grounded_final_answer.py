"""Replace the final answer tags while preserving Grounded questions and answers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding='utf-8'))
    if Counter(row['source'] for row in rows) != {'point_cot': 2000, 'gqa': 2000}:
        raise ValueError('Expected 2,000 Point-CoT and 2,000 GQA rows')
    for row in rows:
        turns = row['qwen3vl']['conversations']
        if [turn['from'] for turn in turns] != ['human', 'gpt']:
            raise ValueError(f"Unexpected conversation roles: {row['id']}")
        target = turns[1]['value']
        match = re.fullmatch(r'(.*\n)<answer>(.*?)</answer>(\s*)', target, flags=re.S)
        if (not match or not match[2].strip()
                or target.count('<answer>') != 1 or target.count('</answer>') != 1):
            raise ValueError(f"Expected one final answer block: {row['id']}")
        turns[1]['value'] = f'{match[1]}Final answer: {match[2]}{match[3]}'

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f'Converted {len(rows)} targets to {args.output}')


if __name__ == '__main__':
    main()
