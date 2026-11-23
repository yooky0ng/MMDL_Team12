#!/usr/bin/env python3
"""Generate a reproducible 48,000-conversation MMVU subset from prepared data.

No third-party Python packages. Keeps all conversations sharing the same image together.
This is a project-defined deterministic image-group sample, not an asserted
reconstruction of the unspecified 48K subset from the reference paper.

Default paths assume the script runs from the user's mmvu_ready directory:
  mmvu/processed/train.json
  mmvu/processed/val.json
  mmvu/images/
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_image_path(row: dict, idx: int) -> str:
    if not isinstance(row, dict):
        raise ValueError(f'Row {idx} is not a dictionary.')
    image = row.get('image')
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f'Row {idx} missing image path.')
    image = image.replace('\\', '/')
    if image.startswith('/') or image.startswith('./../') or '..' in image.split('/') or ':' in image.split('/')[0]:
        raise ValueError(f'Row {idx} has unsafe image path: {image}')
    conversations = row.get('conversations')
    if not isinstance(conversations, list) or not conversations or len(conversations) % 2:
        raise ValueError(f'Row {idx} must have complete human/assistant pairs.')
    for j in range(0, len(conversations), 2):
        u, a = conversations[j], conversations[j + 1]
        if not isinstance(u, dict) or not isinstance(a, dict) or u.get('from') != 'human' or a.get('from') != 'gpt':
            raise ValueError(f'Row {idx} has invalid alternating roles at pair {j//2}.')
        if not isinstance(u.get('value'), str) or not isinstance(a.get('value'), str):
            raise ValueError(f'Row {idx} contains a non-text message.')
    return image


def write_gz_json(path: Path, rows: list) -> None:
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(',', ':'))
    # mtime=0 gives byte-identical gzip output across repeated runs.
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as z:
            for piece in encoder.iterencode(rows):
                z.write(piece.encode('utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Select and package 48K MMVU conversations for GitHub')
    parser.add_argument('--train', type=Path, default=Path('mmvu/processed/train.json'))
    parser.add_argument('--val', type=Path, default=Path('mmvu/processed/val.json'))
    parser.add_argument('--images-dir', type=Path, default=Path('mmvu/images'))
    parser.add_argument('--verify-images', action='store_true', help='Check each selected image exists on disk')
    parser.add_argument('--target', type=int, default=48000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('mmvu_48k_repo/datasets/mmvu_48k'))
    args = parser.parse_args()

    if not args.train.is_file() or not args.val.is_file():
        parser.error(f'Missing train / val file: {args.train} or {args.val}')
    if args.target <= 0:
        parser.error('--target must be >0')
    if args.verify_images and not args.images_dir.is_dir():
        parser.error(f'Images dir not found: {args.images_dir}')

    print(f'[1/6] Reading {args.train}', flush=True)
    with args.train.open('r', encoding='utf-8-sig') as f:
        train = json.load(f)
    print(f'[2/6] Reading {args.val}', flush=True)
    with args.val.open('r', encoding='utf-8-sig') as f:
        val = json.load(f)
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError('Both train and val must be JSON arrays.')
    if args.target > len(train):
        raise ValueError(f'Target {args.target} > available train records {len(train)}')

    print('[3/6] Validating rows, checking image split, grouping train conversations', flush=True)
    groups = defaultdict(list)
    for idx, row in enumerate(train):
        image = safe_image_path(row, idx)
        groups[image].append(idx)
    val_images = set()
    for idx, row in enumerate(val):
        val_images.add(safe_image_path(row, idx))
    overlap = val_images.intersection(groups)
    if overlap:
        raise ValueError(f'Train/val image leakage: {len(overlap)} common images. Sample: {next(iter(overlap))}')

    print(f'[4/6] Deterministic selection: seed={args.seed}, target={args.target}', flush=True)
    image_keys = sorted(groups)
    random.Random(args.seed).shuffle(image_keys)
    selected_images = set()
    count = 0
    skipped_oversized = 0
    for image in image_keys:
        n = len(groups[image])
        if count + n <= args.target:
            selected_images.add(image)
            count += n
            if count == args.target:
                break
        else:
            skipped_oversized += 1
    if count != args.target:
        raise RuntimeError(f'Exact target could not be reached without splitting an image group: selected {count}.')
    # Preserve original input row order and all turns inside each conversation.
    selected = [row for row in train if row['image'].replace('\\', '/') in selected_images]
    if len(selected) != args.target:
        raise AssertionError(f'Unexpected selected row count: {len(selected)}')

    if args.verify_images:
        print('[5/6] Verifying selected images exist', flush=True)
        missing = [path for path in sorted(selected_images) if not (args.images_dir / path).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} selected images missing. Example: {missing[:3]}')
    else:
        print('[5/6] Image existence check skipped (previous full preprocessing report is assumed)', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / 'train_48k.json.gz'
    val_out = args.output_dir / 'val.json.gz'
    print('[6/6] Writing compressed, Git-friendly JSON and reproducibility manifest', flush=True)
    write_gz_json(train_out, selected)
    write_gz_json(val_out, val)
    (args.output_dir / 'selected_images.txt').write_text('\n'.join(sorted(selected_images)) + '\n', encoding='utf-8')
    qa_pairs = sum(len(row['conversations']) // 2 for row in selected)
    manifest = {
        'dataset': 'BAAI MMVU/MMR-data, converted by local preprocessing kit',
        'subset_name': 'mmvu_48k',
        'selection_method': 'Seeded image-group shuffle, greedy whole-image-group fill without replacement',
        'is_exact_reference_paper_48k_subset': False,
        'seed': args.seed,
        'target_conversation_records': args.target,
        'selected_conversation_records': len(selected),
        'selected_qa_pairs': qa_pairs,
        'selected_unique_images': len(selected_images),
        'source_train_records': len(train),
        'source_train_unique_images': len(groups),
        'heldout_val_records': len(val),
        'heldout_val_unique_images': len(val_images),
        'train_val_image_overlap': 0,
        'positive_negative_labels_available': False,
        'note': 'Positive/Negative labels absent in merged source; preserve multi-turn conversations but do not claim balanced class sampling.',
        'sampling_skips_due_to_overshoot': skipped_oversized,
        'input_train_sha256': sha256_file(args.train),
        'input_val_sha256': sha256_file(args.val),
        'train_gz_sha256': sha256_file(train_out),
        'val_gz_sha256': sha256_file(val_out),
        'images_included_in_repository': False,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('selected_conversation_records','selected_qa_pairs','selected_unique_images','heldout_val_records','train_val_image_overlap','seed')}, ensure_ascii=False, indent=2))
    for p in (train_out, val_out):
        mb = p.stat().st_size / (1024 * 1024)
        print(f'  {p}: {mb:.1f} MiB')
        if mb >= 95:
            print('WARNING: >=95 MiB file; do not git push normally. Use Git LFS or host externally.', file=sys.stderr)
    print('DONE. Output: ', args.output_dir.resolve())

if __name__ == '__main__':
    main()
