#!/usr/bin/env python3
"""Deterministic stratified subset of existing 30K; never rewrites targets.

MMVU pair_id denotes a local grouping, NOT verified POS/NEG pairing.
The authoritative SFT target remains qwen3vl.conversations[1].value;
the legacy top-level answer field is deliberately left unchanged.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import tempfile
import unicodedata

QUOTAS = {'ChartQA': 3320, 'DVQA': 2840, 'PlotQA': 1584,
          'CLEVR': 152, 'TallyQA': 104}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def image_path(row, point_root, mmvu_root):
    raw = row['qwen3vl']['image'].replace('\\', '/')
    if row['source'] == 'point_cot' and 'source_images/' in raw:
        return (point_root / raw.split('source_images/', 1)[1]).resolve()
    if row['source'] == 'mmvu':
        return (mmvu_root / raw).resolve()
    return Path(raw).resolve()


def target(row):
    return row['qwen3vl']['conversations'][1]['value']


def lengths(rows):
    values = sorted(len(target(r)) for r in rows)
    return {'unit': 'characters, not tokens', 'count': len(values),
            'mean': statistics.mean(values), 'median': statistics.median(values),
            'p90': values[int((len(values)-1)*.90)],
            'p95': values[int((len(values)-1)*.95)], 'max': values[-1]}


def select(rows, paths, hashes, seed):
    rng = random.Random(seed)
    pools = defaultdict(list)
    groups = defaultdict(list)
    excluded = []
    for r in rows:
        if r['source'] == 'point_cot':
            pools[r['source_dataset']].append(r)
        elif r['source'] == 'mmvu':
            groups[r['pair_id']].append(r)
        else:
            raise ValueError(f"Unknown source: {r['source']}")

    def key(r):
        c = r['qwen3vl']['conversations']
        return (hashes[str(paths[r['id']])],
                normalized(c[0]['value'].replace('<image>', '')),
                normalized(c[1]['value']))

    def invalid(r):
        c = r['qwen3vl']['conversations']
        if len(c) != 2 or [m['from'] for m in c] != ['human', 'gpt']:
            return 'invalid_single_turn_schema'
        if not c[0]['value'].replace('<image>', '').strip() or not target(r).strip():
            return 'empty_question_or_target'
        if str(paths[r['id']]) not in hashes:
            return 'missing_image'
        if r['source'] == 'point_cot':
            if not re.fullmatch(r'\s*<think>.+</think>\s*<answer>.+</answer>\s*',
                                target(r), flags=re.S):
                return 'invalid_point_cot_target_tags'
        elif not re.fullmatch(r'\s*<answer>.+</answer>\s*', target(r), flags=re.S):
            return 'invalid_mmvu_target_tags'
        return None

    chosen_point = []
    used = set()
    for source, count in QUOTAS.items():
        pool = sorted(pools[source], key=lambda r: r['id'])
        rng.shuffle(pool)
        valid = []
        for r in pool:
            reason = invalid(r)
            if reason is None and key(r) in used:
                reason = 'duplicate_image_content_question_target'
            if reason:
                excluded.append({'id': r['id'], 'reason': reason})
            else:
                used.add(key(r))
                valid.append(r)
        if len(valid) < count:
            raise ValueError(f'{source}: {len(valid)} eligible, need {count}')
        chosen_point.extend(valid[:count])

    # Apply duplicate checks to selected Point-CoT plus all eligible MMVU groups.
    used = {key(r) for r in chosen_point}
    candidates = sorted(groups.items())
    rng.shuffle(candidates)
    eligible = []
    for group_id, members in candidates:
        members = sorted(members, key=lambda r: r['id'])
        reason = None
        if len(members) != 2:
            reason = 'group_size_not_two'
        elif len({(r['original_id'], str(paths[r['id']])) for r in members}) != 1:
            reason = 'group_original_id_or_image_mismatch'
        else:
            reason = next((s for r in members if (s := invalid(r))), None)
            if reason is None:
                keys = [key(r) for r in members]
                if len(set(keys)) != 2 or any(k in used for k in keys):
                    reason = 'duplicate_image_content_question_target'
        if reason:
            excluded.append({'group_id': group_id, 'ids': [r['id'] for r in members],
                             'reason': reason})
        else:
            used.update(keys)
            eligible.append(members)
    if len(eligible) < 1000:
        raise ValueError(f'MMVU: {len(eligible)} eligible groups, need 1000')
    chosen_mmvu = [r for members in eligible[:1000] for r in members]
    mixed = chosen_point + chosen_mmvu
    rng.shuffle(mixed)
    return chosen_point, chosen_mmvu, mixed, excluded, key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--point-image-root', required=True, type=Path)
    p.add_argument('--mmvu-image-root', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {a.output_dir}')
    input_hash = sha(a.input)
    rows = json.loads(a.input.read_text())
    assert len(rows) == 30000
    assert Counter(r['source'] for r in rows) == {'point_cot': 24000, 'mmvu': 6000}
    assert len({r['id'] for r in rows}) == len(rows)
    paths = {r['id']: image_path(r, a.point_image_root, a.mmvu_image_root) for r in rows}
    unique_paths = sorted(set(paths.values()))
    present = [p for p in unique_paths if p.is_file()]
    print(f'Checking {len(rows)} records, hashing {len(present)} unique image paths', flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = dict(zip(map(str, present), executor.map(sha, present)))
    pc, mm, mixed, excluded, key = select(rows, paths, hashes, a.seed)
    pc2, mm2, mixed2, ex2, _ = select(rows, paths, hashes, a.seed)
    assert [r['id'] for r in mixed] == [r['id'] for r in mixed2] and excluded == ex2
    assert len(pc) == 8000 and len(mm) == 2000 and len(mixed) == 10000
    assert len({key(r) for r in mixed}) == 10000
    originals = {r['id']: r for r in rows}
    assert all(r == originals[r['id']] for r in mixed)
    assert input_hash == sha(a.input), 'Input changed while sampling'
    by_source = {}
    for source in ['ChartQA', 'DVQA', 'PlotQA', 'CLEVR', 'TallyQA', 'MMVU']:
        before = [r for r in rows if r['source_dataset'] == source]
        after = [r for r in mixed if r['source_dataset'] == source]
        by_source[source] = {'before': lengths(before), 'selected': lengths(after)}
    stats = {'seed': a.seed, 'input_sha256': input_hash,
             'counts': dict(Counter(r['source'] for r in mixed)),
             'point_cot_source_counts': dict(Counter(r['source_dataset'] for r in pc)),
             'mmvu_local_groups': 1000, 'excluded_candidate_units': len(excluded),
             'length_statistics': by_source,
             'assistant_token_statistics': 'not measured; no tokenizer/model loaded'}
    report = {'status': 'passed_subset_checks', 'total': 10000,
              'missing_selected_images': 0, 'duplicate_selected_content': 0,
              'selected_rows_identical_to_input': True, 'reproducible_selection': True,
              'input_unchanged': True, 'target_field': 'qwen3vl.conversations[1].value',
              'legacy_answer_field_warning': 'Top-level answer is NOT the full SFT target; retained unchanged.',
              'point_cot_think_answer_structure_valid': True,
              'mmvu_answer_structure_valid': True,
              'mmvu_grouping': 'Same original_id and image, two distinct records; POS/NEG relation NOT verified.',
              'mmvu_scope': 'Only first QA already present in 30K; no reconstruction of full original conversations.',
              'image_hash_method': 'SHA256 file bytes; detects identical files, not re-encoded visual duplicates.',
              'evaluation_leakage_check': 'not performed in this subset extraction; no leakage-free claim',
              'processor_token_length_and_masking_validation': 'not performed; required before training',
              'excluded_candidates': excluded}
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.mixed10k-', dir=a.output_dir.parent))
    try:
        for name, obj in [('point_cot_8k.json', pc), ('mmvu_2k.json', mm),
                          ('mixed_10k.json', mixed), ('dataset_statistics.json', stats),
                          ('validation_report.json', report)]:
            (tmp / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
        with (tmp / 'sample_manifest.jsonl').open('w') as f:
            index = {r['id']: i for i, r in enumerate(rows)}
            for position, r in enumerate(mixed):
                entry = {'position': position, 'id': r['id'], 'source': r['source'],
                         'source_dataset': r['source_dataset'], 'original_id': r['original_id'],
                         'input_file': str(a.input.resolve()), 'input_sha256': input_hash,
                         'input_index': index[r['id']], 'seed': a.seed,
                         'resolved_image': str(paths[r['id']]),
                         'image_sha256': hashes[str(paths[r['id']])],
                         'assistant_target_sha256': hashlib.sha256(target(r).encode()).hexdigest()}
                if r['source'] == 'mmvu':
                    entry.update({k: r[k] for k in ['pair_id', 'pair_label', 'pair_member']})
                f.write(json.dumps(entry, ensure_ascii=False)+'\n')
        (tmp / 'file_checksums.json').write_text(json.dumps(
            {p.name: sha(p) for p in sorted(tmp.iterdir())}, indent=2)+'\n')
        tmp.rename(a.output_dir)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
