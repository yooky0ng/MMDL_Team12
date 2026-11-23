#!/usr/bin/env python3
"""Single-GPU, LLM-attention-only mixed LoRA SFT. Dry run unless --execute.

Resume only from a complete checkpoint produced by this version. Old 30K
adapters are deliberately never used to initialize a new run.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time

# Must be set before CUDA/cuBLAS initialization for repeatable GPU reductions.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

from mixed_sft_inputs import prepare_batch
from mixed_sft_training import (
    backward_token_sum, file_sha, load_checkpoint, optimizer_update,
    prune_checkpoints, reserve_output, restore_rng, save_checkpoint,
    seed_all, supervised_tokens, write_json_atomic,
)


def collate(items):
    return items


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/mmvu_repro/train.json')
    p.add_argument('--point-image-root', default='datasets/point_cot/source_images')
    p.add_argument('--mmvu-image-root', default='/home/hyk/mmvu_48k_images')
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--revision', default='ebb281ec70b05090aa6165b016eac8ec08e71b17')
    p.add_argument('--output-dir', default='outputs/mmvu_repro')
    p.add_argument('--max-samples', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--grad-accumulation-steps', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--save-steps', type=int, default=250)
    p.add_argument('--save-total-limit', type=int, default=3)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                   help='Use deterministic GPU algorithms for repeatable checkpoint resume')
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--stop-after-updates', type=int,
                   help='Save and stop after this global update; for resume validation')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error('learning-rate must be positive and finite')
    for name in ['batch_size', 'grad_accumulation_steps', 'save_steps', 'save_total_limit', 'max_length']:
        if getattr(a, name) <= 0:
            p.error(f'{name} must be positive')
    if a.num_workers < 0 or (a.max_samples is not None and a.max_samples <= 0):
        p.error('num-workers must be nonnegative, max-samples positive')
    if a.batch_size != 1:
        p.error('MMVU reproduction requires batch size 1 for per-sample loss averaging')
    if a.stop_after_updates is not None and a.stop_after_updates <= 0:
        p.error('stop-after-updates must be positive')
    return a


def prepare_run(a):
    rows = json.loads(Path(a.dataset).read_text())
    if not rows or any(row['source'] != 'mmvu' for row in rows):
        raise ValueError('Expected a nonempty MMVU-only training dataset')
    data_sha = file_sha(a.dataset)
    report_path = Path(a.dataset).parent / 'input_checks' / 'input_validation.json'
    report = json.loads(report_path.read_text())
    if (report['dataset_sha256'] != data_sha or report['scan_errors']
            or report['max_length'] != a.max_length or not report['ready_for_4096_training']
            or report['processor_examples'] < 20 or not report['padding_masking_passed']):
        raise ValueError('Missing, stale or failed input validation; revalidate before training')
    if a.max_samples is not None:
        if a.max_samples > len(rows):
            raise ValueError('max-samples exceeds dataset size')
        rows = rows[:a.max_samples]
    batches = math.ceil(len(rows) / a.batch_size)
    updates = math.ceil(batches / a.grad_accumulation_steps)
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'transformers', 'peft', 'accelerate', 'numpy']}
    code = [Path(__file__), Path(__file__).with_name('mixed_sft_inputs.py'),
            Path(__file__).with_name('mixed_sft_training.py')]
    cfg = {
        'algorithm_version': 2, 'dataset': str(Path(a.dataset).resolve()),
        'dataset_sha256': data_sha, 'samples': len(rows),
        'source_counts': dict(Counter(r['source'] for r in rows)),
        'target_field': 'qwen3vl.conversations[1].value',
        'point_image_root': str(Path(a.point_image_root).resolve()),
        'mmvu_image_root': str(Path(a.mmvu_image_root).resolve()),
        'model': a.model, 'revision': a.revision, 'seed': a.seed,
        'batch_size': a.batch_size, 'accumulation_steps': a.grad_accumulation_steps,
        'num_workers': a.num_workers, 'max_length': a.max_length,
        'epochs': 1, 'learning_rate': a.learning_rate, 'weight_decay': 0.01,
        'warmup_ratio': 0.03, 'warmup_updates': math.ceil(updates * .03),
        'total_updates': updates, 'max_grad_norm': 1.0,
        'r': 32, 'alpha': 64, 'dropout': .05,
        'dtype': 'bfloat16', 'gradient_checkpointing': True,
        'deterministic_algorithms': a.deterministic,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'loss_reduction': 'sum per-sample mean loss / samples per accumulation window',
        'save_steps': a.save_steps, 'save_total_limit': a.save_total_limit,
        'versions': versions, 'code_sha256': {p.name: file_sha(p) for p in code},
    }
    return rows, cfg


def main():
    a = arguments()
    if a.deterministic and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in [':4096:8', ':16:8']:
        raise ValueError('Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    torch.use_deterministic_algorithms(a.deterministic)
    rows, cfg = prepare_run(a)
    print(json.dumps(cfg, ensure_ascii=False, indent=2), flush=True)
    if not a.execute:
        print('DRY RUN: no model loaded, no training, no output directory created.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported')
    state = load_checkpoint(a.resume_from_checkpoint, cfg) if a.resume_from_checkpoint else None
    if state and (state['update'] >= cfg['total_updates'] or
                  a.stop_after_updates is not None and state['update'] >= a.stop_after_updates):
        raise ValueError('Checkpoint already reached requested update limit')
    reserve_output(a.output_dir, a.resume_from_checkpoint)
    output = Path(a.output_dir)
    write_json_atomic(output / 'run_config.json', cfg)
    (output / 'logs').mkdir(exist_ok=True)
    seed_all(a.seed)
    processor = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, revision=a.revision, dtype=torch.bfloat16, device_map={'': 0},
        trust_remote_code=True)
    expected = {
        f'model.language_model.layers.{i}.self_attn.{projection}'
        for i in range(base.config.text_config.num_hidden_layers)
        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    }
    modules = dict(base.named_modules())
    if not expected.issubset(modules):
        raise RuntimeError(f'Unexpected model layout, missing: {sorted(expected - modules.keys())}')
    targets = sorted(expected)
    if state:
        model = PeftModel.from_pretrained(base, a.resume_from_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=.05, target_modules=targets,
            bias='none', task_type='CAUSAL_LM'))
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any('lora_' not in n or '.language_model.layers.' not in n
                                  or '.self_attn.' not in n for n in trainable_names):
        raise RuntimeError('Unexpected trainable parameter outside LLM attention LoRA')
    injected = sorted(n for n, m in model.named_modules() if hasattr(m, 'lora_A'))
    if len(injected) != len(targets) or any(not n.endswith(tuple(targets)) for n in injected):
        raise RuntimeError('LoRA injection scope mismatch')
    trainable = [p for p in model.parameters() if p.requires_grad]
    write_json_atomic(output / 'lora_scope.json', {
        'targets': targets, 'injected_modules': injected, 'trainable_names': trainable_names,
        'trainable_parameters': sum(p.numel() for p in trainable),
        'all_parameters': sum(p.numel() for p in model.parameters()),
        'base_weights_frozen': True, 'vision_and_merger_frozen': True})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=cfg['learning_rate'], weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg['warmup_updates'], num_training_steps=cfg['total_updates'])
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(a.seed)).tolist()
    cursor = update = total_tokens = 0
    source_counts = Counter()
    elapsed_before = 0.0
    if state:
        if sorted(state['order']) != list(range(len(rows))) or state['order'] != order:
            raise ValueError('Checkpoint data order mismatch')
        cursor, update = state['cursor'], state['update']
        if not (0 <= cursor <= len(rows)) or sum(state['source_counts'].values()) != cursor:
            raise ValueError('Checkpoint progress mismatch')
        if cursor != min(update * a.batch_size * a.grad_accumulation_steps, len(rows)):
            raise ValueError('Checkpoint is not at the expected optimizer boundary')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if scheduler.last_epoch != update:
            raise ValueError('Scheduler step mismatch')
        total_tokens = state['total_tokens']
        source_counts.update(state['source_counts'])
        elapsed_before = state['elapsed_seconds']
    remaining = order[cursor:]
    sampler = [remaining[i:i+a.batch_size] for i in range(0, len(remaining), a.batch_size)]
    # Dedicated generator keeps DataLoader worker seeding out of model/dropout RNG.
    loader = DataLoader(rows, batch_sampler=sampler, num_workers=a.num_workers,
                        persistent_workers=a.num_workers > 0, pin_memory=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(a.seed + 1))
    iterator = iter(loader)
    if state:
        restore_rng(state['rng'])
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    window_tokens = 0
    window_weight = 0
    window_loss = 0.0
    source_loss = Counter()
    source_tokens = Counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - start

    def checkpoint():
        torch.cuda.synchronize()
        destination = output / 'checkpoints' / f'checkpoint-{update}'
        save_checkpoint(destination, model, processor, optimizer, scheduler, {
            'cursor': cursor, 'update': update, 'order': order,
            'source_counts': dict(source_counts), 'total_tokens': total_tokens,
            'elapsed_seconds': elapsed()}, cfg)
        prune_checkpoints(output / 'checkpoints', a.save_total_limit)
        return str(destination)

    try:
        with (output / 'logs' / 'updates.jsonl').open('a') as log:
            for micro_step, samples in enumerate(iterator, 1):
                full = prepare_batch(processor, samples, a.point_image_root,
                                     a.mmvu_image_root, a.max_length)
                tokens = supervised_tokens(full['labels'])
                batch = {k: v.to('cuda:0', non_blocking=True) for k, v in full.items()
                         if torch.is_tensor(v)}
                loss = model(**batch).loss
                weight = 1
                backward_token_sum(loss, weight)
                window_loss += float(loss.detach()) * weight
                window_weight += weight
                window_tokens += tokens
                cursor += len(samples)
                source_counts.update(s['source'] for s in samples)
                if len(samples) == 1:
                    source_loss[samples[0]['source']] += float(loss.detach()) * weight
                    source_tokens[samples[0]['source']] += weight
                del loss, full, batch
                if micro_step % a.grad_accumulation_steps and cursor < len(rows):
                    continue
                norm, lr_used = optimizer_update(trainable, optimizer, scheduler,
                                                 window_weight, 1.0)
                update += 1
                total_tokens += window_tokens
                duration = elapsed()
                entry = {'epoch': cursor / len(rows), 'optimizer_step': update,
                         'processed_samples': cursor, 'loss': window_loss / window_weight,
                         'assistant_tokens_in_update': window_tokens,
                         'learning_rate_used': lr_used,
                         'next_learning_rate': optimizer.param_groups[0]['lr'],
                         'gradient_norm': norm, 'source_counts': dict(source_counts),
                         'source_loss': {k: source_loss[k] / source_tokens[k] for k in source_loss},
                         'elapsed_seconds': duration, 'samples_per_sec': cursor / duration,
                         'tokens_per_sec': total_tokens / duration,
                         'estimated_remaining_seconds': (len(rows) - cursor) * duration / cursor,
                         'gpu_memory_allocated_mib': torch.cuda.memory_allocated() / 2**20,
                         'gpu_memory_reserved_mib': torch.cuda.memory_reserved() / 2**20}
                log.write(json.dumps(entry) + '\n')
                log.flush()
                print(json.dumps(entry), flush=True)
                window_tokens = 0
                window_weight = 0
                window_loss = 0.0
                source_loss.clear()
                source_tokens.clear()
                stopping = a.stop_after_updates is not None and update >= a.stop_after_updates
                if update % a.save_steps == 0 or cursor == len(rows) or stopping:
                    saved = checkpoint()
                if stopping and cursor < len(rows):
                    write_json_atomic(output / 'paused_state.json',
                                      {'checkpoint': saved, 'processed_samples': cursor,
                                       'optimizer_step': update})
                    print('STOPPED at optimizer boundary; resume with --resume-from-checkpoint', flush=True)
                    return
        if cursor != len(rows) or update != cfg['total_updates']:
            raise RuntimeError('Training sample/update count mismatch')
        tmp = Path(tempfile.mkdtemp(prefix='.final-adapter-', dir=output))
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        tmp.rename(output / 'adapter')
        write_json_atomic(output / 'training_summary.json', {
            'status': 'completed', 'samples': cursor, 'source_counts': dict(source_counts),
            'updates': update, 'elapsed_seconds': elapsed(), 'total_assistant_tokens': total_tokens,
            'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'lora_targets': len(targets), 'config': cfg})
    except BaseException as exc:
        write_json_atomic(output / 'failure.json', {
            'error': repr(exc), 'processed_samples_in_this_state': cursor,
            'completed_optimizer_updates': update,
            'note': 'Only complete checkpoint-* folders are resumable; partial window is discarded.'})
        raise


if __name__ == '__main__':
    main()
