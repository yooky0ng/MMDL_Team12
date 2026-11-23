"""Testable single-GPU training state and exact token-mean accumulation."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint requires CUDA')
        torch.cuda.set_rng_state(state['cuda'], 0)


def supervised_tokens(labels):
    # The model shifts labels one position for causal next-token prediction.
    return int((labels[:, 1:] != -100).sum().item())


def backward_token_sum(mean_loss, tokens):
    if tokens <= 0 or not torch.isfinite(mean_loss.detach()):
        raise FloatingPointError('Empty target or non-finite loss')
    (mean_loss.float() * tokens).backward()


def optimizer_update(parameters, optimizer, scheduler, tokens, max_grad_norm):
    if tokens <= 0:
        raise ValueError('No tokens in accumulation window')
    parameters = list(parameters)
    for p in parameters:
        if p.grad is not None:
            p.grad.div_(tokens)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Missing, zero or non-finite LoRA gradient')
    lr_used = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm), lr_used


def save_checkpoint(directory, model, processor, optimizer, scheduler, state, config):
    """Publish only complete checkpoints, always at an optimizer boundary."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.checkpoint-', dir=directory.parent))
    try:
        model.save_pretrained(tmp)
        if processor is not None:
            processor.save_pretrained(tmp)
        training = dict(state, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), rng=capture_rng(),
                        config_sha256=config_sha(config))
        torch.save(training, tmp / 'training_state.pt')
        (tmp / 'run_config.json').write_text(json.dumps(config, indent=2))
        (tmp / 'checksums.json').write_text(json.dumps(
            {p.name: file_sha(p) for p in tmp.iterdir() if p.is_file()}, indent=2))
        tmp.rename(directory)
    except BaseException:
        shutil.rmtree(tmp)
        raise


def load_checkpoint(directory, config):
    directory = Path(directory)
    checks = json.loads((directory / 'checksums.json').read_text())
    for filename, digest in checks.items():
        if file_sha(directory / filename) != digest:
            raise ValueError(f'Checkpoint checksum mismatch: {filename}')
    if json.loads((directory / 'run_config.json').read_text()) != config:
        raise ValueError('Resume configuration or input dataset/code hash changed')
    # Only load locally generated, trusted checkpoints (contains RNG Python objects).
    state = torch.load(directory / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['config_sha256'] != config_sha(config):
        raise ValueError('Resume state/config mismatch')
    return state


def prune_checkpoints(root, keep):
    paths = sorted(Path(root).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    for old in paths[:-keep]:
        shutil.rmtree(old)


def reserve_output(output, resume):
    output = Path(output)
    if resume:
        if Path(resume).resolve().parent.parent != output.resolve():
            raise ValueError('Resume checkpoint must belong to this output directory')
        if (output / 'adapter').exists():
            raise FileExistsError('Final adapter exists; refusing to overwrite completed training')
    else:
        # Exclusive creation also protects against two simultaneous launches.
        output.mkdir(parents=True, exist_ok=False)


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)
