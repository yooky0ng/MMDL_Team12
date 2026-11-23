"""CPU algorithm tests; no pretrained model or GPU execution."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mixed_sft_training import (
    backward_token_sum, load_checkpoint, optimizer_update, prune_checkpoints,
    reserve_output, restore_rng, save_checkpoint, seed_all, supervised_tokens,
)
from mixed_sft_inputs import messages_for_sample


class TinyAdapter(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear(self.dropout(x))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / 'tiny_adapter.pt')


def setup(dropout=0):
    model = TinyAdapter(dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=5)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler, batches):
    tokens = 0
    for x, y in batches:
        n = int((y != -100).sum())
        backward_token_sum(F.cross_entropy(model(x), y), n)
        tokens += n
    optimizer_update(model.parameters(), optimizer, scheduler, tokens, 1.0)


class TrainingTests(unittest.TestCase):
    def test_regression_full_target_instead_of_legacy_answer(self):
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / 'image.png'
            image.write_bytes(b'existence check only')
            full = '<think>Step 1 [point: [[10, 20]]]</think>\n<answer>7</answer>'
            sample = {'id': 'example', 'source': 'point_cot', 'answer': '7',
                      'question': 'wrong legacy question',
                      'qwen3vl': {'image': str(image), 'conversations': [
                          {'from': 'human', 'value': '<image>\nActual question'},
                          {'from': 'gpt', 'value': full}]}}
            messages = messages_for_sample(sample, root, root)
            self.assertEqual(messages[1]['content'][0]['text'], full)
            self.assertEqual(messages[0]['content'][1]['text'], 'Actual question')
            self.assertNotEqual(messages[1]['content'][0]['text'], sample['answer'])

    def test_causal_shift_token_count(self):
        labels = torch.tensor([[5, -100, 3, 4, -100]])
        self.assertEqual(supervised_tokens(labels), 2)

    def test_accumulated_token_gradient_matches_concatenated_batch(self):
        torch.manual_seed(1)
        a = TinyAdapter().double()
        b = deepcopy(a)
        x1, x2 = torch.randn(2, 4, dtype=torch.float64), torch.randn(7, 4, dtype=torch.float64)
        y1 = torch.tensor([0, -100])
        y2 = torch.tensor([1, 2, -100, 0, 2, 1, 0])
        F.cross_entropy(a(torch.cat([x1, x2])), torch.cat([y1, y2])).backward()
        # Use the real production function; float32 sum scaling tolerance applies.
        for x, y in [(x1, y1), (x2, y2)]:
            backward_token_sum(F.cross_entropy(b(x), y), int((y != -100).sum()))
        n = int((torch.cat([y1, y2]) != -100).sum())
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad / n, rtol=1e-6, atol=1e-7)

    def test_partial_window_update_matches_reference(self):
        torch.manual_seed(2)
        a, oa, sa = setup()
        b = deepcopy(a)
        ob = torch.optim.AdamW(b.parameters(), lr=.01, weight_decay=.01)
        sb = get_cosine_schedule_with_warmup(ob, num_warmup_steps=2, num_training_steps=5)
        batches = [(torch.randn(2, 4), torch.tensor([1, 0])),
                   (torch.randn(5, 4), torch.tensor([1, 2, -100, 1, 0]))]
        # First advance warmup, then compare an incomplete accumulation window.
        update(a, oa, sa, batches[:1]); update(b, ob, sb, batches[:1])
        update(a, oa, sa, batches)
        x = torch.cat([r[0] for r in batches]); y = torch.cat([r[1] for r in batches])
        F.cross_entropy(b(x), y).backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
        ob.step(); sb.step(); ob.zero_grad(set_to_none=True)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q)

    def test_warmup_then_cosine_on_optimizer_updates(self):
        m, opt, scheduler = setup()
        lr = [opt.param_groups[0]['lr']]
        for _ in range(5):
            update(m, opt, scheduler, [(torch.randn(2, 4), torch.tensor([1, 0]))])
            lr.append(opt.param_groups[0]['lr'])
        self.assertEqual(lr[0], 0)
        self.assertLess(lr[1], lr[2])
        self.assertAlmostEqual(lr[2], .01)
        self.assertGreater(lr[2], lr[3])
        self.assertAlmostEqual(lr[-1], 0)
        self.assertEqual(scheduler.last_epoch, 5)

    def test_checkpoint_resume_matches_uninterrupted_with_dropout_and_rng(self):
        seed_all(42)
        data = [(torch.randn(i+1, 4), torch.tensor([i % 3] * (i+1))) for i in range(5)]
        initial = setup(dropout=.2)
        initial_weights = deepcopy(initial[0].state_dict())
        # Uninterrupted reference.
        seed_all(7)
        for batch in data:
            update(*initial, [batch])
        reference = deepcopy(initial[0].state_dict())
        reference_rng = (random.random(), np.random.rand(), torch.rand(3))
        seed_all(123)
        split = setup(dropout=.2)
        split[0].load_state_dict(initial_weights)
        seed_all(7)
        for batch in data[:2]:
            update(*split, [batch])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'checkpoint-2'
            config = {'dataset_sha256': 'example', 'total_updates': 5}
            save_checkpoint(path, split[0], None, split[1], split[2],
                            {'cursor': 2, 'update': 2, 'order': list(range(5))}, config)
            resumed = setup(dropout=.2)  # Consumes RNG, as loading a real model does.
            state = load_checkpoint(path, config)
            resumed[0].load_state_dict(torch.load(path / 'tiny_adapter.pt', weights_only=True))
            resumed[1].load_state_dict(state['optimizer'])
            resumed[2].load_state_dict(state['scheduler'])
            restore_rng(state['rng'])
            visited = list(range(state['cursor']))
            for i in range(state['cursor'], len(data)):
                update(*resumed, [data[i]]); visited.append(i)
            self.assertEqual(visited, list(range(5)))
            self.assertEqual(resumed[2].last_epoch, 5)
            for key in reference:
                torch.testing.assert_close(resumed[0].state_dict()[key], reference[key], rtol=0, atol=0)
            self.assertEqual(random.random(), reference_rng[0])
            self.assertEqual(np.random.rand(), reference_rng[1])
            torch.testing.assert_close(torch.rand(3), reference_rng[2], rtol=0, atol=0)

    def test_resume_rejects_changed_configuration_and_corruption(self):
        m, opt, sch = setup()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'checkpoint-0'
            cfg = {'dataset_sha256': 'a'}
            save_checkpoint(path, m, None, opt, sch, {}, cfg)
            with self.assertRaises(ValueError):
                load_checkpoint(path, {'dataset_sha256': 'b'})
            with (path / 'tiny_adapter.pt').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaises(ValueError):
                load_checkpoint(path, cfg)

    def test_existing_output_protection_and_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'run'
            reserve_output(out, None)
            with self.assertRaises(FileExistsError):
                reserve_output(out, None)
            cp = out / 'checkpoints'
            for i in range(1, 6):
                (cp / f'checkpoint-{i}').mkdir(parents=True)
            prune_checkpoints(cp, 3)
            self.assertEqual(sorted(p.name for p in cp.iterdir()),
                             ['checkpoint-3', 'checkpoint-4', 'checkpoint-5'])
            (out / 'adapter').mkdir()
            with self.assertRaises(FileExistsError):
                reserve_output(out, cp / 'checkpoint-5')


if __name__ == '__main__':
    unittest.main()
