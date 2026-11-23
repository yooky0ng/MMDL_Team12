"""Shared, fail-closed target construction for training and CPU validation."""
from pathlib import Path
import torch


def messages_for_sample(sample, point_root, mmvu_root):
    conv = sample['qwen3vl']['conversations']
    if len(conv) != 2 or [m['from'] for m in conv] != ['human', 'gpt']:
        raise ValueError(f"{sample['id']}: expected one human/assistant turn")
    # Never use the legacy top-level `answer`: it omits Point-CoT reasoning.
    target = conv[1]['value']
    question = conv[0]['value'].replace('<image>', '').strip()
    if not target.strip() or not question:
        raise ValueError(f"{sample['id']}: empty question or target")
    raw = sample['qwen3vl']['image'].replace('\\', '/')
    if sample['source'] == 'point_cot' and 'source_images/' in raw:
        path = Path(point_root) / raw.split('source_images/', 1)[1]
    elif sample['source'] == 'mmvu':
        path = Path(mmvu_root) / raw
    else:
        path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        {'role': 'user', 'content': [{'type': 'image', 'image': str(path)},
                                     {'type': 'text', 'text': question}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]},
    ]


def prepare_batch(processor, samples, point_root, mmvu_root, max_length=4096):
    conversations = [messages_for_sample(s, point_root, mmvu_root) for s in samples]
    full = processor.apply_chat_template(
        conversations, tokenize=True, return_dict=True, return_tensors='pt',
        padding=True, add_generation_prompt=False)
    prefix = processor.apply_chat_template(
        [c[:1] for c in conversations], tokenize=True, return_dict=True,
        return_tensors='pt', padding=True, add_generation_prompt=True)
    labels = torch.full_like(full['input_ids'], -100)
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    for i, (s, c) in enumerate(zip(samples, conversations)):
        positions = full['attention_mask'][i].nonzero(as_tuple=True)[0]
        tokens = full['input_ids'][i, positions]
        prompt = prefix['input_ids'][i, prefix['attention_mask'][i].bool()]
        if max_length is not None and len(tokens) > max_length:
            raise ValueError(f"{s['id']}: {len(tokens)} tokens > {max_length}; no truncation applied")
        n = len(prompt)
        if len(tokens) <= n or not torch.equal(tokens[:n], prompt):
            raise ValueError(f"{s['id']}: full/prompt prefix mismatch or empty target")
        supervised = tokens[n:]
        decoded = processor.tokenizer.decode(supervised, skip_special_tokens=False)
        expected = c[1]['content'][0]['text'] + '<|im_end|>\n'
        if decoded != expected:
            raise ValueError(f"{s['id']}: supervised tokens do not exactly preserve target and end marker")
        if eos not in supervised.tolist():
            raise ValueError(f"{s['id']}: missing assistant end token")
        labels[i, positions[n:]] = supervised
        if not (labels[i] != -100).any():
            raise ValueError(f"{s['id']}: no supervised tokens")
    full['labels'] = labels
    return full
