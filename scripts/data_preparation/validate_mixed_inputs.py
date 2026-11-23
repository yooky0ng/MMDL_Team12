#!/usr/bin/env python3
"""CPU-only length audit and actual processor/assistant-mask checks; no weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixed_sft_inputs import messages_for_sample, prepare_batch


def summary(values):
    v=sorted(values)
    return {'count':len(v),'mean':sum(v)/len(v),'median':v[len(v)//2],
            'p95':v[int((len(v)-1)*.95)],'max':max(v)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--point-image-root',required=True)
    parser.add_argument('--mmvu-image-root',required=True)
    a=parser.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.dataset.read_text()); original_hash=hashlib.sha256(a.dataset.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    proc=AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct',revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',local_files_only=True)
    ip=proc.image_processor; sizes={}; scans=[]; errors=[]
    for i,s in enumerate(rows):
        try:
            msgs=messages_for_sample(s,a.point_image_root,a.mmvu_image_root)
            path=msgs[0]['content'][0]['image']
            if path not in sizes:
                with Image.open(path) as im:
                    # Processor reads EXIF-oriented RGB images.
                    exif_orientation=im.getexif().get(274)
                    w,h=im.size
                    if exif_orientation in (5,6,7,8):w,h=h,w
                rh,rw=smart_resize(h,w,factor=ip.patch_size*ip.merge_size,
                                  min_pixels=ip.size['shortest_edge'],max_pixels=ip.size['longest_edge'])
                sizes[path]=(rh,rw)
            rh,rw=sizes[path]
            visual=(rh//ip.patch_size)*(rw//ip.patch_size)//(ip.merge_size**2)
            text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
            prompt=proc.apply_chat_template(msgs[:1],tokenize=False,add_generation_prompt=True)
            if not text.startswith(prompt):raise ValueError('text prefix mismatch')
            if text[len(prompt):] != msgs[1]['content'][0]['text']+'<|im_end|>\n':raise ValueError('target string changed')
            ids=proc.tokenizer.encode(text,add_special_tokens=False)
            pre=proc.tokenizer.encode(prompt,add_special_tokens=False)
            image_id=proc.tokenizer.convert_tokens_to_ids(proc.image_token)
            if ids.count(image_id)!=1 or pre.count(image_id)!=1:raise ValueError('expected one image token')
            if ids[:len(pre)]!=pre:raise ValueError('text token prefix mismatch')
            scans.append({'id':s['id'],'source':s['source'],'source_dataset':s['source_dataset'],
                          'total_tokens':len(ids)+visual-1,'assistant_tokens':len(ids)-len(pre),
                          'visual_tokens':visual,'resized_hw':[rh,rw]})
        except Exception as e:errors.append({'id':s['id'],'error':str(e)})
        if (i+1)%1000==0:print(f'Length audit {i+1}/{len(rows)}',flush=True)
    with (a.output_dir/'token_lengths.jsonl').open('w') as f:
        for s in scans:f.write(json.dumps(s)+'\n')
    lookup={r['id']:r for r in rows}; byid={s['id']:s for s in scans}; selected=[]
    sources = sorted({row['source'] for row in rows})
    for source in sources:
        pool=sorted([s for s in scans if s['source']==source],key=lambda s:(s['total_tokens'],s['id']))
        selected.extend(lookup[pool[round(i*(len(pool)-1)/9)]['id']] for i in range(10))
    checks=[]
    for s in selected:
        print('Actual processor check',s['id'],flush=True)
        b=prepare_batch(proc,[s],a.point_image_root,a.mmvu_image_root,max_length=None)
        active=b['attention_mask'][0].bool(); lab=b['labels'][0]; supervised=lab[lab!=-100]
        expected=byid[s['id']]
        grid=b['image_grid_thw']; visual=int(grid.prod(-1).sum())//(ip.merge_size**2)
        assert int(active.sum())==expected['total_tokens']
        assert len(supervised)==expected['assistant_tokens'] and visual==expected['visual_tokens']
        assert b['pixel_values'].shape[0]==int(grid.prod(-1).sum())
        checks.append({'id':s['id'],'source':s['source'],'total_tokens':int(active.sum()),
                       'assistant_tokens':len(supervised),'image_grid_thw':grid.tolist(),
                       'supervised_decoded':proc.tokenizer.decode(supervised,skip_special_tokens=False),
                       'prefix_and_target_checks_passed':True})
        del b
    # A mixed batch with unequal lengths exercises actual padding masking too.
    pair=[selected[0],selected[10]]
    b=prepare_batch(proc,pair,a.point_image_root,a.mmvu_image_root,max_length=None)
    assert (b['attention_mask']==0).any()
    assert (b['labels'][b['attention_mask']==0]==-100).all()
    for i,s in enumerate(pair):
        assert int((b['labels'][i]!=-100).sum())==byid[s['id']]['assistant_tokens']
    assert hashlib.sha256(a.dataset.read_bytes()).hexdigest()==original_hash
    report={'dataset_sha256':original_hash,'model_weights_loaded':False,'training_executed':False,
            'length_method':'Qwen smart_resize + official chat template/tokenizer image-token expansion; cross-checked against actual processor on 20 examples',
            'image_processor':{'patch_size':ip.patch_size,'merge_size':ip.merge_size,'size':dict(ip.size)},
            'scanned':len(scans),'scan_errors':errors,
            'length_statistics':{src:{k:summary([x[k] for x in scans if x['source']==src]) for k in ['total_tokens','assistant_tokens','visual_tokens']} for src in sources},
            'max_length':4096,'overlength_count':sum(x['total_tokens']>4096 for x in scans),
            'overlength_by_source':dict(Counter(x['source'] for x in scans if x['total_tokens']>4096)),
            'processor_examples':len(checks),'padding_masking_passed':True,
            'full_target_roundtrip_passed':True,'dataset_unchanged':True,
            'ready_for_4096_training':len(scans)==len(rows) and not errors and all(x['total_tokens']<=4096 for x in scans)}
    (a.output_dir/'processor_examples.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    (a.output_dir/'input_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
