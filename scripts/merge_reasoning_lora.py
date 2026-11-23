#!/usr/bin/env python3
"""Merge a completed training run, compare inference, save and independently reload.

No training and no MMMU evaluation. Existing adapters/models are never overwritten.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from mixed_sft_inputs import messages_for_sample


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args(); root=a.run_dir.resolve()
    out=root/'merged_model'; pending=root/'.merged_model.pending'
    if out.exists() or pending.exists():
        raise FileExistsError(f'Refusing to overwrite: {out} or {pending}')
    summary=json.loads((root/'training_summary.json').read_text()); cfg=summary['config']
    if summary['status']!='completed' or summary['samples']!=10000 or summary['updates']!=625:
        raise ValueError('Expected completed 10K/625-update run')
    if cfg['target_field']!='qwen3vl.conversations[1].value':
        raise ValueError('Wrong training target field')
    if sha(cfg['dataset'])!=cfg['dataset_sha256']:
        raise ValueError('Dataset differs from completed training input')
    adapter=root/'adapter'; before_hash=sha(adapter/'adapter_model.safetensors')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.use_deterministic_algorithms(True)
    processor=AutoProcessor.from_pretrained(adapter,local_files_only=True)
    base=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True,dtype=torch.bfloat16,device_map={'':0})
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
    injected=[n for n,m in model.named_modules() if hasattr(m,'lora_A')]
    if len(injected)!=144 or any('.language_model.layers.' not in n or '.self_attn.' not in n for n in injected):
        raise ValueError('Adapter injection scope differs from LLM-only training')
    rows=json.loads(Path(cfg['dataset']).read_text())
    samples=[next(x for x in rows if x['source']==s) for s in ['point_cot','mmvu']]
    messages=[messages_for_sample(s,cfg['point_image_root'],cfg['mmvu_image_root'])[:1] for s in samples]
    def inputs(proc):
        return [{k:v.to('cuda:0') for k,v in proc.apply_chat_template(
            m,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True).items()}
            for m in messages]
    batches=inputs(processor)
    @torch.inference_mode()
    def infer(m,bs):
        tokens=[]
        for batch in bs:
            generated=m.generate(**batch,max_new_tokens=64,do_sample=False,
                                 temperature=None,top_p=None,top_k=None)
            tokens.append(generated[:,batch['input_ids'].shape[1]:].cpu())
        return tokens
    print('1. Original revision + completed 10K adapter loaded',flush=True)
    before=infer(model,batches)
    print('2. safe_merge started',flush=True)
    merged=model.merge_and_unload(safe_merge=True).eval()
    if any('lora_' in n for n,_ in merged.named_parameters()):
        raise ValueError('LoRA parameters remain after merge')
    after=infer(merged,batches)
    report={'base_model':cfg['model'],'base_revision':cfg['revision'],
            'training_samples':summary['samples'],'adapter_path':str(adapter),
            'adapter_sha256':before_hash,'output_path':str(out),'dtype':'bfloat16',
            'max_new_tokens_for_smoke':64,'mmmu_evaluation_executed':False,'tests':[]}
    for s,b,t in zip(samples,before,after):
        report['tests'].append({'source':s['source'],'id':s['id'],
            'adapter_vs_merged_tokens_equal':torch.equal(b,t),
            'adapter_token_ids':b[0].tolist(),'merged_token_ids':t[0].tolist(),
            'adapter_output':processor.decode(b[0],skip_special_tokens=False),
            'merged_output':processor.decode(t[0],skip_special_tokens=False)})
    print('3. Saving standalone merged model',flush=True)
    pending.mkdir()
    merged.save_pretrained(pending,safe_serialization=True)
    processor.save_pretrained(pending)
    # Drop every reference to the original/adapter model before independent reload.
    del model,base,merged
    gc.collect(); torch.cuda.empty_cache()
    print('4. Reloading saved model without PEFT/adapter',flush=True)
    reloaded=Qwen3VLForConditionalGeneration.from_pretrained(
        pending,local_files_only=True,dtype=torch.bfloat16,device_map={'':0}).eval()
    # Explicitly disable a known false-positive Mistral regex patch for this Qwen tokenizer.
    reproc=AutoProcessor.from_pretrained(pending,local_files_only=True,fix_mistral_regex=False)
    rebatches=inputs(reproc)
    for old,new in zip(batches,rebatches):
        if not torch.equal(old['input_ids'],new['input_ids']):
            raise ValueError('Tokenizer input changed after save/reload')
    reoutputs=infer(reloaded,rebatches)
    for test,t,r in zip(report['tests'],after,reoutputs):
        test['merged_vs_reloaded_tokens_equal']=torch.equal(t,r)
        test['reloaded_output']=reproc.decode(r[0],skip_special_tokens=False)
        if r.numel()==0 or not test['merged_vs_reloaded_tokens_equal']:
            raise ValueError('Saved model reload inference mismatch')
    if sha(adapter/'adapter_model.safetensors')!=before_hash:
        raise ValueError('Adapter modified during merge')
    report['adapter_unchanged']=True
    report['standalone_reload_without_adapter']=True
    report['all_merge_generation_tests_equal']=all(t['adapter_vs_merged_tokens_equal'] for t in report['tests'])
    report['all_reload_generation_tests_equal']=True
    if not report['all_merge_generation_tests_equal']:
        report['warning']='BF16 merge generated different tokens; inspect before evaluating. Reload output is consistent.'
    out_parent=out.parent
    if out.exists(): raise FileExistsError(out)
    pending.rename(out)
    (root/'merge_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (root/'merged_model_checksums.json').write_text(json.dumps(
        {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('DONE:',out,flush=True)

if __name__=='__main__':main()
