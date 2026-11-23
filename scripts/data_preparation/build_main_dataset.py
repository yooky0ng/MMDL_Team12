import argparse, gzip, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

def norm_source(s):
    s=s.lower()
    if 'chartqa' in s: return 'ChartQA'
    if 'dvqa' in s: return 'DVQA'
    if 'plotqa' in s: return 'PlotQA'
    if 'tallyqa' in s: return 'TallyQA'
    if 'clevr' in s or 'super-clevr' in s or 'mathv360k' in s and 'clevr' in s: return 'CLEVR'
    return None

def main(a):
    rng=random.Random(a.seed); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    point=json.load(open(a.point,encoding='utf-8')); target={'ChartQA':9960,'DVQA':8520,'PlotQA':4752,'CLEVR':456,'TallyQA':312}
    groups=defaultdict(list); unknown=[]; stats=Counter()
    for i,x in enumerate(point):
        src=norm_source(x.get('source','')); stats[src or 'unclassified']+=1
        if src: groups[src].append((i,x))
        else: unknown.append(i)
    selected=[]; excluded=[]
    for src,n in target.items():
        if len(groups[src])<n: raise RuntimeError(f'{src} has {len(groups[src])}, needs {n}')
        pool=groups[src][:]; rng.shuffle(pool); selected.extend(x for _,x in pool[:n])
    point_out=[]; manifest=[]
    for j,x in enumerate(selected):
        q=x['qwen3vl']; item={'id':f'point_cot_{j:05d}','source':'point_cot','source_dataset':norm_source(x['source']),'original_id':x.get('intra_source_id'),'image':q['image'],'question':x['question'],'answer':x['original_answer'],'qwen3vl':q}
        point_out.append(item); manifest.append({'id':item['id'],'source':'point_cot','source_dataset':item['source_dataset'],'original_id':item['original_id'],'image':q['image']})
    with gzip.open(a.mmvu,'rt',encoding='utf-8') as f: mmvu=json.load(f)
    mgroup=defaultdict(list)
    for x in mmvu: mgroup[(x['id'],x['image'])].append(x)
    pairs=[(k,v) for k,v in mgroup.items() if len(v)>=2]
    rng.shuffle(pairs); chosen=pairs[:3000]
    if len(chosen)<3000: raise RuntimeError(f'MMVU unlabeled groups only {len(chosen)}')
    mmvu_out=[]
    for pi,((pid,image),rows) in enumerate(chosen):
        for mi,x in enumerate(rows[:2]):
            conv=x['conversations']; q=conv[0]['value']; ans=conv[1]['value']; item={'id':f'mmvu_{pi:05d}_{mi}','source':'mmvu','source_dataset':'MMVU','original_id':pid,'pair_id':f'mmvu_pair_{pi:05d}','pair_label':'unlabeled','pair_member':mi,'image':image,'question':q,'answer':ans,'qwen3vl':{'image':image,'conversations':[{'from':'human','value':q if '<image>' in q else '<image>\n'+q},{'from':'gpt','value':'<answer>'+ans+'</answer>'}]}}
            mmvu_out.append(item); manifest.append({'id':item['id'],'source':'mmvu','original_id':pid,'pair_id':item['pair_id'],'pair_label':'unlabeled','pair_member':mi,'image':image})
    mixed=point_out+mmvu_out; rng.shuffle(mixed)
    json.dump(point_out,open(out/'point_cot_24k.json','w'),ensure_ascii=False,indent=2); json.dump(mmvu_out,open(out/'mmvu_6k.json','w'),ensure_ascii=False,indent=2); json.dump(mixed,open(out/'mixed_30k.json','w'),ensure_ascii=False,indent=2)
    with open(out/'sample_manifest.jsonl','w') as f:
        for x in manifest:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    stats_out={'seed':a.seed,'point_cot_total':len(point),'point_cot_source_distribution':dict(stats),'point_cot_selected':Counter(x['source_dataset'] for x in point_out),'mmvu_input_records':len(mmvu),'mmvu_unlabeled_groups':len(mgroup),'mmvu_selected_pairs':3000,'mmvu_selected_samples':len(mmvu_out),'mixed_total':len(mixed),'pair_labels':'unlabeled; source data has no positive/negative field'}
    json.dump(stats_out,open(out/'dataset_statistics.json','w'),ensure_ascii=False,indent=2,default=lambda x:dict(x))
    missing=[]; bad=[]; dup=set(); seen=set()
    for x in mixed:
        if x['id'] in seen: dup.add(x['id'])
        seen.add(x['id'])
        if not x['question'].strip() or not x['answer'].strip(): bad.append(x['id'])
        raw=x['image'].replace('\\','/')
        root=a.point_image if x['source']=='point_cot' else a.mmvu_image
        if x['source']=='point_cot': path=Path(root)/raw.split('source_images/',1)[-1] if 'source_images/' in raw else Path(raw)
        else: path=Path(root)/raw
        if not path.is_file(): missing.append({'id':x['id'],'path':str(path)})
    report={'point_cot_count':len(point_out),'mmvu_count':len(mmvu_out),'mixed_count':len(mixed),'missing_images':missing,'empty_text':bad,'duplicate_ids':sorted(dup),'pair_integrity':len(chosen)==3000 and all(len(v)>=2 for _,v in chosen),'pair_labels_available':False,'processor_validation':'schema/image validation only; model weights not loaded'}
    json.dump(report,open(out/'validation_report.json','w'),ensure_ascii=False,indent=2)
    print(json.dumps({'point':len(point_out),'mmvu':len(mmvu_out),'mixed':len(mixed),'missing':len(missing),'empty':len(bad),'duplicates':len(dup),'mmvu_pairs':len(chosen)},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--point',default='datasets/point_cot/Point-CoT_qwen3vl_datasets.json'); p.add_argument('--mmvu',default='/home/hyk/mmdl_point_cot/MMDL_Team12/datasets/mmvu_48k/train_48k.json.gz'); p.add_argument('--point-image',default='datasets/point_cot/source_images'); p.add_argument('--mmvu-image',default='/home/hyk/mmvu_48k_images'); p.add_argument('--output-dir',default='datasets/mixed_sft_3k'); p.add_argument('--seed',type=int,default=42); main(p.parse_args())
