import json, re
from pathlib import Path

src = Path('processed/Point-CoT_filtered.json')
out = Path('processed/Point-CoT_qwen3vl.json')
data = json.loads(src.read_text(encoding='utf-8'))

def image_path(item):
    name = Path(item['image']).name
    source = item['source']
    if source == 'dvqa_train_200k_new.json':
        n = int(Path(name).stem); p = Path('source_images/DVQA/images') / f'bar_train_{n:08d}.png'
    elif source == 'plotqa_new.json':
        n = int(re.search(r'(\d+)\.png$', name).group(1)); p = Path('source_images/PlotQA/Train/png') / f'{n}.png'
    elif 'MathV360K' in source:
        p = Path('source_images/MathV360K/extracted') / name
    elif source == 'tallyqa(cauldron,llava_format)_processed_new.json':
        p = Path('source_images/TallyQA/extracted') / name
    elif source == 'chartqa(cauldron,llava_format)_processed_new.json':
        p = Path('source_images/ChartQA/extracted') / name
    elif source == 'HuggingFaceM4/ChartQA':
        p = Path('source_images/ChartQA/hf_extracted') / name
    else:
        raise ValueError(source)
    return p.resolve()

converted=[]; missing=[]
for i,item in enumerate(data):
    p=image_path(item)
    if not p.exists(): missing.append((i,str(p))); continue
    q=item['question'].replace('<image>','').strip()
    lines=[]
    for j,text in enumerate(item['steps_text']):
        point=item['steps_point'][j]
        desc=item.get('steps_point_description',[])
        d=desc[j] if j < len(desc) else ''
        lines.append(f'Step {j+1}: {text} [point: {point}]' + (f' ({d})' if d else ''))
    answer=f"<think>\n"+'\n'.join(lines)+f"\n</think>\n<answer>{item['original_answer']}</answer>"
    converted.append({'messages':[{'role':'user','content':[{'type':'image','image':str(p)},{'type':'text','text':q}]},{'role':'assistant','content':answer}]})

if missing: raise RuntimeError(f'missing images: {len(missing)}; first={missing[:3]}')
out.write_text(json.dumps(converted,ensure_ascii=False),encoding='utf-8')
print('written',len(converted),'output',out)
