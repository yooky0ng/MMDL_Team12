import json
from pathlib import Path
src=Path('processed/Point-CoT_filtered.json'); out=Path('processed/Point-CoT_qwen3vl_integrated.json')
d=json.loads(src.read_text(encoding='utf-8'))
# Reuse the already resolved local image paths from the converted file.
old=json.loads(Path('processed/Point-CoT_qwen3vl.json').read_text(encoding='utf-8'))
for item, conv in zip(d, old):
    image=conv['messages'][0]['content'][0]['image']
    q=item['question'].replace('<image>','').strip()
    lines=[]
    for i,text in enumerate(item['steps_text']):
        point=item['steps_point'][i]
        desc=item.get('steps_point_description',[])
        extra=f" ({desc[i]})" if i < len(desc) and desc[i] else ''
        lines.append(f"Step {i+1}: {text} [point: {point}]{extra}")
    ans='<think>\n'+'\n'.join(lines)+'\n</think>\n<answer>'+item['original_answer']+'</answer>'
    item['qwen3vl']={'image':image,'conversations':[{'from':'human','value':'<image>\n'+q},{'from':'gpt','value':ans}]}
out.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
print('written',len(d),out)
