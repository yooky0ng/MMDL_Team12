"""Merge the trained adapter and compare deterministic multimodal inference."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

root=Path('outputs/mixed_sft_lora')
out=root/'merged_model'
if out.exists(): raise FileExistsError(out)
revision='ebb281ec70b05090aa6165b016eac8ec08e71b17'
processor=AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct',revision=revision,local_files_only=True)
base=Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-4B-Instruct',revision=revision,local_files_only=True,dtype=torch.bfloat16,device_map='cuda')
model=PeftModel.from_pretrained(base,root/'adapter').eval()
rows=json.load(open('datasets/mixed_sft_3k/mixed_30k.json'))
samples=[next(x for x in rows if x['source']==s) for s in ('point_cot','mmvu')]
batches=[]
for s in samples:
 raw=s['image'].replace('\\','/')
 image=Path('datasets/point_cot/source_images')/raw.split('source_images/',1)[1] if s['source']=='point_cot' else Path('/home/hyk/mmvu_48k_images')/raw
 msg=[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':s['question'].replace('<image>','')}]}]
 b=processor.apply_chat_template(msg,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True)
 batches.append({k:v.to('cuda') for k,v in b.items()})
def infer(m):
 with torch.inference_mode():
  return [m.generate(**b,max_new_tokens=64,do_sample=False)[:,b['input_ids'].shape[1]:].cpu() for b in batches]
print('Adapter inference',flush=True)
before=infer(model)
print('Merging',flush=True)
merged=model.merge_and_unload(safe_merge=True).eval()
after=infer(merged)
report={'revision':revision,'tests':[{'source':s['source'],'same_tokens':torch.equal(x,y),'adapter_output':processor.decode(x[0],skip_special_tokens=True),'merged_output':processor.decode(y[0],skip_special_tokens=True)} for s,x,y in zip(samples,before,after)]}
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
merged.save_pretrained(out); processor.save_pretrained(out)
json.dump(report,open(root/'merge_validation.json','w'),ensure_ascii=False,indent=2)
print('SAVED',out,flush=True)
