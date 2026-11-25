import json,hashlib
from pathlib import Path
import imageio,numpy as np
from PIL import Image,ImageDraw
from v2v_empty_common import DATA,GROUPS,atomic_json
root=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
scenes=json.loads((root/'validation_scenes.json').read_text());rows=[];images={}
for scene in scenes:
 reader=imageio.get_reader(str(DATA/'train'/scene/'videos/cam01.mp4'))
 try:frame=reader.get_data(40)
 finally:reader.close()
 rows.append({'scene':scene,'mean_luminance':float((frame[...,0]*.2126+frame[...,1]*.7152+frame[...,2]*.0722).mean())});images[scene]=Image.fromarray(frame)
chosen=[];used_ids=set()
for group in GROUPS:
 candidates=[r for r in rows if r['scene'].startswith(group+'/')];median=float(np.median([r['mean_luminance'] for r in candidates]))
 candidates.sort(key=lambda r:(abs(r['mean_luminance']-median),r['scene']))
 c=next(r for r in candidates if r['scene'].split('/')[-1] not in used_ids);chosen.append(c['scene']);used_ids.add(c['scene'].split('/')[-1])
ordered=chosen+[s for s in scenes if s not in chosen]
atomic_json(ordered,root/'validation_scenes.json');atomic_json({'method':'before training, near median mid-frame luminance within each focal group; unique numeric scene IDs across groups','candidates':rows,'chosen':chosen},root/'validation_probe_selection.json')
proto=json.loads((root/'protocol.json').read_text());proto['validation_probe_scenes']=chosen;proto['validation_probe_selection']='pretraining median-luminance per focal group, unique scene IDs; not selected using trained model output';atomic_json(proto,root/'protocol.json')
canvas=Image.new('RGB',(4*336,360),'white');draw=ImageDraw.Draw(canvas)
for i,s in enumerate(chosen):
 im=images[s];im.thumbnail((336,336));canvas.paste(im,(i*336,24));draw.text((i*336+2,4),s,fill='black')
canvas.save(root/'validation_probes.png');print(json.dumps({'chosen':chosen,'luminance':[next(r['mean_luminance'] for r in rows if r['scene']==s) for s in chosen]}))
