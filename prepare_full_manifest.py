"""Freeze split and source hashes before full-cache workers start."""
import csv, hashlib, json
from pathlib import Path
from v2v_empty_common import DATA, WEIGHTS, CACHE, sha, atomic_json

ROOT=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
ROOT.mkdir(parents=True,exist_ok=True)
source=DATA.parent.parent/'MultiCamVideo-download/extracted_manifest.jsonl'
rows=[]
for line in source.read_text().splitlines():
    r=json.loads(line);prefix='MultiCamVideo-Dataset/train/'
    if r['path'].startswith(prefix) and r['path'].endswith('.mp4'):
        rel=r['path'][len(prefix):];rows.append(dict(source=rel,source_sha256=r['sha256'],scene=rel.split('/videos/')[0],camera=int(Path(rel).stem[3:])))
rows.sort(key=lambda x:x['source']);assert len(rows)==136000
ids=sorted(range(1,3400),key=lambda n:hashlib.sha256(f'recam_split_20261114:{n}'.encode()).hexdigest())
val_ids=set([3400]+ids[:9]);test_ids=set(ids[9:19])
for r in rows:
    scene_id=int(r['scene'].split('/scene')[1]);r['split']='validation' if scene_id in val_ids else 'test' if scene_id in test_ids else 'train'
path=ROOT/'source_manifest.jsonl'
with path.open('x') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
for split in ['train','validation','test']:
    subset=[r for r in rows if r['split']==split]
    with (ROOT/f'metadata_{split}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file_name','text']);w.writerows((r['source'],'') for r in subset)
    atomic_json(sorted(set(r['scene'] for r in subset)),ROOT/f'{split}_scenes.json')
contract=dict(version=1,source_manifest_sha256=sha(path),vae_sha256=sha(WEIGHTS/'Wan2.1_VAE.pth'),text_sha256=sha(CACHE/'empty_text.pt'),preprocess_id='official_81_384_672_tiled34x34_stride18x16_bf16_v1',shape=[16,21,48,84],dtype='bfloat16',cache_root=str(CACHE),data_root=str(DATA),counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']})
atomic_json(contract,ROOT/'cache_contract.json');print(json.dumps(contract,indent=2))
