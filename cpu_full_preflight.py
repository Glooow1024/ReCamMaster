import json,torch,numpy as np
from pathlib import Path
from v2v_empty_common import atomic_json
root=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
x=torch.randn(1,21,4).repeat(1,2,1)
a=x.unsqueeze(2).unsqueeze(3).repeat(1,1,30,52,1).reshape(1,-1,4);b=x.repeat_interleave(30*52,dim=1);assert torch.equal(a,b)
assert x.repeat_interleave(24*42,dim=1).shape==(1,42336,4)
scenes=json.loads((root/'train_scenes.json').read_text());assert len(scenes)==13520
for epoch in range(2):
 order=np.random.default_rng(20261114+epoch).permutation(len(scenes));assert len(set(order.tolist()))==len(scenes)
 ranksets=[set(order[r::8].tolist()) for r in range(8)]
 assert sum(len(s) for s in ranksets)==len(set.union(*ranksets))==len(scenes)
rows=[json.loads(l) for l in (root/'source_manifest.jsonl').read_text().splitlines()]
shards=[set(range(r,len(rows),8)) for r in range(8)];assert sum(len(s) for s in shards)==len(set.union(*shards))==136000
r={'dynamic_grid_480_bitwise_equivalent':True,'384_token_shape':[1,42336,4],'train_scenes':len(scenes),'sampler_two_epochs_complete_no_rank_overlap':True,'cache_shards':[len(s) for s in shards],'cache_shards_disjoint_complete':True};atomic_json(r,root/'cpu_preflight.json');print(json.dumps(r))
