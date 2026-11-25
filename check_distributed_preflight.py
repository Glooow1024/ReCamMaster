import json,time
from pathlib import Path
from v2v_empty_common import sha,atomic_json
root=Path('/data/nvs/recammaster/runs/v2v_full_20261114');test=Path('/data/nvs/recammaster/runs/v2v_ddp8x5_fixed_20261114');comparison={};positions=[]
for rank in range(8):
 a=json.loads((test/f'continuous_rank{rank}_fingerprint_step2.json').read_text());b=json.loads((test/f'resumed_rank{rank}_fingerprint_step2.json').read_text())
 comparison[str(rank)]={k:a[k]==b[k] for k in ['records','model','optimizer','cpu_rng','cuda_rng']}
 positions.extend(x['global_position'] for x in a['records'])
 assert all(comparison[str(rank)].values()),f'rank{rank} replay mismatch: {comparison[str(rank)]}'
 assert json.loads((test/f'continuous_rank{rank}_result.json').read_text())['status']=='passed'
 assert json.loads((test/f'resumed_rank{rank}_result.json').read_text())['status']=='passed'
assert set(positions)==set(range(40,80)) and len(positions)==40
hashes=[json.loads((test/f'continuous_rank{r}_fingerprint_step2.json').read_text())['model'] for r in range(8)];assert len(set(hashes))==1
result={'status':'passed','ranks':comparison,'global_batch40_no_duplicate_positions':True,'all_rank_model_identical':True,'time':time.time()}
atomic_json(result,root/'distributed_preflight.json');print(json.dumps(result,indent=2))
