import json,math,time,statistics
from pathlib import Path
import torch
import torch.distributed as dist

class StabilityGate:
    def __init__(self,run,start):
        self.run=Path(run);self.rows=[]
        p=self.run/'stability_gate.json'
        if p.exists():self.rows=[x for x in json.loads(p.read_text()).get('rows',[]) if x['step']<=start]
    def update(self,step,row,rank):
        v=torch.tensor([row['weighted_loss'],row['raw_loss'],row['grad_norm']],device='cuda',dtype=torch.float64)
        dist.all_reduce(v);v/=dist.get_world_size()
        self.rows.append(dict(step=step,loss=v[0].item(),raw=v[1].item(),grad=v[2].item()))
        tail=self.rows[-50:];last10=self.rows[-10:]
        failure=None
        if len(last10)==10 and statistics.mean(x['loss'] for x in last10)>0.23:failure='10-update mean loss exceeds 5x stable BS40 baseline'
        if step==5200:
            if len(tail)!=50:failure='insufficient validation history'
            elif statistics.mean(x['loss'] for x in tail)>.092:failure='last50 mean exceeds 2x stable loss'
            elif statistics.mean(x['grad'] for x in tail)>.268:failure='last50 gradient exceeds 4x stable gradient'
            elif statistics.mean(x['loss'] for x in tail[-25:])>1.5*statistics.mean(x['loss'] for x in self.rows[:25]):failure='sustained upward loss trend'
        if rank==0 and (step<=5200 or failure):
            record=dict(status='failed' if failure else ('passed' if step==5200 else 'validating'),step=step,time=time.time(),failure=failure,rows=self.rows,criteria={'baseline_loss':.045934,'baseline_grad':.067063,'last50_loss_max':.092,'last50_grad_max':.268,'early_last10_loss_max':.23},note='Empirical safety guards, not universal convergence thresholds')
            p=self.run/'stability_gate.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(p)
        if failure:raise RuntimeError('STABILITY_GATE_FAILED: '+failure)
