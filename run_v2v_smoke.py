"""Finite-lifetime supervisor. No auto-restart or silent configuration changes."""
import json, os, subprocess, sys, time, traceback
from v2v_empty_common import *

def main():
    manifest=[]
    for scene in TRAIN+VAL:
        for c in range(1,11):
            p=cache_path(scene,c);v=torch.load(p,weights_only=True,map_location='cpu')
            assert v['source_sha256']==sha(DATA/'train'/v['source'])
            assert v['latents'].shape==(16,21,48,84) and torch.isfinite(v['latents']).all()
            manifest.append(dict(source=v['source'],path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    atomic_json(manifest,RUN/'cache_manifest.json')
    atomic_json({p.name:sha(p) for p in WEIGHTS.iterdir() if p.suffix in ('.pth','.safetensors')},RUN/'initial_weight_hashes.json')
    atomic_json({p.name:sha(p) for p in Path('.').glob('*v2v*.py')},RUN/'code_hashes.json')
    atomic_json(dict(status='running',stage='preflight',time=time.time()),RUN/'supervisor.json')
    for tag,extra in [('continuous',[]),('resumed',['--resume',str(RUN/'checkpoints/continuous_step4.pt'),'--stop','6','--tag','resumed'])]:
        with (RUN/f'{tag}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u','train_v2v_empty.py',*extra],stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                mem=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                atomic_json(dict(status='running',stage=tag,pid=proc.pid,time=time.time(),gpu6=mem),RUN/'supervisor.json')
                time.sleep(15)
            if proc.returncode:raise RuntimeError(f'{tag} exited {proc.returncode}; see {tag}.log; no automatic restart')
    a=json.loads((RUN/'continuous_step6_fingerprint.json').read_text());b=json.loads((RUN/'resumed_step6_fingerprint.json').read_text())
    keys=['step','cursor','scene','target','condition','timestep_index','loss','raw_loss','weight','grad_norms','probe_update_max']
    comparison={k:a['record'][k]==b['record'][k] for k in keys}
    comparison.update(model_bitwise_equal=a['model_sha256']==b['model_sha256'],optimizer_bitwise_equal=a['optimizer_sha256']==b['optimizer_sha256'],torch_rng_equal=a['rng']==b['rng'])
    # Check both actually replayed updates, not just the final cursor.
    continuous=json.loads((RUN/'continuous_result.json').read_text());resumed=json.loads((RUN/'resumed_result.json').read_text())
    comparison['both_replayed_updates_equal']=all(all(x[k]==y[k] for k in keys) for x,y in zip(continuous['records'][4:6],resumed['records']))
    atomic_json(comparison,RUN/'resume_comparison.json')
    assert all(comparison.values()), 'Resume comparison differs; investigate, do not report exact restart'
    atomic_json(dict(status='passed',stage='finished',time=time.time(),monitor_stopped=True,no_permanent_tasks=True,external_notification='not_configured',cache_videos=len(manifest),vae_bytes=sum(x['bytes'] for x in manifest),shared_text_bytes=(CACHE/'empty_text.pt').stat().st_size),RUN/'supervisor.json')
    event('smoke_verified_complete',exact_resume=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(dict(status='failed',stage='needs_diagnosis',time=time.time(),error=str(exc),monitor_stopped=True),RUN/'supervisor.json')
        event('incident',error=str(exc));traceback.print_exc();raise
