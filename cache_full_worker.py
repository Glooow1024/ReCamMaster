"""Full cache with bounded progress records and append-only provenance journal."""
import argparse, json, shutil, time, traceback
import torch
from diffsynth import ModelManager,WanVideoReCamMasterPipeline
from v2v_empty_common import *

ROOT=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--limit',type=int);args=ap.parse_args()
    assert args.shard in range(8)
    import fcntl
    shard_lock=(ROOT/f'cache_shard{args.shard}.lock').open('w');fcntl.flock(shard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=gpu_lease();torch.set_num_threads(4);torch.set_grad_enabled(False)
    contract=json.loads((ROOT/'cache_contract.json').read_text())
    assert sha(ROOT/'source_manifest.jsonl')==contract['source_manifest_sha256']
    assert sha(WEIGHTS/'Wan2.1_VAE.pth')==contract['vae_sha256']
    rows=[json.loads(s) for s in (ROOT/'source_manifest.jsonl').read_text().splitlines()][args.shard::8]
    if args.limit is not None:rows=rows[:args.limit]
    # Reuse the official load_video method without silent __getitem__ fallback.
    ds=TextVideoDataset(str(DATA),str(ROOT/'metadata_train.csv'),height=384,width=672)
    ds.text=['']*len(ds)
    manager=ModelManager(torch_dtype=torch.bfloat16,device='cpu');manager.load_models([str(WEIGHTS/'Wan2.1_VAE.pth')])
    pipe=WanVideoReCamMasterPipeline.from_model_manager(manager,device='cuda');pipe.vae.to('cuda').eval()
    begun=time.time();created=0;reused=0;last=0;total_bytes=0
    journal=ROOT/f'cache_journal_{args.shard}.jsonl'
    for i,r in enumerate(rows):
        tick=time.time();p=cache_path(r['scene'],r['camera'])
        assert shutil.disk_usage(CACHE).free>20*1024**3,'Less than20GiB free; stop without deleting anything'
        if p.exists():
            v=torch.load(p,weights_only=True,map_location='cpu')
            assert v['source_sha256']==r['source_sha256'] and v['preprocess_id']==contract['preprocess_id']
            assert list(v['latents'].shape)==contract['shape'] and v['latents'].dtype==torch.bfloat16 and torch.isfinite(v['latents']).all()
            reused+=1
        else:
            assert sha(DATA/'train'/r['source'])==r['source_sha256'],'Source hash differs from verified archive'
            video=ds.load_video(str(DATA/'train'/r['source']))
            assert video is not None and tuple(video.shape)==(3,81,384,672)
            z=pipe.encode_video(video.unsqueeze(0).to('cuda',dtype=torch.bfloat16),tiled=True,tile_size=(34,34),tile_stride=(18,16))[0].detach().cpu()
            assert list(z.shape)==contract['shape'] and z.dtype==torch.bfloat16 and torch.isfinite(z).all()
            atomic_save(dict(latents=z,source=r['source'],source_sha256=r['source_sha256'],preprocess_id=contract['preprocess_id']),p)
            created+=1
        rec=dict(index=args.shard+8*i,source=r['source'],cache_sha256=sha(p),bytes=p.stat().st_size,seconds=time.time()-tick,time=time.time())
        with journal.open('a') as f:f.write(json.dumps(rec)+'\n')
        total_bytes+=rec['bytes']
        if time.time()-last>=30 or i+1==len(rows):
            elapsed=time.time()-begun;rate=(created/elapsed) if elapsed else 0
            state=dict(status='complete' if i+1==len(rows) else 'running',pid=os.getpid(),shard=args.shard,completed=i+1,total=len(rows),created=created,reused=reused,bytes=total_bytes,seconds=elapsed,generated_videos_per_second=rate,estimated_remaining_seconds=(len(rows)-i-1)/rate if rate else None,last_source=r['source'],time=time.time())
            atomic_json(state,ROOT/f'cache_state_{args.shard}.json');print(json.dumps(state),flush=True);last=time.time()

if __name__=='__main__':
    try:main()
    except Exception as e:
        import sys
        shard=int(sys.argv[sys.argv.index('--shard')+1])
        atomic_json(dict(status='failed',pid=os.getpid(),time=time.time(),error=str(e)),ROOT/f'cache_state_{shard}.json');traceback.print_exc();raise
