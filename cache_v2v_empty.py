"""Resumable VAE-only per-video cache; one shared T5 empty embedding."""
import argparse, csv, time
import torch
from diffsynth import ModelManager, WanVideoReCamMasterPipeline
from v2v_empty_common import *

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--all',action='store_true'); ap.add_argument('--text-only',action='store_true'); args=ap.parse_args()
    lease=gpu_lease(); torch.set_num_threads(4); torch.set_grad_enabled(False)
    CACHE.mkdir(parents=True,exist_ok=True); RUN.mkdir(parents=True,exist_ok=True)
    text_path=CACHE/'empty_text.pt'
    if args.text_only:
        if text_path.exists():
            t=torch.load(text_path,weights_only=True); assert t['text']=='' and tuple(t['context'].shape)==(1,512,4096)
            event('text_cache_reused',sha256=sha(text_path)); return
        m=ModelManager(torch_dtype=torch.bfloat16,device='cpu')
        m.load_models([str(WEIGHTS/'models_t5_umt5-xxl-enc-bf16.pth')])
        pipe=WanVideoReCamMasterPipeline.from_model_manager(m,device='cuda'); pipe.text_encoder.to('cuda').eval()
        context=pipe.encode_prompt('')['context'].detach().cpu()
        assert context.shape==(1,512,4096) and torch.isfinite(context).all()
        atomic_save(dict(text='',context=context,encoder_sha256=sha(WEIGHTS/'models_t5_umt5-xxl-enc-bf16.pth')),text_path)
        event('text_cache_complete',shape=list(context.shape),bytes=text_path.stat().st_size,sha256=sha(text_path)); return
    scenes=sorted(str(p.parent.parent.relative_to(DATA/'train')) for p in (DATA/'train').glob('*/*/cameras/camera_extrinsics.json')) if args.all else TRAIN+VAL
    rows=[dict(file_name=f'{s}/videos/cam{c:02}.mp4',text='') for s in scenes for c in range(1,11)]
    meta=CACHE/f'metadata_{"all" if args.all else "smoke"}_{args.shard}.csv'
    with meta.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file_name','text']);w.writeheader();w.writerows(rows)
    ds=TextVideoDataset(str(DATA),str(meta),height=384,width=672); ds.text=['']*len(ds) # pandas otherwise converts empty cells to NaN.
    m=ModelManager(torch_dtype=torch.bfloat16,device='cpu'); m.load_models([str(WEIGHTS/'Wan2.1_VAE.pth')])
    pipe=WanVideoReCamMasterPipeline.from_model_manager(m,device='cuda');pipe.vae.to('cuda').eval()
    records=[]; begun=time.time(); selected=list(range(args.shard,len(rows),args.shards))
    for i in selected:
        rel=rows[i]['file_name'];scene=rel.rsplit('/videos/',1)[0];cam=int(Path(rel).stem[3:]);dest=cache_path(scene,cam)
        if dest.exists():
            cached=torch.load(dest,weights_only=True,map_location='cpu')
            assert cached['source_sha256']==sha(DATA/'train'/rel) and cached['preprocess_id']=='official_81_384_672_tiled34x34_stride18x16_bf16_v1'
            assert tuple(cached['latents'].shape)==(16,21,48,84) and torch.isfinite(cached['latents']).all()
            records.append(dict(path=rel,reused=True));continue
        tick=time.time();torch.cuda.reset_peak_memory_stats()
        video=ds.load_video(str(DATA/'train'/rel)) # fail loudly; never use official silent sample fallback.
        assert video is not None and tuple(video.shape)==(3,81,384,672)
        z=pipe.encode_video(video.unsqueeze(0).to('cuda',dtype=torch.bfloat16),tiled=True,tile_size=(34,34),tile_stride=(18,16))[0].detach().cpu()
        assert tuple(z.shape)==(16,21,48,84) and z.dtype==torch.bfloat16 and torch.isfinite(z).all()
        atomic_save(dict(latents=z,source=rel,source_sha256=sha(DATA/'train'/rel),preprocess_id='official_81_384_672_tiled34x34_stride18x16_bf16_v1'),dest)
        r=dict(path=rel,seconds=time.time()-tick,bytes=dest.stat().st_size,allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(r);event('cache_video',shard=args.shard,completed=len(records),total=len(selected),**r)
        atomic_json(dict(status='running',records=records),CACHE/f'cache_shard{args.shard}.json')
    atomic_json(dict(status='complete',seconds=time.time()-begun,records=records),CACHE/f'cache_shard{args.shard}.json')
    event('cache_complete',shard=args.shard,count=len(records),seconds=time.time()-begun)

if __name__=='__main__':main()
