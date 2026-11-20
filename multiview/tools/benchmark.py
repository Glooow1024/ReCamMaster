"""Finite GPU benchmark; updates discarded. Never starts formal training."""
import argparse,json,os,time,threading,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from multiview.model import build_model
from multiview.data.cache import load_example

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--assets',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    def save(name,obj):
        q=out/name;t=q.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(q)
    result={'sources':args.sources,'status':'starting','pid':os.getpid(),'time':time.time(),'formal_training':False}
    save('status.json',result)
    try:
        import fcntl
        lock=Path('/home/users/ly/nvs/recamaster/benchmark_gpu0.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>512:raise RuntimeError(f'GPU0 not idle: {used} MiB')
        torch.set_num_threads(4);torch.manual_seed(20261116);torch.cuda.set_device(0)
        assets=json.loads(Path(args.assets).read_text());begin=time.time()
        data=load_example('f24_aperture5/scene1',1,list(range(2,args.sources+2)),Path(args.assets).parent/'geometry_v1',assets['existing_vae_cache'])
        io_seconds=time.time()-begin
        model=build_model(assets['wan_base'],dtype=torch.float32).cuda().train()
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        # Same precision policy as running reproduction: FP32 parameters/Adam, BF16 AMP.
        clean=data['target'].cuda().float();sources=data['sources'].cuda();rays=data['ray_tokens'].cuda();context=data['context'].cuda()
        noise=torch.randn_like(clean);noisy=.5*(clean+noise);flow=noise-clean
        timestep=torch.tensor([500.],device='cuda',dtype=torch.bfloat16)
        del data
        stop=threading.Event();memory=[]
        def sample_memory():
            while not stop.wait(.5):
                try:memory.append(int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip()))
                except Exception:pass
        thread=threading.Thread(target=sample_memory,daemon=True);thread.start()
        times=[];losses=[];peaks=[]
        for i in range(4):
            result.update(status='running',iteration=i,heartbeat=time.time());save('status.json',result)
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(noisy,sources,timestep,context,rays,use_gradient_checkpointing=True)
                loss=torch.nn.functional.mse_loss(pred.float(),flow)
            loss.backward();torch.cuda.synchronize();fb=time.perf_counter()-tick
            grad=torch.nn.utils.clip_grad_norm_(params,float('inf'))
            if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite loss or gradient')
            tick=time.perf_counter();opt.step();torch.cuda.synchronize();optimizer=time.perf_counter()-tick
            row={'iteration':i,'warmup':i==0,'forward_backward_seconds':fb,'optimizer_seconds':optimizer,'loss':float(loss.detach()),'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
            times.append(row);save('iterations.json',times)
        stop.set();thread.join(timeout=6)
        measured=times[1:]
        result.update(status='complete',heartbeat=time.time(),gpu=torch.cuda.get_device_name(),micro_batch=1,frames=81,resolution=[384,672],latent_frames=21,total_tokens=(args.sources+1)*21*24*42,precision='FP32 parameters and AdamW states; BF16 autocast; gradient checkpointing',iterations=times,mean_forward_backward_seconds=float(np.mean([r['forward_backward_seconds'] for r in measured])),mean_optimizer_seconds=float(np.mean([r['optimizer_seconds'] for r in measured])),max_allocated_gib=max(r['allocated_gib'] for r in times),max_reserved_gib=max(r['reserved_gib'] for r in times),sampled_device_memory_gib=max(memory)/1024 if memory else None,cache_and_ray_prepare_seconds=io_seconds,limitations='single GPU, fixed real sample and t=0.5, 1 warmup+3 measured updates; no DDP communication, checkpoint save, evaluation, asynchronous data pipeline timing; all weights discarded')
        save('result.json',result);save('status.json',result)
    except Exception:
        result.update(status='failed',heartbeat=time.time(),error=traceback.format_exc());save('result.json',result);save('status.json',result);raise

if __name__=='__main__':main()
