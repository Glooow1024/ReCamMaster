"""Eight-rank, stateless curriculum trainer. Never starts from zero in an existing run."""
import argparse, contextlib, hashlib, json, os, random, time, traceback
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiview.data.curriculum import GeometrySampler, schedule
from multiview.training.lr_policy import lr_for_update, POLICY
from multiview.training.stability_gate import StabilityGate
from multiview.training.selective_checkpoint import install
install()
from multiview.data.cache import load_example
from multiview.model import build_model
from diffsynth.schedulers.flow_match import FlowMatchScheduler

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def scheduler():
    s=FlowMatchScheduler(shift=5,sigma_min=0,extra_one_step=True);s.set_timesteps(1000,training=True);return s

class Samples:
    def __init__(self,root,seed=20261116,accumulation=2,origin_step=5000,origin_samples=200000):
        self.accumulation=accumulation;self.origin_step=origin_step;self.origin_samples=origin_samples
        self.root=Path(root);self.seed=seed
        q=np.load(self.root/'planning_20k_v1/train_geometry_metrics.npz',allow_pickle=False)
        self.sampler=GeometrySampler(q['metrics'],q['groups'],q['scenes'])
        self.assets=json.loads((self.root/'assets.json').read_text())
    def position(self,step,world=8):
        return self.origin_samples+(step-self.origin_step)*world*self.accumulation
    def plan(self,step,world=8):
        accum=self.accumulation
        # Common source count across ranks/update; deterministic collision-free anchors.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,0]))
        vp,_=schedule(step,self.sampler.natural);n=int(rng.choice(np.arange(1,5),p=vp))
        rows=[];seen=set()
        while len(rows)<world*accum:
            r=self.sampler.anchor(step,rng);key=(r['scene'],r['target'],r['sources'][0])
            if key in seen:continue
            seen.add(key);r['source_count']=n;r['sample_position']=self.position(step,world)+len(rows);rows.append(r)
        return rows
    def load(self,row):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,row['sample_position'],1]))
        with np.load(self.root/'geometry_v1/scenes'/(row['scene'].replace('/','__')+'.npz')) as g:
            bits=np.unpackbits(g['coverage_bits'][2:4],axis=-1).astype(bool)
        r=self.sampler.complete(row,bits,rng)
        data=load_example(r['scene'],r['target']+1,[s+1 for s in r['sources']],self.root/'geometry_v1',self.assets['existing_vae_cache'])
        if data['split']!='train':raise ValueError('non-training sample')
        r['reference_camera']=data['reference_camera'];return data,r

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())

def restore_rng(s):
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda']);random.setstate(s['python']);np.random.set_state(s['numpy'])

def trainable_state(model):
    return {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}

def load_trainable(model,state):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    if names!=set(state):raise ValueError('trainable state key mismatch')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in names:p.copy_(state[n])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run',required=True)
    ap.add_argument('--stop-at',type=int,default=20000);ap.add_argument('--resume');ap.add_argument('--save-every',type=int,default=250)
    ap.add_argument('--diagnostic',action='store_true');ap.add_argument('--eval-every',type=int,default=500)
    args=ap.parse_args();root=Path(args.root);run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8:raise ValueError('this protocol requires eight ranks')
    torch.cuda.set_device(local);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=45))
    status=dict(rank=rank,pid=os.getpid(),status='loading',step=0,time=time.time())
    def heartbeat(**kw):
        status.update(**kw,time=time.time());save_json(run/f'rank{rank}.json',status)
    heartbeat()
    try:
        torch.manual_seed(20261116);random.seed(20261116);np.random.seed(20261116)
        protocol=json.loads((run/'protocol.json').read_text());accum=protocol['accumulation']
        assert accum in (2,5) and protocol['effective_batch']==8*accum
        data=Samples(root,accumulation=accum,origin_step=protocol['sample_cursor_origin']['step'],origin_samples=protocol['sample_cursor_origin']['samples']);model=build_model(data.assets['wan_base'],dtype=torch.float32).cuda()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=4e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        sch=scheduler();start=0
        protocol=json.loads((run/'protocol.json').read_text())
        ck=None
        if args.resume:
            cp=Path(args.resume)
            if not cp.with_suffix('.complete.json').exists():raise ValueError('checkpoint incomplete')
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            if ck['identity']!=protocol['identity']:raise ValueError('checkpoint protocol/code/base/data mismatch')
            load_trainable(model,ck['model']);optimizer.load_state_dict(ck['optimizer']);start=ck['step']
            assert ck['sampler_next_position']==data.position(start)
            assert ck['effective_batch']==world*accum and ck['accumulation']==accum
        elif any(run.glob('train_rank*.jsonl')):raise ValueError('existing log: explicit resume required')
        ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
        if ck:restore_rng(ck['rng'][rank]);del ck
        else:torch.manual_seed(20261116+rank);random.seed(20261116+rank);np.random.seed(20261116+rank)
        log=(run/f'train_rank{rank}.jsonl').open('a',buffering=1)
        log.write(json.dumps(dict(event='resume' if args.resume else 'start',step=start,rank=rank,time=time.time()))+'\n')
        if not args.diagnostic and start==0:
            heartbeat(status='evaluating',step=0)
            if rank==0:
                from multiview.evaluation.probes import evaluate
                saved=rng_state();evaluate(model,root,run,0);restore_rng(saved)
            dist.barrier();model.train()
        def checkpoint(step):
            heartbeat(status='saving',step=step)
            states=[None]*world;dist.all_gather_object(states,rng_state())
            if rank==0:
                folder=run/'checkpoints';folder.mkdir(exist_ok=True);path=folder/f'step{step:06}.pt'
                tmp=path.with_suffix('.tmp')
                torch.save(dict(model=trainable_state(model),optimizer=optimizer.state_dict(),rng=states,step=step,
                                identity=protocol['identity'],sampler_next_position=data.position(step),world_size=world,effective_batch=world*accum,accumulation=accum,
                                lr_scheduler=POLICY,flow_scheduler=dict(shift=5,sigma_min=0,extra_one_step=True),amp_scaler=None),tmp)
                tmp.replace(path);save_json(path.with_suffix('.complete.json'),dict(step=step,size=path.stat().st_size,time=time.time()))
                save_json(run/'latest_checkpoint.json',dict(step=step,path=str(path)))
                # Keep evaluation/milestone checkpoints plus last two transient checkpoints.
                if not args.diagnostic:
                    transient=[p for p in sorted(folder.glob('step*.pt')) if int(p.stem[4:])%args.eval_every!=0]
                    for old in transient[:-2]:old.unlink();old.with_suffix('.complete.json').unlink()
            dist.barrier()
        gate=StabilityGate(run,start)
        pool=ThreadPoolExecutor(max_workers=1)
        pending=None
        if start<args.stop_at:pending=pool.submit(data.load,data.plan(start)[rank])
        for step in range(start,args.stop_at):
            for group in optimizer.param_groups:group['lr']=lr_for_update(step+1)
            begun=time.time();plan=data.plan(step);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            raws=[];weighted=[];rows=[]
            for micro in range(accum):
                heartbeat(status='training',step=step,micro=micro,source_count=plan[0]['source_count'])
                batch,row=pending.result()
                nextrow=plan[(micro+1)*world+rank] if micro<accum-1 else (data.plan(step+1)[rank] if step+1<args.stop_at else None)
                pending=pool.submit(data.load,nextrow) if nextrow is not None else None
                clean=batch['target'].cuda().float();src=batch['sources'].cuda();rays=batch['ray_tokens'].cuda();ctx=batch['context'].cuda()
                del batch
                noise=torch.randn_like(clean);idx=torch.randint(0,1000,(1,));t=sch.timesteps[idx].to(device='cuda',dtype=torch.bfloat16)
                noisy=sch.add_noise(clean,noise,t);weight=sch.training_weight(t).cuda()
                with (ddp.no_sync() if micro<accum-1 else contextlib.nullcontext()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=ddp(noisy,src,t,ctx,rays,True)
                        raw=torch.nn.functional.mse_loss(pred.float(),noise-clean);loss=raw*weight
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                    (loss/accum).backward()
                raws.append(float(raw.detach()));weighted.append(float(loss.detach()));rows.append(dict(**row,timestep_index=int(idx),weight=float(weight)))
                del pred,loss,raw,noisy,noise,clean,src,rays,ctx
            grad=torch.nn.utils.clip_grad_norm_(params,float('inf'))
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient')
            optimizer.step();torch.cuda.synchronize()
            row=dict(event='update',step=step+1,rank=rank,raw_loss=float(np.mean(raws)),weighted_loss=float(np.mean(weighted)),grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],
                     seconds=time.time()-begun,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                     samples=rows,supervised_target_samples=accum,effective_batch=world*accum,seen_target_samples=data.position(step+1),training_revision=protocol["run_id"],time=time.time())
            log.write(json.dumps(row)+'\n');heartbeat(status='training',step=step+1,micro=0,raw_loss=row['raw_loss'],weighted_loss=row['weighted_loss'],seconds=row['seconds'])
            gate.update(step+1,row,rank)
            if (step+1)%args.save_every==0 or step+1==args.stop_at or step+1==5200:checkpoint(step+1)
            if not args.diagnostic and (step+1)%args.eval_every==0:
                heartbeat(status='evaluating',step=step+1)
                if rank==0:
                    from multiview.evaluation.probes import evaluate
                    saved=rng_state();evaluate(model,root,run,step+1);restore_rng(saved)
                dist.barrier();model.train()
        pool.shutdown()
        if not args.diagnostic:
            heartbeat(status='final_evaluating',step=args.stop_at)
            if rank==0:
                from multiview.evaluation.probes import evaluate
                evaluate(model,root,run,args.stop_at,split='test')
            dist.barrier()
        heartbeat(status='complete' if args.diagnostic else 'training_complete',step=args.stop_at)
        log.close();dist.barrier()
    except Exception:
        heartbeat(status='failed',error=traceback.format_exc());raise
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
