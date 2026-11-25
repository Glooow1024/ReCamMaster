"""Two-rank V2V training with deterministic global sampling and complete restart."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, contextlib, hashlib, json, random, time
import numpy as np
import torch
import torch.distributed as dist
from train_recammaster import LightningModelForTrain
from v2v_empty_common import *

PREP=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
def tensor_hash(named):
    h=hashlib.sha256()
    for n,t in named:
        h.update(str(n).encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng():return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore(d):
    random.setstate(d['py']);np.random.set_state(d['np']);torch.set_rng_state(d['cpu']);torch.cuda.set_rng_state(d['cuda'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--resume',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--tag',default='continuous');args=ap.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl');seed=20261114;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    run=args.run;run.mkdir(parents=True,exist_ok=True)
    proto=json.loads((run/'protocol.json').read_text(encoding='utf-8-sig'));accum=proto['gradient_accumulation'];assert accum*world==40
    scenes=TRAIN if args.smoke else json.loads((PREP/'train_scenes.json').read_text())
    heldout=VAL if args.smoke else [next(s for s in json.loads((PREP/'validation_scenes.json').read_text()) if s.startswith(g+'/')) for g in GROUPS]
    identity=dict(protocol=sha(run/'protocol.json'),code=sha(__file__),dit_code=sha(Path(__file__).parent/'diffsynth/models/wan_video_dit.py'),cache_contract=sha(PREP/'cache_contract.json'),shared_text=sha(CACHE/'empty_text.pt'),world_size=world,smoke=args.smoke)
    if not args.smoke:
        assert json.loads((PREP/'cache_verified.json').read_text())['status']=='verified'
    with (run/f'{args.tag}_rank{rank}.jsonl').open('x') as log:
        def emit(kind,**kw):
            r=dict(time=time.time(),kind=kind,rank=rank,tag=args.tag,**kw);log.write(json.dumps(r)+'\n');log.flush()
            atomic_json(r,run/f'rank{rank}_state.json')
            if rank==0:print(json.dumps(r),flush=True)
        wrapper=LightningModelForTrain(str(WEIGHTS/'diffusion_pytorch_model.safetensors'),learning_rate=proto['lr'],use_gradient_checkpointing=True)
        pipe=wrapper.pipe;pipe.device='cuda';model=pipe.dit.float().to(local).train();trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad];frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
        assert not any(p.requires_grad for n,p in model.named_parameters() if 'cross_attn' in n)
        context=torch.load(CACHE/'empty_text.pt',weights_only=True,map_location='cpu')['context'].to(local)
        opt=torch.optim.AdamW([p for n,p in trainable],lr=proto['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        start=0;loaded_rng=None
        if args.resume:
            receipt=json.loads(args.resume.with_suffix('.json').read_text());assert sha(args.resume)==receipt['sha256']
            ck=torch.load(args.resume,weights_only=False,map_location='cpu');assert ck['identity']==identity
            model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['lr_scheduler']);pipe.scheduler.__dict__.update(ck['noise_scheduler']);start=ck['step'];assert ck['cursor']==start*40;loaded_rng=ck['rng_by_rank'][rank];del ck
        # Fixed-order data-parallel reduction avoids DDP bucket-rebuild differences
        # between an uninterrupted step and the first step of a new process.
        ddp=model
        if loaded_rng is not None:restore(loaded_rng)
        else:
            torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
        frozen_hash=tensor_hash(frozen);initial_train_hash=tensor_hash(trainable)
        initial_hashes=[None]*world;dist.all_gather_object(initial_hashes,(frozen_hash,initial_train_hash))
        assert len(set(initial_hashes))==1, 'Ranks must start from identical weights'
        emit('started',step=start,world_size=world,accumulation=accum,identity=identity)
        def forward_loss(x,cam,tidx=None,use_ddp=True,ckpt=True):
            noise=torch.randn_like(x)
            if tidx is None:tidx=int(torch.randint(0,1000,(1,)).item())
            t=pipe.scheduler.timesteps[tidx:tidx+1].to(local,dtype=torch.bfloat16)
            z=pipe.scheduler.add_noise(x,noise,t);z[:,:,21:]=x[:,:,21:]
            target=pipe.scheduler.training_target(x,noise,t)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=(ddp if use_ddp else model)(z,timestep=t,cam_emb=cam,context=context,use_gradient_checkpointing=ckpt)
                raw=torch.nn.functional.mse_loss(pred[:,:,:21].float(),target[:,:,:21].float());weight=pipe.scheduler.training_weight(t);loss=raw*weight
            return loss,raw,weight,tidx
        def validate(step):
            result=[]
            for j,scene in enumerate(heldout):
                if j%world!=rank:continue
                x,cam=batch(scene,1,2)
                for tidx in ([500] if args.smoke else [100,400,700,900]):
                    with torch.random.fork_rng(devices=[local]):
                        torch.manual_seed(310000+j*1000+tidx);torch.cuda.manual_seed(310000+j*1000+tidx)
                        with torch.no_grad():loss,raw,weight,_=forward_loss(x,cam,tidx,False,False)
                    assert torch.isfinite(loss);result.append(dict(scene=scene,timestep_index=tidx,raw=float(raw),weight=float(weight),loss=float(loss)))
            gathered=[None]*world;dist.all_gather_object(gathered,result)
            if rank==0:
                flat=sum(gathered,[]);v=dict(step=step,probes=flat,mean_weighted_mse=float(np.mean([a['loss'] for a in flat])))
                atomic_json(v,run/f'{args.tag}_validation_step{step}.json');emit('validation',**v)
            dist.barrier()
        def save(step):
            states=[None]*world;dist.all_gather_object(states,rng())
            if rank==0:
                path=run/'checkpoints'/f'step{step:06}.pt'
                atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),lr_scheduler=scheduler.state_dict(),noise_scheduler=pipe.scheduler.__dict__,rng_by_rank=states,step=step,cursor=step*40,epoch=(step*40)//len(scenes),amp_scaler=None),path)
                atomic_json(dict(step=step,sha256=sha(path),bytes=path.stat().st_size),path.with_suffix('.json'));atomic_json(dict(path=str(path),step=step),run/'latest_checkpoint.json');emit('checkpoint_complete',step=step,path=str(path))
            dist.barrier()
        def fingerprint(step,records):
            value=dict(rank=rank,step=step,records=records,model=tensor_hash(trainable),optimizer=tensor_hash((f'{i}:{k}',v) for i,s in enumerate(opt.state.values()) for k,v in sorted(s.items()) if torch.is_tensor(v)),cpu_rng=tensor_hash([('rng',torch.get_rng_state())]),cuda_rng=tensor_hash([('rng',torch.cuda.get_rng_state())]))
            atomic_json(value,run/f'{args.tag}_rank{rank}_fingerprint_step{step}.json')
        if not args.resume:validate(0)
        orders={}
        for step in range(start,args.steps):
            tick=time.time();torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);records=[];sums=np.zeros(4)
            for micro in range(accum):
                pos=step*40+micro*world+rank;epoch,index=divmod(pos,len(scenes))
                if epoch not in orders:orders={epoch:np.random.default_rng(seed+epoch).permutation(len(scenes))}
                scene=scenes[int(orders[epoch][index])];target,cond=random.Random(seed+pos).sample(range(1,11),2)
                x,cam=batch(scene,target,cond)
                with contextlib.nullcontext():
                    loss,raw,weight,tidx=forward_loss(x,cam)
                    assert torch.isfinite(loss);(loss/accum).backward()
                rec=dict(global_position=pos,scene=scene,target=target,condition=cond,timestep_index=tidx,loss=float(loss.detach()),raw=float(raw.detach()),weight=float(weight))
                records.append(rec);sums+=np.array([rec['raw'],rec['loss'],rec['weight'],int(rec['weight']>0)])
                emit('microbatch',step=step+1,micro=micro+1,accumulation=accum,**rec)
            # All ranks concatenate exactly the same parameter order on every update.
            assert all(p.grad is not None for n,p in trainable)
            flat=torch.cat([p.grad.reshape(-1) for n,p in trainable]);dist.all_reduce(flat);flat.div_(world)
            offset=0
            for n,p in trainable:
                p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
            del flat
            group_norms={}
            for n,p in trainable:
                assert p.grad is not None and torch.isfinite(p.grad).all(),n
                group='camera' if 'cam_encoder' in n else 'projector' if 'projector' in n else 'self_attn'
                group_norms[group]=group_norms.get(group,0.)+float(p.grad.square().sum())
            group_norms={k:v**.5 for k,v in group_norms.items()};assert all(np.isfinite(v) and v>0 for v in group_norms.values())
            assert all(p.grad is None for n,p in frozen)
            opt.step();scheduler.step();torch.cuda.synchronize()
            totals=torch.tensor(sums,device=local);dist.all_reduce(totals)
            emit('update',step=step+1,cursor=(step+1)*40,loss=float(totals[1]/40),raw_loss=float(totals[0]/40),effective_supervision=int(totals[3]),lr=opt.param_groups[0]['lr'],grad_norms=group_norms,seconds=time.time()-tick,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            if args.smoke:fingerprint(step+1,records)
            if (args.smoke and not args.resume) or (not args.smoke and ((step+1)%50==0 or step+1==args.steps)):save(step+1)
            if not args.smoke and (step+1)%50==0:validate(step+1)
        if args.smoke and not args.resume:validate(args.steps)
        assert tensor_hash(frozen)==frozen_hash and tensor_hash(trainable)!=initial_train_hash
        atomic_json(dict(status='passed',rank=rank,start=start,end=args.steps,frozen_unchanged=True,trainable_changed=True),run/f'{args.tag}_rank{rank}_result.json');emit('complete',step=args.steps)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
