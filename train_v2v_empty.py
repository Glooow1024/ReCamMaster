"""Auditable single-GPU smoke training; official ReCamMaster loss, complete restart."""
import argparse, copy, hashlib, random, time, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from train_recammaster import LightningModelForTrain
from v2v_empty_common import *

def digest_tensors(items):
    h=hashlib.sha256()
    for name,t in items:
        h.update(name.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(d):
    random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['torch']);torch.cuda.set_rng_state_all(d['cuda'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',type=Path);ap.add_argument('--stop',type=int,default=8);ap.add_argument('--tag',default='continuous');args=ap.parse_args()
    lease=gpu_lease();torch.set_num_threads(4);seed=20261114
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8-sig'))
    identity=dict(protocol_sha256=sha(RUN/'protocol.json'),shared_text_sha256=sha(CACHE/'empty_text.pt'),cache_manifest_sha256=sha(RUN/'cache_manifest.json'),code_sha256=sha(__file__))
    cache_hashes={v['path']:v['sha256'] for v in json.loads((RUN/'cache_manifest.json').read_text())}
    # Require complete scenes, fail before loading model if anything is absent/corrupt.
    for scene in TRAIN+VAL:
        for c in range(1,11):
            assert sha(cache_path(scene,c))==cache_hashes[str(cache_path(scene,c))], 'cache checksum mismatch'
            v=torch.load(cache_path(scene,c),map_location='cpu',weights_only=True)
            assert tuple(v['latents'].shape)==(16,21,48,84) and torch.isfinite(v['latents']).all()
    text=torch.load(CACHE/'empty_text.pt',weights_only=True,map_location='cpu')
    assert text['text']=='' and text['context'].shape==(1,512,4096)
    context=text['context'].to('cuda')
    wrapper=LightningModelForTrain(str(WEIGHTS/'diffusion_pytorch_model.safetensors'),learning_rate=protocol['lr'],use_gradient_checkpointing=True)
    pipe=wrapper.pipe;pipe.device='cuda';model=pipe.dit.float().to('cuda');model.train()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    assert all(any(k in n for k in ('self_attn','projector','cam_encoder')) for n,p in trainable)
    assert not any(p.requires_grad for n,p in model.named_parameters() if 'cross_attn' in n)
    opt=torch.optim.AdamW([p for n,p in trainable],lr=protocol['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    lr_sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    start_step=0
    if args.resume:
        receipt=json.loads(args.resume.with_suffix('.json').read_text())
        assert sha(args.resume)==receipt['sha256'], 'checkpoint checksum mismatch'
        ck=torch.load(args.resume,weights_only=False,map_location='cpu')
        assert ck['identity']==identity
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);lr_sched.load_state_dict(ck['lr_scheduler'])
        pipe.scheduler.__dict__.update(ck['diffusion_scheduler']);start_step=ck['step']
        assert ck['data_cursor']==start_step and ck['epoch']==start_step//len(TRAIN)
        restore_rng(ck['rng']);del ck
    frozen_hash=digest_tensors(frozen);initial_hash=digest_tensors(trainable)
    event('train_start',tag=args.tag,start_step=start_step,stop_step=args.stop,trainable_parameters=sum(p.numel() for n,p in trainable),frozen_parameters=sum(p.numel() for n,p in frozen),shared_text=True,cross_attention_frozen=True)
    def loss_for(x,cam,t_id=None,checkpoint=True):
        noise=torch.randn_like(x)
        if t_id is None:t_id=int(torch.randint(0,pipe.scheduler.num_train_timesteps,(1,)).item())
        timestep=pipe.scheduler.timesteps[t_id:t_id+1].to('cuda',dtype=torch.bfloat16)
        noisy=pipe.scheduler.add_noise(x,noise,timestep);noisy[:,:,21:]=x[:,:,21:]
        target=pipe.scheduler.training_target(x,noise,timestep)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(noisy,timestep=timestep,cam_emb=cam,context=context,use_gradient_checkpointing=checkpoint)
            raw=torch.nn.functional.mse_loss(pred[:,:,:21].float(),target[:,:,:21].float())
            weight=pipe.scheduler.training_weight(timestep)
            weighted=raw*weight
        return weighted,raw,weight,t_id,float(timestep.item())
    def validation(label):
        # A fixed heldout denoising probe, not video-generation quality or convergence proof.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(314159);torch.cuda.manual_seed_all(314159)
            x,cam=batch(VAL[0],1,2)
            with torch.no_grad():l,r,w,_,_=loss_for(x,cam,t_id=500,checkpoint=False)
            result=dict(label=label,weighted_mse=float(l),raw_mse=float(r),weight=float(w),scene=VAL[0],target=1,condition=2,timestep_index=500)
            assert np.isfinite(result['weighted_mse']);event('validation',tag=args.tag,**result)
            return result
    def save(step):
        d=dict(version=1,identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),lr_scheduler=lr_sched.state_dict(),diffusion_scheduler=pipe.scheduler.__dict__,rng=rng_state(),step=step,data_cursor=step,epoch=step//len(TRAIN),amp_scaler=None,precision='bf16 autocast with fp32 parameters',sample_policy='round-robin scene; Python RNG distinct cameras')
        p=RUN/'checkpoints'/f'{args.tag}_step{step}.pt';atomic_save(d,p)
        atomic_json(dict(step=step,bytes=p.stat().st_size,sha256=sha(p)),p.with_suffix('.json'));event('checkpoint_complete',tag=args.tag,step=step,path=str(p))
    baseline=validation('initial') if not args.resume else None
    records=[];started=time.time()
    for step in range(start_step,args.stop):
        scene=TRAIN[step%len(TRAIN)];target,cond=random.sample(range(1,11),2)
        x,cam=batch(scene,target,cond);opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();tick=time.time()
        loss,raw,weight,t_id,timestep=loss_for(x,cam)
        assert torch.isfinite(loss), 'nonfinite loss'
        loss.backward()
        grad_groups={};no_grad=[]
        for n,p in trainable:
            if p.grad is None:no_grad.append(n);continue
            assert torch.isfinite(p.grad).all(), 'nonfinite gradient '+n
            group='cam_encoder' if 'cam_encoder' in n else 'projector' if 'projector' in n else 'self_attn'
            grad_groups[group]=grad_groups.get(group,0.)+float(p.grad.float().square().sum())
        assert not no_grad, str(no_grad)
        assert all(p.grad is None for n,p in frozen)
        grad_groups={k:v**.5 for k,v in grad_groups.items()};assert all(v>0 for v in grad_groups.values())
        # No clipping / no skipped updates. Preserve official AdamW default behavior.
        probes={n:p.detach().clone() for n,p in trainable if n in ('blocks.0.cam_encoder.weight','blocks.0.projector.weight','blocks.0.self_attn.q.weight')}
        opt.step();lr_sched.step();torch.cuda.synchronize()
        updates={n:float((dict(trainable)[n].detach()-v).abs().max()) for n,v in probes.items()};assert all(v>0 for v in updates.values())
        record=dict(step=step+1,cursor=step+1,epoch=step//len(TRAIN),scene=scene,target=target,condition=cond,requested_equals_actual=True,raw_loss=float(raw),weight=float(weight),loss=float(loss),timestep_index=t_id,timestep=timestep,grad_norms=grad_groups,probe_update_max=updates,lr=opt.param_groups[0]['lr'],seconds=time.time()-tick,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30,input_shape=list(x.shape),camera_shape=list(cam.shape))
        records.append(record);event('train_step',tag=args.tag,**record)
        if step+1==6:
            fingerprint=dict(record=record,model_sha256=digest_tensors(trainable),optimizer_sha256=digest_tensors((f'{i}:{k}',v) for i,s in enumerate(opt.state.values()) for k,v in sorted(s.items()) if torch.is_tensor(v)),rng=rng_state()['torch'].tolist())
            atomic_json(fingerprint,RUN/f'{args.tag}_step6_fingerprint.json')
        if not args.resume and step+1 in (4,8):save(step+1)
    final=validation('final') if not args.resume else None
    final_hash=digest_tensors(trainable);assert final_hash!=initial_hash;assert digest_tensors(frozen)==frozen_hash
    result=dict(status='passed',tag=args.tag,updates=len(records),start_step=start_step,end_step=args.stop,seconds=time.time()-started,records=records,initial_validation=baseline,final_validation=final,frozen_unchanged=True,trainable_changed=True,identity=identity)
    atomic_json(result,RUN/f'{args.tag}_result.json');event('train_complete',tag=args.tag,step=args.stop)

if __name__=='__main__':main()
