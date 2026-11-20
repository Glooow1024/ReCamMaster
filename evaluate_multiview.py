"""Fixed validation probes, target withheld from generation, CPU/GPU RNG preserved by caller."""
import gc,json,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from diffsynth import ModelManager
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from multiview_data import load_example

def write_video(path,tensor):
    # Wan decode is B,C,T,H,W in [-1,1]. Stream frames to avoid giant copies.
    with imageio.get_writer(str(path),fps=15,codec='libx264',quality=7,macro_block_size=1) as writer:
        for frame in tensor[0].permute(1,2,3,0):
            writer.append_data(((frame.float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy())

@torch.no_grad()
def evaluate(model,root,run,step,split='validation'):
    from train_multiview import save_json,scheduler
    root,run=Path(root),Path(run);assets=json.loads((root/'assets.json').read_text());model.eval()
    protocol=json.loads((run/'protocol.json').read_text());probes=protocol['evaluation'][split]
    folder=run/'quality'/f'{split}_step{step:06}';folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'status.json',dict(status='running',step=step,time=time.time()));rows=[]
    manager=ModelManager(torch_dtype=torch.bfloat16,device='cpu')
    manager.load_models([str(Path(assets['wan_base'])/'Wan2.1_VAE.pth')]);vae=manager.fetch_model('wan_video_vae').to('cuda')
    for i,p in enumerate(probes):
        data=load_example(p['scene'],p['target'],p['sources'],root/'geometry_v1',assets['existing_vae_cache'])
        assert data['split']==split
        clean=data['target'].cuda().float();src=data['sources'].cuda();rays=data['ray_tokens'].cuda();ctx=data['context'].cuda()
        generator=torch.Generator(device='cuda').manual_seed(123+i);noise=torch.randn(clean.shape,device='cuda',generator=generator)
        s=FlowMatchScheduler(shift=5,sigma_min=0,extra_one_step=True);s.set_timesteps(30)
        z=noise.clone()
        for t in s.timesteps:
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,src,t.reshape(1).to(device='cuda',dtype=torch.bfloat16),ctx,rays)
            z=s.step(pred.float(),t,z)
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite generation')
        decoded=vae.decode(z.to(torch.bfloat16),device='cuda',tiled=True,tile_size=(34,34),tile_stride=(18,16)).cpu()
        target=vae.decode(clean.to(torch.bfloat16),device='cuda',tiled=True,tile_size=(34,34),tile_stride=(18,16)).cpu()
        source=vae.decode(src[:,0].to(torch.bfloat16),device='cuda',tiled=True,tile_size=(34,34),tile_stride=(18,16)).cpu()
        if not torch.isfinite(decoded).all():raise FloatingPointError('nonfinite decode')
        write_video(folder/f'probe{i}_prediction.mp4',decoded);write_video(folder/f'probe{i}_target_vae.mp4',target)
        write_video(folder/f'probe{i}_source0_vae.mp4',source)
        mse=((decoded.float().clamp(-1,1)-target.float().clamp(-1,1))/2).square().mean().item()
        copy_mse=((source.float().clamp(-1,1)-target.float().clamp(-1,1))/2).square().mean().item()
        # Fixed teacher-forced denoising diagnostic; not an image metric.
        train_s=scheduler();tt=train_s.timesteps[500:501].to(device='cuda',dtype=torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):pflow=model(train_s.add_noise(clean,noise,tt),src,tt,ctx,rays)
        rows.append(dict(**p,source_count=len(p['sources']),seed=123+i,latent_flow_mse=float((pflow.float()-(noise-clean)).square().mean()),
                         psnr_vs_vae_target=-10*np.log10(max(mse,1e-12)),source_copy_psnr=-10*np.log10(max(copy_mse,1e-12)),prediction=f'probe{i}_prediction.mp4',target_video=f'probe{i}_target_vae.mp4',source_video=f'probe{i}_source0_vae.mp4'))
        save_json(folder/'partial.json',rows);del data,clean,src,rays,ctx,noise,z,pred,decoded,target,source,pflow
    del vae,manager;gc.collect();torch.cuda.empty_cache()
    result=dict(status='complete',step=step,time=time.time(),split=split,probes=rows,mean_psnr=float(np.mean([r['psnr_vs_vae_target'] for r in rows])),
                mean_flow_mse=float(np.mean([r['latent_flow_mse'] for r in rows])),protocol='30 steps, CFG1 empty text, fixed seeds, target absent from generation; PSNR vs VAE reconstruction, higher better; latent flow MSE lower better. Not camera or motion metric.')
    save_json(folder/'result.json',result);save_json(folder/'status.json',dict(status='complete',step=step,time=time.time()))
    save_json(run/'latest_quality.json',result)
    return result
