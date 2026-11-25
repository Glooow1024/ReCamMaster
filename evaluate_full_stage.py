"""Fixed-seed generated-video probes, source/prediction/GT composites."""
import argparse,json,math,time
import imageio,numpy as np,torch
from diffsynth import ModelManager,WanVideoReCamMasterPipeline
from train_recammaster import LightningModelForTrain
from v2v_empty_common import *
ROOT=Path('/data/nvs/recammaster/runs/v2v_full_20261114')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path);ap.add_argument('--label',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    lease=gpu_lease();torch.set_grad_enabled(False);torch.set_num_threads(4)
    out=ROOT/'quality'/args.label;out.mkdir(parents=True,exist_ok=False)
    wrapper=LightningModelForTrain(str(WEIGHTS/'diffusion_pytorch_model.safetensors'))
    model=wrapper.pipe.dit
    if args.checkpoint:
        assert sha(args.checkpoint)==json.loads(args.checkpoint.with_suffix('.json').read_text())['sha256']
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
    pipe=wrapper.pipe;pipe.device='cuda';model.bfloat16().to('cuda').eval()
    manager=ModelManager(torch_dtype=torch.bfloat16,device='cpu');manager.load_models([str(WEIGHTS/'Wan2.1_VAE.pth')]);vae_pipe=WanVideoReCamMasterPipeline.from_model_manager(manager,device='cuda');vae_pipe.vae.to('cuda').eval()
    context=torch.load(CACHE/'empty_text.pt',weights_only=True)['context'].to('cuda')
    ds=TextVideoDataset(str(DATA),str(ROOT/'metadata_validation.csv'),height=384,width=672)
    scenes=json.loads((ROOT/'validation_scenes.json').read_text());probes=VAL if args.smoke else [next(s for s in scenes if s.startswith(g+'/')) for g in GROUPS];results=[]
    for j,scene in enumerate(probes):
        tick=time.time();torch.manual_seed(1000+j);torch.cuda.manual_seed_all(1000+j)
        x,cam=batch(scene,1,2);cond=x[:,:,21:];target=torch.randn_like(x[:,:,:21]);pipe.scheduler.set_timesteps(30)
        for t in pipe.scheduler.timesteps:
            timestep=t.reshape(1).to('cuda',dtype=torch.bfloat16)
            pred=model(torch.cat([target,cond],dim=2),timestep=timestep,cam_emb=cam,context=context)
            target=pipe.scheduler.step(pred[:,:,:21],timestep,target)
        decoded=vae_pipe.decode_video(target,tiled=True,tile_size=(34,34),tile_stride=(18,16))[0]
        frames=[np.asarray(im) for im in vae_pipe.tensor2video(decoded)]
        inputs=[]
        for c in [2,1]:
            v=ds.load_video(str(DATA/'train'/scene/'videos'/f'cam{c:02}.mp4'));inputs.append([a for a in ((v.permute(1,2,3,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)])
        pred_path=out/f'probe{j}_prediction.mp4';comparison=out/f'probe{j}_source_prediction_gt.mp4'
        imageio.mimsave(pred_path,frames,fps=15,quality=7)
        with imageio.get_writer(comparison,fps=15,quality=7) as writer:
            for src,pred,gt in zip(inputs[0],frames,inputs[1]):writer.append_data(np.concatenate([src,pred,gt],axis=1))
        mse=float(np.mean([np.mean((a.astype(np.float32)-b.astype(np.float32))**2) for a,b in zip(frames,inputs[1])]))
        results.append(dict(scene=scene,target=1,condition=2,seed=1000+j,steps=30,cfg=1,pixel_mse=mse,psnr=10*math.log10(255**2/max(mse,1e-12)),prediction=str(pred_path),comparison=str(comparison),frames=len(frames),seconds=time.time()-tick))
        atomic_json(dict(status='running',results=results),out/'summary.json')
    atomic_json(dict(status='complete',checkpoint=str(args.checkpoint) if args.checkpoint else 'Wan original + new camera modules',results=results,limitations='Pixel PSNR is a supplementary paired-view diagnostic, not paper camera/motion metrics; inspect videos for action and camera control.'),out/'summary.json')
if __name__=='__main__':main()
