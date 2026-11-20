"""CPU acceptance: geometry, view association, gradients, checkpointing, real cache IO."""
import argparse, copy, json, tempfile
from pathlib import Path
import torch
from diffsynth.models.plucker_geometry import cropped_intrinsics, relative_cameras, plucker_patch_tokens
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_multiview import MultiViewPluckerWan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets');ap.add_argument('--official-camera');ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(123)
    checks=[]
    k=cropped_intrinsics(24)
    torch.testing.assert_close(k[:2,2],torch.tensor([336.,192.]))
    pose=torch.eye(4).expand(1,3,2,4,4).clone();pose[:,1,:,0,3]=1;pose[:,2,:,1,3]=2
    intr=k.expand(1,3,2,3,3)
    rel=relative_cameras(pose,pose[:,1,0])
    rays=plucker_patch_tokens(rel,intr,(32,32),(4,4))
    lines=rays.reshape(-1,6)
    torch.testing.assert_close(lines[:,3:].norm(dim=-1),torch.ones(len(lines)))
    torch.testing.assert_close((lines[:,:3]*lines[:,3:]).sum(-1),torch.zeros(len(lines)),atol=2e-6,rtol=0)
    # Camera transformations shared by all views must cancel in the reference frame.
    transform=torch.eye(4);transform[:3,:3]=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]]);transform[:3,3]=torch.tensor([30.,-50,70])
    moved=transform@pose
    torch.testing.assert_close(relative_cameras(moved,moved[:,1,0]),rel)
    shifted=rel.clone();shifted[:,2,:,0,3]+=3
    changed=plucker_patch_tokens(shifted,intr,(32,32),(4,4))
    assert torch.equal(changed[:,:2],rays[:,:2]) and not torch.equal(changed[:,2],rays[:,2])
    checks+=['crop K','unit ray directions','direction orthogonal to moment','shared world frame invariance','camera binds only its own tokens']
    if args.official_camera:
        import importlib.util
        spec=importlib.util.spec_from_file_location('official_camera',args.official_camera)
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        w2c=torch.linalg.inv(rel)[...,:3,:]
        reference=official.Camera.get_plucker_rays(w2c,intr,(32,32))
        reference=reference.reshape(1,3,2,2,16,2,16,6).permute(0,1,2,3,5,4,6,7).reshape(1,3,2,2,2,1536)
        torch.testing.assert_close(rays,reference,atol=2e-6,rtol=1e-5)
        checks+=['numerical parity with pinned official Camera.get_plucker_rays and 16x16 flatten']
    dit=WanModel(dim=96,in_dim=16,ffn_dim=192,out_dim=16,text_dim=32,freq_dim=32,eps=1e-6,patch_size=(1,2,2),num_heads=4,num_layers=2,has_image_input=False)
    model=MultiViewPluckerWan(dit);model.trainable_conditioning()
    for enc in model.camera_encoders:
        std=96**-.5
        assert enc.bias is None and enc.weight.abs().max()<=3*std
        assert .9*std < enc.weight.std() < 1.1*std
    checks+=['official trunc_normal std=1/sqrt(D), bounds +/-3std, no bias']
    target=torch.randn(1,16,2,4,4);sources=torch.randn(1,2,16,2,4,4);context=torch.randn(1,5,32);step=torch.tensor([500.])
    # Make camera conditioning active before testing equivariance and responsiveness.
    for enc in model.camera_encoders:torch.nn.init.normal_(enc.weight,std=.01)
    out=model(target,sources,step,context,rays)
    perm=model(target,sources[:,[1,0]],step,context,rays[:,[0,2,1]])
    torch.testing.assert_close(out,perm,atol=2e-6,rtol=2e-5)
    other=model(target,sources,step,context,changed)
    assert not torch.allclose(out,other)
    out.square().mean().backward()
    assert all(e.weight.grad is not None and e.weight.grad.abs().sum()>0 for e in model.camera_encoders)
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    checkpointed=model(target,sources,step,context,rays,use_gradient_checkpointing=True)
    checkpointed.square().mean().backward()
    torch.testing.assert_close(out,checkpointed)
    for n,p in model.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-6,rtol=1e-5)
    checks+=['two-source order equivariance with fixed reference','camera change affects target','camera nonzero gradients','checkpoint forward/backward equivalence']
    assert len(model.projectors)==len(model.dit.blocks)
    for proj in model.projectors:
        torch.testing.assert_close(proj.weight,torch.eye(96),rtol=0,atol=0)
        assert torch.count_nonzero(proj.bias)==0
        assert proj.weight.grad is not None and proj.weight.grad.abs().sum()>0
    assert all(e.bias is None and e.in_features==1536 for e in model.camera_encoders)
    for n in (1,2,4,5):
        src=sources[:,:1].expand(-1,n,-1,-1,-1,-1)
        cr=torch.cat([rays[:,:1],rays[:,1:2].expand(-1,n,-1,-1,-1,-1)],1)
        assert model(target,src,step,context,cr).shape==target.shape
    checks+=['ReCamMaster identity projector and nonzero gradient, dense bias-free camera projection','1/2/4/5 source shapes']
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    def update():
        opt.zero_grad(set_to_none=True);loss=model(target,sources,step,context,rays,True).square().mean();loss.backward();opt.step();return loss.detach()
    update()
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'resume.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},path)
        expected=update();params=copy.deepcopy(model.state_dict())
        ck=torch.load(path,weights_only=True);model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng'])
        actual=update();torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        for n,p in model.state_dict().items():torch.testing.assert_close(p,params[n],rtol=0,atol=0)
    checks+=['tiny model optimizer and RNG resume exact next update']
    real=[]
    if args.assets:
        from multiview_data import load_example
        assets=json.loads(Path(args.assets).read_text())
        for group in ('f18_aperture10','f24_aperture5','f35_aperture2.4','f50_aperture2.4'):
            data=load_example(group+'/scene1',1,[2,10],Path(args.assets).parent/'geometry_v1',assets['existing_vae_cache'])
            assert data['ray_tokens'].shape==(1,3,21,24,42,1536)
            assert torch.isfinite(data['ray_tokens']).all()
            real.append({'scene':data['scene'],'split':data['split'],'ray_shape':list(data['ray_tokens'].shape),'context_shape':list(data['context'].shape)})
        checks+=['four focal groups real geometry + three actual video latent caches + shared empty text']
        swapped=load_example(data['scene'],1,[10,2],Path(args.assets).parent/'geometry_v1',assets['existing_vae_cache'],reference_camera=2)
        torch.testing.assert_close(swapped['ray_tokens'],data['ray_tokens'][:,[0,2,1]])
        assert swapped['reference_camera']==data['reference_camera']
        checks+=['real adapter source permutation keeps reference and camera association']
        first=load_example(data['scene'],1,[10,2],Path(args.assets).parent/'geometry_v1',assets['existing_vae_cache'])
        assert first['reference_camera']==10
        # First source / first frame has zero ray moments in its own origin.
        assert first['ray_tokens'][0,1,0].reshape(-1,6)[:,:3].abs().max()<1e-5
        checks+=['default first source first frame anchor, including nonminimal camera ID']
        # Exercise the actual cached data in a reduced-size model with flow matching.
        small=MultiViewPluckerWan(WanModel(dim=96,in_dim=16,ffn_dim=192,out_dim=16,text_dim=4096,freq_dim=32,eps=1e-6,patch_size=(1,2,2),num_heads=4,num_layers=2,has_image_input=False))
        small.trainable_conditioning();optim=torch.optim.AdamW([p for p in small.parameters() if p.requires_grad],lr=1e-4)
        clean=data['target'][:,:,:2,:4,:4].float()
        noise=torch.randn_like(clean);noisy=.5*clean+.5*noise
        losses=[]
        for _ in range(3):
            optim.zero_grad(set_to_none=True)
            pred=small(noisy,data['sources'][:,:,:,:2,:4,:4].float(),step,data['context'].float(),data['ray_tokens'][:,:,:2,:2,:2],True)
            loss=(pred-(noise-clean)).square().mean();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in small.parameters() if p.grad is not None)
            optim.step();losses.append(float(loss.detach()))
        real.append({'reduced_real_cache_flow_losses':losses,'scope':'spatial/temporal subset, random tiny model, plumbing only'})
        checks+=['three real cached-data flow-matching updates with empty text and target-only loss']
    result={'passed':True,'device':'cpu','checks':checks,'real_assets':real,'scope':'small randomly initialized Wan model; no full-size GPU training or generation quality validation'}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
