"""Load the original Wan base, then initialize new Plucker adaptation modules."""
import argparse,json
from pathlib import Path
import torch
from diffsynth import ModelManager
from diffsynth.models.wan_video_multiview import MultiViewPluckerWan


def build_model(wan_path, device='cpu', dtype=torch.bfloat16):
    manager=ModelManager(torch_dtype=dtype,device=device)
    manager.load_models([str(Path(wan_path)/'diffusion_pytorch_model.safetensors')])
    model=MultiViewPluckerWan(manager.fetch_model('wan_video_dit'))
    model.trainable_conditioning()
    return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wan',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    model=build_model(args.wan)
    assert len(model.projectors)==len(model.dit.blocks)
    for proj in model.projectors:
        assert torch.equal(proj.weight,torch.eye(model.dit.dim,dtype=proj.weight.dtype))
        assert torch.count_nonzero(proj.bias)==0
    for enc in model.camera_encoders:
        assert enc.bias is None and enc.in_features==1536
        assert torch.count_nonzero(enc.weight)>0
        assert float(enc.weight.detach().float().abs().max()) <= 3/model.dit.dim**.5 + .001
    assert all(torch.isfinite(p).all() for p in model.parameters())
    result={'base':str(Path(args.wan)/'diffusion_pytorch_model.safetensors'),
            'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'blocks':len(model.dit.blocks),'dim':model.dit.dim,'passed':True,
            'scope':'CPU original Wan load and adapter initialization; no full-size training or generation'}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
