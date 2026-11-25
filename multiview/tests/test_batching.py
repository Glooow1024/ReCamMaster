"""CPU regression: independent sample times, weighted gradients and source counts."""
import copy,contextlib,json
from unittest.mock import patch
import torch
from diffsynth.models.wan_video_dit import WanModel,Head
from diffsynth.models.wan_video_multiview import MultiViewPluckerWan
from multiview.training.batching import pack,loss_for

def main():
    torch.set_num_threads(2);torch.manual_seed(317)
    head=Head(8,2,(1,2,2),1e-6);x=torch.randn(2,5,8);t=torch.randn(2,8)
    reference=torch.cat([head(x[i:i+1],t[i:i+1]) for i in range(2)])
    torch.testing.assert_close(head(x,t[:,None]),reference)
    assert not torch.allclose(head(x,t),reference)
    assert torch.equal(head(x[:1],t[:1]),head(x[:1],t[:1,None]))
    checks=[]
    for source_count in (1,2,3,4):
        dit=WanModel(dim=96,in_dim=16,ffn_dim=192,out_dim=16,text_dim=32,freq_dim=32,eps=1e-6,patch_size=(1,2,2),num_heads=4,num_layers=2,has_image_input=False)
        a=MultiViewPluckerWan(dit);a.trainable_conditioning();b=copy.deepcopy(a)
        samples=[]
        for i in range(2):
            samples.append(dict(noisy=torch.randn(1,16,2,4,4),sources=torch.randn(1,source_count,16,2,4,4),timestep=torch.tensor([137.+i*611]),context=torch.randn(1,5,32),rays=torch.randn(1,source_count+1,2,2,2,1536),target=torch.randn(1,16,2,4,4),weight=torch.tensor([.2+i*1.3])))
        # Exercise the same reduction on CPU, without CUDA autocast.
        with patch('torch.autocast',lambda *args,**kw:contextlib.nullcontext()):
            values=[]
            for sample in samples:
                loss,_,weighted=loss_for(a,sample);values.extend(weighted.tolist());(loss/2).backward()
            loss,_,weighted=loss_for(b,pack(samples));loss.backward()
        torch.testing.assert_close(weighted,torch.tensor(values),atol=2e-6,rtol=2e-5)
        maximum=0.
        for p,q in zip(a.parameters(),b.parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
                maximum=max(maximum,float((p.grad-q.grad).abs().max()))
        checks.append(dict(sources=source_count,gradient_max_abs=maximum))
    print(json.dumps(dict(passed=True,checks=checks,head_single_sample_unchanged=True)))
if __name__=='__main__':main()
