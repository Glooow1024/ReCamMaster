"""Batch independent diffusion samples without merging their attention sequences."""
import torch

def pack(samples):
    keys=('noisy','sources','timestep','context','rays','target','weight')
    return {key:torch.cat([sample[key] for sample in samples],dim=0) for key in keys}

def loss_for(model,batch):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=model(batch['noisy'],batch['sources'],batch['timestep'],batch['context'],batch['rays'],True)
        per_sample=(prediction.float()-batch['target']).square().flatten(1).mean(1)
        weighted=per_sample*batch['weight'].reshape(-1)
    return weighted.mean(),per_sample.detach(),weighted.detach()
