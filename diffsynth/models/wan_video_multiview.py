"""Opt-in multi-source Wan wrapper; official inference/training entry points unchanged."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .wan_video_dit import modulate, sinusoidal_embedding_1d


class MultiViewPluckerWan(nn.Module):
    def __init__(self, dit):
        super().__init__()
        if dit.has_image_input or tuple(dit.patch_size) != (1,2,2):
            raise ValueError('v1 supports T2V Wan with (1,2,2) patches')
        if any(hasattr(b, 'cam_encoder') or hasattr(b, 'projector') for b in dit.blocks):
            raise ValueError('Load a plain Wan base, not an adapted ReCamMaster model')
        self.dit = dit
        self.camera_encoders = nn.ModuleList([nn.Linear(1536, dit.dim, bias=False) for _ in dit.blocks])
        self.projectors = nn.ModuleList([nn.Linear(dit.dim, dit.dim) for _ in dit.blocks])
        for projector in self.projectors:
            nn.init.eye_(projector.weight)
            nn.init.zeros_(projector.bias)
        self.roles = nn.Embedding(2, dit.dim)  # target=0, all interchangeable sources=1
        for encoder in self.camera_encoders:
            std = 1.0 / math.sqrt(dit.dim)
            nn.init.trunc_normal_(encoder.weight, std=std, a=-3*std, b=3*std)
        nn.init.zeros_(self.roles.weight)
        # Match base model placement. Geometric computation remains float32 upstream.
        self.to(device=dit.patch_embedding.weight.device, dtype=dit.patch_embedding.weight.dtype)

    def trainable_conditioning(self):
        """Explicit proposed policy: attention + ReCamMaster projectors + cameras + roles."""
        self.requires_grad_(False)
        for block in self.dit.blocks: block.self_attn.requires_grad_(True)
        self.projectors.requires_grad_(True)
        self.camera_encoders.requires_grad_(True)
        self.roles.requires_grad_(True)

    def forward(self, target, sources, timestep, context, ray_tokens, use_gradient_checkpointing=False):
        """target [B,C,T,H,W], clean sources [B,S,C,T,H,W].

        ray_tokens [B,1+S,T,H/2,W/2,1536], target first. Same source count within
        a batch; use view-count buckets instead of unmasked zero padding.
        Return ONLY the target prediction. Loss/noise scheduling belongs to trainer.
        """
        b,c,t,h,w = target.shape
        if sources.ndim != 6 or sources.shape[0] != b or sources.shape[2:] != target.shape[1:]:
            raise ValueError('source and target dimensions disagree')
        s = sources.shape[1]; v = s+1
        if s < 1: raise ValueError('at least one source is required')
        if h%2 or w%2 or ray_tokens.shape != (b,v,t,h//2,w//2,1536):
            raise ValueError('ray tokens do not align with visual patches')
        videos = torch.cat((target[:,None], sources), dim=1)
        x = videos.permute(0,2,1,3,4,5).reshape(b,c,v*t,h,w)
        x, (_, gh, gw) = self.dit.patchify(x)
        role_ids = torch.cat((torch.zeros(t*gh*gw,device=x.device,dtype=torch.long),
                              torch.ones(s*t*gh*gw,device=x.device,dtype=torch.long)))
        x = x + self.roles(role_ids)[None]
        cameras = ray_tokens.reshape(b,v*t*gh*gw,1536).to(x)
        # Every view starts at the SAME latent time coordinate. Never offset time by view.
        ft,fh,fw = self.dit.freqs
        if t>len(ft) or gh>len(fh) or gw>len(fw): raise ValueError('RoPE grid exceeds table')
        freqs = torch.cat((ft[:t,None,None].expand(t,gh,gw,-1),
                           fh[None,:gh,None].expand(t,gh,gw,-1),
                           fw[None,None,:gw].expand(t,gh,gw,-1)),dim=-1)
        freqs = freqs.reshape(t*gh*gw,1,-1).repeat(v,1,1).to(x.device)
        temb = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim,timestep))
        tmod = self.dit.time_projection(temb).unflatten(1,(6,self.dit.dim))
        context = self.dit.text_embedding(context)
        for block, encoder, projector in zip(self.dit.blocks,self.camera_encoders,self.projectors):
            def run(hidden, ctx, cam, tm, block=block, encoder=encoder, projector=projector):
                shift, scale, gate, fshift, fscale, fgate = (block.modulation.to(tm)+tm).chunk(6,dim=1)
                attn_in = modulate(block.norm1(hidden),shift,scale)+encoder(cam)
                hidden = hidden+gate*projector(block.self_attn(attn_in,freqs))
                hidden = hidden+block.cross_attn(block.norm3(hidden),ctx)
                return hidden+fgate*block.ffn(modulate(block.norm2(hidden),fshift,fscale))
            if self.training and use_gradient_checkpointing:
                x = checkpoint(run,x,context,cameras,tmod,use_reentrant=False)
            else: x = run(x,context,cameras,tmod)
        prediction = self.dit.head(x[:,:t*gh*gw],temb)
        return self.dit.unpatchify(prediction,(t,gh,gw))
