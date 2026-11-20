"""Save self-attention kernels during checkpoint recomputation; preserve text cross-attention."""
import collections
import torch
from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts
from diffsynth.models import wan_video_multiview as module
ATTENTION_OPS={getattr(torch.ops.aten,n).default for n in (
    '_scaled_dot_product_flash_attention','_flash_attention_forward',
    '_scaled_dot_product_efficient_attention','_scaled_dot_product_cudnn_attention',
    '_scaled_dot_product_flash_attention_for_cpu')}
COUNTS=collections.Counter()
def policy(ctx,op,*args,**kwargs):
    if op in ATTENTION_OPS:
        # SDPA uses [B,H,T,D]; the lower-level flash op uses [B,T,H,D].
        axis=1 if op==torch.ops.aten._flash_attention_forward.default else -2
        if args[0].shape[axis]!=args[1].shape[axis]:
            COUNTS['cross_not_saved:'+('recompute' if ctx.is_recompute else 'forward')]+=1
            return CheckpointPolicy.PREFER_RECOMPUTE
        COUNTS[('recompute:' if ctx.is_recompute else 'forward:')+str(op)]+=1
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

ORIGINAL=module.checkpoint
def selective(fn,*args,**kwargs):
    return ORIGINAL(fn,*args,context_fn=lambda:create_selective_checkpoint_contexts(policy),**kwargs)

def install():
    module.checkpoint=selective
