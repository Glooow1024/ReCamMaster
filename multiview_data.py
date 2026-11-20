"""Read-only adapter for existing geometry and VAE caches; no resampling fallback."""
from pathlib import Path
import numpy as np
import torch
from diffsynth.models.plucker_geometry import cropped_intrinsics, relative_cameras, plucker_patch_tokens

PREPROCESS_ID='official_81_384_672_tiled34x34_stride18x16_bf16_v1'


def load_example(scene, target_camera, source_cameras, geometry_root, cache_root, reference_camera=None):
    if len(source_cameras) < 1:
        raise ValueError('require at least one source camera')
    ids=[target_camera]+list(source_cameras)
    if len(set(ids))!=len(ids) or any(c not in range(1,11) for c in ids):
        raise ValueError('camera IDs must be distinct and within 1..10')
    if Path(scene).is_absolute() or '..' in Path(scene).parts:
        raise ValueError('scene must be a relative dataset identifier')
    with np.load(Path(geometry_root)/'scenes'/(scene.replace('/','__')+'.npz'),allow_pickle=False) as geo:
        if str(geo['scene'])!=scene or int(geo['version'])!=1:raise ValueError('geometry identity mismatch')
        poses=torch.from_numpy(geo['c2w'][np.array(ids)-1].copy()).unsqueeze(0)
        focal=float(geo['focal_mm']);split=str(geo['split'])
    # The first source anchors the world for this sample; adding sources must not
    # reorder it. Explicit reference is retained for fixed-anchor comparisons.
    reference_camera=source_cameras[0] if reference_camera is None else reference_camera
    if reference_camera not in source_cameras:raise ValueError('reference must be an available source camera')
    poses=relative_cameras(poses,poses[:,ids.index(reference_camera),0])[:,:,::4] # representative timestamps 0,4,...80
    k=cropped_intrinsics(focal).reshape(1,1,1,3,3).expand(1,len(ids),21,3,3)
    rays=plucker_patch_tokens(poses,k,(384,672),(48,84))
    latents=[]
    for camera in ids:
        item=torch.load(Path(cache_root)/'latents'/scene/'videos'/f'cam{camera:02}.pt',map_location='cpu',weights_only=True)
        expected=f'{scene}/videos/cam{camera:02}.mp4'
        if item['preprocess_id']!=PREPROCESS_ID or item['source']!=expected:raise ValueError('latent provenance mismatch')
        z=item['latents']
        if z.shape!=(16,21,48,84) or not torch.isfinite(z).all():raise ValueError('invalid latent')
        latents.append(z)
    text=torch.load(Path(cache_root)/'empty_text.pt',map_location='cpu',weights_only=True)
    return {'target':latents[0][None],'sources':torch.stack(latents[1:])[None],
            'ray_tokens':rays,'context':text['context'],'scene':scene,'split':split,
            'camera_ids':ids,'reference_camera':reference_camera,
            'latent_frame_indices':list(range(0,81,4))}
