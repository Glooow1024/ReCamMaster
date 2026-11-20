"""Plucker rays in a shared reference frame. Pixel convention: centers at i + .5."""
import torch
import torch.nn.functional as F


def cropped_intrinsics(focal_mm, native_hw=(1280, 1280), output_hw=(384, 672), sensor_mm=(23.76, 23.76)):
    """Match official aspect-preserving resize + torchvision CenterCrop (edge coordinates)."""
    h, w = native_hw
    oh, ow = output_hw
    scale = max(oh / h, ow / w)
    rh, rw = round(h * scale), round(w * scale)
    top, left = round((rh - oh) / 2), round((rw - ow) / 2)
    return torch.tensor([[focal_mm / sensor_mm[0] * rw, 0, rw / 2 - left],
                         [0, focal_mm / sensor_mm[1] * rh, rh / 2 - top],
                         [0, 0, 1]], dtype=torch.float32)


def relative_cameras(c2w, reference):
    """c2w [B,V,T,4,4], reference [B,4,4]; keep one reference for every view/time."""
    c2w, reference = c2w.float(), reference.float()
    if not torch.isfinite(c2w).all() or not torch.isfinite(reference).all():
        raise ValueError('nonfinite camera')
    result = torch.linalg.inv(reference)[:, None, None] @ c2w
    r = result[..., :3, :3]
    eye = torch.eye(3, device=r.device)
    if not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=2e-4, rtol=0):
        raise ValueError('camera rotation is not orthogonal')
    if not torch.allclose(torch.linalg.det(r), torch.ones_like(r[..., 0, 0]), atol=2e-4, rtol=0):
        raise ValueError('relative cameras must have proper rotations')
    return result


def plucker_patch_tokens(c2w, intrinsics, image_hw, latent_hw, patch_hw=(2, 2)):
    """Return [B,V,T,Hpatch,Wpatch,1536], order per ray = (moment,direction).

    Intrinsics are [B,V,T,3,3] in processed-image pixel units. Cameras are already
    sampled at latent times and expressed in one common reference frame.
    Compute every processed-image pixel ray; flatten each 16x16 patch without averaging.
    """
    b, v, t = c2w.shape[:3]
    if c2w.shape != (b, v, t, 4, 4) or intrinsics.shape != (b, v, t, 3, 3):
        raise ValueError('camera and intrinsic shapes must agree')
    lh, lw = latent_hw; ph, pw = patch_hw
    if lh % ph or lw % pw:
        raise ValueError('latent size must be divisible by patch size')
    ih, iw = image_hw
    if ih != lh*8 or iw != lw*8 or (ph,pw)!=(2,2):
        raise ValueError("dense v2 requires spatial VAE stride 8 and 2x2 DiT patches")
    c2w, intrinsics = c2w.float(), intrinsics.float()
    if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all():
        raise ValueError('nonfinite camera input')
    if torch.any(intrinsics[..., 0, 0] <= 0) or torch.any(intrinsics[..., 1, 1] <= 0):
        raise ValueError('positive focal lengths required')
    y = (torch.arange(ih, device=c2w.device, dtype=torch.float32) + .5)
    x = (torch.arange(iw, device=c2w.device, dtype=torch.float32) + .5)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1)
    local = torch.einsum('bvtij,hwj->bvthwi', torch.linalg.inv(intrinsics), pixels)
    direction = F.normalize(torch.einsum('bvtij,bvthwj->bvthwi', c2w[..., :3, :3], local), dim=-1)
    origin = c2w[..., :3, 3][..., None, None, :].expand_as(direction)
    moment = torch.cross(origin, direction, dim=-1)
    rays = torch.cat((moment, direction), -1)
    return rays.reshape(b, v, t, ih//16, 16, iw//16, 16, 6).permute(0,1,2,3,5,4,6,7).reshape(b,v,t,ih//16,iw//16,1536)
