"""V1: pure V2V, shared empty text, official spatial preprocessing/camera convention."""
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from train_recammaster import TextVideoDataset, TensorDataset, Camera

DATA = Path('/data/nvs/recammaster/datasets/MultiCamVideo-extracted/MultiCamVideo-Dataset')
WEIGHTS = Path('/data/nvs/recammaster/Wan2.1-T2V-1.3B')
CACHE = Path('/home/3dvision/nvs/recammaster/cache/v2v_empty_384x672_v1')
RUN = Path(os.environ.get('RECAM_RUN', '/data/nvs/recammaster/runs/v2v_empty_smoke_20261114'))
GROUPS = ['f18_aperture10', 'f24_aperture5', 'f35_aperture2.4', 'f50_aperture2.4']
TRAIN = [g + '/scene1' for g in GROUPS]
VAL = [GROUPS[0] + '/scene3400']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)

def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp); os.replace(tmp, path)

def event(kind, **fields):
    RUN.mkdir(parents=True, exist_ok=True)
    value = dict(time=time.time(), kind=kind, pid=os.getpid(), **fields)
    with (RUN/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)

def cache_path(scene, cam):
    return CACHE/'latents'/scene/'videos'/f'cam{cam:02}.pt'

def camera(scene, cond, target):
    d = json.loads((DATA/'train'/scene/'cameras/camera_extrinsics.json').read_text())
    helper = TensorDataset.__new__(TensorDataset)
    views = []
    for v in (cond, target):
        arr = np.stack([helper.parse_matrix(d[f'frame{i}'][f'cam{v:02}']) for i in range(0,81,4)]).transpose(0,2,1)
        poses = []
        for a in arr:
            a = a[:,[1,2,0,3]]; a[:3,1] *= -1; a[:3,3] /= 100
            poses.append(Camera(a))
        views.append(poses)
    x = np.stack([helper.get_relative_pose([views[0][0], c])[1,:3,:].reshape(12) for c in views[1]])
    return torch.tensor(x, dtype=torch.bfloat16).unsqueeze(0)

def batch(scene, target, cond, device='cuda'):
    assert target != cond
    values = [torch.load(cache_path(scene, c), map_location='cpu', weights_only=True) for c in (target, cond)]
    for v in values:
        assert tuple(v['latents'].shape) == (16,21,48,84)
        assert v['preprocess_id'] == 'official_81_384_672_tiled34x34_stride18x16_bf16_v1'
    x = torch.cat([v['latents'] for v in values],dim=1).unsqueeze(0).to(device)
    return x, camera(scene, cond, target).to(device)

def gpu_lease():
    import fcntl, subprocess
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in tuple(str(i) for i in range(8)), 'Explicit authorized GPU required'
    p = Path('/root/nvs/recammaster_ws') / f'gpu{gpu}.v2v.lock'
    f = p.open('w'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    used = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used < 100, f'GPU {gpu} already uses {used} MiB; abort instead of interfering'
    f.write(str(os.getpid())); f.flush()
    return f
