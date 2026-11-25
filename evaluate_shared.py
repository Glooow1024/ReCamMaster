"""Isolated evaluation on authorized query GPU7; training code is unchanged."""
import os,fcntl,subprocess,json
from pathlib import Path
import torch
import evaluate_full_stage as evaluation
def shared_lease():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    lock=open('/root/nvs/recammaster_ws/gpu7.evaluation.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>32000,'Insufficient memory for concurrent evaluation'
    # Reject unrelated GPU tenants; only this experiment may share the device.
    pids=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    for pid in pids:
        if int(pid)==os.getpid():continue
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'train_full_v2v.py' in cmd and b'v2v_full_20261114' in cmd, 'Unrelated GPU process'
    torch.cuda.set_per_process_memory_fraction(.35)
    return lock
evaluation.gpu_lease=shared_lease
if __name__=='__main__':evaluation.main()
