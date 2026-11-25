"""Own all eight authorized GPUs for the lifetime of a torchrun subprocess."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume');ap.add_argument('--tag',default='continuous');a=ap.parse_args()
    locks=[]
    for gpu in range(8):
        f=open(f'/root/nvs/recammaster_ws/gpu{gpu}.v2v.lock','w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<100,f'GPU{gpu} busy: {used}MiB'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7';env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='4';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['NCCL_ALGO']='Ring';env['NCCL_PROTO']='Simple'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--','train_full_v2v.py','--run',a.run,'--steps',str(a.steps),'--tag',a.tag]
    if a.smoke:cmd.append('--smoke')
    if a.resume:cmd+=['--resume',a.resume]
    p=subprocess.Popen(cmd,env=env)
    while p.poll() is None:
        path=Path(a.run)/'launcher.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='running',pid=p.pid,time=time.time(),tag=a.tag)));os.replace(tmp,path);time.sleep(15)
    path.write_text(json.dumps(dict(status='exited',returncode=p.returncode,time=time.time(),tag=a.tag)));sys.exit(p.returncode)
if __name__=='__main__':main()
