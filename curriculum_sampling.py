"""Deterministic 20K curriculum proposal; no model or training side effects."""
import numpy as np

STAGES=[(0,4000,[1,0,0,0],[.8,.2,0],[.6,.35,.05]),
        (4000,8000,[.3,.7,0,0],[.6,.35,.05],[.4,.45,.15]),
        (8000,12000,[.2,.3,.5,0],[.4,.45,.15],[.25,.4,.35]),
        (12000,20000,[.1,.2,.3,.4],[.25,.4,.35],None)]

def classify(metrics):
    a,b,c,d=[metrics[...,i] for i in range(4)]
    trivial=(a<2)&(b<.05)
    easy=(a<=20)&(b<=1.5)&(c>=.6)&(d>=.3)
    medium=(a<=40)&(b<=2.5)&(c>=.4)&(d>=.1)
    return np.where(trivial,3,np.where(easy,0,np.where(medium,1,2))).astype('uint8')

def schedule(step,natural):
    if not 0<=step<20000:raise ValueError('step must be in [0,20000)')
    stage=next(s for s in STAGES if s[0]<=step<s[1])
    start,end,view,startp,endp=stage
    hold=0 if start==0 else 500
    ramp_end=16000 if start==12000 else end
    alpha=np.clip((step-start-hold)/max(1,ramp_end-start-hold),0,1)
    finish=np.asarray(natural if endp is None else endp)
    probs=(1-alpha)*np.array(startp)+alpha*finish
    return np.array(view),probs

class GeometrySampler:
    """Training scenes only; input is collect_curriculum_stats output.

    Anchor chosen by curriculum bin, then uniform focal group, eligible scene, pair.
    Added sources condition on same target, draw available bins with renormalized
    curriculum probabilities, then choose randomly among top three union gains.
    Renormalization and exhaustion fallback are explicit in returned diagnostics.
    """
    def __init__(self,metrics,groups,scenes):
        self.metrics=metrics;self.groups=groups;self.scenes=scenes;self.bins=classify(metrics)
        self.pairs=np.array([(t,s) for t in range(10) for s in range(10) if t!=s])
        count=np.bincount(self.bins.ravel(),minlength=4)[:3];self.natural=count/count.sum()
        self.focals=sorted(set(groups.tolist()))
        self.eligible={(g,b):np.flatnonzero((groups==g)&(self.bins==b).any(1)) for g in self.focals for b in range(3)}
        if any(len(x)==0 for x in self.eligible.values()):raise ValueError('empty global focal/bin pool')

    def anchor(self,step,rng):
        views,prob=schedule(step,self.natural)
        n=int(rng.choice(np.arange(1,5),p=views));b=int(rng.choice(3,p=prob));g=int(rng.choice(self.focals))
        scene=int(rng.choice(self.eligible[g,b]));pair=int(rng.choice(np.flatnonzero(self.bins[scene]==b)))
        target,source=map(int,self.pairs[pair])
        return {'scene_index':scene,'scene':str(self.scenes[scene]),'target':target,'sources':[source],'anchor_bin':b,'source_count':n,'prob':prob.tolist()}

    def complete(self,sample,bits,rng):
        # bits [2 depth ranges, target, source, 9 times, 128 point samples]
        row=sample['scene_index'];target=sample['target'];sources=sample['sources'][:]
        covered=bits[:,target,sources[0]].copy();diag=[];prob=np.array(sample['prob'])
        while len(sources)<sample['source_count']:
            candidates=[c for c in range(10) if c!=target and c not in sources]
            indices=[int(np.flatnonzero((self.pairs==[target,c]).all(1))[0]) for c in candidates]
            labels=self.bins[row,indices]
            avail=np.array([(labels==b).any() for b in range(3)]);weights=prob*avail
            fallback=bool(weights.sum()==0)
            if fallback:weights=avail.astype(float)
            if weights.sum()==0:raise ValueError('only near-identity candidates remain')
            weights/=weights.sum();chosen_bin=int(rng.choice(3,p=weights))
            choices=[c for c,b in zip(candidates,labels) if b==chosen_bin]
            gains=[float(np.minimum(*(covered|bits[:,target,c]).mean(-1)).mean()) for c in choices]
            top=np.argsort(gains)[-min(3,len(choices)):];j=int(rng.choice(top));source=choices[j]
            before=float(np.minimum(*covered.mean(-1)).mean());covered|=bits[:,target,source];sources.append(source)
            diag.append({'bin':chosen_bin,'available_prob':weights.tolist(),'fallback':fallback,'union_gain':gains[j]-before})
        sample={**sample,'sources':sources,'extra_sources':diag,'joint_coverage':float(np.minimum(*covered.mean(-1)).mean())}
        return sample
