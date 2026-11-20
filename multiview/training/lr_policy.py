import math
POLICY=dict(kind='constant_then_cosine',peak=4e-5,start=8000,end=20000,floor=1e-5,clock='global optimizer update number; no restart warmup')
def lr_for_update(step):
    t=min(1.,max(0.,(step-8000)/12000))
    return 1e-5+0.5*3e-5*(1+math.cos(math.pi*t))
