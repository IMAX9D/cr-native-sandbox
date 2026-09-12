import sys,random,json,time
from types import SimpleNamespace as S
sys.path.insert(0,'D:/Deepseek/outputs/hokoff-match-box-20260911/worktree')
import numpy as np
from expert_selfplay_v1.native_observation import _grid as original
from vector_encoder import _grid as vector
rng=random.Random(123);actors=[]
for n in range(513):
    entities=[S(x=rng.randrange(18)*1000+500,y=rng.randrange(32)*1000+500,relation=rng.randrange(2),hp=rng.randrange(1001),max_hp=1000) for _ in range(n)]
    if n%3==0:
        for e in entities:e.x=9500;e.y=15500
    actor=S(towers=[],entities=entities)
    assert np.array_equal(original(actor),vector(actor)),n
    actors.append(actor)
timings={}
for name,fn in [('reference',original),('vector',vector)]:
    t=time.perf_counter()
    for _ in range(3):
        for a in actors:fn(a)
    timings[name]=time.perf_counter()-t
print(json.dumps(dict(exact_grids=len(actors),seconds=timings)))
