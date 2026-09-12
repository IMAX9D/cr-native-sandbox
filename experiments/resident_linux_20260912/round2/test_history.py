"""CPU-only exact output/guard equivalence including reset and same-tick boundaries."""
import random, sys, time, json
from pathlib import Path
sys.path.insert(0, 'D:/Deepseek/outputs/hokoff-match-box-20260911/worktree')
from hokoff_model.online_history import OnlinePlayHistory, PublicPlay
from fast_history import FastHistory
import torch

torch.set_num_threads(1)
rng=random.Random(1234);checks=0
for length in (1,4,16):
    old,new=OnlinePlayHistory(length),FastHistory(length)
    for tick in range(300):
        if tick % 97 == 0: old.reset();new.reset()
        for side in (0,1):
            if rng.random()<.35:
                event=PublicPlay(tick,side,rng.randint(1,180),rng.randrange(576),rng.random()>.1)
                assert old.record(event)==new.record(event)
                assert old.record(event)==new.record(event)
        for side in (0,1):
            a,b=old.query(side,tick),new.query(side,tick)
            assert list(a)==list(b)
            for key in a: assert torch.equal(a[key],b[key]),(length,tick,side,key)
            checks+=1
old,new=OnlinePlayHistory(4),FastHistory(4)
for tick in range(8):
    event=PublicPlay(tick,tick%2,tick+1,tick*5);old.record(event);new.record(event)
times={}
for name,h in [('reference',old),('numpy',new)]:
    start=time.perf_counter()
    for _ in range(3000):h.query(0,100)
    times[name]=time.perf_counter()-start
print(json.dumps(dict(exact_queries=checks,seconds=times,speedup=times['reference']/times['numpy'])))
