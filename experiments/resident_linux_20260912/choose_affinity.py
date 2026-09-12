import json,os,time
from pathlib import Path
# GPU0 NUMA affinity observed from nvidia-smi topo: node0, CPUs0..51/104..155.
# Use two distinct physical cores, not sibling hyperthreads; only our processes.
def snapshot():
    out={}
    for line in Path('/proc/stat').read_text().splitlines():
        p=line.split()
        if p and p[0].startswith('cpu') and p[0][3:].isdigit():
            n=int(p[0][3:]);v=list(map(int,p[1:]));out[n]=(sum(v),v[3]+v[4])
    return out
a=snapshot();time.sleep(.5);b=snapshot();allowed=os.sched_getaffinity(0)
scores=[]
for n in range(2,52):
    if n in allowed:
        total=b[n][0]-a[n][0];idle=b[n][1]-a[n][1]
        scores.append(((total-idle)/max(total,1),n))
scores.sort();chosen=[n for _,n in scores[:2]]
assert len(chosen)==2
r=dict(native_cpu=chosen[0],policy_cpu=chosen[1],sample_utilization=scores[:8],gpu_numa=0)
Path(__file__).with_name('affinity.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
