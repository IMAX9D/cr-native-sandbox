"""Attribute CUDA activity to CPU launch scopes; do not equate stream spans with kernels."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
folder=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'cpu-gpu-profile')
aggregate=collections.defaultdict(float);by_phase=collections.defaultdict(float);kernels=collections.defaultdict(float);cpu_ops=collections.defaultdict(float);counts=collections.Counter();window_us=0
for path in sorted(p for p in folder.glob('trace-*.json') if p.stem.removeprefix('trace-').isdigit()):
    trace=json.loads(path.read_text());events=trace['traceEvents']
    phases=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('phase/')]
    cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    runtime={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime'}
    steps=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('ProfilerStep#')]
    window_us+=sum(e.get('dur',0) for e in steps)
    for e in events:
        cat=e.get('cat','');dur=e.get('dur',0);name=e.get('name','')
        if cat in ('kernel','gpu_memcpy','gpu_memset'):
            kind='kernel' if cat=='kernel' else ('h2d' if 'HtoD' in name else 'd2h' if 'DtoH' in name else cat)
            aggregate[kind]+=dur;counts[kind]+=1
            if cat=='kernel':kernels[name]+=dur
            parent=cpu.get(e.get('args',{}).get('External id')) or runtime.get(e.get('args',{}).get('correlation'))
            candidates=[p for p in phases if parent and p.get('tid')==parent.get('tid') and p['ts']<=parent['ts']<=p['ts']+p.get('dur',0)]
            phase=min(candidates,key=lambda p:p.get('dur',0))['name'] if candidates else 'unattributed'
            by_phase[phase+'/'+kind]+=dur
        elif cat=='cpu_op':cpu_ops[name]+=dur
result=dict(cuda_activity_us=dict(aggregate),cuda_event_counts=dict(counts),sampled_step_wall_us=window_us,
            cuda_by_phase_us=dict(by_phase),top_cuda_kernels_us=sorted(kernels.items(),key=lambda x:-x[1])[:20],
            top_cpu_op_inclusive_us=sorted(cpu_ops.items(),key=lambda x:-x[1])[:20],
            note='CUDA activity durations may overlap; CPU ops are inclusive. Stream-span metrics are reported separately.')
(folder/'trace-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
