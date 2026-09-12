"""CPU-thread work, exclusive wall phases, CUDA spans and sampled CUDA traces."""
import argparse,collections,contextlib,copy,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from hokoff_model.match_agent import load_release
from split_agent import MatchAgent
from policy_test import ResidentEnv,collate
from profile_client import JsonLineClient

class Meter:
    def __init__(self,trace=False):self.stats=collections.defaultdict(lambda:dict(calls=0,wall_ns=0,cpu_ns=0));self.stack=[];self.trace=trace
    @contextlib.contextmanager
    def phase(self,name):
        w=time.perf_counter_ns();c=time.thread_time_ns();entry=[0,0];self.stack.append(entry)
        scope=torch.profiler.record_function('phase/'+name) if self.trace else contextlib.nullcontext()
        with scope:
            try:yield
            finally:
                dw=time.perf_counter_ns()-w;dc=time.thread_time_ns()-c;self.stack.pop();r=self.stats[name]
                r['calls']+=1;r['wall_ns']+=dw-entry[0];r['cpu_ns']+=dc-entry[1]
                if self.stack:self.stack[-1][0]+=dw;self.stack[-1][1]+=dc

class Client(JsonLineClient):
    def __init__(self,meter,**kw):super().__init__(profile={},**kw);self.meter=meter
    def request(self,payload):
        name='legality_rpc' if payload.get('mode')=='probe_grid' else 'simulation_rpc'
        with self.meter.phase(name):return super().request(payload)

def delta(a,b):
    if isinstance(a,dict):return {k:delta(a[k],b[k]) for k in a if k in b}
    if type(a) in (int,float) and type(b) in (int,float):return b-a
    return b

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--native-profile',action='store_true')
    p.add_argument('--label',default='pipeline-profile');p.add_argument('--max-cycles',type=int,default=2000)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,encoder,identity=load_release(ROOT/'model319051/inference.pt',contract_path=ROOT/'model319051/encoder-contract.json',device='cuda')
    out=ROOT/a.label;out.mkdir(exist_ok=True);m=Meter(a.trace);c=Client(m,port=41431,timeout=30)
    replay=json.loads(Path('/root/autodl-tmp/bc-cloud-bench-20260911/core/examples/hog-2.6-evo-hero.json').read_text())
    envs=[ResidentEnv(c,i,41431) for i in range(4)];states=[];agents=[]
    for i,e in enumerate(envs):
        r=copy.deepcopy(replay);r['rndSeed']=424242+i;states.append(e.reset(r))
        agents.append([MatchAgent(model,encoder,side=s,device='cuda') for s in (0,1)])
        for agent in agents[-1]:agent.reset(12)
    native_before=c.request({'op':'profile_stats'}) if a.native_profile else None
    m.stats.clear();c.profile.clear();events=[];active=list(range(4));finals={};game_hashes=[hashlib.sha256() for _ in range(4)];plays=[0]*4
    @contextlib.contextmanager
    def gpu_phase(name):
        with m.phase(name),torch.inference_mode():
            if a.trace or a.native_profile:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                yield
                end.record();events.append((name,start,end))
            else:yield
    trace_paths=[]
    def schedule(step):
        for lo in (32,500,1200):
            if step==lo-1:return torch.profiler.ProfilerAction.WARMUP
            if lo<=step<lo+31:return torch.profiler.ProfilerAction.RECORD
            if step==lo+31:return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.NONE
    def save_trace(prof):
        path=out/f'trace-{len(trace_paths)}.json';prof.export_chrome_trace(str(path));trace_paths.append(str(path))
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                   schedule=schedule,on_trace_ready=save_trace,record_shapes=True) if a.trace else contextlib.nullcontext()
    total_ticks=0;iterations=0;profile_step_wall=0.;start_wall=time.perf_counter();start_cpu=time.process_time()
    try:
        with profiler as prof:
            for cycle in range(a.max_cycles):
                prepared=[];who=[];actions={i:[] for i in active}
                with m.phase('observation_encode'):
                    for i in active:
                        for s,agent in enumerate(agents[i]):prepared.append(agent.prepare(states[i],envs[i].decks,batch_device='cpu'));who.append((i,s))
                with m.phase('batch_collate'),torch.inference_mode():cpu_batch=collate([x[0] for x in prepared])
                with gpu_phase('hidden_gather'):hidden=tuple(torch.cat([agents[i][s].hidden[k] for i,s in who],1) for k in (0,1))
                with gpu_phase('h2d'):batch={k:v.to('cuda') for k,v in cpu_batch.items()}
                with gpu_phase('model_forward'):
                    with torch.inference_mode():output,h=model.forward_stream(batch,hidden)
                with gpu_phase('gpu_guard_pack'):
                    assert bool(torch.stack([torch.isfinite(v).all() for v in [*output.values(),*h]]).all())
                    shapes={k:v.shape[1:] for k,v in output.items()};widths=[v[0].numel() for v in output.values()]
                    packed=torch.cat([v.reshape(len(who),-1) for v in output.values()],1)
                with gpu_phase('d2h'):host=packed.cpu()
                with gpu_phase('prediction_split'):
                    pieces=host.split(widths,1);output={k:v.reshape(len(who),*shapes[k]) for k,v in zip(output,pieces)}
                    predictions=[({k:v[n:n+1] for k,v in output.items()},tuple(v[:,n:n+1].clone() for v in h)) for n in range(len(who))]
                with m.phase('cuda_sync_wait'):torch.cuda.synchronize()
                with m.phase('action_decode'):
                    for p0,(i,s),(o,h0) in zip(prepared,who,predictions):
                        action,_=agents[i][s].finish(p0,o,h0,states[i],envs[i].decks,envs[i],batch_validated=True)
                        if action is not None:actions[i].append(action)
                with m.phase('request_prepare'):entries=[dict(slot=i,steps=4,actions=envs[i]._joint_payload(actions[i])) for i in active]
                values=c.request(dict(op='resident_batch',entries=entries,raw_response=True))['results']
                with m.phase('history_and_enrichment'):
                    for value in values:
                        i=value['slot'];before=states[i]['tick'];after=value['state']['tick'];done=value['step']['episode']['terminated']
                        assert done or after-before==4
                        for item in value['joint_action']['actions']:assert item['result']['accepted']
                        for agent in agents[i]:agent.record_transition(before,after,actions[i],value['joint_action'],envs[i].decks,terminal=done)
                        total_ticks+=after-before;plays[i]+=len(actions[i]);game_hashes[i].update((json.dumps((before,actions[i]),sort_keys=True)+'\n').encode())
                        states[i]=envs[i]._enrich_state(value['state'])
                        if done:
                            ep=value['step']['episode'];finals[i]=dict(tick=after,outcome=ep['outcome'],crowns=ep['crowns'],plays=plays[i],action_digest=game_hashes[i].hexdigest());active.remove(i)
                iterations+=1
                if a.trace:
                    t=time.perf_counter();prof.step();profile_step_wall+=time.perf_counter()-t
                if not active:break
        wall=time.perf_counter()-start_wall;cpu=time.process_time()-start_cpu
        rpc=dict(c.profile);phases=dict(m.stats);native_after=c.request({'op':'profile_stats'}) if a.native_profile else None
        cuda=collections.defaultdict(float)
        for name,s,e in events:cuda[name]+=s.elapsed_time(e)
        summary=dict(identity=identity,trace=a.trace,native_profile=a.native_profile,iterations=iterations,ticks=total_ticks,
                     wall_seconds=wall,actor_process_cpu_seconds=cpu,profiler_step_wall_seconds=profile_step_wall,
                     exclusive_phases=phases,rpc=rpc,cuda_stream_span_ms=dict(cuda),
                     native_delta=delta(native_before,native_after) if a.native_profile else None,
                     finals=finals,traces=trace_paths,torch_gpu_allocated_peak=torch.cuda.max_memory_allocated(),torch_gpu_reserved_peak=torch.cuda.max_memory_reserved())
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('identity','finals')}),flush=True)
    finally:c.close()
if __name__=='__main__':main()
