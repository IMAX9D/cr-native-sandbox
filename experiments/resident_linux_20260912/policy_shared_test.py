"""Four resident matches / eight player views, fixed4, frozen release weights."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE=Path('/root/autodl-tmp/bc-cloud-bench-20260911/core')
sys.path.insert(0,str(CORE));sys.path.insert(0,str(ROOT))
import torch
from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv
from hokoff_model.match_agent import load_release,MatchAgent as OriginalAgent,DEFAULT_CONTRACT
from split_agent import MatchAgent

def collate(batches):
    out={}
    for key in batches[0]:
        values=[b[key] for b in batches]
        if key.startswith('entity_'):
            width=max(v.shape[2] for v in values);padded=[]
            for v in values:
                shape=list(v.shape);shape[2]=width
                target=v.new_zeros(shape);target[:,:,:v.shape[2]]=v;padded.append(target)
            values=padded
        out[key]=torch.cat(values,0)
    return out

class ResidentEnv(NativeRoyaleEnv):
    def __init__(self,c,slot,port=39331):
        super().__init__(port=port);self.client=c;self.slot=slot
    def reset(self,replay):
        self._configure_replay(replay)
        self.client.request(dict(op='resident',slot=self.slot,mode='create',replay=replay))
        self.client.request(dict(op='resident',slot=self.slot,mode='step',steps=12))
        return self._enrich_state(self.client.request(dict(op='resident',slot=self.slot,mode='observe'))['state'])
    def probe_grid(self,*,side,deck_index):
        hi,lo=self.accounts[side]
        return self.client.request(dict(op='resident',slot=self.slot,mode='probe_grid',side=side,deck_index=deck_index,account_hi=hi,account_lo=lo))['result']
    def close(self):pass

def run(model,encoder,c,args,mode,seed):
    replay=json.loads((CORE/'examples/hog-2.6-evo-hero.json').read_text())
    envs=[ResidentEnv(c,i,args.port) for i in range(args.slots)]
    states=[];agents=[];refs=[]
    for i,env in enumerate(envs):
        config=copy.deepcopy(replay);config['rndSeed']=seed+i
        states.append(env.reset(config))
        agents.append([MatchAgent(model,encoder,side=s,device=args.device) for s in (0,1)])
        refs.append([OriginalAgent(model,encoder,side=s,device=args.device) for s in (0,1)])
        for a in agents[-1]+refs[-1]:a.reset(12)
    active=list(range(args.slots));ticks=plays=decisions=0;calls=0;verify_count=0;max_error=0.
    times=dict(encode=0.,forward=0.,decode=0.,native=0.)
    begin=time.perf_counter();finals={};peak_entities=0;action_log=[]
    for cycle in range(2200):
        if time.perf_counter()-begin>args.timeout:raise TimeoutError('policy case time limit')
        prepared=[];who=[];actions={i:[] for i in active}
        t=time.perf_counter()
        for i in active:
            peak_entities=max(peak_entities,len(states[i]['entities']))
            for side,a in enumerate(agents[i]):
                p=a.prepare(states[i],envs[i].decks,batch_device='cpu' if mode=='packed' else None);assert p is not None
                prepared.append(p);who.append((i,side))
        times['encode']+=time.perf_counter()-t;t=time.perf_counter()
        with torch.inference_mode():
            if mode in ('batch','packed'):
                batch={k:v.to(args.device) for k,v in collate([p[0] for p in prepared]).items()}
                hidden=tuple(torch.cat([agents[i][s].hidden[k] for i,s in who],1) for k in (0,1))
                output,h=model.forward_stream(batch,hidden);calls+=1
                if mode=='packed':
                    if not bool(torch.stack([torch.isfinite(v).all() for v in [*output.values(),*h]]).all()):
                        raise FloatingPointError('nonfinite batched policy')
                    shapes={k:v.shape[1:] for k,v in output.items()}
                    widths={k:v[0].numel() for k,v in output.items()}
                    packed=torch.cat([v.reshape(len(who),-1) for v in output.values()],1).cpu()
                    parts=packed.split(list(widths.values()),1)
                    output={k:v.reshape(len(who),*shapes[k]) for k,v in zip(output,parts)}
                predictions=[({k:v[n:n+1] for k,v in output.items()},tuple(v[:,n:n+1].clone() for v in h)) for n in range(len(who))]
            else:
                predictions=[model.forward_stream(p[0],agents[i][s].hidden) for p,(i,s) in zip(prepared,who)];calls+=len(who)
        if args.device=='cuda':torch.cuda.synchronize()
        times['forward']+=time.perf_counter()-t;t=time.perf_counter()
        for p,(i,s),(output,h) in zip(prepared,who,predictions):
            if args.verify:
                ref=refs[i][s]
                with torch.inference_mode():o0,h0=model.forward_stream({k:v.to(args.device) for k,v in p[0].items()},ref.hidden)
                for k in output:
                    reference=o0[k].to(output[k].device)
                    err=float((output[k]-reference).abs().max());max_error=max(max_error,err)
                    torch.testing.assert_close(output[k],reference,rtol=2e-4,atol=1e-4)
                for x,y in zip(h,h0):torch.testing.assert_close(x,y,rtol=2e-4,atol=1e-4)
                expected,audit0=ref.decide(states[i],envs[i].decks,envs[i])
            a,audit=agents[i][s].finish(p,output,h,states[i],envs[i].decks,envs[i],batch_validated=(mode=='packed'));decisions+=1
            if args.verify:
                assert a==expected,dict(slot=i,side=s,tick=states[i]['tick'],actual=a,expected=expected)
                verify_count+=1
            if a is not None:actions[i].append(a)
        times['decode']+=time.perf_counter()-t;t=time.perf_counter()
        entries=[dict(slot=i,steps=4,actions=envs[i]._joint_payload(actions[i])) for i in active]
        results=c.request(dict(op='resident_batch',entries=entries,raw_response=args.raw))['results']
        assert [r['slot'] for r in results]==active
        for r in results:
            i=r['slot'];before=states[i]['tick'];after=r['state']['tick'];ep=r['step']['episode'];done=bool(ep['terminated'])
            if not done and after-before!=4:
                (ROOT/'interval-failure.json').write_text(json.dumps(dict(before=before,after=after,result=r),indent=2))
                raise AssertionError(('native interval mismatch',i,before,after,ep))
            for item in r['joint_action']['actions']:assert item['result']['accepted'],item
            for a in agents[i]+(refs[i] if args.verify else []):
                a.record_transition(before,after,actions[i],r['joint_action'],envs[i].decks,terminal=done)
            plays+=len(actions[i]);ticks+=after-before
            action_log.append((i,before,actions[i]))
            states[i]=envs[i]._enrich_state(r['state'])
            if done:finals[i]=dict(tick=after,outcome=ep['outcome'],crowns=ep['crowns']);active.remove(i)
        times['native']+=time.perf_counter()-t
        if not active or (args.max_ticks and all(states[i]['tick']>=args.max_ticks for i in active)):break
        if cycle%250==0:print('progress',mode,cycle,plays,round(time.perf_counter()-begin,2),flush=True)
    else:raise AssertionError('cycle limit')
    wall=time.perf_counter()-begin
    result=dict(device=args.device,mode=mode,raw_response=args.raw,slots=args.slots,seed=seed,seconds=wall,ticks=ticks,tps=ticks/wall,
                plays=plays,decisions=decisions,model_calls=calls,verified_decisions=verify_count,max_logit_error=max_error,
                times=times,finals=finals,complete=len(finals)==args.slots,peak_entities=peak_entities,
                action_digest=hashlib.sha256(json.dumps(action_log,sort_keys=True).encode()).hexdigest())
    print(json.dumps(result),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--slots',type=int,default=4)
    p.add_argument('--verify',action='store_true');p.add_argument('--max-ticks',type=int,default=0)
    p.add_argument('--raw',action='store_true')
    p.add_argument('--port',type=int,default=39331)
    p.add_argument('--seed',type=int,default=424242)
    p.add_argument('--seed-stride',type=int,default=0)
    p.add_argument('--barrier')
    p.add_argument('--ready-file')
    p.add_argument('--checkpoint',default='/root/autodl-tmp/bc-cloud-bench-20260911/model.pt')
    p.add_argument('--contract',default=str(DEFAULT_CONTRACT))
    p.add_argument('--policy-server',type=int,default=41580)
    p.add_argument('--timeout',type=float,default=240);p.add_argument('--modes',default='scalar,batch');p.add_argument('--output',default='policy-results.json')
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.device='cpu'
    model,encoder,identity=load_release(a.checkpoint,contract_path=a.contract,device='cpu')
    from shared_policy import RemotePolicy
    model=RemotePolicy(model.config,identity['checkpoint_sha256'],a.policy_server)
    if a.ready_file:Path(a.ready_file).write_text(json.dumps(dict(pid=os.getpid(),ready=time.time(),identity=identity)))
    if a.barrier:
        until=time.monotonic()+120
        while not Path(a.barrier).exists():
            if time.monotonic()>until:raise TimeoutError('start barrier')
            time.sleep(.02)
    c=JsonLineClient(host='127.0.0.1',port=a.port,timeout=30);rows=[]
    try:
        for round_index,mode in enumerate(a.modes.split(',')):
            rows.append(run(model,encoder,c,a,mode,a.seed+round_index*a.seed_stride))
            (ROOT/a.output).write_text(json.dumps(dict(identity=identity,rows=rows),indent=2))
    finally:c.close();model.close()
if __name__=='__main__':main()
