"""Isolated real-data recurrent PPO head canary, bounded asynchronous collector/learner.

Backbone/RNN stay frozen, so head publication at segment boundaries does not invalidate
carried recurrent state. This is NOT full-network training, league training, or a release.
"""
import os,sys,time,json,copy,hashlib,queue,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/root/autodl-tmp/resident-linux-20260912')
CORE=Path('/root/autodl-tmp/bc-cloud-bench-20260911/core')
sys.path[:0]=[str(CORE),str(BASE),'/root/autodl-tmp/resident-opt-20260912',str(ROOT)]
import numpy as np
import torch
from torch import nn
from hokoff_model.match_agent import load_release
from hokoff_model.live import NativeMaskProvider,canonical_position_to_native
from policy_test import ResidentEnv,collate
from optimized_agent import MatchAgent
from fast_history import FastHistory
from dense_forward import install as install_dense
from native_core.client import JsonLineClient
from expert_v1.tick_store_v1.schema import normalize_native_state

HEADS=('timing.','kind.','card_score.','ability_score.','position_head.','position_skip.')
GAMMA=.999;LAMBDA=.95;STEPS=32;LANES=4;MAX_LAG=1
def head(name):return name.startswith(HEADS)
def digest(state,selector=lambda k:True):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if selector(k):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(123)
    model,encoder,identity=load_release(BASE/'model319051/inference.pt',BASE/'model319051/encoder-contract.json',device='cuda')
    install_dense(model)
    torch.manual_seed(457)
    value=nn.Sequential(nn.Linear(model.config.width,128),nn.Tanh(),nn.Linear(128,1)).cuda().eval()
    return model,encoder,value,identity
def arrays(batch):return {k:v.detach().cpu().numpy().copy() for k,v in batch.items()}
def tensor_batch(batch):return {k:torch.from_numpy(v).cuda() for k,v in batch.items()}
def snapshot(model,value,version):
    return dict(version=version,heads={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items() if head(k)},
                value=arrays(value.state_dict()),published=time.time())
def apply_snapshot(model,value,snap):
    state=model.state_dict()
    with torch.no_grad():
        for k,v in snap['heads'].items():
            if not head(k) or k not in state:raise ValueError('publication modifies frozen backbone')
            state[k].copy_(torch.from_numpy(v).to(state[k]))
    value.load_state_dict({k:torch.from_numpy(v).cuda() for k,v in snap['value'].items()},strict=True)
def mask_arrays(m):return dict(cards=m.cards.copy(),positions=m.positions.copy(),abilities=m.abilities.copy())
def dist(logits,mask):
    mask=torch.as_tensor(mask,device=logits.device,dtype=torch.bool)
    if not bool(mask.any()):raise ValueError('empty categorical distribution')
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-torch.inf))
def joint_logp(out,row,m,record=None):
    kindmask=np.array([m['cards'].any(),m['abilities'].any()])
    if not kindmask.any():return out['timing'][row,0]*0.,dict(play=0,kind=-1,slot=-1,position=-1)
    timing=torch.distributions.Bernoulli(logits=out['timing'][row,0])
    play=int(timing.sample()) if record is None else record['play']
    logp=timing.log_prob(torch.tensor(float(play),device='cuda'))
    if not play:return logp,dict(play=0,kind=-1,slot=-1,position=-1)
    kd=dist(out['kind'][row,0],kindmask);kind=int(kd.sample()) if record is None else record['kind'];logp=logp+kd.log_prob(torch.tensor(kind,device='cuda'))
    if kind==0:
        cd=dist(out['card'][row,0],m['cards']);slot=int(cd.sample()) if record is None else record['slot'];logp=logp+cd.log_prob(torch.tensor(slot,device='cuda'))
        pd=dist(out['position'][row,0,slot],m['positions'][slot]);pos=int(pd.sample()) if record is None else record['position'];logp=logp+pd.log_prob(torch.tensor(pos,device='cuda'))
    else:
        ad=dist(out['ability'][row,0],m['abilities']);slot=int(ad.sample()) if record is None else record['slot'];pos=-1;logp=logp+ad.log_prob(torch.tensor(slot,device='cuda'))
    return logp,dict(play=1,kind=kind,slot=slot,position=pos)
def command(record,m,side,deck):
    if not record['play']:return None
    if record['kind']==0:
        di=int(m.hand[record['slot']]);x,y=canonical_position_to_native(record['position'],side)
        return dict(side=side,deck_index=di,x=x,y=y,card_id=int(deck[di]['card_id']))
    return dict(type='ability',side=side,entity_id=int(m.ability_keys[record['slot']]))
def potential(raw):
    s=normalize_native_state(raw)
    own=sum(t.hp/max(1,t.max_hp) for t in s.towers if t.side==0)
    enemy=sum(t.hp/max(1,t.max_hp) for t in s.towers if t.side==1)
    return (own-enemy)/3.

def collect(rollouts,published,stop,outdir,segments):
    outdir=Path(outdir);stats=dict(segments=[],accepted_plays=0,nonempty_frames=0,frames=0,queue_wait_seconds=0.,versions=[],
                                  new_entity_frames=0,moving_entity_frames=0,peak_new_entities=0,
                                  behavior_mode='stochastic_joint_fixed4_v1',opponent_mode='greedy_fixed4')
    model,encoder,value,identity=setup();opponent=copy.deepcopy(model).eval();opponent.lstm.flatten_parameters()
    # Restore opponent's method bound to its own copied parameters.
    install_dense(opponent)
    old_opp=digest(opponent.state_dict());old_backbone=digest(model.state_dict(),lambda k:not head(k))
    c=JsonLineClient(port=25431,timeout=30);envs=[ResidentEnv(c,i,25431) for i in range(LANES)]
    replay=json.loads((CORE/'examples/hog-2.6-evo-hero.json').read_text());states=[];agents=[];epochs=[0]*LANES
    version=0;initial_keys=[set() for _ in range(LANES)];positions=[{} for _ in range(LANES)];grid_hashes=set()
    def reset(i):
        r=copy.deepcopy(replay);r['rndSeed']=818000+i+epochs[i]*1000;epochs[i]+=1
        state=envs[i].reset(r)
        normalized=normalize_native_state(state)
        initial_keys[i]={e.key for e in normalized.entities};positions[i]={e.key:(e.x,e.y) for e in normalized.entities}
        for agent in agents[i]:agent.reset(12)
        return state
    try:
        for i in range(LANES):
            agents.append([MatchAgent(model,encoder,side=0,device='cuda'),MatchAgent(opponent,encoder,side=1,device='cuda')])
            for agent in agents[-1]:agent.history=FastHistory(4)
            states.append(reset(i))
        for segment_id in range(segments):
            if stop.is_set():break
            while True:
                try:snap=published.get_nowait()
                except queue.Empty:break
                if snap['version']>version:apply_snapshot(model,value,snap);version=snap['version']
            start=time.time();steps=[]
            h0=[torch.cat([agents[i][0].hidden[k] for i in range(LANES)],1).detach().cpu().numpy().copy() for k in (0,1)]
            for tick_index in range(STEPS):
                prepared=[[agent.prepare(states[i],envs[i].decks,batch_device='cpu') for agent in agents[i]] for i in range(LANES)]
                cpu_batch=collate([p[0][0] for p in prepared]);batch={k:v.cuda() for k,v in cpu_batch.items()}
                hidden=tuple(torch.cat([agents[i][0].hidden[k] for i in range(LANES)],1) for k in (0,1))
                with torch.inference_mode():output,h=model.forward_stream(batch,hidden);values=value(output['context']).squeeze(-1)[:,0]
                actions=[[] for _ in range(LANES)];records=[];masks=[];logps=[];before_phi=[]
                for i in range(LANES):
                    agent=agents[i][0];p=prepared[i][0];normalized=agent._prepared_frame[1]
                    if agent.mask_provider is None:agent.mask_provider=NativeMaskProvider(envs[i])
                    m=agent.mask_provider.for_side(normalized,0,envs[i].decks[0],p[1].ability_entity_keys[0],p[1].ability_mask[0,0].numpy())
                    ma=mask_arrays(m)
                    with torch.inference_mode():lp,record=joint_logp(output,i,ma)
                    action=command(record,m,0,envs[i].decks[0])
                    if action is not None:actions[i].append(action)
                    agent.hidden=tuple(v[:,i:i+1].clone().detach() for v in h);agent.last_decision_tick=p[2];agent.decisions+=1
                    other=agents[i][1];op=prepared[i][1]
                    with torch.inference_mode():oo,oh=opponent.forward_stream({k:v.cuda() for k,v in op[0].items()},other.hidden)
                    oa,_=other.finish(op,oo,oh,states[i],envs[i].decks,envs[i])
                    if oa is not None:actions[i].append(oa)
                    masks.append(ma);records.append(record);logps.append(float(lp));before_phi.append(potential(states[i]))
                    stats['frames']+=1;stats['nonempty_frames']+=int(len(normalized.entities)>0)
                    new_count=sum(e.key not in initial_keys[i] for e in normalized.entities)
                    moving=any(e.key in positions[i] and positions[i][e.key]!=(e.x,e.y) for e in normalized.entities)
                    stats['new_entity_frames']+=int(new_count>0);stats['moving_entity_frames']+=int(moving)
                    stats['peak_new_entities']=max(stats['peak_new_entities'],new_count)
                    positions[i]={e.key:(e.x,e.y) for e in normalized.entities}
                    grid_hashes.add(hashlib.sha256(p[0]['grid'].numpy().tobytes()).hexdigest())
                receipt=c.request(dict(op='resident_batch',raw_response=True,entries=[dict(slot=i,steps=4,actions=envs[i]._joint_payload(actions[i])) for i in range(LANES)]))['results']
                rewards=[];dones=[];resets=[]
                for r in receipt:
                    i=r['slot'];before=states[i]['tick'];after=r['state']['tick'];ep=r['step']['episode'];done=bool(ep['terminated'])
                    assert done or after-before==4
                    assert all(item['result']['accepted'] for item in r['joint_action']['actions']),'native rejection'
                    for a in agents[i]:a.record_transition(before,after,actions[i],r['joint_action'],envs[i].decks,terminal=done)
                    stats['accepted_plays']+=len(actions[i]);states[i]=envs[i]._enrich_state(r['state'])
                    terminal=(1. if ep['outcome']=='side0_win' else -1. if ep['outcome']=='side1_win' else 0.) if done else 0.
                    rewards.append(terminal+GAMMA*(0. if done else potential(states[i]))-before_phi[i]);dones.append(done)
                    if done:states[i]=reset(i)
                steps.append(dict(batch=arrays(cpu_batch),masks=masks,actions=records,old_logp=np.asarray(logps,np.float32),
                                  old_value=values.cpu().numpy().copy(),reward=np.asarray(rewards,np.float32),done=np.asarray(dones,np.bool_)))
            # Peek next state without committing another recurrent update.
            next_prepared=[agents[i][0].prepare(states[i],envs[i].decks,batch_device='cpu')[0] for i in range(LANES)]
            nb={k:v.cuda() for k,v in collate(next_prepared).items()};nh=tuple(torch.cat([agents[i][0].hidden[k] for i in range(LANES)],1) for k in (0,1))
            with torch.inference_mode():no,_=model.forward_stream(nb,nh);bootstrap=value(no['context']).squeeze(-1)[:,0].cpu().numpy().copy()
            segment=dict(id=segment_id,version=version,behavior_mode='stochastic_joint_fixed4_v1',backbone=old_backbone,h0=h0,steps=steps,bootstrap=bootstrap,started=start,ended=time.time())
            t=time.time()
            while not stop.is_set():
                try:rollouts.put(segment,timeout=.2);break
                except queue.Full:continue
            stats['queue_wait_seconds']+=time.time()-t
            stats['segments'].append(dict(id=segment_id,version=version,start=start,end=segment['ended']))
            stats['versions'].append(version)
            (outdir/'collector-progress.json').write_text(json.dumps(stats))
        if not stop.is_set():rollouts.put(None,timeout=30)
        stats['opponent_unchanged']=digest(opponent.state_dict())==old_opp
        stats['backbone_unchanged']=digest(model.state_dict(),lambda k:not head(k))==old_backbone
        stats['distinct_grid_frames']=len(grid_hashes)
        (outdir/'collector.json').write_text(json.dumps(stats,indent=2))
    except BaseException as e:
        (outdir/'collector-error.json').write_text(json.dumps(dict(error=repr(e))));stop.set();raise
    finally:c.close()

def learn(rollouts,published,stop,outdir):
    outdir=Path(outdir);model,encoder,value,identity=setup()
    for name,p in model.named_parameters():p.requires_grad_(head(name))
    parameters=[p for p in model.parameters() if p.requires_grad]+list(value.parameters())
    optimizer=torch.optim.Adam(parameters,lr=1e-5)
    initial_backbone=digest(model.state_dict(),lambda k:not head(k));initial_head=digest(model.state_dict(),head)
    version=0;stats=dict(updates=[],dropped=[],source_identity=identity,trainable_parameters=sum(p.numel() for p in parameters))
    try:
        while not stop.is_set():
            try:segment=rollouts.get(timeout=10)
            except queue.Empty:continue
            if segment is None:break
            lag=version-segment['version']
            if lag<0 or segment['backbone']!=initial_backbone or segment['behavior_mode']!='stochastic_joint_fixed4_v1':raise ValueError('rollout contract mismatch')
            if lag>MAX_LAG:stats['dropped'].append(dict(id=segment['id'],lag=lag));continue
            started=time.time();rows=segment['steps'];T=len(rows)
            oldv=np.stack([r['old_value'] for r in rows]);rew=np.stack([r['reward'] for r in rows]);done=np.stack([r['done'] for r in rows])
            adv=np.zeros_like(rew);carry=np.zeros(LANES,np.float32);nextv=segment['bootstrap']
            for t in range(T-1,-1,-1):
                mask=1.-done[t];delta=rew[t]+GAMMA*mask*nextv-oldv[t]
                carry=delta+GAMMA*LAMBDA*mask*carry;adv[t]=carry;nextv=oldv[t]
            returns=torch.from_numpy(adv+oldv).cuda();advantages=torch.from_numpy((adv-adv.mean())/(adv.std()+1e-8)).cuda()
            oldlog=torch.from_numpy(np.stack([r['old_logp'] for r in rows])).cuda()
            saved={k:v.detach().clone() for k,v in model.state_dict().items() if head(k)};saved_value=copy.deepcopy(value.state_dict());saved_opt=copy.deepcopy(optimizer.state_dict())
            batches=[tensor_batch(r['batch']) for r in rows]
            hidden=tuple(torch.from_numpy(v).cuda() for v in segment['h0']);lp=[];pred=[]
            torch.cuda.synchronize();optimizer.zero_grad(set_to_none=True)
            for t,(r,b) in enumerate(zip(rows,batches)):
                output,hidden=model.forward_stream(b,hidden)
                values=value(output['context']).squeeze(-1)[:,0];pred.append(values)
                lp.append(torch.stack([joint_logp(output,i,r['masks'][i],r['actions'][i])[0] for i in range(LANES)]))
                if np.any(r['done']):
                    mask=torch.from_numpy(r['done']).cuda();hidden=tuple(h.masked_fill(mask[None,:,None],0) for h in hidden)
            logp=torch.stack(lp);predicted=torch.stack(pred)
            ratio=(logp-oldlog).exp();ploss=-torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()
            vloss=.5*(predicted-returns).square().mean();loss=ploss+vloss
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite PPO loss')
            pre_difference=float((logp-oldlog).detach().abs().max())
            if lag==0 and version==0 and pre_difference>2e-4:raise ValueError('behavior logp recomputation mismatch')
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
            if not np.isfinite(grad):raise FloatingPointError('nonfinite gradient')
            optimizer.step();torch.cuda.synchronize()
            # Post-update joint KL estimate on the same recorded masks/actions.
            h=tuple(torch.from_numpy(v).cuda() for v in segment['h0']);post=[]
            with torch.inference_mode():
                for r,b in zip(rows,batches):
                    output,h=model.forward_stream(b,h);post.append(torch.stack([joint_logp(output,i,r['masks'][i],r['actions'][i])[0] for i in range(LANES)]))
                    if np.any(r['done']):h=tuple(v.masked_fill(torch.from_numpy(r['done']).cuda()[None,:,None],0) for v in h)
                diff=torch.stack(post)-oldlog;kl=float((diff.exp()-1.-diff).mean())
            if not np.isfinite(kl) or kl>.02:
                with torch.no_grad():
                    for k,v in saved.items():model.state_dict()[k].copy_(v)
                value.load_state_dict(saved_value);optimizer.load_state_dict(saved_opt)
                raise RuntimeError('PPO KL guard failed; unpublished update rolled back')
            if digest(model.state_dict(),lambda k:not head(k))!=initial_backbone:raise ValueError('frozen backbone changed')
            version+=1;ended=time.time()
            metric=dict(version=version,segment=segment['id'],behavior_version=segment['version'],lag=lag,
                        queue_age_seconds=started-segment['ended'],started=started,ended=ended,samples=T*LANES,
                        loss=float(loss.detach()),policy_loss=float(ploss.detach()),value_loss=float(vloss.detach()),gradient_norm=grad,kl=kl,
                        behavior_logp_max_error=pre_difference,reward_abs_sum=float(np.abs(rew).sum()))
            stats['updates'].append(metric)
            torch.save(dict(kind='isolated_head_ppo_canary_v1',version=version,source_identity=identity,
                            frozen_backbone_sha256=initial_backbone,actor_heads={k:v.detach().cpu() for k,v in model.state_dict().items() if head(k)},
                            value={k:v.detach().cpu() for k,v in value.state_dict().items()},optimizer=optimizer.state_dict(),metrics=metric),outdir/f'update-{version:03d}.pt')
            snap=snapshot(model,value,version)
            try:published.put_nowait(snap)
            except queue.Full:
                try:published.get_nowait()
                except queue.Empty:pass
                published.put(snap,timeout=5)
            (outdir/'learner-progress.json').write_text(json.dumps(stats,indent=2))
        stats['head_changed']=digest(model.state_dict(),head)!=initial_head
        stats['backbone_unchanged']=digest(model.state_dict(),lambda k:not head(k))==initial_backbone
        stats['final_version']=version
        (outdir/'learner.json').write_text(json.dumps(stats,indent=2))
    except BaseException as e:
        (outdir/'learner-error.json').write_text(json.dumps(dict(error=repr(e))));stop.set();raise
    finally:
        # Collector may have finished; don't wait forever for unused final publications.
        published.cancel_join_thread()

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='learner-canary');parser.add_argument('--segments',type=int,default=8)
    args=parser.parse_args();assert args.label.replace('-','').isalnum() and 4<=args.segments<=16
    out=ROOT/args.label;out.mkdir(exist_ok=False)
    ctx=mp.get_context('spawn');rollouts=ctx.Queue(maxsize=2);published=ctx.Queue(maxsize=2);stop=ctx.Event()
    ps=[ctx.Process(target=learn,args=(rollouts,published,stop,str(out))),ctx.Process(target=collect,args=(rollouts,published,stop,str(out),args.segments))]
    for p in ps:p.start()
    (out/'pids.json').write_text(json.dumps([p.pid for p in ps]));deadline=time.monotonic()+300
    try:
        while any(p.is_alive() for p in ps):
            if time.monotonic()>deadline:raise TimeoutError('bounded learner canary')
            if any(p.exitcode not in (None,0) for p in ps):raise RuntimeError('canary child failed')
            time.sleep(.2)
        for p in ps:p.join();assert p.exitcode==0
        c=json.loads((out/'collector.json').read_text());l=json.loads((out/'learner.json').read_text())
        overlap=any(max(s['start'],u['started'])<min(s['end'],u['ended']) for s in c['segments'] for u in l['updates'])
        assert l['final_version']>=2 and l['head_changed'] and l['backbone_unchanged'] and c['opponent_unchanged'] and c['new_entity_frames']>0 and c['moving_entity_frames']>0 and c['distinct_grid_frames']>1 and overlap
        result=dict(passed=True,updates=l['final_version'],segments=len(c['segments']),samples=len(c['segments'])*STEPS*LANES,
                    accepted_plays=c['accepted_plays'],nonempty_frames=c['nonempty_frames'],frames=c['frames'],
                    overlap_observed=overlap,collector_versions=c['versions'],dropped_segments=l['dropped'],
                    max_policy_lag=max(u['lag'] for u in l['updates']),head_changed=True,backbone_unchanged=True,opponent_unchanged=True)
        result.update({k:c[k] for k in ('new_entity_frames','moving_entity_frames','peak_new_entities','distinct_grid_frames','behavior_mode','opponent_mode')})
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        stop.set()
        for p in ps:
            p.join(timeout=5)
            if p.is_alive():p.terminate();p.join(timeout=5)
        for q in (rollouts,published):q.cancel_join_thread();q.close()
if __name__=='__main__':main()
