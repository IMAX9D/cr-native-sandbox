"""Read-only delay evaluation on raw recorded states; no UI or game simulation."""
import argparse
from bisect import bisect_right
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from policy_v1.data import ENTITY_FIELDS, ROW_TOKENS, close_arrays, digest
from policy_v1.train import load_checkpoint, move
from .decision_data import DecisionWindows, collate_decisions, ragged_indices
from .decision_model import DecisionConfig, DecisionPolicy
from .metrics import bc_loss as action_loss, summarize as action_summary


class LocatedWindows(DecisionWindows):
    def __getitem__(self, i):
        b = super().__getitem__(i)
        sh = bisect_right(self.prefix, i)-1
        local = i-self.prefix[sh]
        seq = int(np.searchsorted(self.sequence_prefix[sh], local, side='right')-1)
        begin, end = self.records[sh]['offsets'][seq:seq+2]
        target = begin+(local-int(self.sequence_prefix[sh][seq]))*self.targets
        start = max(begin, target-self.frame_window+1)
        _, d = self._open(sh)
        b['source_row'] = torch.from_numpy(d['rows'][start:start+len(b['frame_ticks'])].copy())
        b['source_shard'] = torch.full_like(b['source_row'], sh)
        b['source_segment'] = torch.full_like(b['source_row'], seq)
        return b


class Timelines:
    def __init__(self, dataset):
        self.ds = dataset
        self.arrays, self.indices, self.segments = {}, {}, {}

    def open(self, shard):
        if shard not in self.arrays:
            a = self.ds._open_source(shard)
            for key in ('sequence_offsets', 'delta_ticks', 'timing_exposure_ticks'):
                a[key] = np.load(self.ds.shard_paths[shard]/(key+'.npy'), mmap_mode='r')
            if not np.all(a['delta_ticks']==1) or not np.all(a['timing_exposure_ticks']==1):
                raise ValueError('raw evaluation requires contiguous one-tick source')
            self.arrays[shard] = a
            with np.load(self.ds.index_paths[shard]) as z:
                self.indices[shard] = {k:z[k] for k in ('rows','offsets','owners','elapsed')}
        return self.arrays[shard]

    def segment(self, shard, seq):
        key = (shard, seq)
        if key not in self.segments:
            a = self.open(shard); d = self.indices[shard]
            owner = int(d['owners'][seq])
            lo, hi = map(int, a['sequence_offsets'][owner:owner+2])
            start = int(d['rows'][d['offsets'][seq]])
            invalid = np.flatnonzero(~np.asarray(a['timing_label_mask'][start:hi], dtype=bool))
            end = start+int(invalid[0]) if len(invalid) else hi
            actions = np.flatnonzero(a['play_now'][start:end])+start
            initial_tick = int(round(float(a['public_scalars'][lo,0])*6000))
            identity = self.ds.records[shard]['identities'][seq]
            self.segments[key] = dict(shard=shard, seq=seq, start=start, end=end, actions=actions,
                tick_offset=initial_tick-lo, battle=identity['battle_tag'], side=identity['actor_side'])
        return self.segments[key]

    def close(self):
        for a in self.arrays.values(): close_arrays(a)
        self.arrays.clear()


def observations(a, rows, ticks, elapsed):
    """Gather current public observations only, with no expert action labels."""
    rows = np.asarray(rows, dtype=np.int64); T = len(rows)
    b = {k:torch.from_numpy(np.asarray(a[k][rows],dtype=np.int64)) for k in ROW_TOKENS}
    b['public_scalars'] = torch.from_numpy(np.asarray(a['public_scalars'][rows], dtype=np.float32))
    b['action_kind_mask'] = torch.from_numpy(np.asarray(a['action_kind_mask'][rows],dtype=bool))
    b['frame_ticks'] = torch.as_tensor(ticks, dtype=torch.int64)
    b['prev_elapsed_ticks'] = torch.as_tensor(elapsed, dtype=torch.float32)
    b['frame_mask'] = torch.ones(T,dtype=torch.bool)
    b['event_ticks'] = torch.empty(0,dtype=torch.int64)  # shared collate; never consumed by this model
    grid = np.zeros((T,8*576),dtype=np.float32)
    q, _, idx = ragged_indices(a['grid_offsets'], rows)
    grid[q,a['grid_indices'][idx]] = a['grid_values'][idx]/255.
    b['grid'] = torch.from_numpy(grid.reshape(T,8,32,18))
    q, col, idx = ragged_indices(a['entity_offsets'],rows)
    N = max(1,int(col.max())+1) if len(col) else 1
    for k in ENTITY_FIELDS[:-1]:
        shape = (T,N,a[k].shape[-1]) if k=='entity_numeric' else (T,N)
        v = np.zeros(shape,dtype=np.float32 if k=='entity_numeric' else np.int64)
        v[q,col] = a[k][idx]; b[k] = torch.from_numpy(v)
    mask = np.zeros((T,N),dtype=bool); mask[q,col] = True
    b['entity_mask'] = torch.from_numpy(mask)
    return b


def choose_mode(timing, kind, legal):
    deploy = timing > 0
    deploy = deploy & legal.any(-1)
    kind = kind.float().masked_fill(~legal, -1e9).argmax(-1)
    return torch.where(deploy,kind+1,0)


def choose_delay(logits, mode):
    return logits.gather(-2, mode[...,None,None].expand(*mode.shape,1,logits.shape[-1])).squeeze(-2).argmax(-1)+1


def streaming_step(model, b, state=None):
    # Evaluation adapter only: gameplay forward_stream remains unchanged.
    x = model.encode(b)
    out, state = model.lstm(x,state)
    context = model.context(out)
    logits = model.delay(context).reshape(*context.shape[:2],3,model.config.max_delay)
    mode = choose_mode(model.timing(context).squeeze(-1),model.kind(context),b['action_kind_mask'])
    return choose_delay(logits,mode).squeeze(1),state


def rolling_step(model, b, history=None):
    """Diagnostic: recompute LSTM over the last frame_window observed frames only."""
    x=model.encode(b).squeeze(1); B,H=x.shape; W=model.config.frame_window
    if history is None:
        frames=x.new_zeros(B,W,H); lengths=torch.zeros(B,dtype=torch.int64,device=x.device)
    else:
        frames,lengths=history
        frames=frames.clone()
    full=lengths==W
    frames=torch.where(full[:,None,None],torch.roll(frames,-1,dims=1),frames)
    frames[torch.arange(B,device=x.device),lengths.clamp_max(W-1)]=x
    lengths=(lengths+1).clamp_max(W)
    out,_=model.recurrent(frames,lengths)
    last=out[torch.arange(B,device=x.device),lengths-1].unsqueeze(1)
    context=model.context(last)
    logits=model.delay(context).reshape(B,1,3,model.config.max_delay)
    mode=choose_mode(model.timing(context).squeeze(-1),model.kind(context),b['action_kind_mask'])
    return choose_delay(logits,mode).squeeze(1),(frames,lengths)


def point_metrics(wait, next_action):
    wait, next_action = np.asarray(wait),np.asarray(next_action)
    late = np.maximum(wait-next_action,0)
    crossing = late>0
    near = next_action < 8
    return dict(points=len(wait),mean_wait_ticks=float(wait.mean()),
        cross_next_action_rate=float(crossing.mean()),
        cross_rate_when_action_within_8_ticks=float(crossing[near].mean()) if near.any() else 0.,
        late_ticks_per_point=float(late.mean()),
        late_ticks_per_crossing=float(late[crossing].mean()) if crossing.any() else 0.,
        late_p95_ticks=float(np.quantile(late[crossing],.95)) if crossing.any() else 0.)


def timing_ranking(probability, actual):
    probability=np.asarray(probability);actual=np.asarray(actual,dtype=bool)
    order=np.argsort(-probability,kind='stable');scores=probability[order];labels=actual[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    tp=np.cumsum(labels)[ends];precision=tp/(ends+1)
    ap=float(np.sum(precision*np.diff(np.r_[0,tp]))/max(int(actual.sum()),1))
    thresholds={}
    for threshold in (.01,.02,.05,.1,.2,.5):
        pred=probability>threshold;correct=int((pred&actual).sum())
        thresholds[str(threshold)]=dict(precision=correct/max(int(pred.sum()),1),
            recall=correct/max(int(actual.sum()),1),predicted_action_rate=float(pred.mean()))
    return dict(average_precision=ap,positive_rate=float(actual.mean()),
        mean_probability_on_actions=float(probability[actual].mean()) if actual.any() else 0.,
        mean_probability_on_waits=float(probability[~actual].mean()) if (~actual).any() else 0.,
        max_probability=float(probability.max()),thresholds=thresholds)


def stage_one(model, ds, timelines, args, device, amp):
    ids=list(range(len(ds))); random.Random(123).shuffle(ids)
    ids=ids[:args.window_batches*args.batch_size]
    loader=DataLoader(ds,batch_size=args.batch_size,sampler=ids,num_workers=args.workers,
        collate_fn=collate_decisions,pin_memory=device.type=='cuda')
    traces=[]; action_stats={}; timing_probability=[]; timing_actual=[]
    for bi,b in enumerate(loader):
        gpu=move(b,device)
        with torch.inference_mode(),amp(): out=model(gpu)
        with torch.inference_mode():
            _,stats=action_loss(out,gpu)
            for key,value in stats.items():action_stats[key]=action_stats.get(key,0)+value
            mask=gpu['frame_mask'] & gpu['loss_mask'] & gpu['timing_label_mask']
            timing_probability.append(out['timing'][mask].float().sigmoid().cpu().numpy())
            timing_actual.append(gpu['play_now'][mask].cpu().numpy())
        mode=choose_mode(out['timing'],out['kind'],gpu['action_kind_mask'])
        oracle=torch.where(gpu['play_now'],gpu['action_kind']+1,0).clamp(0,2)
        predicted=choose_delay(out['delay'],mode).cpu().numpy()
        teacher=choose_delay(out['delay'],oracle).cpu().numpy()
        mode=mode.cpu().numpy()
        valid=(b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']).numpy()
        for sh,seq,row,pred,teach,m in zip(b['source_shard'].numpy()[valid],b['source_segment'].numpy()[valid],
                b['source_row'].numpy()[valid],predicted[valid],teacher[valid],mode[valid]):
            seg=timelines.segment(int(sh),int(seq))
            # Require the whole common 8-tick horizon to lie inside valid raw data.
            if row+8>=seg['end']: continue
            actions=seg['actions']; j=np.searchsorted(actions,row,side='right')
            distance=int(actions[j]-row) if j<len(actions) else 32767
            traces.append((sh,seq,row,pred,teach,m,distance))
        if (bi+1)%25==0: print('point batches',bi+1,flush=True)
    trace=np.asarray(traces,dtype=np.int64)
    if not len(trace): raise ValueError('no fully known evaluation points')
    np.savez_compressed(args.output/'point_predictions.npz',columns=np.array(['shard','segment','row','model_delay','expert_mode_delay','model_mode','next_action_ticks']),rows=trace)
    d=trace[:,6]; model_wait=trace[:,3]
    result={f'fixed{k}':point_metrics(np.full(len(d),k),d) for k in (4,6,7,8)}
    result['model']=point_metrics(model_wait,d)
    result['model_expert_current_mode']=point_metrics(trace[:,4],d)
    rng=np.random.default_rng(args.seed)
    for name in ('shuffled','shuffled_within_mode'):
        repeats=[]
        for _ in range(args.shuffle_trials):
            shuffled=model_wait.copy()
            if name=='shuffled': shuffled=rng.permutation(shuffled)
            else:
                for m in range(3):
                    ix=np.flatnonzero(trace[:,5]==m); shuffled[ix]=rng.permutation(shuffled[ix])
            repeats.append(point_metrics(shuffled,d))
        result[name]={k:float(np.mean([r[k] for r in repeats])) for k in repeats[0]}
        result[name]['cross_rate_interval95']=np.quantile([r['cross_next_action_rate'] for r in repeats],[.025,.975]).tolist()
    return dict(metrics=result,action_metrics=action_summary(action_stats),
        timing_ranking=timing_ranking(np.concatenate(timing_probability),np.concatenate(timing_actual)),
        raw_valid_points=len(trace),source_windows=len(ids),shuffle_trials=args.shuffle_trials)


def score_schedule(times, seg, reserve=8):
    times=np.asarray(times,dtype=np.int64)
    if not len(times) or times[0]!=seg['start'] or np.any(np.diff(times)<=0) or times[-1]>=seg['end']:
        raise ValueError('invalid observation schedule')
    events=seg['actions'][seg['actions']<seg['end']-reserve]
    indices=np.searchsorted(times,events,side='left')
    if np.any(indices==len(times)): raise ValueError('common event horizon has no next observation')
    delays=times[indices]-events
    return dict(observations=len(times),valid_ticks=seg['end']-seg['start'],events=len(events),
                late_events=int((delays>0).sum()),delay_sum=int(delays.sum()),
                excluded_tail_events=int(len(seg['actions'])-len(events)),delays=delays.tolist())


def aggregate(scores):
    total={k:sum(s[k] for s in scores) for k in ('observations','valid_ticks','events','late_events','delay_sum','excluded_tail_events')}
    delay=np.array([x for s in scores for x in s['delays']])
    total.update(observations_per_1000_ticks=1000*total['observations']/max(total['valid_ticks'],1),
        late_event_rate=total['late_events']/max(total['events'],1),
        mean_event_lateness_ticks=total['delay_sum']/max(total['events'],1),
        event_lateness_p95_ticks=float(np.quantile(delay,.95)) if len(delay) else 0,
        on_time_event_rate=1-total['late_events']/max(total['events'],1))
    return total


def shuffled_schedule(times,rng):
    # Same first/last observation, exact same number and multiset of internal waits.
    return np.r_[times[0],times[0]+np.cumsum(rng.permutation(np.diff(times)))]


def audit_elapsed_shortcut(timelines, output):
    def counts(elapsed,actual):
        predicted=(elapsed>0)&(elapsed<8)
        return np.array([(predicted&actual).sum(),(predicted&~actual).sum(),
                         (~predicted&actual).sum(),(~predicted&~actual).sum()],dtype=np.int64)
    def summary(c):
        tp,fp,fn,tn=map(int,c)
        return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
    with np.load(output/'point_predictions.npz') as z: trace=z['rows']
    point=np.zeros(4,dtype=np.int64)
    for sh in np.unique(trace[:,0]):
        sh=int(sh); rows=trace[trace[:,0]==sh,2]
        a=timelines.open(sh); d=timelines.indices[sh]
        ids=np.searchsorted(d['rows'],rows)
        point+=counts(d['elapsed'][ids],np.asarray(a['play_now'][rows],dtype=bool))
    visited=np.zeros(4,dtype=np.int64)
    for seg in json.loads((output/'state_schedules.json').read_text()):
        rows=np.asarray(seg['observations'])
        a=timelines.open(seg['shard'])
        visited+=counts(np.r_[0,np.diff(rows)],np.asarray(a['play_now'][rows],dtype=bool))
    return dict(rule='0 < prev_elapsed_ticks < 8 predicts play_now',
        point_sampling=timelines.ds.index['decision_contract'],
        at_dataset_decision_points=summary(point),
        at_expert_decision_points=summary(point),at_model_visited_states=summary(visited),
        interpretation='sampling-induced shortcut diagnostic; not proof that the network uses only this rule')


def stage_two(model,ds,timelines,args,device,amp):
    tags=sorted({t for r in ds.records for t in r['battle_tags']})
    chosen=set(random.Random(args.seed).sample(tags,min(args.battles,len(tags))))
    segments=[]
    for sh,r in enumerate(ds.records):
        for seq,identity in enumerate(r['identities']):
            if identity['battle_tag'] in chosen:
                seg=timelines.segment(sh,seq)
                if seg['end']-seg['start']>8: segments.append(seg)
    schedules=[]
    for begin in range(0,len(segments),args.replay_batch_size):
        chunk=segments[begin:begin+args.replay_batch_size]; N=len(chunk)
        current=np.array([s['start'] for s in chunk]); elapsed=np.zeros(N,dtype=np.int64)
        end=np.array([s['end'] for s in chunk]); times=[[] for _ in chunk]
        state=None; full_state=None; full_history=None
        while (current<end).any():
            active=np.flatnonzero(current<end)
            items=[]
            for i in active:
                seg=chunk[i]; row=current[i]
                items.append(observations(timelines.open(seg['shard']),[row],[row+seg['tick_offset']],[elapsed[i]]))
                times[i].append(int(row))
            b=move(collate_decisions(items),device)
            active_gpu=torch.as_tensor(active,device=device)
            with torch.inference_mode(),amp():
                if args.replay_history=='rolling':
                    history=None if full_history is None else tuple(s.index_select(0,active_gpu) for s in full_history)
                    wait,new=rolling_step(model,b,history)
                    if full_history is None:
                        full_history=tuple(s.new_zeros((N,)+s.shape[1:]) for s in new)
                    for saved,updated in zip(full_history,new): saved.index_copy_(0,active_gpu,updated)
                else:
                    state=None if full_state is None else tuple(s.index_select(1,active_gpu) for s in full_state)
                    wait,new=streaming_step(model,b,state)
                    if full_state is None:
                        full_state=tuple(s.new_zeros(1,N,s.shape[-1]) for s in new)
                    for saved,updated in zip(full_state,new): saved.index_copy_(1,active_gpu,updated)
            wait=wait.cpu().numpy()
            current[active]+=wait; elapsed[active]=wait
        schedules.extend(times)
        print('state replay segments',min(begin+N,len(segments)),'/',len(segments),flush=True)
    policies={'model':schedules}
    for k in (4,6,7,8): policies[f'fixed{k}']=[list(range(s['start'],s['end'],k)) for s in segments]
    policies['uniform_same_budget']=[np.linspace(t[0],t[-1],len(t)).astype(np.int64).tolist() for t in schedules]
    scored={name:[score_schedule(t,s) for t,s in zip(ts,segments)] for name,ts in policies.items()}
    result={name:aggregate(scores) for name,scores in scored.items()}
    rng=np.random.default_rng(args.seed)
    trials=[]; shuffled_scores=[]
    for j in range(args.shuffle_trials):
        ss=[score_schedule(shuffled_schedule(t,rng),s) for t,s in zip(schedules,segments)]
        trials.append(aggregate(ss)); shuffled_scores.append(ss)
    result['shuffled_same_budget']={k:float(np.mean([r[k] for r in trials])) for k in trials[0]}
    result['shuffled_same_budget']['late_event_rate_interval95']=np.quantile([r['late_event_rate'] for r in trials],[.025,.975]).tolist()
    # Paired battle bootstrap: account for both actor views belonging to one battle.
    by_battle={tag:np.zeros(4,dtype=float) for tag in sorted(chosen)}
    for i,s in enumerate(segments):
        by_battle[s['battle']]+=np.array([scored['model'][i]['events'],scored['model'][i]['late_events'],
            np.mean([trial[i]['late_events'] for trial in shuffled_scores]),scored['uniform_same_budget'][i]['late_events']])
    blocks=np.stack(list(by_battle.values())); diffs=[]; uniform_diffs=[]
    for _ in range(2000):
        total=blocks[rng.integers(0,len(blocks),len(blocks))].sum(0)
        if total[0]:
            diffs.append((total[1]-total[2])/total[0]); uniform_diffs.append((total[1]-total[3])/total[0])
    paired=dict(model_minus_shuffled_late_rate_interval95=np.quantile(diffs,[.025,.975]).tolist(),
                model_minus_uniform_late_rate_interval95=np.quantile(uniform_diffs,[.025,.975]).tolist())
    trace=[dict(shard=s['shard'],segment=s['seq'],battle=s['battle'],side=s['side'],start=s['start'],end=s['end'],observations=t) for s,t in zip(segments,schedules)]
    (args.output/'state_schedules.json').write_text(json.dumps(trace))
    gaps=np.concatenate([np.diff(t) for t in schedules])
    return dict(metrics=result,battles=len(chosen),actor_segments=len(segments),paired_battle_bootstrap=paired,
                replay_history=args.replay_history,model_internal_gap_histogram=np.bincount(gaps,minlength=9)[1:].tolist(),
                excluded_common_tail_ticks_per_segment=8,shuffle_trials=args.shuffle_trials)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path.home()/'cr-data'
    p.add_argument('--data',type=Path,default=base/'expert-dataset/native-bc-v1')
    p.add_argument('--cache',type=Path,default=base/'hokoff-decision-cache-k8')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window-batches',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--battles',type=int,default=64)
    p.add_argument('--replay-batch-size',type=int,default=64)
    p.add_argument('--replay-history',choices=['continuous','rolling'],default='continuous',
                   help='rolling is a diagnostic using the last frame_window observed frames, not intervening raw frames')
    p.add_argument('--shuffle-trials',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260907)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--allow-evaluation-cache-change',action='store_true',
                   help='evaluate another observation sampling of the same source manifest and validation split')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.window_batches,args.batch_size,args.battles,args.replay_batch_size,args.shuffle_trials)<1 or args.workers<0:
        raise ValueError('invalid evaluation size')
    if args.output.exists(): raise FileExistsError('use a new output directory')
    torch.set_num_threads(4)
    saved=load_checkpoint(args.checkpoint)
    config=DecisionConfig(**saved['config'])
    if config.max_delay!=8: raise ValueError('this evaluation contract uses max_delay=8')
    if saved['contract']['decision_cache_sha256']!=digest(args.cache/'index.json'):
        index=json.loads((args.cache/'index.json').read_text())
        if not args.allow_evaluation_cache_change or index['manifest_sha256']!=saved['contract'].get('manifest_sha256'):
            raise ValueError('checkpoint cache differs; same-source sampling comparison needs explicit opt-in')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    model=DecisionPolicy(config).to(device); model.load_state_dict(saved['model']); model.eval()
    amp=lambda:torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
    ds=LocatedWindows(args.data,args.cache,saved['contract']['val_split'],targets=saved['contract']['targets'],frame_window=config.frame_window,max_delay=8)
    timelines=Timelines(ds)
    if any(any(r.get('segment_roles',[])) for r in ds.records):
        raise ValueError('schedule evaluation requires primary observation sequences without auxiliary action examples')
    args.output.mkdir(parents=True)
    results=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),checkpoint_step=saved['step'],
        evaluation_cache_sha256=digest(args.cache/'index.json'),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        limitations=['recorded expert world; no counterfactual actions executed','only public current observations reach model',
        'labels used only by scoring and expert-current-mode diagnostic','validation subset, not test or game win rate'])
    started=time.monotonic()
    try:
        results['pointwise']=stage_one(model,ds,timelines,args,device,amp)
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print('POINTWISE',json.dumps(results['pointwise']),flush=True)
        results['state_replay']=stage_two(model,ds,timelines,args,device,amp)
        results['elapsed_shortcut_audit']=audit_elapsed_shortcut(timelines,args.output)
        results['elapsed_seconds']=time.monotonic()-started
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print('STATE_REPLAY',json.dumps(results['state_replay']),flush=True)
    finally:
        ds.close(); timelines.close()


if __name__=='__main__': main()
