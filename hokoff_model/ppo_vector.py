"""Concurrent native games with one batched GPU policy and per-game RNG/LSTM."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy,deepcopy
import threading
import time
import torch
from .ppo_actions import action_heads,action_distribution,sample_actions
from .ppo_rollout import episode_stream
from .ppo_decks import episode_replay
from run_hokoff_fixed import show,write_json

COLLECTOR_MODE='batched_fixed4_episode_rng_v1'


def collate_live_batches(batches,device):
    result={}
    for key in batches[0]:
        values=[b[key] for b in batches]
        if key.startswith('entity_'):
            size=max(v.shape[2] for v in values);padded=[]
            for v in values:
                shape=list(v.shape);shape[2]=size
                out=v.new_zeros(shape);out[:,:,:v.shape[2]]=v;padded.append(out)
            values=padded
        result[key]=torch.cat(values,0).to(device)
    return result


@torch.no_grad()
def infer_requests(model,trainer,requests,generators,opponent):
    device=trainer.device
    batch=collate_live_batches([r['batch'] for r in requests],device)
    zeros=lambda:torch.zeros(model.lstm.num_layers,2,model.config.hidden_size,device=device)
    states=[r['hidden'] if r['hidden'] is not None else (zeros(),zeros()) for r in requests]
    hidden=tuple(torch.cat([s[k] for s in states],1).to(device) for k in (0,1))
    output,next_hidden=model.forward_stream(batch,hidden)
    features=dict(context=output['context'][:,0],hand_tokens=batch['hand_tokens'][:,0],
                  ability_tokens=batch['ability_tokens'][:,0])
    masks={k:torch.cat([r['masks'][k] for r in requests],0).to(device) for k in requests[0]['masks']}
    logits={k:output[k][:,0] for k in ('timing','kind','card','position','ability')}
    if opponent is not None:
        other=action_heads(opponent,features)
        learner=torch.tensor([side==r['learner_side'] for r in requests for side in (0,1)],device=device)
        logits={k:torch.where(learner.reshape(-1,*([1]*(v.ndim-1))),v,other[k]) for k,v in logits.items()}
    dist=action_distribution(logits,masks)
    indices=torch.cat([sample_actions({k:v[2*i:2*i+2] for k,v in dist.items()},g)
                       for i,g in enumerate(generators)])
    probabilities=dist['log_prob'].gather(1,indices[:,None]).squeeze(1)
    values=trainer.values(features).cpu()
    # Complete GPU work before returning; environment threads never run CUDA ops.
    features={k:v.detach().cpu() for k,v in features.items()}
    indices=indices.cpu();probabilities=probabilities.cpu()
    return [dict(hidden=tuple(s[:,2*i:2*i+2].detach() for s in next_hidden),
                 indices=indices[2*i:2*i+2],probabilities=probabilities[2*i:2*i+2],
                 features={k:v[2*i:2*i+2] for k,v in features.items()},
                 value=float(values[2*i+r['learner_side']])) for i,r in enumerate(requests)]


class LockedLog:
    def __init__(self,log):self.log=log;self.lock=threading.Lock()
    def write(self,value):
        with self.lock:self.log.write(value)
    def flush(self):
        with self.lock:self.log.flush()


def advance_stream(stream,prediction=None):
    try:return False,next(stream) if prediction is None else stream.send(prediction)
    except StopIteration as done:return True,done.value


def collect_parallel(agent,trainer,envs,fixture,*,episode_ids,seed,opponent,opponent_number,
                     log=None,max_decisions=2500,full_elixir_penalty_per_tick=.001,deck_config=None,learner_deck_config=None,replays_dir=None):
    if not envs or not episode_ids:raise ValueError('nonempty environment and episode lists required')
    if len(set(episode_ids))!=len(episode_ids):raise ValueError('episode IDs must be unique')
    started=time.perf_counter();pending=iter(episode_ids);active={};finished={};all_streams=[];replays={}
    if replays_dir is not None:replays_dir.mkdir(parents=True,exist_ok=True)
    locked_log=None if log is None else LockedLog(log)
    metrics=dict(inference_seconds=0.,environment_wait_seconds=0.,inference_batches=0,inference_actor_rows=0)
    def launch(slot):
        try:episode_id=next(pending)
        except StopIteration:return None
        local=copy(agent);local.device='cpu'
        local.encoder=copy(agent.encoder);local.encoder._deck_cache={};local.reset()
        replay=episode_replay(fixture,seed=seed,episode_id=episode_id,learner_side=episode_id%2,config=deck_config,learner_config=learner_deck_config)
        replays[episode_id]=replay
        if replays_dir is not None:write_json(replays_dir/f'episode-{episode_id:06d}.json',replay)
        # Scheduling and completion order cannot change another game's RNG.
        generator=torch.Generator(device=trainer.device);generator.manual_seed(seed+episode_id)
        stream=episode_stream(local,envs[slot],replay,learner_side=episode_id%2,episode_id=episode_id,
            log=locked_log,max_decisions=max_decisions,full_elixir_penalty_per_tick=full_elixir_penalty_per_tick,
            opponent_number=opponent_number,progress=False)
        all_streams.append(stream)
        active[slot]=dict(stream=stream,generator=generator,episode_id=episode_id)
        return stream
    executor=ThreadPoolExecutor(max_workers=len(envs),thread_name_prefix='native-game')
    try:
        futures={slot:executor.submit(advance_stream,stream) for slot in range(len(envs)) if (stream:=launch(slot)) is not None}
        while futures:
            wait_started=time.perf_counter();requests={};replacement={}
            for slot,future in futures.items():
                done,value=future.result()
                if done:
                    finished[active[slot]['episode_id']]=value
                    del active[slot]
                    if (stream:=launch(slot)) is not None:replacement[slot]=executor.submit(advance_stream,stream)
                else:requests[slot]=value
            # New games join the next inference batch with cleared LSTM/reveal state.
            for slot,future in replacement.items():
                done,value=future.result()
                if done:raise RuntimeError('episode ended before its first decision')
                requests[slot]=value
            metrics['environment_wait_seconds']+=time.perf_counter()-wait_started
            if not requests:break
            slots=sorted(requests)
            inference_started=time.perf_counter()
            predictions=infer_requests(agent.model,trainer,[requests[s] for s in slots],
                                       [active[s]['generator'] for s in slots],opponent)
            metrics['inference_seconds']+=time.perf_counter()-inference_started
            metrics['inference_batches']+=1;metrics['inference_actor_rows']+=2*len(slots)
            futures={s:executor.submit(advance_stream,active[s]['stream'],p) for s,p in zip(slots,predictions)}
            if metrics['inference_batches']%200==0:
                show('Parallel PPO collection',dict(active_games=len(active),completed_games=len(finished),
                     target_games=len(episode_ids),inference_batches=metrics['inference_batches'],
                     collection_seconds=time.perf_counter()-started))
                if locked_log:locked_log.flush()
    finally:
        # Bounded native socket timeout; never retry or restart an ambiguous game.
        executor.shutdown(wait=True,cancel_futures=True)
        for stream in all_streams:stream.close()
    ordered=[finished[i] for i in episode_ids]
    for _,episode in ordered:
        replay=replays[episode['episode']];side=episode['learner_side']
        if 'battle' in replay:
            episode['learner_deck_ids']=[c['d'] for c in replay['battle'][f'deck{side}']['sp']]
            episode['opponent_deck_ids']=[c['d'] for c in replay['battle'][f'deck{1-side}']['sp']]
    metrics['collection_seconds']=time.perf_counter()-started
    metrics['native_ticks']=sum(e['native_ticks'] for _,e in ordered)
    metrics['collection_ticks_per_second']=metrics['native_ticks']/metrics['collection_seconds']
    metrics['mean_inference_batch_games']=metrics['inference_actor_rows']/max(2*metrics['inference_batches'],1)
    return [d for d,_ in ordered],[e for _,e in ordered],metrics
