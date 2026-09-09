#!/usr/bin/env python3
"""Small fixed4 self-play PPO run with frozen IL anchor and exact KL rollback."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import torch
from policy_v1.data import digest
from policy_v1.train import load_checkpoint
from native_core.env import NativeRoyaleEnv
from run_hokoff_fixed import LIBG_SHA256,write_json,show
from hokoff_model.live import FixedLiveAgent
from hokoff_model.ppo import PPOConfig,FixedPPO,state_digest
from hokoff_model.ppo_rollout import collect_episode
from hokoff_model.ppo_vector import COLLECTOR_MODE,collect_parallel
from hokoff_model.ppo_decks import opponent_deck_config
from hokoff_model.ppo_opponent import FrozenOpponent,OPPONENT_MODE,resolve_win_rate_window


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path('/home/lenovo/cr-data')
    p.add_argument('--il-checkpoint',type=Path,default=base/'runs/hokoff-fixed-p4/last.pt')
    p.add_argument('--manifest',type=Path,default=base/'expert-dataset/native-bc-v1/manifest.json')
    p.add_argument('--worker-dir',type=Path,default=Path('/data/local/tmp/cr-native-direct-0'))
    p.add_argument('--replay',type=Path,help='new runs default to hokoff_model/fixtures/hog_2_6.json; resume keeps saved fixture')
    p.add_argument('--learner-deck',choices=['random','fixed'],default=None,
                   help='new runs: random legal base deck; resume inherits saved mode')
    p.add_argument('--opponent-deck',choices=['random','fixed'],default=None,
                   help='new runs: random legal base deck; resume inherits saved mode')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=39031)
    p.add_argument('--iterations',type=int,default=1,help='total complete iterations, including any resumed iterations')
    p.add_argument('--episodes-per-iteration',type=int,default=None,help='new runs: max(workers,2); resume inherits')
    p.add_argument('--workers',type=int,default=None,help='concurrent games on consecutive ports; new runs default 12, resume inherits')
    p.add_argument('--ppo-epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--actor-lr',type=float,default=1e-5)
    p.add_argument('--critic-lr',type=float,default=3e-4)
    p.add_argument('--il-kl-limit',type=float,default=.02)
    p.add_argument('--il-action-kl-limit',type=float,default=.03)
    p.add_argument('--update-kl-limit',type=float,default=.01)
    p.add_argument('--max-state-kl',type=float,default=.2)
    p.add_argument('--kl-coefficient',type=float,default=1.)
    p.add_argument('--anchor-size',type=int,default=512)
    p.add_argument('--full-elixir-penalty-per-tick',type=float,default=None,
                   help='nonnegative penalty; new runs default 0.001, resume inherits saved value; 0 disables')
    p.add_argument('--win-rate-window',type=int,default=None,
                   help='even rolling game window; new runs default 100, resume inherits saved value')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--max-decisions',type=int,default=2500)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path,help='explicitly resume last.pt; partial collection is discarded, never retried')
    return p


def config_from_args(a):
    return PPOConfig(actor_lr=a.actor_lr,critic_lr=a.critic_lr,epochs=a.ppo_epochs,batch_size=a.batch_size,
        il_kl_limit=a.il_kl_limit,il_action_kl_limit=a.il_action_kl_limit,update_kl_limit=a.update_kl_limit,
        max_state_kl=a.max_state_kl,kl_coefficient=a.kl_coefficient,anchor_size=a.anchor_size)


def resolve_full_elixir_penalty(requested,contract=None):
    saved=None if contract is None else float(contract.get('full_elixir_penalty_per_tick',0.))
    value=(.001 if saved is None else saved) if requested is None else requested
    if not math.isfinite(value) or value<0:raise ValueError('full elixir penalty must be finite and nonnegative')
    if saved is not None and value!=saved:raise ValueError('resume full_elixir_penalty_per_tick differs; start a new experiment')
    return value


def save_checkpoint(path,trainer,agent,contract,generator,iteration,opponent):
    value=dict(kind='hokoff_fixed_ppo_checkpoint_v1',config=asdict(agent.model.config),
        model=trainer.actor.state_dict(),contract=dict(manifest_sha256=contract['manifest_sha256']),
        step=contract['source_step']+trainer.accepted_updates,critic=trainer.critic.state_dict(),
        actor_optimizer=trainer.actor_optimizer.state_dict(),critic_optimizer=trainer.critic_optimizer.state_dict(),
        ppo_config=asdict(trainer.config),run_contract=contract,next_iteration=iteration,
        anchor=trainer.anchor,beta=trainer.beta,accepted_updates=trainer.accepted_updates,
        rejected_updates=trainer.rejected_updates,generator_state=generator.get_state(),
        opponent=opponent.state_dict())
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)


def run(args):
    config=config_from_args(args);config.validate()
    if min(args.iterations,args.max_decisions,args.cpu_threads)<1:
        raise ValueError('positive iteration/episode/decision/thread settings required')
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(args.worker_dir/'libg.so')!=LIBG_SHA256:raise ValueError('wrong frozen native runtime')
    resumed=None
    if args.resume:
        resumed=load_checkpoint(args.resume)
        if resumed.get('kind')!='hokoff_fixed_ppo_checkpoint_v1':raise ValueError('not a fixed4 PPO checkpoint')
        output=args.resume.resolve().parent
        if args.output and args.output.resolve()!=output:raise ValueError('resume must use its existing run directory')
        contract=resumed['run_contract']
        if contract.get('opponent')!=OPPONENT_MODE or 'opponent' not in resumed:
            raise ValueError('legacy self-play checkpoint has no fixed-opponent stage; start a new run from the original IL')
        args.workers=contract.get('workers',1) if args.workers is None else args.workers
        if args.workers!=contract.get('workers',1):raise ValueError('resume worker count differs; start a new parallel experiment')
        args.episodes_per_iteration=contract['episodes_per_iteration'] if args.episodes_per_iteration is None else args.episodes_per_iteration
        deck_config=opponent_deck_config(args.opponent_deck,json.loads(args.manifest.read_text()),saved=contract.get('opponent_deck_config',dict(mode='fixed')))
        learner_deck_config=opponent_deck_config(args.learner_deck,json.loads(args.manifest.read_text()),
            saved=contract.get('learner_deck_config',dict(mode='fixed')),role='learner')
        window=resolve_win_rate_window(args.win_rate_window,contract)
        penalty=resolve_full_elixir_penalty(args.full_elixir_penalty_per_tick,contract)
        if resumed['ppo_config']!=asdict(config):raise ValueError('resume PPO settings differ')
        for key in ('seed','device','episodes_per_iteration','max_decisions'):
            if getattr(args,key)!=contract[key]:raise ValueError('resume '+key+' differs')
        source=output/'il_source.pt'
        if digest(source)!=contract['source_sha256']:raise ValueError('frozen IL source changed')
        if digest(args.manifest)!=contract['manifest_sha256']:raise ValueError('resume vocabulary changed')
        fixture=json.loads((output/'fixture.json').read_text())
        if digest(output/'fixture.json')!=contract['fixture_sha256']:raise ValueError('resume fixture changed')
        if args.replay and digest(args.replay)!=contract['replay_source_sha256']:raise ValueError('resume replay differs')
    else:
        args.workers=12 if args.workers is None else args.workers
        args.episodes_per_iteration=max(args.workers,2) if args.episodes_per_iteration is None else args.episodes_per_iteration
        if args.workers<1 or args.episodes_per_iteration<args.workers:raise ValueError('episodes per iteration must be >= workers >=1')
        penalty=resolve_full_elixir_penalty(args.full_elixir_penalty_per_tick)
        window=resolve_win_rate_window(args.win_rate_window)
        deck_config=opponent_deck_config(args.opponent_deck,json.loads(args.manifest.read_text()))
        learner_deck_config=opponent_deck_config(args.learner_deck,json.loads(args.manifest.read_text()),role='learner')
        output=args.output or Path('/home/lenovo/cr-data/runs')/('hokoff-fixed-ppo-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        output.mkdir(parents=True,exist_ok=False)
        source=output/'il_source.pt';original_hash=digest(args.il_checkpoint)
        shutil.copyfile(args.il_checkpoint,source)
        if digest(source)!=original_hash or digest(args.il_checkpoint)!=original_hash:raise ValueError('IL checkpoint changed during snapshot')
        replay_path=args.replay or Path(__file__).resolve().parent/'hokoff_model/fixtures/hog_2_6.json'
        fixture=NativeRoyaleEnv.read_replay(replay_path);fixture['cmd']=[]
        write_json(output/'fixture.json',fixture)
        original=load_checkpoint(source)
        if original.get('kind') in ('hokoff_fixed_ppo_checkpoint_v1','hokoff_fixed_beat_snapshot_v1'):
            raise ValueError('use --resume to preserve the original IL anchor; a PPO checkpoint is not a new IL source')
        contract=dict(kind='hokoff_fixed_ppo_run_v1',source_sha256=original_hash,
            source_checkpoint=str(args.il_checkpoint.resolve()),source_step=original['step'],
            manifest_sha256=digest(args.manifest),fixture_sha256=digest(output/'fixture.json'),
            replay_source_sha256=digest(replay_path),device=args.device,seed=args.seed,
            episodes_per_iteration=args.episodes_per_iteration,max_decisions=args.max_decisions,
            workers=args.workers,collector_mode=COLLECTOR_MODE,opponent_deck_config=deck_config,learner_deck_config=learner_deck_config,
            decision_period=4,policy_distribution='masked_bernoulli_kind_card_position_or_ability_v1',
            kl_direction='IL_reference_to_current',conditional_action_kl=True,
            trainable_actor='action_heads_only',critic='independent_mlp_on_detached_frozen_context',
            opponent=OPPONENT_MODE,promotion_window=window,promotion_threshold=.95,
            promotion_evidence='rolling_training_games',reward='defensive_tower_damage_v1',
            full_elixir_penalty_per_tick=penalty,
            full_elixir_penalty_rule='pre_tick_elixir_raw_ge_100000_without_accepted_card_v1',
            gamma_per_tick=.99995,gae_lambda_per_tick=.995,local_libg_sha256=LIBG_SHA256,
            ppo_config=asdict(config))
        write_json(output/'contract.json',contract)
    if args.workers<1 or args.episodes_per_iteration<args.workers:raise ValueError('episodes per iteration must be >= workers >=1')
    agent=FixedLiveAgent.load(source,args.manifest,device=args.device)
    reference=deepcopy(agent.model).eval().requires_grad_(False)
    trainer=FixedPPO(agent.model,reference,config,device=args.device)
    opponent=FrozenOpponent(reference,window=window)
    source_contract=load_checkpoint(source)['contract']
    generator=torch.Generator(device=args.device);generator.manual_seed(args.seed)
    for side in (0,1):agent.encoder._deck(fixture['battle'][f'deck{side}']['sp'])
    start_iteration=0
    if resumed:
        trainer.actor.load_state_dict(resumed['model']);trainer.critic.load_state_dict(resumed['critic'])
        opponent.load_state_dict(resumed['opponent'])
        trainer.actor_optimizer.load_state_dict(resumed['actor_optimizer'])
        trainer.critic_optimizer.load_state_dict(resumed['critic_optimizer'])
        trainer.anchor=resumed['anchor'];trainer.beta=resumed['beta']
        trainer.accepted_updates=resumed['accepted_updates'];trainer.rejected_updates=resumed['rejected_updates']
        generator.set_state(resumed['generator_state'].cpu());start_iteration=resumed['next_iteration']
        if state_digest(trainer.actor,frozen_only=True)!=trainer.frozen_hash:raise ValueError('resume modified frozen encoder')
    else:save_checkpoint(output/'last.pt',trainer,agent,contract,generator,0,opponent)
    if args.iterations<=start_iteration:
        raise ValueError(f'--iterations is a cumulative target; set it above {start_iteration} to continue')
    opponent.check_body(trainer.actor)
    opponent.export_latest(output,source_contract=source_contract)
    write_json(output/'opponents.json',dict(current=opponent.statistics(),promotions=opponent.promotions))
    result=dict(opponent=opponent.statistics(),status='running',output=str(output),start_iteration=start_iteration,next_iteration=start_iteration,iterations=[])
    exit_code=0;started=time.perf_counter()
    try:
        with ExitStack() as stack:
            envs=[stack.enter_context(NativeRoyaleEnv(host=args.host,port=args.port+i,timeout=args.timeout)) for i in range(args.workers)]
            for env in envs:env._request({'op':'ping'})
            for iteration in range(start_iteration,args.iterations):
                iteration_started=time.perf_counter();episodes=[];batches=[]
                opponent.assert_unchanged();opponent.check_body(trainer.actor)
                collection_policy_step=contract['source_step']+trainer.accepted_updates
                collection_opponent_hash=opponent.model_hash
                # Completed iterations are atomic. Failed/interrupt partial iterations
                # are kept as diagnostics and resumed from the previous checkpoint.
                logpath=output/f'iteration-{iteration:04d}-{datetime.now().strftime("%H%M%S-%f")}-collection.jsonl'
                if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
                collection_started=time.perf_counter()
                with logpath.open('w') as log:
                    if contract.get('collector_mode')==COLLECTOR_MODE:
                        ids=list(range(iteration*args.episodes_per_iteration,(iteration+1)*args.episodes_per_iteration))
                        batches,episodes,collection=collect_parallel(agent,trainer,envs,fixture,episode_ids=ids,
                            seed=args.seed,opponent=opponent.model,opponent_number=opponent.number,log=log,
                            max_decisions=args.max_decisions,full_elixir_penalty_per_tick=penalty,
                            deck_config=deck_config,learner_deck_config=learner_deck_config,replays_dir=output/'episode-replays')
                    else:
                        # Keep old one-env checkpoint behavior/RNG when resuming.
                        for index in range(args.episodes_per_iteration):
                            episode_id=iteration*args.episodes_per_iteration+index
                            replay=deepcopy(fixture);replay['rndSeed']=args.seed+episode_id
                            data,episode=collect_episode(agent,trainer,envs[0],replay,learner_side=episode_id%2,
                                generator=generator,log=log,episode_id=episode_id,max_decisions=args.max_decisions,
                                full_elixir_penalty_per_tick=penalty,opponent=opponent.model,opponent_number=opponent.number)
                            batches.append(data);episodes.append(episode)
                        seconds=time.perf_counter()-collection_started
                        collection=dict(collection_seconds=seconds,native_ticks=sum(e['native_ticks'] for e in episodes))
                        collection['collection_ticks_per_second']=collection['native_ticks']/seconds
                    # Commit game statistics in episode order, never thread completion order.
                    for episode in episodes:
                        opponent.record(episode,policy_step=collection_policy_step)
                        show('PPO collection',dict(**opponent.statistics(),learner_deck_mode=learner_deck_config['mode'],opponent_deck_mode=deck_config['mode'],iteration=iteration+1,episode=episode['episode'],
                             decisions=episode['decisions'],accepted_actions=episode['accepted_actions'],
                             reward_sum=episode['reward_sum'],base_reward_sum=episode['base_reward_sum'],
                             full_elixir_penalty_sum=episode['full_elixir_penalty_sum'],
                             full_elixir_idle_ticks=episode['full_elixir_idle_ticks'] if penalty else 'not_measured',
                             learner_card_plays=episode['learner_card_plays'],first_card_tick=episode['first_card_tick'],
                             outcome=episode['terminal']['outcome']))
                data={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
                torch.save(dict(data=data,episodes=episodes),output/f'rollout-{iteration:04d}.pt')
                opponent.assert_unchanged()
                win_stats=opponent.statistics()
                promotion=opponent.maybe_promote(trainer.actor,policy_step=collection_policy_step,iteration=iteration+1)
                update_started=time.perf_counter()
                stats=trainer.update(data,generator=generator)
                update_seconds=time.perf_counter()-update_started
                opponent.assert_unchanged()
                record=dict(iteration=iteration+1,episodes=episodes,win_rate=win_stats,promotion=promotion,
                    workers=args.workers,collection=collection,update_seconds=update_seconds,
                    peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else 0.,
                    collection_opponent_sha256=collection_opponent_hash,opponent_sha256=opponent.model_hash,
                    collection_policy_step=collection_policy_step,
                    elapsed_seconds=time.perf_counter()-iteration_started,**stats)
                write_json(output/f'iteration-{iteration:04d}-update.json',record)
                save_checkpoint(output/'last.pt',trainer,agent,contract,generator,iteration+1,opponent)
                opponent.export_latest(output,source_contract=source_contract)
                write_json(output/'opponents.json',dict(current=opponent.statistics(),promotions=opponent.promotions))
                result['opponent']=opponent.statistics()
                if promotion:
                    show('Opponent surpassed',dict(saved_model=promotion['filename'],
                         defeated_opponent=promotion['opponent_number'],win_rate=promotion['win_rate'],
                         wins=promotion['wins'],games=promotion['win_rate_games'],
                         next_opponent=opponent.name,opponents_beaten=len(opponent.promotions)))
                result['iterations'].append(record);result['next_iteration']=iteration+1
                display={k:record[k] for k in ('iteration','accepted_updates','rejected_updates',
                    'il_kl','il_action_kl','il_action_kl_max','update_kl','update_action_kl','beta','actor_lr','elapsed_seconds')}
                # Small KLs and the conservative learning rate must not print as 0.0000.
                for key,value in display.items():
                    if isinstance(value,float):display[key]=f'{value:.6g}'
                show('Anchored PPO update',display)
                show('PPO throughput',dict(workers=args.workers,**collection,update_seconds=update_seconds,
                     iteration_ticks_per_second=collection['native_ticks']/record['elapsed_seconds'],peak_cuda_mb=record['peak_cuda_mb']))
                write_json(output/'summary.json',result)
            result['status']='completed'
    except KeyboardInterrupt:result['status']='interrupted';exit_code=130
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        (output/'error.txt').write_text(traceback.format_exc());exit_code=1
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        result['opponent_parameters_frozen']=state_digest(opponent.model)==opponent.model_hash and not any(p.requires_grad for p in opponent.model.parameters())
        result['frozen_il_file_unchanged']=digest(source)==contract['source_sha256']
        result['reference_parameters_unchanged']=state_digest(reference)==trainer.reference_hash
        result['encoder_lstm_unchanged']=state_digest(trainer.actor,frozen_only=True)==trainer.frozen_hash
        if not all(result[k] for k in ('frozen_il_file_unchanged','reference_parameters_unchanged','encoder_lstm_unchanged','opponent_parameters_frozen')):
            result['status']='failed';exit_code=1
        write_json(output/'summary.json',result)
        show('PPO run finished',dict(status=result['status'],next_iteration=result['next_iteration'],output=str(output)))
        if result.get('error'):print(result['error'],flush=True)
    return exit_code


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
