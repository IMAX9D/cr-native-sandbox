from copy import deepcopy
import io
import json
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hokoff_model.tests.test_live import fixture
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.live import FixedLiveAgent,LiveMasks
from hokoff_model.ppo import FixedPPO,PPOConfig
from hokoff_model.ppo_vector import collect_parallel,infer_requests
from hokoff_model.ppo_rollout import collect_episode
from expert_selfplay_v1.native_observation import NativeObservationEncoder


class ProbeActivity:
    def __init__(self):self.lock=threading.Lock();self.active=0;self.peak=0


class Env:
    decks=[list(fixture.DECK),list(fixture.DECK)]
    def __init__(self,activity=None,fail=False):self.activity=activity;self.fail=fail;self.resets=[]
    def reset(self,replay,*,warmup_steps):
        self.seed=replay['rndSeed'];self.resets.append(self.seed)
        self.tick=warmup_steps;self.end=100+4*(1+self.seed%3)
    def observe_train(self):
        raw=fixture.state();raw['tick']=self.tick
        raw['players'][0]['elixir_raw']=100000
        raw['episode']['tower_snapshot_complete']=True
        return raw
    def joint_training_transition(self,commands,*,steps):
        assert not commands
        if self.activity:
            with self.activity.lock:
                self.activity.active+=1;self.activity.peak=max(self.activity.peak,self.activity.active)
            time.sleep(.005)
            with self.activity.lock:self.activity.active-=1
        self.tick+=steps;raw=self.observe_train()
        if self.tick==self.end:
            terminal=deepcopy(raw['episode'])
            terminal.update(terminated=True,truncated=False,terminal_tick=self.tick,
                            rewards=[1,-1],outcome='side0_win',tower_snapshot_complete=not self.fail)
            return dict(joint_action=dict(actions=[]),step=dict(episode=terminal))
        return dict(joint_action=dict(actions=[]),step=dict(episode=raw['episode']),state=raw)


class VectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(12)
        model=FixedPolicy(FixedConfig(9,3,width=8,hidden_size=16,frame_window=3)).eval()
        encoder=NativeObservationEncoder(card_id_to_token=fixture.CARD_MAP,ability_id_to_token=fixture.ABILITY_MAP,max_ability_slots=16)
        self.agent=FixedLiveAgent(model,encoder)
        self.reference=deepcopy(model).requires_grad_(False)
        self.trainer=FixedPPO(model,self.reference,PPOConfig())
        self.empty=LiveMasks(np.zeros(4,bool),np.zeros((4,576),bool),np.zeros(16,bool),(-1,)*4,(),np.zeros(4))

    def request(self,side):
        raw=fixture.state();raw['tick']=100
        _,_,batch=self.agent.observations(raw,Env.decks)
        masks=dict(cards=torch.tensor([[1,0,0,0]]*2,dtype=torch.bool),
                   positions=torch.zeros(2,4,576,dtype=torch.bool),abilities=torch.zeros(2,16,dtype=torch.bool))
        masks['positions'][:,0,:4]=True
        return dict(batch=batch,hidden=None,masks=masks,learner_side=side)

    def test_batched_inference_matches_independent_games_with_padding_and_reset(self):
        r0=self.request(0);r1=self.request(1)
        p=infer_requests(self.agent.model,self.trainer,[r1],[torch.Generator().manual_seed(1)],self.reference)[0]
        r1['hidden']=p['hidden']
        for k,v in list(r1['batch'].items()):
            if k.startswith('entity_'):
                shape=list(v.shape);shape[2]+=3
                out=v.new_zeros(shape);out[:,:,:v.shape[2]]=v;r1['batch'][k]=out
        batched=infer_requests(self.agent.model,self.trainer,[r0,r1],
                               [torch.Generator().manual_seed(i) for i in (20,21)],self.reference)
        for i,request in enumerate((r0,r1)):
            serial=infer_requests(self.agent.model,self.trainer,[request],[torch.Generator().manual_seed(20+i)],self.reference)[0]
            torch.testing.assert_close(batched[i]['indices'],serial['indices'])
            torch.testing.assert_close(batched[i]['probabilities'],serial['probabilities'],atol=2e-5,rtol=2e-5)
            for a,b in zip(batched[i]['hidden'],serial['hidden']):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(batched[i]['features']['context'],serial['features']['context'],atol=2e-5,rtol=2e-5)

    def test_parallel_games_match_serial_rewards_gae_and_state_resets(self):
        activity=ProbeActivity();envs=[Env(activity),Env(activity)];log=io.StringIO()
        with patch('hokoff_model.ppo_rollout.NativeMaskProvider') as provider:
            provider.return_value.for_side.return_value=self.empty
            batches,episodes,stats=collect_parallel(self.agent,self.trainer,envs,{},episode_ids=list(range(4)),
                seed=42,opponent=self.reference,opponent_number=1,log=log,full_elixir_penalty_per_tick=.001)
            for i,(data,episode) in enumerate(zip(batches,episodes)):
                serial,summary=collect_episode(self.agent,self.trainer,Env(),dict(rndSeed=42+i),
                    learner_side=i%2,generator=torch.Generator().manual_seed(42+i),episode_id=i,
                    opponent=self.reference,opponent_number=1,full_elixir_penalty_per_tick=.001)
                for k in data:torch.testing.assert_close(data[k],serial[k],rtol=3e-5,atol=3e-5)
                self.assertEqual(episode['episode'],i)
                self.assertEqual(episode['full_elixir_idle_ticks'],summary['full_elixir_idle_ticks'])
        self.assertGreaterEqual(activity.peak,2)
        self.assertEqual(sum(len(e.resets) for e in envs),4)
        rows=[json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(len(rows),sum(e['decisions'] for e in episodes))
        self.assertGreater(stats['mean_inference_batch_games'],1.)
        self.assertTrue(self.agent.model.timing.weight.requires_grad)

    def test_failure_discards_entire_batch_and_does_not_restart_games(self):
        envs=[Env(fail=True),Env()]
        with patch('hokoff_model.ppo_rollout.NativeMaskProvider') as provider:
            provider.return_value.for_side.return_value=self.empty
            with self.assertRaisesRegex(RuntimeError,'incomplete tower'):
                collect_parallel(self.agent,self.trainer,envs,{},episode_ids=[0,1],seed=42,
                    opponent=self.reference,opponent_number=1)
        self.assertEqual([len(e.resets) for e in envs],[1,1])

    def test_interrupt_closes_streams_without_restarting_workers(self):
        envs=[Env(),Env()]
        with patch('hokoff_model.ppo_rollout.NativeMaskProvider') as provider:
            provider.return_value.for_side.return_value=self.empty
            with patch('hokoff_model.ppo_vector.infer_requests',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    collect_parallel(self.agent,self.trainer,envs,{},episode_ids=[0,1],seed=42,
                        opponent=self.reference,opponent_number=1)
        self.assertEqual([len(e.resets) for e in envs],[1,1])
        self.assertFalse(any(t.name.startswith('native-game') for t in threading.enumerate()))


if __name__=='__main__':unittest.main()
