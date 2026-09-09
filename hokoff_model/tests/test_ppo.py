from copy import deepcopy
from unittest.mock import patch
import unittest
import torch
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.ppo_actions import (action_heads,action_distribution,exact_kl,entropy,
                                      sample_actions,train_action_heads_only,CARD_ACTIONS)
from hokoff_model.ppo import FixedPPO,PPOConfig,state_digest


class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.model=FixedPolicy(FixedConfig(9,3,width=8,hidden_size=16,frame_window=3)).eval()
        self.reference=deepcopy(self.model).requires_grad_(False)
        n=12
        self.data=dict(context=torch.randn(n,8),hand_tokens=torch.randint(1,9,(n,4)),
                       ability_tokens=torch.ones(n,16,dtype=torch.long),cards=torch.ones(n,4,dtype=torch.bool),
                       positions=torch.zeros(n,4,576,dtype=torch.bool),abilities=torch.zeros(n,16,dtype=torch.bool))
        self.data['positions'][:,:,[0,17]]=True;self.data['abilities'][:,0]=True
        self.data['cards'][0]=False;self.data['abilities'][0]=False
        self.generator=torch.Generator().manual_seed(22)
        self.refresh_behavior()
        self.data.update(advantage=torch.linspace(-1,1,n),**{'return':torch.linspace(1,3,n)})

    def refresh_behavior(self):
        with torch.no_grad():d=action_distribution(action_heads(self.model,self.data),self.data)
        actions=sample_actions(d,self.generator)
        self.data.update(action=actions,old_log_prob=d['log_prob'].gather(1,actions[:,None]).squeeze(1))

    def trainer(self):return FixedPPO(self.model,self.reference,PPOConfig(batch_size=12,epochs=1))

    def test_heads_equal_original_BC(self):
        recurrent=torch.randn(12,1,16)
        b={k:self.data[k][:,None] for k in ('hand_tokens','ability_tokens')}
        with torch.no_grad():
            original=self.model.heads(recurrent,b)
            data=dict(self.data,context=original['context'][:,0])
            new=action_heads(self.model,data)
        for k in new:torch.testing.assert_close(new[k],original[k][:,0],rtol=1e-5,atol=1e-6)

    def test_joint_distribution_matches_explicit_hierarchy(self):
        out=dict(timing=torch.tensor([-1.0986122887]),kind=torch.tensor([[.4,.6]]).log(),
            card=torch.zeros(1,4),position=torch.zeros(1,4,576),ability=torch.zeros(1,16))
        out['position'][0,0,:2]=torch.tensor([.3,.7]).log()
        masks=dict(cards=torch.tensor([[True,False,False,False]]),positions=torch.zeros(1,4,576,dtype=torch.bool),
                   abilities=torch.zeros(1,16,dtype=torch.bool))
        masks['positions'][0,0,:2]=True;masks['abilities'][0,0]=True
        p=action_distribution(out,masks)['log_prob'].exp()[0]
        expected=torch.zeros(2321);expected[0]=.75;expected[1]=.03;expected[2]=.07;expected[2305]=.15
        torch.testing.assert_close(p,expected,rtol=1e-5,atol=1e-7)

    def test_distribution_normalized_and_impossible_actions_never_sampled(self):
        d=action_distribution(action_heads(self.model,self.data),self.data)
        torch.testing.assert_close(d['log_prob'].exp().sum(-1),torch.ones(12))
        self.assertEqual(float(d['log_prob'][0,0].detach()),0.)
        for _ in range(50):
            actions=sample_actions(d,self.generator);self.assertEqual(int(actions[0]),0)
            for i,index in enumerate(actions):
                index=int(index)
                if 1<=index<=CARD_ACTIONS:
                    slot,pos=divmod(index-1,576);self.assertTrue(self.data['cards'][i,slot]);self.assertTrue(self.data['positions'][i,slot,pos])
                elif index>CARD_ACTIONS:self.assertTrue(self.data['abilities'][i,index-1-CARD_ACTIONS])

    def test_kl_exact_and_conditional_detects_wait_dilution(self):
        out=action_heads(self.model,self.data)
        out={k:v.detach().clone() for k,v in out.items()};out['timing'].fill_(-25)
        reference=action_distribution(out,self.data)
        changed={k:v.clone() for k,v in out.items()};changed['card'][:,0]+=20
        current=action_distribution(changed,self.data)
        joint,mark=exact_kl(reference,current)
        self.assertLess(float(joint.max()),1e-6);self.assertGreater(float(mark[1:].mean()),1.)
        same,_=exact_kl(reference,reference);self.assertEqual(float(same.max()),0.)
        direct=(reference['log_prob'].exp()*(reference['log_prob']-current['log_prob'])).sum(-1)
        torch.testing.assert_close(joint,direct.clamp_min(0))

    def test_update_preserves_reference_and_encoder_changes_heads(self):
        trainer=self.trainer();before=deepcopy(self.model.state_dict())
        stats=trainer.update(self.data,generator=self.generator)
        self.assertGreater(stats['accepted_updates'],0);self.assertEqual(stats['rejected_updates'],0)
        self.assertTrue(stats['reference_unchanged']);self.assertTrue(stats['encoder_lstm_unchanged'])
        names={n for n,p in self.model.named_parameters() if p.requires_grad}
        changed=[n for n,v in self.model.state_dict().items() if not torch.equal(v,before[n])]
        self.assertTrue(changed);self.assertTrue(set(changed)<=names)

    def test_large_update_rolls_back_actor_and_adam_state(self):
        trainer=self.trainer();trainer.update(self.data,generator=self.generator)
        self.refresh_behavior()
        trainer.actor_optimizer.param_groups[0]['lr']=1.
        before=deepcopy(self.model.state_dict());optim=deepcopy(trainer.actor_optimizer.state_dict())
        result=trainer.update(self.data,generator=self.generator)
        self.assertEqual(result['accepted_updates'],0);self.assertEqual(result['rejected_updates'],1)
        for name,value in before.items():torch.testing.assert_close(value,self.model.state_dict()[name],rtol=0,atol=0)
        restored=trainer.actor_optimizer.state_dict()
        for key,state in optim['state'].items():
            for field,v in state.items():torch.testing.assert_close(v,restored['state'][key][field],rtol=0,atol=0)
        self.assertEqual(restored['param_groups'][0]['lr'],.5)

    def test_critic_cannot_send_gradients_into_actor(self):
        trainer=self.trainer()
        trainer.values(self.data).sum().backward()
        self.assertTrue(all(p.grad is None for p in trainer.actor.parameters()))

    def collect_fake(self,*,truncated=False,complete_snapshot=True,penalty=0.,full=False):
        from hokoff_model.tests.test_live import fixture
        from hokoff_model.live import FixedLiveAgent,LiveMasks
        from expert_selfplay_v1.native_observation import NativeObservationEncoder
        from hokoff_model.ppo_rollout import collect_episode
        import numpy as np
        raw=fixture.state();raw['tick']=100;raw['episode']['tower_snapshot_complete']=True
        if full:raw['players'][0]['elixir_raw']=100000
        terminal=deepcopy(raw['episode'])
        for tower in terminal['crown_towers']:
            if tower['side']==1 and tower.get('type')=='king':tower['hp']-=100
        terminal.update(terminated=not truncated,truncated=truncated,rewards=[1,-1],
                        outcome='side0_win',terminal_tick=104,tower_snapshot_complete=complete_snapshot)
        class Env:
            decks=[list(fixture.DECK),list(fixture.DECK)]
            def reset(self,*a,**k):pass
            def observe_train(self):return raw
            tick=100
            def joint_training_transition(self,actions,*,steps):
                assert not actions and steps in (1,4)
                self.tick+=steps
                if self.tick==104:return dict(joint_action=dict(actions=[]),step=dict(episode=terminal))
                current=deepcopy(raw);current['tick']=self.tick
                return dict(joint_action=dict(actions=[]),step=dict(episode=current['episode']),state=current)
        encoder=NativeObservationEncoder(card_id_to_token=fixture.CARD_MAP,ability_id_to_token=fixture.ABILITY_MAP,max_ability_slots=16)
        agent=FixedLiveAgent(self.model,encoder);trainer=self.trainer()
        empty=LiveMasks(np.zeros(4,bool),np.zeros((4,576),bool),np.zeros(16,bool),(-1,)*4,(),np.zeros(4))
        with patch('hokoff_model.ppo_rollout.NativeMaskProvider') as provider:
            provider.return_value.for_side.return_value=empty
            return collect_episode(agent,trainer,Env(),{},learner_side=0,generator=self.generator,max_decisions=1,full_elixir_penalty_per_tick=penalty)

    def test_native_rollout_reward_and_actual_tick_gae(self):
        data,result=self.collect_fake()
        self.assertEqual(data['action'].tolist(),[0]);self.assertEqual(data['delta_ticks'].tolist(),[4])
        self.assertEqual(data['terminated'].tolist(),[True])
        self.assertAlmostEqual(float(data['reward'][0]),10.1,places=5)
        self.assertAlmostEqual(float(data['return'][0]),10.1,places=5)
        self.assertEqual(float(data['old_log_prob'][0]),0.)

    def test_full_elixir_penalty_enters_rollout_and_gae_once(self):
        data,result=self.collect_fake(penalty=.001,full=True)
        self.assertEqual(data['delta_ticks'].tolist(),[4])
        self.assertEqual(data['full_elixir_idle_ticks'].tolist(),[4])
        self.assertAlmostEqual(float(data['base_reward'][0]),10.1,places=5)
        self.assertAlmostEqual(float(data['full_elixir_penalty'][0]),-.004,places=7)
        self.assertAlmostEqual(float(data['reward'][0]),10.096,places=5)
        self.assertAlmostEqual(float(data['return'][0]),10.096,places=5)
        self.assertEqual(result['full_elixir_idle_ticks'],4)
        data,result=self.collect_fake(penalty=.001,full=False)
        self.assertEqual(result['full_elixir_idle_ticks'],0)
        self.assertAlmostEqual(float(data['reward'][0]),10.1,places=5)

    def test_truncation_and_missing_tower_reward_evidence_are_rejected(self):
        with self.assertRaises(RuntimeError):self.collect_fake(truncated=True)
        with self.assertRaises(RuntimeError):self.collect_fake(complete_snapshot=False)

    def test_guard_exception_also_restores_candidate(self):
        trainer=self.trainer();before=deepcopy(self.model.state_dict())
        real_guard=trainer.guard;calls=0
        def fail(*args):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt()
            return real_guard(*args)
        with patch.object(trainer,'guard',side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):trainer.update(self.data,generator=self.generator)
        for k,v in before.items():torch.testing.assert_close(v,self.model.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(len(trainer.actor_optimizer.state),0)


if __name__=='__main__':unittest.main()
