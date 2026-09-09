from copy import deepcopy
import unittest
from hokoff_model.ppo_rollout import advance_with_elixir_ticks,full_elixir_idle_ticks
from train_hokoff_ppo import resolve_full_elixir_penalty


def state(tick,elixir):
    return dict(tick=tick,players=[dict(side=s,elixir_raw=e) for s,e in enumerate(elixir)],
                episode=dict(terminated=False,truncated=False))


class ElixirEnv:
    def __init__(self,levels,terminal_tick=None,late_terminal=False):
        self.levels=levels;self.tick=100;self.calls=[]
        self.terminal_tick=terminal_tick;self.late_terminal=late_terminal

    def joint_training_transition(self,commands,*,steps):
        assert steps==1
        self.calls.append(deepcopy(commands))
        previous=self.tick
        if self.terminal_tick is None or self.tick<self.terminal_tick:self.tick+=1
        done=self.tick==self.terminal_tick and (not self.late_terminal or previous==self.tick)
        receipt=dict(actions=[dict(side=a['side'],result=dict(accepted=True,**{k:v for k,v in a.items() if k not in ('type','side')})) for a in commands])
        episode=dict(terminated=done,truncated=False,terminal_tick=self.tick)
        result=dict(joint_action=receipt,step=dict(episode=episode))
        if not done:result['state']=state(self.tick,self.levels[self.tick-100])
        return result


class FullElixirTests(unittest.TestCase):
    def test_full_wait_normal_wait_and_mid_interval_refill(self):
        levels=[[99900,100000],[99950,100000],[100000,100000],[100000,100000],[100000,100000]]
        env=ElixirEnv(levels)
        result,advanced,trace=advance_with_elixir_ticks(env,state(100,levels[0]),[])
        self.assertEqual(advanced,4);self.assertEqual(len(env.calls),4)
        full,idle=full_elixir_idle_ticks(trace,[],set())
        self.assertEqual(idle,{0:2,1:4});self.assertEqual(full,idle)
        self.assertAlmostEqual(-.001*idle[0],-.002)
        self.assertEqual(full_elixir_idle_ticks([[90000,99999]]*4,[],set())[1],{0:0,1:0})

    def test_card_submitted_once_and_only_own_execution_tick_exempt(self):
        card=dict(type='play',side=0,deck_index=0,x=1,y=2)
        levels=[[100000,100000],[60000,100000],[60020,100000],[60040,100000],[60060,100000]]
        env=ElixirEnv(levels)
        result,advanced,trace=advance_with_elixir_ticks(env,state(100,levels[0]),[card])
        self.assertEqual(env.calls,[[card],[],[],[]]);self.assertEqual(advanced,4)
        full,idle=full_elixir_idle_ticks(trace,[card],{0})
        self.assertEqual(full,{0:1,1:4});self.assertEqual(idle,{0:0,1:4})
        self.assertEqual(result['joint_action']['actions'][0]['side'],0)
        self.assertEqual(full_elixir_idle_ticks([[100000,100000]]*4,[card],{0})[1],{0:3,1:4})
        self.assertEqual(full_elixir_idle_ticks([[100000,100000]],[card],set())[1],{0:1,1:1})

    def test_skill_cost_removes_full_condition_on_following_ticks(self):
        skill=dict(type='ability',side=0,entity_id=5000001)
        levels=[[100000,100000],[90000,100000],[90020,100000],[90040,100000],[90060,100000]]
        env=ElixirEnv(levels)
        _,_,trace=advance_with_elixir_ticks(env,state(100,levels[0]),[skill])
        self.assertEqual(env.calls,[[skill],[],[],[]])
        self.assertEqual(full_elixir_idle_ticks(trace,[skill],{0})[1],{0:1,1:4})

    def test_short_and_delayed_zero_tick_terminal_are_not_overcharged(self):
        levels=[[100000,100000]]*5
        for end,late in ((102,False),(102,True),(100,False)):
            with self.subTest(end=end,late=late):
                env=ElixirEnv(levels,terminal_tick=end,late_terminal=late)
                result,advanced,trace=advance_with_elixir_ticks(env,state(100,levels[0]),[])
                self.assertTrue(result['step']['episode']['terminated'])
                self.assertEqual(advanced,end-100);self.assertEqual(len(trace),advanced)
                self.assertEqual(full_elixir_idle_ticks(trace,[],set())[1],{0:advanced,1:advanced})

    def test_resume_preserves_legacy_and_new_reward_contracts(self):
        self.assertEqual(resolve_full_elixir_penalty(None),.001)
        self.assertEqual(resolve_full_elixir_penalty(None,{}),0.)
        self.assertEqual(resolve_full_elixir_penalty(None,{'full_elixir_penalty_per_tick':.002}),.002)
        self.assertEqual(resolve_full_elixir_penalty(0.),0.)
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):resolve_full_elixir_penalty(value)
        with self.assertRaisesRegex(ValueError,'resume'):resolve_full_elixir_penalty(.001,{})


if __name__=='__main__':unittest.main()
