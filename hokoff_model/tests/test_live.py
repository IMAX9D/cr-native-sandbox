from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from hokoff_model.live import (FixedLiveAgent,NativeMaskProvider,LiveMasks,decode_action,
                               canonical_position_to_native)
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from expert_selfplay_v1.native_observation import NativeObservationEncoder
from expert_v1.compile_native_bc_dataset import _grid,_public_scalars
from expert_v1.tick_store_v1.schema import actor_projection,normalize_native_state
from run_hokoff_fixed import advance_fixed,accepted_actions

fixture_path=Path(__file__).resolve().parents[2]/'tests/test_expert_selfplay_native_observation.py'
spec=importlib.util.spec_from_file_location('native_encoder_fixture',fixture_path)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class Probes:
    def __init__(self):self.calls=[]
    def probe_grid(self,*,side,deck_index):
        self.calls.append((side,deck_index))
        cost=9 if deck_index==0 else 1
        return dict(width=18,height=32,cell_size=1000,rows=['1'*18]*32,valid_cells=576,
                    packed_selection=1,card_cost=cost,card_cost_raw=cost*10000,
                    selection_strategy='canonical',resolved_data_id=26000000,
                    selection_form_index=-1,selection_builder_rva='0x1',selection_root_vtable_rva='0x2')


class LiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9)
        encoder=NativeObservationEncoder(card_id_to_token=fixture.CARD_MAP,ability_id_to_token=fixture.ABILITY_MAP,
                                         max_ability_slots=16)
        self.model=FixedPolicy(FixedConfig(9,3,width=16,hidden_size=32,frame_window=3))
        self.agent=FixedLiveAgent(self.model,encoder)
        self.state=fixture.state();self.decks=[list(fixture.DECK),list(fixture.DECK)]

    def test_live_features_match_compiler_and_do_not_include_enemy_private_state(self):
        state,_,b=self.agent.observations(self.state,self.decks)
        for side in (0,1):
            actor=actor_projection(state,actor_side=side)
            np.testing.assert_array_equal(b['grid'][side,0].numpy(),_grid(actor).astype(np.float32)/255.)
            np.testing.assert_array_equal(b['public_scalars'][side,0].numpy(),_public_scalars(actor,state))
        changed=deepcopy(self.state);changed['players'][1]['elixir_raw']=10000
        changed['players'][1]['hand_deck_indices']=[0,5,6,7];changed['players'][1]['next_deck_index']=1
        _,_,other=self.agent.observations(changed,self.decks)
        for key in b:torch.testing.assert_close(b[key][0],other[key][0],rtol=0,atol=0)

    def test_streaming_matches_full_sequence_and_keeps_sides_independent(self):
        _,_,b=self.agent.observations(self.state,self.decks)
        both={k:torch.cat([v,v],1) for k,v in b.items()}
        both['frame_ticks'][:,1]+=4;both['prev_elapsed_ticks'][:,1]=4
        both['loss_mask']=both['frame_mask'].clone()
        with torch.no_grad():
            full=self.model(both);first,hidden=self.model.forward_stream(b)
            next_b={k:v[:,1:] for k,v in both.items() if k!='loss_mask'}
            latter,after=self.model.forward_stream(next_b,hidden)
            for name in first:
                torch.testing.assert_close(first[name],full[name][:,:1],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(latter[name],full[name][:,1:],atol=2e-6,rtol=2e-5)
            for side in (0,1):
                _,solo=self.model.forward_stream({k:v[side:side+1] for k,v in b.items()})
                for x,y in zip(solo,hidden):torch.testing.assert_close(x,y[:,side:side+1],atol=2e-6,rtol=2e-5)
            reset_out,_=self.model.forward_stream(b,after,torch.ones(2,dtype=torch.bool))
            for k in first:torch.testing.assert_close(first[k],reset_out[k],rtol=0,atol=0)

    def test_fixed_clock_rejects_missing_reset(self):
        self.agent.last_tick=1200
        with self.assertRaises(ValueError):self.agent.observations(self.state,self.decks)
        self.agent.reset();self.assertIsNone(self.agent.hidden)
        self.assertIsNone(self.agent.last_tick)
        self.agent.tracker.record_play(played_side=0,card=fixture.DECK[0]);self.agent.reset()
        self.assertEqual(self.agent.tracker.tokens_for(1),())

    def test_native_costs_and_live_hero_identity_are_used(self):
        state,encoded,_=self.agent.observations(self.state,self.decks)
        env=Probes();provider=NativeMaskProvider(env)
        m=provider.for_side(state,0,self.decks[0],encoded.ability_entity_keys[0],encoded.ability_mask[0,0].numpy())
        self.assertFalse(m.cards[0]);self.assertTrue(m.cards[1]);self.assertEqual(m.card_cost_raw[0],90000)
        self.assertTrue(m.abilities[0]);self.assertEqual(m.ability_keys[0],5000001)
        provider.for_side(state,0,self.decks[0],m.ability_keys,m.abilities);self.assertEqual(len(env.calls),4)
        provider.reset();provider.for_side(state,0,self.decks[0],m.ability_keys,m.abilities)
        self.assertEqual(len(env.calls),8)
        self.state['players'][0]['elixir_raw']=10000
        state=normalize_native_state(self.state)
        self.assertFalse(provider.for_side(state,0,self.decks[0],m.ability_keys,m.abilities).abilities[0])

    def test_cached_position_mask_changes_when_enemy_tower_falls(self):
        state,encoded,_=self.agent.observations(self.state,self.decks)
        provider=NativeMaskProvider(Probes())
        args=(0,self.decks[0],encoded.ability_entity_keys[0],encoded.ability_mask[0,0].numpy())
        before=provider.for_side(state,*args)
        changed=deepcopy(self.state)
        for tower in changed['episode']['crown_towers']:
            if tower['side']==1 and tower.get('lane')=='left':tower['hp']=0
        after=provider.for_side(normalize_native_state(changed),*args)
        self.assertFalse(before.positions[1,18*18+2]);self.assertTrue(after.positions[1,18*18+2])
        provider.reset();self.assertEqual(provider.position_cache,{})

    def test_mirror_is_reprobed_at_each_decision(self):
        state,encoded,_=self.agent.observations(self.state,self.decks)
        deck=deepcopy(self.decks[0]);deck[0]={'card_id':28000006}
        env=Probes();provider=NativeMaskProvider(env)
        for _ in range(2):provider.for_side(state,0,deck,encoded.ability_entity_keys[0],encoded.ability_mask[0,0].numpy())
        self.assertEqual(env.calls.count((0,0)),2)
        self.assertEqual(env.calls.count((0,1)),1)

    def output(self):
        return dict(timing=torch.full((2,1),10.),kind=torch.tensor([[[0.,10.]],[[10.,0.]]]),
                    card=torch.zeros(2,1,4),position=torch.zeros(2,1,4,576),ability=torch.zeros(2,1,16))

    def test_skill_has_no_position_and_masked_card_uses_correct_side(self):
        m=LiveMasks(np.array([False,True,False,False]),np.zeros((4,576),bool),
                    np.array([True]+[False]*15),(-1,3,4,5),(5000123,),np.array([0,10000,0,0]))
        m.positions[1,0]=True;out=self.output()
        a,_=decode_action(out,0,m,.7)
        self.assertEqual(a,dict(type='ability',side=0,entity_id=5000123))
        out['card'][1,0,0]=100  # illegal card must never win
        a,_=decode_action(out,1,m,.7)
        self.assertEqual(a,dict(type='play',side=1,deck_index=3,x=17500,y=31500))
        out['timing'][1,0]=-100
        self.assertIsNone(decode_action(out,1,m,.7)[0])

    def test_empty_masks_force_wait(self):
        m=LiveMasks(np.zeros(4,bool),np.zeros((4,576),bool),np.zeros(16,bool),(-1,)*4,(),np.zeros(4))
        a,r=decode_action(self.output(),0,m,.7)
        self.assertIsNone(a);self.assertEqual(r['reason'],'no_legal_action')

    def test_advance_finishes_four_ticks_without_resubmitting_actions(self):
        class Env:
            def __init__(self):self.calls=[]
            def joint_training_transition(self,actions,*,steps):
                self.calls.append((actions,steps));n=len(self.calls)
                return dict(state=dict(tick=[100,102,104][n-1]),joint_action=dict(actions=[]),
                            step=dict(episode=dict(terminated=False,truncated=False)))
        env=Env();actions=[dict(type='play',side=0)]
        _,advance=advance_fixed(env,dict(tick=100),actions)
        self.assertEqual(advance,4);self.assertEqual(env.calls,[(actions,4),([],4),([],2)])

    def test_accepted_receipt_must_match_selected_hero_or_card(self):
        expected=[dict(type='ability',side=0,entity_id=5000006)]
        receipt=dict(actions=[dict(side=0,result=dict(accepted=True,entity_id=5000007))])
        with self.assertRaises(RuntimeError):accepted_actions(receipt,expected)
        receipt['actions'][0]['result']['entity_id']=5000006
        self.assertEqual(accepted_actions(receipt,expected),{0})

    def test_terminal_and_rejected_receipts_are_not_fabricated_success(self):
        class Env:
            def joint_training_transition(self,actions,*,steps):
                return dict(joint_action=dict(actions=[]),step=dict(episode=dict(terminated=True,truncated=False,terminal_tick=101)))
        _,advance=advance_fixed(Env(),dict(tick=100),[]);self.assertEqual(advance,1)
        receipt=dict(actions=[dict(side=0,result=dict(accepted=False,result_code=9))])
        with self.assertRaises(RuntimeError):accepted_actions(receipt,[dict(side=0)])
        receipt['actions'][0]['result']['result_code']=4
        self.assertEqual(accepted_actions(receipt,[dict(side=0)],terminal_gate=True),set())
        receipt['actions'][0]['side']=1
        with self.assertRaises(RuntimeError):accepted_actions(receipt,[dict(side=0)],terminal_gate=True)


if __name__=='__main__':unittest.main()
