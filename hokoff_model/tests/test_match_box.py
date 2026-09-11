from copy import deepcopy
import unittest
import tempfile
import numpy as np
import torch
from hokoff_model.history import HistoryIndex
from hokoff_model.online_history import OnlinePlayHistory,PublicPlay
from hokoff_model.match_agent import HistoryOnlinePolicy,MatchAgent,verify_runtime,LIBG_SHA
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from hokoff_model.tests.test_live import fixture,Probes
from expert_selfplay_v1.native_observation import NativeObservationEncoder,NativeActorFrame
from hokoff_model.match_lock import match_lease


class OnlineHistoryTests(unittest.TestCase):
    def test_endpoint_lease_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with match_lease('127.0.0.1',12345,directory=directory):
                with self.assertRaises(RuntimeError):
                    with match_lease('127.0.0.1',12345,directory=directory):pass
            with match_lease('127.0.0.1',12345,directory=directory):pass

    def test_matches_offline_both_sides_strict_past_and_age(self):
        n=32
        a={key:np.zeros(2*n,dtype=np.int64) for key in ('play_now','kind_label_mask','action_kind','timing_label_mask','card_label_mask','position_label_mask','card_slot','position')}
        a['public_scalars']=np.zeros((2*n,16),np.float32)
        a['public_scalars'][:,0]=np.tile(np.arange(n),2)/6000
        a['hand_tokens']=np.ones((2*n,4),np.int64)
        events=[PublicPlay(t,side,1+side,20+t) for t in (2,5,9,12,16,20,25) for side in (0,1)]
        for event in events:
            row=event.side*n+event.tick
            for key in ('play_now','kind_label_mask','timing_label_mask','card_label_mask','position_label_mask'):a[key][row]=1
            a['hand_tokens'][row,0]=event.token
            a['position'][row]=event.absolute_cell if event.side==0 else 575-event.absolute_cell
        offline=HistoryIndex(a,[0,n,2*n]);online=OnlinePlayHistory(4)
        for tick in range(n):
            for event in events:
                if event.tick==tick:online.record(event)
            for side in (0,1):
                expected=offline.query(np.array([side*n+tick]),np.array([tick]),4)
                actual=online.query(side,tick)
                for key in expected:torch.testing.assert_close(actual[key][0],expected[key],rtol=0,atol=0)

    def test_duplicate_unknown_and_reset(self):
        history=OnlinePlayHistory(4);event=PublicPlay(10,0,0,0,False)
        self.assertTrue(history.record(event));self.assertFalse(history.record(event))
        with self.assertRaises(ValueError):history.record(PublicPlay(10,0,1,0))
        q=history.query(1,11)
        self.assertTrue(q['history_mask'][0,0,1,0]);self.assertFalse(q['history_known'].any())
        self.assertFalse(history.query(1,10)['history_mask'].any())
        history.reset();self.assertFalse(history.query(1,12)['history_mask'].any())


class MatchAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(24)
        self.encoder=NativeObservationEncoder(card_id_to_token=fixture.CARD_MAP,
                        ability_id_to_token=fixture.ABILITY_MAP,max_ability_slots=16)
        self.model=HistoryOnlinePolicy(FixedConfig(9,3,width=16,hidden_size=32,history_length=4,spatial_skip_channels=4)).eval()
        with torch.no_grad():self.model.timing.bias.fill_(-100)
        self.state=fixture.state();self.decks=[list(fixture.DECK),list(fixture.DECK)]
        self.agent=MatchAgent(self.model,self.encoder);self.agent.reset(1200)

    def test_fixed_clock_receipts_and_reset(self):
        _,first=self.agent.decide(self.state,self.decks,Probes())
        hidden=tuple(value.clone() for value in self.agent.hidden)
        actions=[dict(side=0,deck_index=1,x=2500,y=4500)]
        self.agent.record_transition(1200,1201,actions,dict(actions=[dict(side=0,result=dict(accepted=True,deck_index=1,x=2500,y=4500))]),self.decks)
        self.assertEqual(self.agent.history.accepted_plays,[1,0])
        self.state['tick']=1201
        self.assertTrue(self.agent.decide(self.state,self.decks,Probes())[1]['policy_skipped'])
        for a,b in zip(hidden,self.agent.hidden):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for tick in range(1201,1204):self.agent.record_transition(tick,tick+1,[],dict(actions=[]),self.decks)
        self.state['tick']=1204
        _,info=self.agent.decide(self.state,self.decks,Probes())
        self.assertEqual(info['history_slots'],1);self.assertGreater(info['encoded_entities'],0)
        with self.assertRaises(ValueError):self.agent.decide(self.state,self.decks,Probes())
        self.agent.reset(0);self.assertEqual(self.agent.history.accepted_plays,[0,0]);self.assertFalse(torch.count_nonzero(self.agent.hidden[0]))

    def test_rejected_and_skills_not_fabricated_as_normal_history(self):
        action=dict(side=0,deck_index=1,x=2500,y=4500)
        self.agent.record_transition(1200,1201,[action],dict(actions=[dict(side=0,result=dict(accepted=False))]),self.decks)
        self.agent.record_transition(1201,1202,[dict(type='ability',side=0,entity_id=5)],dict(actions=[dict(side=0,result=dict(accepted=True,entity_id=5))]),self.decks)
        self.assertEqual(self.agent.history.accepted_plays,[0,0])
        with self.assertRaises(ValueError):self.agent.record_transition(1202,1203,[action],dict(actions=[]),self.decks)
        with self.assertRaises(ValueError):self.agent.record_transition(1203,1204,[],dict(actions=[]),self.decks)
        with self.assertRaisesRegex(ValueError,'different command'):
            self.agent.record_transition(1202,1203,[action],dict(actions=[dict(side=0,result=dict(accepted=True,deck_index=2,x=2500,y=4500))]),self.decks)

    def test_enemy_private_state_does_not_change_actor(self):
        self.agent.decide(self.state,self.decks,Probes());before=self.agent.hidden
        changed=deepcopy(self.state);changed['players'][0]['elixir_raw']=10000
        changed['players'][0]['hand_deck_indices']=[0,5,6,7];changed['players'][0]['next_deck_index']=1
        other=MatchAgent(self.model,self.encoder);other.reset(1200);other.decide(changed,self.decks,Probes())
        for a,b in zip(before,other.hidden):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_stream_matches_offline_and_old_guard_remains(self):
        encoded=self.encoder.encode_batch([NativeActorFrame(self.state,1,self.decks[1])])
        b=dict(encoded);b['frame_ticks']=torch.tensor([[1200]]);b['prev_elapsed_ticks']=b.pop('delta_ticks')
        b['frame_mask']=torch.ones(1,1,dtype=torch.bool);b.update(self.agent.history.query(1,1200))
        with torch.no_grad():
            streamed,hidden=self.model.forward_stream(b)
            full=self.model({**b,'loss_mask':b['frame_mask']})
        for key in full:torch.testing.assert_close(full[key],streamed[key],rtol=2e-5,atol=2e-6)
        with self.assertRaises(ValueError):self.model.forward_stream({k:v for k,v in b.items() if k!='history_age'})
        original=FixedPolicy(self.model.config)
        with self.assertRaises(NotImplementedError):original.forward_stream(b)

    def test_runtime_hash_guard(self):
        class Client:
            def request(self,payload):return dict(ok=True,identity=dict(libg_sha256=self.sha))
        class Env:client=Client()
        Env.client.sha='0'*64
        with self.assertRaises(ValueError):verify_runtime(Env())
        Env.client.sha=LIBG_SHA;self.assertEqual(verify_runtime(Env())['libg_sha256'],LIBG_SHA)


if __name__=='__main__':unittest.main()
