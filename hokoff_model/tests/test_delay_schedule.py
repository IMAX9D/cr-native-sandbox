import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from policy_v1.data import digest
from policy_v1.smoke import create_fixture
from hokoff_model.decision_data import prepare,collate_decisions
from hokoff_model.decision_model import DecisionConfig,DecisionPolicy
from hokoff_model.evaluate_delay_schedule import (LocatedWindows,Timelines,observations,streaming_step,
    choose_mode,choose_delay,point_metrics,score_schedule,shuffled_schedule,rolling_step,timing_ranking,main)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=40)
        prepare(self.root/'data',self.root/'cache',allow_smoke=True)
        self.ds=LocatedWindows(self.root/'data',self.root/'cache','train',targets=4,frame_window=3)
        self.timelines=Timelines(self.ds)
        self.model=DecisionPolicy(DecisionConfig(12,4,width=16,hidden_size=32,frame_window=3)).eval()

    def tearDown(self):
        self.ds.close();self.timelines.close();self.tmp.cleanup()

    def test_observations_and_actor_boundaries(self):
        for i in range(len(self.ds)):
            item=self.ds[i]; sh=int(item['source_shard'][0]); seq=int(item['source_segment'][0])
            seg=self.timelines.segment(sh,seq)
            obs=observations(self.timelines.open(sh),item['source_row'].numpy(),item['frame_ticks'],item['prev_elapsed_ticks'])
            for k,v in obs.items(): torch.testing.assert_close(v,item[k])
            self.assertNotIn('play_now',obs)
            self.assertNotIn('delay_target_ticks',obs)
            self.assertNotIn('card_slot',obs)
            self.assertLessEqual(seg['end'],40 if seg['side']==0 else 80)
            self.assertTrue(np.all(seg['actions']>=seg['start']))
            self.assertTrue(np.all(seg['actions']<seg['end']))

    def test_stream_only_updates_on_selected_observations(self):
        a=self.timelines.open(0)
        b=collate_decisions([observations(a,[0,3,7],[100,103,107],[0,3,4])])
        with torch.inference_mode():
            x=self.model.encode(b)
            output,expected_state=self.model.lstm(x)
            heads=self.model.heads(output,b)
            mode=choose_mode(heads['timing'],heads['kind'],b['action_kind_mask'])
            expected=choose_delay(heads['delay'],mode)
            state=None; predicted=[]
            for i in range(3):
                step={k:v[:,i:i+1] for k,v in b.items()}
                delay,state=streaming_step(self.model,step,state)
                predicted.append(delay)
            torch.testing.assert_close(torch.stack(predicted,dim=1),expected)
            for actual,want in zip(state,expected_state): torch.testing.assert_close(actual,want)
        # No legal current action forces WAIT even with a positive timing logit.
        mode=choose_mode(torch.tensor([2.]),torch.tensor([[0.,1.]],dtype=torch.float16),torch.tensor([[False,False]]))
        self.assertEqual(int(mode[0]),0)

    def test_crossing_is_strict_and_includes_no_event_intervals(self):
        m=point_metrics(np.array([4,4,8]),np.array([4,3,32767]))
        self.assertAlmostEqual(m['cross_next_action_rate'],1/3)
        self.assertEqual(m['late_ticks_per_crossing'],1)

    def test_timing_ranking_handles_ties_and_low_probabilities(self):
        r=timing_ranking([.05,.05,.05,.05],[1,0,1,0])
        self.assertEqual(r['average_precision'],.5)
        r=timing_ranking([.1,.08,.01,.005],[1,1,0,0])
        self.assertEqual(r['average_precision'],1)
        self.assertEqual(r['thresholds']['0.5']['recall'],0)
        self.assertEqual(r['thresholds']['0.05']['recall'],1)

    def test_rolling_history_matches_last_observed_window(self):
        a=self.timelines.open(0); history=None
        rows=[0,3,7,12,20,21]; elapsed=[0,3,4,5,8,1]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                single=collate_decisions([observations(a,[row],[row+100],[elapsed[i]])])
                pred,history=rolling_step(self.model,single,history)
                start=max(0,i-2)
                b=collate_decisions([observations(a,rows[start:i+1],np.array(rows[start:i+1])+100,elapsed[start:i+1])])
                x=self.model.encode(b); out,_=self.model.lstm(x)
                heads=self.model.heads(out,b)
                mode=choose_mode(heads['timing'],heads['kind'],b['action_kind_mask'])
                expected=choose_delay(heads['delay'],mode)[:,-1]
                torch.testing.assert_close(pred,expected)

    def test_tail_guard_event_scoring_and_exact_budget_shuffle(self):
        seg=dict(start=0,end=25,actions=np.array([0,3,4,9,17,24]))
        times=np.array([0,4,8,12,16,20,24])
        score=score_schedule(times,seg)
        self.assertEqual(score['events'],4)
        self.assertEqual(score['excluded_tail_events'],2)
        self.assertEqual(score['delays'],[0,1,0,3])
        rng=np.random.default_rng(1)
        times=np.array([0,2,6,7,15,20,24])
        for _ in range(20):
            shuffled=shuffled_schedule(times,rng)
            self.assertEqual(len(shuffled),len(times))
            self.assertEqual(shuffled[-1],times[-1])
            np.testing.assert_array_equal(np.sort(np.diff(shuffled)),np.sort(np.diff(times)))
            score_schedule(shuffled,seg)
        with self.assertRaises(ValueError):score_schedule([0,4,4],seg)

    def test_full_cpu_evaluation(self):
        checkpoint=self.root/'model.pt'
        torch.save(dict(config=asdict(self.model.config),model=self.model.state_dict(),step=2000,
            contract=dict(decision_cache_sha256=digest(self.root/'cache/index.json'),val_split='train',targets=4)),checkpoint)
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
                '--checkpoint',str(checkpoint),'--output',str(self.root/'output'),
                '--device','cpu','--workers','0','--window-batches','2','--batch-size','2',
                '--battles','1','--replay-batch-size','2','--shuffle-trials','3'])
        result=json.loads((self.root/'output/results.json').read_text())
        self.assertGreater(result['pointwise']['raw_valid_points'],0)
        replay=result['state_replay']['metrics']
        self.assertEqual(replay['model']['observations'],replay['shuffled_same_budget']['observations'])
        self.assertEqual(replay['model']['observations'],replay['uniform_same_budget']['observations'])
        for k in (4,6,7,8):
            self.assertEqual(replay['model']['events'],replay[f'fixed{k}']['events'])
        self.assertEqual(result['state_replay']['actor_segments'],2)
        self.assertEqual(result['elapsed_shortcut_audit']['at_expert_decision_points']['precision'],1)


if __name__=='__main__':unittest.main()
