import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from policy_v1.data import Windows,collate,prepare
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.model import Config,Policy
from hokoff_model.horizon import future_targets,HorizonWindows
from hokoff_model.metrics import bc_loss
from hokoff_model.compare_horizon import lead_events,parser,run
from hokoff_model.train import parser as train_parser,run as train


class HorizonTests(unittest.TestCase):
    def test_inclusive_horizon_gaps_and_censoring(self):
        y=np.zeros(24,dtype=bool);y[10]=True
        valid=np.ones(24,dtype=bool)
        target,known=future_targets(y,valid,10)
        self.assertTrue(target[0])
        self.assertTrue(target[10])
        self.assertFalse(target[11])
        self.assertTrue(known[13])
        self.assertFalse(known[14:].any())
        valid[5]=False
        _,known=future_targets(y,valid,10)
        self.assertFalse(known[:6].any())
        self.assertTrue(known[6])
        _,point_known=future_targets(y,valid,0,10)
        np.testing.assert_array_equal(known,point_known)

    def test_future_outside_input_window_never_changes_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_fixture(root/'data')
            prepare(root/'data',root/'cache',allow_smoke=True)
            common=dict(targets=4,frame_window=8,event_window=1)
            base=Windows(root/'data',root/'cache','train',**common)
            future=HorizonWindows(root/'data',root/'cache','train',timing_horizon_ticks=10,**common)
            point=HorizonWindows(root/'data',root/'cache','train',timing_horizon_ticks=0,timing_mask_horizon_ticks=10,**common)
            for i in range(len(base)):
                old,a,b=base[i],future[i],point[i]
                for key in old:torch.testing.assert_close(old[key],a[key],rtol=0,atol=0)
                torch.testing.assert_close(a['timing_target_mask'],b['timing_target_mask'])
            # Raw observation length does not grow by H, while a label after
            # the first four input rows can make an earlier forecast positive.
            first=future[0]
            self.assertEqual(len(first['frame_ticks']),4)
            self.assertTrue(first['timing_target'].any())
            torch.set_num_threads(1)
            model=Policy(Config(12,4,width=16,hidden_size=32,frame_window=8)).eval()
            a=collate([future[0],future[1]])
            original=collate([base[0],base[1]])
            with torch.no_grad():
                out=model(a);raw=model(original)
            for key in out:torch.testing.assert_close(out[key],raw[key])
            _,s1=bc_loss(raw,original);_,s2=bc_loss(out,a)
            for head in ('card','kind','position','ability','ability_position'):
                self.assertEqual(s1[head+'_sum'],s2[head+'_sum'])

    def test_lead_event_matching_ignores_edge_reference_hits(self):
        n=60;h=10
        y=np.zeros(n,dtype=bool);y[[5,30,55]]=True
        scores=np.zeros(n);scores[[0,20,45]]=1
        record=dict(ticks=np.arange(n),valid=np.ones(n,dtype=bool),labels=y,probabilities=scores)
        r=lead_events([record],.5,10,h,'every_frame')
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,0,0))
        self.assertEqual(r['ignored_edge_matches'],2)
        scores[21]=1
        r=lead_events([record],.5,10,h,'every_frame')
        self.assertEqual(r['fp'],1)
        r=lead_events([record],.5,0,h,'every_frame')
        self.assertEqual(r['tp'],0)

    def test_small_comparison_and_resume_contract_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_fixture(root/'data',steps=64)
            prepare(root/'data',root/'cache',allow_smoke=True)
            args=parser().parse_args(['--data',str(root/'data'),'--cache',str(root/'cache'),
                '--run-dir',str(root/'comparison'),'--train-split','train','--val-split','validation',
                '--device','cpu','--width','16','--hidden-size','32','--frame-window','8','--targets','4',
                '--steps','2','--batch-size','2','--workers','0','--cpu-threads','1','--sequences','2',
                '--eval-batches','1','--allow-smoke'])
            with contextlib.redirect_stdout(io.StringIO()):result=run(args)
            a,b=result['arms']
            self.assertEqual(a['point_target']['valid_frames'],b['point_target']['valid_frames'])
            self.assertEqual(a['within_500ms_target']['actual_actions'],b['within_500ms_target']['actual_actions'])
            path=root/'comparison/forecast_500ms/last.pt'
            saved=load_checkpoint(path)
            self.assertEqual(saved['contract']['timing_horizon_ticks'],10)
            self.assertEqual(saved['contract']['timing_mask_horizon_ticks'],10)
            argv=['--data',str(root/'data'),'--cache',str(root/'cache'),'--run-dir',str(root/'other'),
                  '--resume',str(path),'--train-split','train','--val-split','validation','--device','cpu',
                  '--width','16','--hidden-size','32','--frame-window','8','--targets','4','--batch-size','2',
                  '--workers','0','--seed','42','--allow-smoke']
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'contract differs'):
                train(train_parser().parse_args(argv))
