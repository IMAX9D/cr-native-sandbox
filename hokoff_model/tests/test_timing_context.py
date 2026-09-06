from dataclasses import asdict
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from policy_v1.data import Windows, prepare, digest
from policy_v1.smoke import create_fixture
from hokoff_model.model import Config, Policy
from hokoff_model.diagnose_timing import (
    match_events, event_metrics, sequence_plan, strata_report,
    shifted_control, parser, run,
)


def record(ticks, scores, labels, valid=None):
    n=len(ticks)
    return dict(ticks=np.array(ticks),probabilities=np.array(scores),labels=np.array(labels,dtype=bool),
                valid=np.ones(n,dtype=bool) if valid is None else np.array(valid,dtype=bool),
                available=np.ones(n,dtype=bool),card_available=np.ones(n,dtype=bool),elixir=np.full(n,.5))


class TimingContextTests(unittest.TestCase):
    def test_one_to_one_and_early_late(self):
        tp,fp,fn,delta=match_events([9,10,11],[10],5,5)
        self.assertEqual((tp,fp,fn),(1,2,0))
        self.assertEqual(delta,[-1])
        self.assertEqual(match_events([12],[10],5,0)[:3],(0,1,1))
        self.assertEqual(match_events([12],[10],5,5)[:3],(1,0,0))
        self.assertEqual(match_events([8,12],[10,14],2,0)[:3],(2,0,0))

    def test_no_match_across_actor_or_invalid_interval(self):
        a=record([0,1],[1,0],[0,0])
        b=record([0,1],[0,0],[0,1])
        result=event_metrics([a,b],.5,10,10,'every_frame')
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
        gap=record([0,1,2],[1,0,0],[0,0,1],[1,0,1])
        result=event_metrics([gap],.5,10,10,'every_frame')
        self.assertEqual(result['tp'],0)

    def test_plateau_not_multiple_event_hits_and_causal_edges(self):
        r=record(range(20),[1]*20,[int(i in (5,15)) for i in range(20)])
        raw=event_metrics([r],.5,10,10,'every_frame')
        edge=event_metrics([r],.5,10,10,'rising_edge')
        self.assertEqual((raw['tp'],raw['fp']),(2,18))
        self.assertEqual((edge['tp'],edge['fp'],edge['fn']),(1,0,1))
        self.assertEqual(edge['predicted_events'],1)

    def test_availability_is_not_claimed_forced_wait(self):
        r=record([0,1,2,3],[.1,.2,.3,.4],[0,1,0,1])
        r['available']=np.array([1,0,1,0],dtype=bool)
        report=strata_report([r])
        self.assertEqual(report['action_frames_without_confirmed_mask'],2)
        self.assertEqual(report['no_action_confirmed_by_mask']['actual_actions'],2)
        self.assertIn('do NOT prove forced WAIT',report['mask_caveat'])

    def test_shift_control_preserves_scores_and_input(self):
        r=record(range(100),np.linspace(0,1,100),[i%13==0 for i in range(100)])
        shifted=shifted_control([r])[0]
        np.testing.assert_array_equal(np.sort(r['probabilities']),np.sort(shifted['probabilities']))
        self.assertFalse(np.array_equal(r['probabilities'],shifted['probabilities']))
        np.testing.assert_array_equal(r['labels'],shifted['labels'])

    def test_end_to_end_complete_sequences_and_unchanged_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=create_fixture(root/'data')
            prepare(data,root/'cache',allow_smoke=True)
            ds=Windows(data,root/'cache','validation',frame_window=8,event_window=1,targets=4)
            chosen,indices,owners=sequence_plan(ds,100,123)
            self.assertEqual(len(indices),len(ds))
            self.assertEqual(len(set(indices)),len(indices))
            c=Config(12,4,width=16,hidden_size=32,frame_window=8)
            checkpoint=root/'last.pt'
            torch.save({'config':asdict(c),'model':Policy(c).state_dict(),'step':2,
                        'contract':{'val_split':'validation','train_split':'train','targets':4,
                                    'manifest_sha256':digest(data/'manifest.json')}},checkpoint)
            original=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args=parser().parse_args(['--checkpoint',str(checkpoint),'--data',str(data),
                '--cache',str(root/'cache'),'--device','cpu','--sequences','100',
                '--batch-size','2','--workers','0','--cpu-threads','1','--allow-smoke',
                '--output',str(root/'report.json')])
            with contextlib.redirect_stdout(io.StringIO()):summary=run(args)
            self.assertEqual(summary['arms'][0]['sequences'],len(chosen))
            self.assertTrue((root/'report.json').is_file())
            self.assertEqual(original,hashlib.sha256(checkpoint.read_bytes()).hexdigest())
