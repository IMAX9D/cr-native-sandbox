import contextlib
import io
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.independent_data import independent_indices,CONTRACT
from hokoff_model.decision_data import prepare,DecisionWindows,collate_decisions
from hokoff_model.decision_model import DecisionPolicy,DecisionConfig
from hokoff_model.decision_loss import bc_loss
from hokoff_model.train_decisions import parser,run
import train_hokoff_decisions as launcher


class IndependentIndexTests(unittest.TestCase):
    def inputs(self,n=300):
        scalar=np.zeros((n,16),dtype=np.float32)
        scalar[:,0]=np.minimum(5990+np.arange(n),6000)/6000
        actions=np.zeros(n,dtype=bool);actions[[4,9,12,15,29,44,65,78,90,150,280]]=True
        valid=np.ones(n,dtype=bool);valid[10:15]=False;valid[-3:]=False
        return [np.array([0,n]),scalar,np.ones(n),np.ones(n),actions,valid,8]

    def test_primary_observations_are_invariant_to_action_labels(self):
        args=self.inputs(); first=independent_indices(*args,seed=123)
        args[4]=~args[4];other=independent_indices(*args,seed=123)
        for key in ('rows','ticks','elapsed'):
            np.testing.assert_array_equal(first[key][first['supervision']==1],other[key][other['supervision']==1])
        primary=first['supervision']==1
        short=(first['elapsed']>0)&(first['elapsed']<8)&primary
        self.assertGreater(int((short & ~self.inputs()[4][first['rows']]).sum()),0)

    def test_targets_come_from_future_actions_and_preserve_all_valid_actions(self):
        args=self.inputs(); d=independent_indices(*args,seed=123)
        actions,valid=args[4:6]
        supervised=d['supervision']>0
        rows=d['rows'][supervised]
        action_rows=rows[actions[rows]]
        np.testing.assert_array_equal(np.sort(action_rows),np.flatnonzero(actions&valid))
        self.assertTrue(valid[d['rows']].all())
        for begin,end,role in zip(d['offsets'][:-1],d['offsets'][1:],d['roles']):
            query=d['rows'][begin:end]
            self.assertTrue((np.diff(query)>0).all())
            if role:
                self.assertEqual(d['supervision'][end-1],2)
                self.assertFalse(d['delay_mask'][begin:end].any())
                self.assertTrue((d['supervision'][begin:end-1]==0).all())
            else:
                boundary=10 if query[0]<10 else 297
                events=np.flatnonzero(actions & valid & (np.arange(len(actions))<boundary))
                for i,row in enumerate(query,begin):
                    future=events[events>row]
                    if d['delay_mask'][i]:
                        expected=min(int(future[0]-row),8) if len(future) else 8
                        self.assertEqual(d['delay'][i],expected)
                    elif not len(future):self.assertGreaterEqual(row+8,boundary)

    def test_schedule_prefix_does_not_depend_on_future_extent(self):
        args=self.inputs();args[5][:]=True
        long=independent_indices(*args,seed=71,auxiliary=False)
        cut=128;short_args=[np.array([0,cut])]+[a[:cut] for a in args[1:6]]+[8]
        short=independent_indices(*short_args,seed=71,auxiliary=False)
        np.testing.assert_array_equal(short['rows'],long['rows'][long['rows']<cut])


class IndependentDatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=80)
        prepare(self.root/'data',self.root/'cache',allow_smoke=True,sampling='independent',auxiliary_frame_window=3)
        self.ds=DecisionWindows(self.root/'data',self.root/'cache','validation',frame_window=3,targets=4,sampling='independent')

    def tearDown(self):self.ds.close();self.tmp.cleanup()

    def test_auxiliary_examples_do_not_supervise_timing_or_delay(self):
        index=next(i for i in range(len(self.ds)) if not self.ds[i]['timing_label_mask'].any())
        item=self.ds[index]
        self.assertEqual(int(item['loss_mask'].sum()),1)
        self.assertTrue(item['play_now'][-1])
        b=collate_decisions([item])
        model=DecisionPolicy(DecisionConfig(12,4,width=16,hidden_size=32,frame_window=3))
        out=model(b)
        out['timing'].retain_grad();out['delay'].retain_grad();out['kind'].retain_grad()
        loss,stats=bc_loss(out,b);loss.backward()
        self.assertEqual(stats['timing_count'],0);self.assertEqual(stats['delay_count'],0)
        self.assertEqual(stats['kind_count'],1)
        self.assertEqual(float(out['timing'].grad.abs().sum()),0)
        self.assertEqual(float(out['delay'].grad.abs().sum()),0)
        self.assertGreater(float(out['kind'].grad.abs().sum()),0)
        val=DecisionWindows(self.root/'data',self.root/'cache','train',frame_window=3,targets=4,sampling='independent')
        self.assertFalse(any(val.records[0]['segment_roles']));val.close()
        with self.assertRaises(ValueError):
            DecisionWindows(self.root/'data',self.root/'cache','validation',sampling='legacy')

    def test_independent_training_resumes_exactly(self):
        def train(name,steps,resume=False):
            argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/name),
                '--sampling','independent','--allow-smoke','--device','cpu','--workers','0','--cpu-threads','1',
                '--width','16','--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
                '--max-steps',str(steps),'--epochs','10','--eval-batches','2','--delay-short-weight','128','--timing-positive-weight','8']
            if resume:argv+=['--resume',str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full=train('full',4);train('resumed',2);resumed=train('resumed',4,True)
        self.assertEqual(full['contract']['decision_contract'],CONTRACT)
        for key in full['model']:torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)

    def test_launcher_resolves_separate_paths(self):
        with contextlib.redirect_stdout(io.StringIO()):args=launcher.main(['--sampling','independent','--device','cpu','--dry-run'])
        self.assertIn('independent',args.cache.name)
        self.assertIn('independent',args.run.name)


if __name__=='__main__':unittest.main()
