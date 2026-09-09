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
from policy_v1.loss import bc_loss
from policy_v1.smoke import create_fixture
from hokoff_model.decision_data import prepare,DecisionWindows,collate_decisions
from hokoff_model.diagnostic_metrics import timing_diagnostics,gradient_geometry
from hokoff_model.diagnose_gradients import isolated_task_losses,measure_batch,shared_parameters
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
import diagnose_hokoff_fixed as launcher


class RankingTests(unittest.TestCase):
    def test_ap_known_ranking_and_thresholds(self):
        r,curve=timing_diagnostics([.9,.8,.7,.6],[1,0,1,0])
        self.assertAlmostEqual(r['average_precision'],(1+2/3)/2)
        self.assertEqual(r['action_budgets']['actual_rate']['predicted_count'],2)
        self.assertEqual(r['action_budgets']['actual_rate']['precision'],.5)
        self.assertEqual(r['thresholds']['0.7']['predicted_count'],2)
        self.assertEqual(curve['recall'][-1],1)

    def test_ties_no_order_advantage_or_budget_overrun(self):
        r,_=timing_diagnostics([.5]*4,[1,0,0,1])
        other,_=timing_diagnostics([.5]*4,[1,1,0,0])
        self.assertEqual(r['average_precision'],.5)
        self.assertEqual(r['average_precision'],other['average_precision'])
        self.assertEqual(r['action_budgets']['actual_rate']['predicted_count'],0)
        r,_=timing_diagnostics([.9,.9,.5,.1],[1,0,1,0])
        self.assertEqual(r['action_budgets']['actual_rate']['predicted_count'],2)
        self.assertEqual(r['action_budgets']['actual_rate']['threshold_inclusive'],.9)

    def test_no_positives_and_invalid_inputs(self):
        r,_=timing_diagnostics([.3,.6],[0,0]);self.assertIsNone(r['average_precision'])
        r,_=timing_diagnostics([.3,.6],[1,1]);self.assertEqual(r['average_precision'],1)
        for probability,actual in [([],[]),([float('nan')],[1]),([.5],[2]),([1.1],[1])]:
            with self.assertRaises(ValueError):timing_diagnostics(probability,actual)

    def test_gradient_geometry_known_directions_and_zero(self):
        g=np.array([[1.,0],[-1.,0],[0,1],[0,0]])
        r=gradient_geometry(g@g.T,['a','b','c','absent'])
        self.assertEqual(r['pair_cosines']['a__b'],-1)
        self.assertEqual(r['pair_cosines']['a__c'],0)
        self.assertIsNone(r['pair_cosines']['a__absent'])
        self.assertAlmostEqual(r['cancellation_ratio'],1/3)


class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        create_fixture(self.root/'data',steps=81)
        prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
        self.ds=DecisionWindows(self.root/'data',self.root/'cache','validation',targets=4,frame_window=3,sampling='fixed')
        self.model=FixedPolicy(FixedConfig(12,4,width=16,hidden_size=32,frame_window=3))
        self.batch=collate_decisions([self.ds[0],self.ds[1]])

    def tearDown(self):self.ds.close();self.tmp.cleanup()

    def test_isolated_losses_and_gradients_equal_joint_training_loss(self):
        model=self.model;b=self.batch;out=model(b)
        joint,_=bc_loss(out,b,timing_positive_weight=32)
        parts,_=isolated_task_losses(out,b,32)
        separate=sum(v for v in parts.values() if v is not None)
        torch.testing.assert_close(joint,separate)
        params=[p for entries in shared_parameters(model).values() for _,p in entries]
        expected=torch.autograd.grad(joint,params,retain_graph=True,allow_unused=True)
        actual=torch.autograd.grad(separate,params,allow_unused=True)
        for left,right in zip(expected,actual):
            if left is None:self.assertIsNone(right)
            else:torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)

    def test_measurement_leaves_weights_and_grad_buffers_unchanged(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        r=measure_batch(self.model,self.batch,32)
        self.assertGreater(r['groups']['all_shared']['tasks']['timing']['norm'],0)
        for k,v in before.items():torch.testing.assert_close(v,self.model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        auxiliary=next(self.ds[i] for i in range(len(self.ds)) if not self.ds[i]['timing_label_mask'].any() and self.ds[i]['kind_label_mask'].any())
        r=measure_batch(self.model,collate_decisions([auxiliary]),32)
        self.assertEqual(r['label_counts']['timing'],0)
        self.assertEqual(r['groups']['all_shared']['tasks']['timing']['norm'],0)
        self.assertIsNone(r['groups']['all_shared']['pair_cosines']['timing__position'])

    def test_combined_launcher_freezes_checkpoint_and_produces_reports(self):
        checkpoint=self.root/'last.pt'
        torch.save(dict(config=asdict(self.model.config),model=self.model.state_dict(),step=9,
            contract=dict(decision_cache_sha256=digest(self.root/'cache/index.json'),targets=4,
                          train_split='validation',val_split='train',batch_size_per_rank=2,
                          world_size=1,timing_positive_weight=32)),checkpoint)
        before=digest(checkpoint)
        args=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--checkpoint',str(checkpoint),
            '--output',str(self.root/'report'),'--device','cpu','--workers','0','--cpu-threads','1',
            '--ap-batches','2','--gradient-batches','2']
        with contextlib.redirect_stdout(io.StringIO()):result=launcher.main(args)
        self.assertEqual(result['checkpoint_sha256'],before);self.assertEqual(digest(checkpoint),before)
        self.assertEqual(digest(self.root/'report/model.pt'),before)
        ap=json.loads((self.root/'report/ap/results.json').read_text())
        gr=json.loads((self.root/'report/gradients/results.json').read_text())
        self.assertEqual(gr['timing_positive_weight'],32)
        self.assertEqual(gr['source_split'],'validation')
        self.assertEqual(ap['checkpoint_sha256'],gr['checkpoint_sha256'])
        self.assertTrue((self.root/'report/ap/pr_curve.npz').exists())
        self.assertEqual(len((self.root/'report/gradients/batches.jsonl').read_text().splitlines()),2)
        with self.assertRaises(FileExistsError):launcher.main(args)


if __name__=='__main__':unittest.main()
