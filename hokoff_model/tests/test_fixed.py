import contextlib
from dataclasses import asdict
import io
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from policy_v1.data import open_arrays, close_arrays
from hokoff_model.fixed_data import fixed_indices, action_rejection, check_mask_proof, CONTRACT
from hokoff_model.decision_data import prepare, DecisionWindows, collate_decisions
from hokoff_model.decision_model import DecisionPolicy, DecisionConfig
from hokoff_model.train_fixed import FixedPolicy, FixedConfig, initialize_policy, parser, run
from hokoff_model.metrics import bc_loss
import train_hokoff_fixed as launcher


class FixedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(13)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        create_fixture(self.root/'data', steps=81)
        opened = open_arrays(self.root/'data/shards/validation-00000')
        self.a = {k: v.copy() for k, v in opened.items()}; close_arrays(opened)

    def tearDown(self): self.tmp.cleanup()

    def test_rows_and_labels_boundaries_and_action_preservation(self):
        d, audit = fixed_indices(self.a, 4)
        main = d['supervision'] == 1
        np.testing.assert_array_equal(d['rows'][main], np.r_[np.arange(0,81,4),np.arange(81,162,4)])
        self.assertEqual(d['label_rows'][0], 2)
        self.assertEqual(d['label_rows'][1], -1)  # [4,8) excludes action at 8
        self.assertEqual(d['label_rows'][2], 8)
        selected = d['label_rows'][(d['supervision']>0) & (d['label_rows']>=0)]
        np.testing.assert_array_equal(np.sort(selected), np.flatnonzero(self.a['play_now']))
        self.assertEqual(audit['positive_periods']+audit['auxiliary_actions'], audit['source_actions'])
        other = dict(self.a, play_now=np.zeros_like(self.a['play_now']))
        e, _ = fixed_indices(other, 4)
        for k in ('rows','ticks','elapsed'):
            np.testing.assert_array_equal(d[k][main], e[k][e['supervision']==1])
        self.assertFalse(d['timing_mask'][np.flatnonzero(main)[20]])  # incomplete final tick

    def test_fail_closed_legality(self):
        a = self.a
        self.assertIsNone(action_rejection(a, 0, 2))
        a['card_mask'][0,0] = 0
        self.assertEqual(action_rejection(a,0,2), 'current_card_unavailable')
        a['card_mask'][0,0] = 1
        a['hand_tokens'][0,0] = 9
        self.assertEqual(action_rejection(a,0,2), 'hand_changed')
        a['hand_tokens'][0,0] = 1
        a['public_scalars'][2,6] = .5
        self.assertEqual(action_rejection(a,0,2), 'tower_changed')
        a['public_scalars'][2,6] = 0
        a['grid_values'][2] = 254
        self.assertEqual(action_rejection(a,0,2), 'tower_changed')
        a['grid_values'][2] = 255
        self.assertEqual(action_rejection(a,81,84), 'advanced_ability')
        self.assertIsNone(action_rejection(a,84,84))
        a['selected_position_mask_packed'][0,25] = 0
        self.assertEqual(action_rejection(a,0,2), 'missing_position_mask')
        with self.assertRaises(ValueError): check_mask_proof({})

    def test_multi_and_unavailable_periods_masked_not_noop(self):
        a = self.a
        a['play_now'][3] = 1
        a['card_mask'][8,0] = 0
        d, _ = fixed_indices(a, 4)
        self.assertFalse(d['timing_mask'][0]); self.assertFalse(d['timing_mask'][2])
        self.assertEqual(d['label_rows'][0],-1)
        selected = d['label_rows'][(d['supervision']>0) & (d['label_rows']>=0)]
        np.testing.assert_array_equal(np.sort(selected), np.flatnonzero(a['play_now']))

    def prepare(self):
        prepare(self.root/'data', self.root/'cache', sampling='fixed', allow_smoke=True, auxiliary_frame_window=3)
        return DecisionWindows(self.root/'data', self.root/'cache', 'validation', sampling='fixed',
                               decision_period=4, targets=4, frame_window=3)

    def test_current_observations_future_labels_and_aux_gradients(self):
        ds = self.prepare()
        b = ds[0]
        self.assertAlmostEqual(float(b['public_scalars'][0,0]),100/6000)
        self.assertTrue(b['play_now'][0]); self.assertEqual(int(b['position'][0]),200)
        self.assertTrue(b['position_mask'][0,200]); self.assertFalse(b['delay_label_mask'].any())
        item = next(ds[i] for i in range(len(ds)) if not ds[i]['timing_label_mask'].any() and ds[i]['kind_label_mask'].any())
        batch = collate_decisions([item])
        model = FixedPolicy(FixedConfig(12,4,width=16,hidden_size=32,frame_window=3))
        out = model(batch); self.assertNotIn('delay',out)
        out['timing'].retain_grad(); out['kind'].retain_grad()
        loss,stats = bc_loss(out,batch); loss.backward()
        self.assertEqual(stats['timing_count'],0); self.assertEqual(stats['kind_count'],1)
        self.assertEqual(float(out['timing'].grad.abs().sum()),0)
        self.assertGreater(float(out['kind'].grad.abs().sum()),0)
        self.assertIsNone(model.delay.weight.grad)
        self.assertFalse(model.delay.weight.requires_grad)
        with self.assertRaises(ValueError):
            DecisionWindows(self.root/'data',self.root/'cache','train',decision_period=8)
        ds.close()

    def test_time_limit_saves_after_completed_update(self):
        ds=self.prepare(); ds.close()
        argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/'limited'),
            '--allow-smoke','--device','cpu','--workers','0','--cpu-threads','1','--width','16',
            '--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
            '--max-steps','100','--epochs','10','--eval-batches','2','--hours','0.000000001']
        with contextlib.redirect_stdout(io.StringIO()): run(parser().parse_args(argv))
        saved=load_checkpoint(self.root/'limited/last.pt')
        self.assertEqual(saved['step'],1)
        import json
        rows=[json.loads(s) for s in (self.root/'limited/metrics.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['reason'],'time_limit')

    def test_weight_transfer_resets_only_timing_and_exact_resume(self):
        ds = self.prepare(); ds.close()
        old = DecisionPolicy(DecisionConfig(12,4,width=16,hidden_size=32,frame_window=3))
        source = self.root/'initial.pt'
        torch.save(dict(config=asdict(old.config),model=old.state_dict()),source)
        model = initialize_policy(FixedConfig(12,4,width=16,hidden_size=32,frame_window=3),checkpoint=source)
        for key,value in old.state_dict().items():
            if not key.startswith('timing.'):
                torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(old.timing.weight,model.timing.weight))
        def train(name,steps,resume=False):
            argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),'--run',str(self.root/name),
                '--allow-smoke','--device','cpu','--workers','0','--cpu-threads','1','--width','16',
                '--hidden-size','32','--frame-window','3','--targets','4','--batch-size','2',
                '--max-steps',str(steps),'--epochs','10','--eval-batches','2']
            argv += ['--resume',str(self.root/name/'last.pt')] if resume else ['--init-from',str(source)]
            with contextlib.redirect_stdout(io.StringIO()): run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full=train('full',4); train('resumed',2); resumed=train('resumed',4,True)
        self.assertEqual(full['contract']['decision_contract'],CONTRACT)
        for key in full['model']:
            torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
        torch.testing.assert_close(full['model']['delay.weight'],old.delay.weight,rtol=0,atol=0)
        with contextlib.redirect_stdout(io.StringIO()):
            args=launcher.main(['--device','cpu','--dry-run'])
        self.assertEqual(args.decision_period,4); self.assertIn('fixed',args.run.name)


if __name__ == '__main__': unittest.main()
