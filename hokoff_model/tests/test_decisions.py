import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from policy_v1.data import Windows, collate, prepare as prepare_dense
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
from hokoff_model.decision_data import decision_indices, prepare, DecisionWindows
from hokoff_model.decision_model import DecisionConfig, DecisionPolicy
from hokoff_model.decision_loss import bc_loss, summarize
from hokoff_model.train_decisions import parser, run, initialize_policy


class DecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_fixture(self.root/'data', steps=40)
        prepare(self.root/'data', self.root/'cache', allow_smoke=True, verify_hashes=True)
        self.ds = DecisionWindows(self.root/'data', self.root/'cache', 'train', targets=4, frame_window=3)
        self.model = DecisionPolicy(DecisionConfig(12, 4, width=16, hidden_size=32, frame_window=3))

    def tearDown(self):
        self.ds.close(); self.tmp.cleanup()

    def test_index_preserves_events_and_censors_holes_and_tail(self):
        n = 36
        scalar = np.zeros((n, 16), np.float32)
        scalar[:, 0] = np.minimum(5990+np.arange(n), 6000)/6000
        valid = np.ones(n, bool); valid[11:14] = False; valid[32:] = False
        actions = np.zeros(n, bool); actions[[6, 12, 14, 16, 29, 33]] = True
        d = decision_indices(np.array([0,n]), scalar, np.ones(n), np.ones(n), actions, valid, 8)
        np.testing.assert_array_equal(d['rows'], [0,6,14,16,24,29])
        np.testing.assert_array_equal(d['ticks'], 5990+d['rows'])
        np.testing.assert_array_equal(d['elapsed'], [0,6,0,2,8,5])
        np.testing.assert_array_equal(d['delay'], [6,0,2,8,5,0])
        np.testing.assert_array_equal(d['delay_mask'], [True,False,True,True,True,False])
        np.testing.assert_array_equal(d['offsets'], [0,2,6])
        self.assertEqual(int(actions[d['rows']].sum()), int((actions & valid).sum()))
        with self.assertRaises(ValueError):
            decision_indices(np.array([0,n]), scalar, np.full(n,8), np.ones(n), actions, valid, 8)

    def test_selected_observations_masks_and_entities_match_dense_reader(self):
        prepare_dense(self.root/'data', self.root/'dense', allow_smoke=True)
        dense = Windows(self.root/'data', self.root/'dense', 'train', targets=40, frame_window=1, event_window=1)
        full = dense[0]
        for i in range(int(self.ds.sequence_prefix[0][1])):
            item = self.ds[i]
            rows = item['frame_ticks']-100
            for k in ('public_scalars','hand_tokens','entity_tokens','entity_numeric','entity_positions',
                      'entity_mask','grid','position_mask','ability_position_mask','play_now','card_slot'):
                torch.testing.assert_close(item[k], full[k][rows])
            self.assertEqual(len(item['event_ticks']), 0)
        with self.assertRaises(ValueError):
            DecisionWindows(self.root/'data', self.root/'cache', 'train', max_delay=4)

    def test_burn_in_padding_causality_and_no_target_leakage(self):
        b = collate([self.ds[0], self.ds[1]])
        self.model.eval()
        with torch.no_grad():
            out = self.model(b)
            # Independent full observation scan is a reference, not a runtime integration.
            x = self.model.encode(b)
            recurrent, _ = self.model.recurrent(x, b['frame_mask'].sum(-1))
            reference = self.model.heads(recurrent, b)
            target = b['frame_mask'] & b['loss_mask']
            for k in ('timing','card','position','delay'):
                torch.testing.assert_close(out[k][target], reference[k][target], rtol=1e-5, atol=1e-6)
            changed = {k:v.clone() for k,v in b.items()}
            for k in ('play_now','action_kind','card_slot','delay_target_ticks','delay_label_mask'):
                changed[k].zero_()
            other = self.model(changed)
            torch.testing.assert_close(out['delay'], other['delay'])
            changed['public_scalars'][:,-1] += 20
            future = self.model(changed)
            torch.testing.assert_close(out['timing'][:,:-1], future['timing'][:,:-1])
        b['public_scalars'].requires_grad_()
        loss, _ = bc_loss(self.model(b), b); loss.backward()
        burn = b['frame_mask'] & ~b['loss_mask']
        self.assertEqual(float(b['public_scalars'].grad[burn].abs().sum()), 0)
        self.assertGreater(float(b['public_scalars'].grad[target].abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        with self.assertRaises(NotImplementedError):
            self.model.forward_stream(b)

    def test_censored_delay_and_unknown_kind_have_no_delay_gradient(self):
        b = collate([self.ds[0], self.ds[1]])
        b['delay_label_mask'].zero_()
        out = self.model(b); out['delay'].retain_grad()
        loss, stats = bc_loss(out, b); loss.backward()
        self.assertEqual(stats['delay_count'], 0)
        self.assertEqual(float(out['delay'].grad.abs().sum()), 0)
        self.assertGreater(stats['timing_count'], 0)
        self.assertTrue(torch.isfinite(loss))
        b['delay_label_mask'].fill_(True); b['delay_target_ticks'].fill_(3)
        b['play_now'].fill_(True); b['kind_label_mask'].zero_(); b['action_kind'].fill_(-100)
        loss, stats = bc_loss(self.model(b), b)
        self.assertEqual(stats['delay_count'], 0)
        self.assertGreater(stats['delay_unknown_mode_count'], 0)

    def test_loss_learns_delay_and_reports_majority_baseline(self):
        b = collate([self.ds[0], self.ds[2]])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        first, s = bc_loss(self.model(b), b)
        for _ in range(35):
            optimizer.zero_grad(); loss, last = bc_loss(self.model(b), b); loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()), float(first.detach())*.7)
        self.assertLess(summarize(last)['delay_loss'], summarize(s)['delay_loss'])
        self.assertIn('delay_short_recall', summarize(last))
        self.assertIn('delay_always_max_accuracy', summarize(last))

    def test_short_weight_changes_only_delay_objective_and_keeps_raw_metrics(self):
        b = collate([self.ds[0], self.ds[2]])
        b['play_now'].zero_()
        b['delay_label_mask'].zero_()
        selected = (b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']).nonzero()[:2]
        self.assertEqual(len(selected), 2)
        for row, ticks in zip(selected, (2, 8)):
            i, j = row.tolist()
            b['delay_label_mask'][i,j] = True
            b['delay_target_ticks'][i,j] = ticks
            b['sample_weight'][i,j] = 1
        out = self.model(b)
        logits = torch.zeros_like(out['delay'])
        logits[..., -1] = 2
        out['delay'] = logits.requires_grad_()
        old_loss, old = bc_loss(out, b)
        new_loss, new = bc_loss(out, b, delay_short_weight=8)
        ce = torch.logsumexp(logits[0,0,0], 0)
        expected = (8*ce+(ce-2))/9
        self.assertAlmostEqual(summarize(new)['delay_loss'], float(expected.detach()), places=6)
        self.assertAlmostEqual(float((new_loss-old_loss).detach()), summarize(new)['delay_loss']-summarize(old)['delay_loss'], places=5)
        metrics = summarize(new)
        for key in ('delay_unweighted_loss', 'delay_short_recall', 'delay_short_accuracy',
                    'delay_short_mae_ticks', 'delay_short_late_rate', 'position_accuracy'):
            self.assertEqual(metrics[key], summarize(old)[key])
        self.assertEqual(metrics['delay_predicted_8_rate'], 1)
        self.assertEqual(metrics['delay_target_2_rate'], .5)
        self.assertEqual(metrics['delay_target_8_rate'], .5)
        self.assertEqual(metrics['delay_predicted_mean_ticks'], 8)
        self.assertEqual(metrics['delay_short_mae_ticks'], 6)
        self.assertEqual(metrics['delay_short_late_rate'], 1)
        grad = torch.autograd.grad(new_loss, out['delay'])[0]
        short_grad = grad[tuple(selected[0].tolist())+(0, 0)]
        max_grad = grad[tuple(selected[1].tolist())+(0, 0)]
        self.assertAlmostEqual(float(short_grad/max_grad), 8, places=5)
        b['delay_label_mask'].zero_()
        empty_loss, empty = bc_loss(out, b, delay_short_weight=8)
        self.assertTrue(torch.isfinite(empty_loss))
        self.assertEqual(summarize(empty)['delay_predicted_short_rate'], 0)
        self.assertEqual(float(torch.autograd.grad(empty_loss, out['delay'])[0].abs().sum()), 0)
        for weight in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): bc_loss(out, b, delay_short_weight=weight)

    def test_prepare_partial_shards_does_not_rewrite_source(self):
        before = (self.root/'data/manifest.json').read_bytes()
        index = prepare(self.root/'data', self.root/'partial', max_delay=4,
                        max_shards_per_split=1, splits=['validation'], allow_smoke=True)
        self.assertEqual(set(index['splits']), {'validation'})
        self.assertEqual((self.root/'data/manifest.json').read_bytes(), before)
        self.assertEqual(index['max_delay'],4)

    def test_train_resume_contract_and_validation(self):
        def train(name, steps, resume=False, k=8, weight=1, init=None):
            argv = ['--data', str(self.root/'data'), '--cache', str(self.root/'cache'),
                    '--run-dir', str(self.root/name), '--allow-smoke', '--device','cpu',
                    '--width','16','--hidden-size','32','--frame-window','3','--targets','4',
                    '--batch-size','2','--workers','0','--cpu-threads','1','--max-steps',str(steps),
                    '--epochs','3','--eval-batches','2','--max-delay',str(k),
                    '--delay-short-weight',str(weight)]
            if resume: argv += ['--resume',str(self.root/name/'last.pt')]
            if init: argv += ['--init-from',str(init)]
            with contextlib.redirect_stdout(io.StringIO()):
                run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full = train('full',4)
        train('resumed',2); resumed = train('resumed',4,True)
        self.assertEqual(resumed['config']['architecture'],'hokoff_cr_lstm_decisions_v1')
        self.assertTrue(resumed['contract']['training_only'])
        self.assertNotIn('delay_short_weight', resumed['contract'])
        for key in full['model']:
            torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
        with self.assertRaises(ValueError): train('resumed',5,True,k=4)
        source = self.root/'full/last.pt'
        initialized = initialize_policy(self.model.config, checkpoint=source)
        for key in full['model']:
            torch.testing.assert_close(initialized.state_dict()[key], full['model'][key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            initialize_policy(replace(self.model.config, width=32), checkpoint=source)
        weighted_full = train('weighted-full', 4, weight=8, init=source)
        first = train('weighted', 2, weight=8, init=source)
        self.assertEqual(first['step'], 2)
        self.assertEqual(first['contract']['delay_short_weight'], 8)
        weighted_resumed = train('weighted', 4, True, weight=8)
        for key in weighted_full['model']:
            torch.testing.assert_close(weighted_full['model'][key], weighted_resumed['model'][key], rtol=0, atol=0)
        with self.assertRaises(ValueError): train('weighted', 5, True, weight=1)
        with self.assertRaises(FileExistsError): train('weighted', 5, weight=8, init=source)
        with self.assertRaises(ValueError): train('weighted', 5, True, weight=8, init=source)


if __name__ == '__main__':
    unittest.main()
