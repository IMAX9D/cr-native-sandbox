from dataclasses import asdict
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from policy_v1.data import digest, prepare
from policy_v1.smoke import create_fixture
from hokoff_model.model import Config, Policy
from hokoff_model.evaluate_timing import probability_report, parser, run


class TimingEvalTests(unittest.TestCase):
    def test_constant_and_perfect_ranking_with_ties(self):
        constant = probability_report([.1]*10, [True]+[False]*9)
        self.assertAlmostEqual(constant['average_precision'], .1)
        self.assertAlmostEqual(constant['roc_auc'], .5)
        shuffled = probability_report([.1]*10, [False]*9+[True])
        self.assertEqual(constant['average_precision'], shuffled['average_precision'])
        perfect = probability_report([.8,.8,.01,.01], [True,True,False,False])
        self.assertEqual(perfect['average_precision'], 1.)
        self.assertEqual(perfect['roc_auc'], 1.)
        self.assertEqual(perfect['thresholds'][-1]['precision'], 1.)
        self.assertEqual(perfect['thresholds'][-1]['recall'], 1.)

    def test_empty_classes_and_threshold_boundary(self):
        r = probability_report([.5,.5], [True,False])
        self.assertEqual(r['thresholds'][-1]['predicted_actions'], 0)
        self.assertIsNone(r['thresholds'][-1]['precision'])
        r = probability_report([.1,.2], [False,False])
        self.assertIsNone(r['average_precision'])
        self.assertIsNone(r['action_probabilities'])
        with self.assertRaises(ValueError):
            probability_report([np.nan], [True])

    def test_checkpoint_evaluation_no_weight_changes_and_holdout_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = create_fixture(root/'data')
            prepare(data, root/'cache', allow_smoke=True)
            config = Config(12,4,width=16,hidden_size=32,frame_window=8)
            saved = {'config': asdict(config), 'model': Policy(config).state_dict(),
                     'step': 7, 'contract': {'train_split': 'train', 'val_split': 'validation',
                     'targets': 4, 'manifest_sha256': digest(data/'manifest.json')}}
            checkpoint = root/'last.pt'
            torch.save(saved, checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args = parser().parse_args(['--checkpoint', str(checkpoint), '--data', str(data),
                '--cache', str(root/'cache'), '--device', 'cpu', '--allow-smoke',
                '--batch-size', '2', '--batches', '3', '--workers', '0',
                '--output', str(root/'report.json')])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result['windows'], 6)
            self.assertEqual(result['split'], 'validation')
            self.assertEqual(result['checkpoint_step'], 7)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            self.assertTrue((root/'report.json').is_file())
            saved['contract']['val_split'] = 'train'
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, 'held-out'):
                run(args)
