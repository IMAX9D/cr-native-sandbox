import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from policy_v1.smoke import create_fixture
from policy_v1.data import prepare
from hokoff_model.deck_curve import select_pools, parser, run


class DeckCurveTests(unittest.TestCase):
    def test_selection_uses_training_frequency_and_battle_disjointness(self):
        a, b = tuple(range(1,9)), tuple(range(2,10))
        train = [dict(deck=a, battle='a'), dict(deck=a, battle='a'),
                 dict(deck=a, battle='b'), dict(deck=b, battle='c')]
        val = [dict(deck=a, battle='v'), dict(deck=b, battle='w')]
        deck, pools = select_pools(train, val, 2, 1, 42)
        self.assertEqual(deck, a)
        self.assertEqual(len({r['battle'] for r in pools['fixed_deck'][0]}), 2)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select_pools(train, train, 1, 1, 42)
        with self.assertRaisesRegex(ValueError, 'most frequent'):
            select_pools(train, [dict(deck=b,battle='z')], 1, 1, 42)

    def test_complete_synthetic_learning_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root/'data')
            prepare(root/'data', root/'cache', allow_smoke=True)
            args = parser().parse_args([
                '--data', str(root/'data'), '--cache', str(root/'cache'),
                '--run-dir', str(root/'run'), '--train-split', 'train',
                '--val-split', 'validation', '--device', 'cpu', '--width', '16',
                '--hidden-size', '32', '--frame-window', '8', '--targets', '4',
                '--steps', '2', '--eval-every', '1', '--eval-windows', '6',
                '--train-battles', '1', '--val-battles', '1', '--scan-battles', '2',
                '--workers', '0', '--batch-size', '2', '--cpu-threads', '1', '--allow-smoke'])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(len(result['arms']), 2)
            a, b = result['arms']
            self.assertEqual(a['step'], 2)
            self.assertEqual(a['common_fixed_validation']['actual_actions'],
                             b['common_fixed_validation']['actual_actions'])
            self.assertEqual(len((root/'run/fixed_deck/curve.jsonl').read_text().splitlines()), 3)
