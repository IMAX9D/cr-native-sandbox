import contextlib
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

import torch
from hokoff_model.train_fixed import FixedConfig, FixedPolicy, parser, run
from hokoff_model.decision_data import prepare, DecisionWindows, collate_decisions
from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self, dim=2):
        return FixedPolicy(FixedConfig(12, 4, width=16, hidden_size=32,
                                      frame_window=3, spatial_type_dim=dim))

    def test_sum_side_position_padding_and_gradient(self):
        m = self.model()
        with torch.no_grad():
            m.spatial_types.weight.zero_()
            m.spatial_types.weight[1] = torch.tensor([1., 2.])
            m.spatial_types.weight[2] = torch.tensor([3., 4.])
        b = dict(entity_tokens=torch.tensor([[[1, 1, 2, 999]]]),
                 entity_positions=torch.tensor([[[20, 20, 20, 999]]]),
                 entity_relations=torch.tensor([[[0, 0, 1, 999]]]),
                 entity_mask=torch.tensor([[[True, True, True, False]]]))
        grid = m.spatial_type_grid(b)
        torch.testing.assert_close(grid[0, 0, :, 1, 2], torch.tensor([2., 4., 3., 4.])/16)
        self.assertEqual(int(torch.count_nonzero(grid)), 4)
        reordered = {k: v[:, :, [2, 0, 3, 1]] for k, v in b.items()}
        torch.testing.assert_close(grid, m.spatial_type_grid(reordered))
        grid.sum().backward()
        torch.testing.assert_close(m.spatial_types.weight.grad[1], torch.full((2,), 2/16))
        self.assertEqual(float(m.spatial_types.weight.grad[0].abs().sum()), 0)
        empty = {k: v[:, :, :0] for k, v in b.items()}
        self.assertEqual(float(m.spatial_type_grid(empty).detach().abs().sum()), 0)

    def test_grid_retains_type_location_association(self):
        m = self.model()
        b = dict(entity_tokens=torch.tensor([[[1, 2]]]),
                 entity_positions=torch.tensor([[[10, 400]]]),
                 entity_relations=torch.tensor([[[0, 0]]]),
                 entity_mask=torch.ones(1, 1, 2, dtype=torch.bool))
        swapped = dict(b, entity_positions=b['entity_positions'].flip(-1))
        self.assertFalse(torch.equal(m.spatial_type_grid(b), m.spatial_type_grid(swapped)))

    def test_training_streaming_and_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root/'data', steps=81)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root/'data', root/'cache', sampling='fixed', allow_smoke=True,
                        auxiliary_frame_window=3)
            ds = DecisionWindows(root/'data', root/'cache', 'validation', sampling='fixed',
                                 decision_period=4, targets=4, frame_window=3)
            b = collate_decisions([ds[0], ds[1]])
            m = self.model().eval()
            out = m(b)
            stream, _ = m.forward_stream(b)
            valid = b['frame_mask'] & b['loss_mask']
            for key in ('timing', 'card', 'position'):
                torch.testing.assert_close(out[key][valid], stream[key][valid], atol=1e-6, rtol=1e-5)
            out['timing'][valid].sum().backward()
            self.assertGreater(float(m.spatial_types.weight.grad.abs().sum()), 0)
            changed = {k: v.clone() for k, v in b.items()}
            changed['entity_tokens'][:, -1] = 2
            torch.testing.assert_close(m(changed)['timing'][:, :-1], out['timing'][:, :-1])
            for key in ('play_now', 'card_slot', 'position'):
                changed[key].zero_()
            changed['entity_tokens'] = b['entity_tokens']
            torch.testing.assert_close(m(changed)['timing'], out['timing'])
            base = ['--data', str(root/'data'), '--cache', str(root/'cache'), '--allow-smoke',
                    '--device', 'cpu', '--workers', '0', '--cpu-threads', '1', '--width', '16',
                    '--hidden-size', '32', '--frame-window', '3', '--targets', '4',
                    '--batch-size', '2', '--epochs', '10', '--eval-batches', '1']
            def train(name, steps, dim, resume=False):
                argv = base + ['--run', str(root/name), '--max-steps', str(steps),
                               '--spatial-type-dim', str(dim)]
                if resume: argv += ['--resume', str(root/name/'last.pt')]
                with contextlib.redirect_stdout(io.StringIO()): run(parser().parse_args(argv))
                return load_checkpoint(root/name/'last.pt')
            full = train('full', 4, 2)
            train('resume', 2, 2)
            resumed = train('resume', 4, 2, True)
            for k in full['model']:
                torch.testing.assert_close(full['model'][k], resumed['model'][k], rtol=0, atol=0)
            loaded = FixedPolicy(FixedConfig(**resumed['config'])).eval()
            loaded.load_state_dict(resumed['model'])
            self.assertTrue(torch.isfinite(loaded.forward_stream(b)[0]['position']).all())
            old = train('legacy', 2, 0)
            old['config'].pop('spatial_type_dim')
            torch.save(old, root/'legacy/last.pt')
            train('legacy', 3, 0, True)
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                train('legacy', 4, 2, True)

    def test_disabled_legacy_weights_and_validation(self):
        m = self.model(0)
        self.assertFalse(any(k.startswith('spatial_types') for k in m.state_dict()))
        self.assertEqual(m.grid[0].in_channels, 8)
        old = asdict(m.config)
        old.pop('spatial_type_dim')
        self.model(0).load_state_dict(FixedPolicy(FixedConfig(**old)).state_dict())
        self.assertEqual(self.model(8).grid[0].in_channels, 24)
        with self.assertRaises(ValueError): self.model(-1)


if __name__ == '__main__': unittest.main()
