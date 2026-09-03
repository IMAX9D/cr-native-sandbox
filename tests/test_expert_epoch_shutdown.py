import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.finish_expert_epoch_shutdown import require_completed, validate_and_preserve, power_off_autodl


class EpochShutdownTests(unittest.TestCase):
    def test_provider_script_uses_explicit_shell_without_executing_it(self):
        with patch('scripts.finish_expert_epoch_shutdown.subprocess.run') as command:
            power_off_autodl()
            command.assert_called_once_with(['/bin/bash', '/usr/bin/shutdown'], check=True, timeout=30)

    def test_only_exact_completed_epoch_is_accepted(self):
        good = dict(status='paused', reason='stop_after_epoch', epoch=2, global_step=154674)
        require_completed(good, 2, 154674)
        for key, value in [('status', 'failed'), ('status', 'training'), ('reason', 'user_requested_checkpoint'),
                           ('epoch', 3), ('global_step', 154675)]:
            with self.assertRaises(RuntimeError):
                require_completed({**good, key: value}, 2, 154674)

    def test_artifacts_are_verified_and_backed_up_without_shutdown(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / 'test-run'
            (root / 'checkpoints/epochs').mkdir(parents=True)
            (root / 'exports/epochs').mkdir(parents=True)
            for name in ('manifest.json', 'events.jsonl', 'training-progress.json', 'launch.json'):
                (root / name).write_text(json.dumps({'run_id': root.name}))
            tb = base / 'tb' / root.name
            tb.mkdir(parents=True)
            (tb / 'events.out.tfevents.test').write_bytes(b'test')
            value = dict(run_id=root.name, epoch=2, global_step=10, epoch_complete=True,
                model_state={'weight': torch.ones(2)}, optimizer_state={'state': {'v': torch.ones(2)}},
                scheduler_state={'last_epoch': 2}, rng={'torch': torch.get_rng_state()},
                normalizer_state={'kind': 'test'}, validation_metrics={'loss': 5.4})
            for filename in ('checkpoints/latest.pt', 'checkpoints/best.pt',
                             'checkpoints/epochs/epoch-002.pt', 'exports/epochs/epoch-002-fp16.pt'):
                torch.save(value, root / filename)
            receipt = validate_and_preserve(root, 2, 10, base / 'tb')
            backup = Path(receipt['protected_backup'])
            self.assertTrue((backup / 'latest.pt').is_file())
            self.assertTrue((backup / 'best.pt').is_file())
            self.assertTrue((backup / 'epoch-002-fp16.pt').is_file())
            self.assertTrue((backup / 'receipt.json').is_file())
            value['model_state']['weight'][0] = float('nan')
            torch.save(value, root / 'checkpoints/latest.pt')
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                validate_and_preserve(root, 2, 10, base / 'tb')
