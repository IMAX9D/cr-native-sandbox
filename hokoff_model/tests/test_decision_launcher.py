import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from policy_v1.smoke import create_fixture
from policy_v1.train import load_checkpoint
import train_hokoff_decisions as launcher


class LauncherTests(unittest.TestCase):
    def test_dry_run_resolves_device_paths_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(launcher, 'BASE', base), patch('torch.cuda.is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                args = launcher.main(['--dry-run','--max-delay','4','--device','cpu'])
            self.assertEqual(args.cache, base/'hokoff-decision-cache-k4')
            self.assertEqual(args.run, base/'runs/hokoff-decisions-k4')
            self.assertEqual(args.precision,'fp32')
            self.assertEqual(args.max_steps,1000)
            self.assertEqual(list(base.iterdir()),[])

    def test_prepare_train_and_auto_resume_additional_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            create_fixture(base/'data',steps=40)
            argv = ['--data',str(base/'data'),'--cache',str(base/'cache'),'--run-dir',str(base/'run'),
                    '--device','cpu','--allow-smoke','--width','16','--hidden-size','32',
                    '--frame-window','3','--targets','4','--batch-size','2','--workers','0',
                    '--cpu-threads','1','--steps','2','--eval-batches','1']
            with contextlib.redirect_stdout(io.StringIO()):
                launcher.main(argv)
                before = (base/'cache/index.json').read_bytes()
                launcher.main(argv)
            self.assertEqual(load_checkpoint(base/'run/last.pt')['step'],4)
            self.assertEqual((base/'cache/index.json').read_bytes(),before)
            with contextlib.redirect_stdout(io.StringIO()):
                args = launcher.main(argv+['--dry-run','--max-steps','10'])
            self.assertEqual(args.max_steps,10)
            source = base/'run/last.pt'
            source_bytes = source.read_bytes()
            experiment = argv+['--run-dir',str(base/'short8'),'--delay-short-weight','8']
            with contextlib.redirect_stdout(io.StringIO()):
                launcher.main(experiment+['--init-from',str(source)])
                self.assertEqual(load_checkpoint(base/'short8/last.pt')['step'],2)
                launcher.main(experiment)
            saved = load_checkpoint(base/'short8/last.pt')
            self.assertEqual(saved['step'],4)
            self.assertEqual(saved['contract']['delay_short_weight'],8)
            self.assertEqual(source.read_bytes(),source_bytes)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.main(experiment+['--init-from',str(source)])


if __name__ == '__main__':
    unittest.main()
