"""Synthetic CPU integration check, not a trained game policy."""
from pathlib import Path
import tempfile
from policy_v1.smoke import create_fixture
from policy_v1.data import prepare
from .train import parser,run


def main():
    root = Path(tempfile.mkdtemp(prefix='hokoff-cr-smoke-'))
    create_fixture(root/'data')
    prepare(root/'data',root/'cache',allow_smoke=True,verify_hashes=True)
    args = parser().parse_args(['--data',str(root/'data'),'--cache',str(root/'cache'),
        '--run-dir',str(root/'run'),'--allow-smoke','--device','cpu','--width','16',
        '--hidden-size','32','--frame-window','8','--targets','4','--workers','0',
        '--batch-size','2','--cpu-threads','1','--max-steps','3','--log-every','1','--eval-batches','2'])
    run(args)
    print('Synthetic LSTM smoke passed; not a trained game policy:',root/'run/last.pt')


if __name__ == '__main__':
    main()
