"""Full-data, one-epoch AutoDL LSTM run; flags may override the preset."""
from datetime import datetime
from pathlib import Path
import sys
from train_hokoff import main as launch


def main():
    run = Path('/root/autodl-tmp/runs') / ('hokoff-lstm-long-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    launch(['--run-dir', str(run), '--epochs', '1', '--max-steps', '0',
            '--timing-positive-weight', '32', '--precision', 'fp16',
            '--save-every', '1000', '--eval-every', '5000', '--eval-batches', '200',
            '--eval-shuffle'] + sys.argv[1:])


if __name__ == '__main__':
    main()
