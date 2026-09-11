"""Mirror fixed BC metrics.jsonl without touching the running trainer."""
import argparse
import json
import math
from pathlib import Path
import time


def mirror_available(path, writer, offset=0):
    """Keep incomplete final lines pending until the trainer finishes writing."""
    if not path.is_file():
        return offset
    if path.stat().st_size < offset:
        raise RuntimeError('metrics file was truncated; use a new TensorBoard log directory')
    with path.open('rb') as stream:
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line.endswith(b'\n'):
                return start
            row = json.loads(line)
            phase, step = row.get('phase'), row.get('step')
            if phase in ('train', 'validation', 'amp_overflow') and isinstance(step, int):
                prefix = 'val' if phase == 'validation' else phase
                for name, value in row.items():
                    if name != 'step' and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        writer.add_scalar(prefix+'/'+name, value, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be positive and finite')
    # A fresh destination prevents duplicate events when restarting a mirror.
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        parser.error('--log-dir must be empty/new; a restart replays the full metrics history')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.log_dir), flush_secs=2)
    offset = 0
    try:
        while True:
            offset = mirror_available(args.metrics, writer, offset)
            writer.flush()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()


if __name__ == '__main__':
    main()
