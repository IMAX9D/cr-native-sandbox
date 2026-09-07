"""One-command fixed-period training: prepare missing cache and resume last.pt automatically.

Defaults match ~/cr-data. Each invocation adds up to 1000 updates (within the
configured epoch limit); --hours enables a wall-time budget instead, and
--max-steps overrides the update limit explicitly.
"""
from pathlib import Path
import json

import torch

from hokoff_model.decision_data import prepare
from hokoff_model.train_fixed import parser as trainer_parser, run
from policy_v1.train import load_checkpoint


BASE = Path.home() / 'cr-data'


def parser():
    p = trainer_parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest in ('data', 'cache', 'run'):
            action.required = False
    p.set_defaults(data=BASE/'expert-dataset/native-bc-v1', cache=None, run=None,
                   device='auto', precision=None, batch_size=None, workers=None,
                   epochs=10, max_steps=None, log_every=100, save_every=500,
                   eval_every=500, eval_batches=100, eval_shuffle=True)
    p.add_argument('--steps', type=int, default=1000, help='additional updates this invocation, unless --hours or --max-steps is set')
    p.add_argument('--dry-run', action='store_true', help='print resolved settings without preparing data or training')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    dry_run, steps = args.dry_run, args.steps
    del args.dry_run, args.steps
    if steps < 1:
        p.error('--steps must be positive')
    if args.cache is None:
        args.cache = BASE/('hokoff-fixed-cache-p%d' % args.decision_period)
    if args.run is None:
        args.run = BASE/'runs'/('hokoff-fixed-p%d' % args.decision_period)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.precision is None:
        args.precision = 'fp16' if args.device == 'cuda' else 'fp32'
    if args.batch_size is None:
        args.batch_size = 32 if args.device == 'cuda' else 8
    if args.workers is None:
        args.workers = 4 if args.device == 'cuda' else 0
    if args.init_from is not None and args.resume is not None:
        p.error('--init-from and --resume are mutually exclusive')
    if args.init_from is not None and args.run.exists() and any(args.run.iterdir()):
        p.error('--init-from requires an empty/new --run directory')
    if args.init_from is None and args.resume is None and (args.run/'last.pt').is_file():
        args.resume = args.run/'last.pt'
    completed_step = 0
    if args.resume is not None:
        saved = load_checkpoint(args.resume)
        if saved.get('config', {}).get('architecture') != 'hokoff_cr_lstm_fixed_period_v1':
            p.error('resume requires a fixed-period checkpoint')
        completed_step = int(saved['step'])
        del saved
    if args.max_steps is None:
        args.max_steps = 0 if args.hours else completed_step+steps
    cache_missing = not (args.cache/'index.json').is_file()
    print(json.dumps({'phase':'fixed_launcher', 'prepare_cache':cache_missing,
                      'completed_step':completed_step, 'arguments':vars(args)}, default=str, indent=2), flush=True)
    if dry_run:
        return args
    if not (args.data/'manifest.json').is_file():
        p.error('dataset manifest is missing: '+str(args.data/'manifest.json'))
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu --precision fp32')
    if args.device == 'cpu' and args.precision != 'fp32':
        p.error('CPU training requires --precision fp32')
    if args.evaluate_only and args.resume is None:
        p.error('--evaluate-only needs a checkpoint')
    if cache_missing:
        print('Preparing full decision indices (no archive extraction):', args.cache, flush=True)
        prepare(args.data, args.cache, max_delay=args.max_delay,
                splits=list(dict.fromkeys([args.train_split, args.val_split])),
                allow_smoke=args.allow_smoke,sampling='fixed',sampling_seed=args.seed,
                auxiliary_split=args.train_split,auxiliary_frame_window=args.frame_window,
                decision_period=args.decision_period)
    run(args)
    return args


if __name__ == '__main__':
    main()
