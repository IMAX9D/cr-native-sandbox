"""Train HoKoff on bounded decision sequences; does not change gameplay inference."""
from functools import partial
from dataclasses import asdict
import math
from pathlib import Path

from policy_v1.data import digest
from policy_v1.train import parser as base_parser, run as base_run, load_checkpoint
from .train import adapt_parser
from .decision_data import collate_decisions, DecisionWindows, CONTRACT, INDEPENDENT_CONTRACT
from .decision_model import DecisionPolicy, config_from_args
from .decision_loss import bc_loss, summarize
from .console_log import format_console


def parser():
    p = adapt_parser(base_parser())
    p.description = __doc__
    p.set_defaults(frame_window=17, targets=32, train_split='validation', val_split='train')
    p.add_argument('--max-delay', type=int, default=8)
    p.add_argument('--sampling', choices=['legacy','independent'], default='legacy')
    p.add_argument('--delay-weight', type=float, default=1.0)
    p.add_argument('--delay-short-weight', type=float, default=1.0,
                   help='relative loss weight for exact delays below max-delay; 1 keeps the original objective')
    p.add_argument('--init-from', type=Path,
                   help='start a new experiment from model weights only; resets optimizer and step, requires a new run directory')
    p.add_argument('--timing-positive-weight', type=float, default=1.0)
    return p


def initialize_policy(config, *, checkpoint):
    saved = load_checkpoint(checkpoint)
    if saved['config'] != asdict(config):
        raise ValueError('initial checkpoint model config differs; use matching model arguments')
    model = DecisionPolicy(config)
    model.load_state_dict(saved['model'])
    return model


def run(args):
    if not 1 <= args.max_delay <= 32767:
        raise ValueError('max-delay must be in 1..32767')
    if any(not math.isfinite(x) or x <= 0 for x in (args.delay_weight, args.timing_positive_weight, args.delay_short_weight)):
        raise ValueError('loss weights must be finite and positive')
    if args.init_from is not None and args.resume is not None:
        raise ValueError('--init-from and --resume are mutually exclusive')
    if args.init_from is not None and args.run.exists() and any(args.run.iterdir()):
        raise FileExistsError('--init-from requires an empty/new run directory')
    contract = dict(decision_contract=CONTRACT if args.sampling=='legacy' else INDEPENDENT_CONTRACT, max_delay=args.max_delay,
                    decision_cache_sha256=digest(args.cache/'index.json'), delay_weight=args.delay_weight,
                    timing_positive_weight=args.timing_positive_weight, training_only=True)
    # Missing key historically means weight 1. Preserve exact resume compatibility.
    if args.delay_short_weight != 1.0:
        contract['delay_short_weight'] = args.delay_short_weight
    model_factory = DecisionPolicy if args.init_from is None else partial(initialize_policy, checkpoint=args.init_from)
    return base_run(args, model_factory=model_factory, config_factory=config_from_args,
                    collate_fn=collate_decisions,
                    dataset_factory=partial(DecisionWindows, max_delay=args.max_delay,sampling=args.sampling),
                    bc_loss=partial(bc_loss, delay_weight=args.delay_weight, timing_positive_weight=args.timing_positive_weight,
                                    delay_short_weight=args.delay_short_weight),
                    summarize=summarize, console_formatter=format_console, contract_extra=contract)


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
