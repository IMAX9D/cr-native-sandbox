"""Fixed-period action baseline. Reuses the recurrent engine without a delay loss."""
from dataclasses import dataclass, asdict
from functools import partial
import math
from pathlib import Path

from torch import nn
from policy_v1.data import digest
from policy_v1.train import parser as base_parser, run as base_run, load_checkpoint
from .train import adapt_parser
from .model import Policy
from .decision_model import DecisionConfig, DecisionPolicy
from .decision_data import DecisionWindows, collate_decisions
from .fixed_data import CONTRACT
from .metrics import bc_loss, summarize
from .console_log import format_console


@dataclass
class FixedConfig(DecisionConfig):
    architecture: str = 'hokoff_cr_lstm_fixed_period_v1'
    decision_period: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.decision_period <= 32767:
            raise ValueError('invalid fixed decision period')


class FixedPolicy(DecisionPolicy):
    def __init__(self, config):
        super().__init__(config)
        # Retain keys for weight transfer, but never compute or optimize this head.
        self.delay.requires_grad_(False)
        nn.init.zeros_(self.timing.bias)

    def heads(self, recurrent, b):
        return Policy.heads(self, recurrent, b)


def config_from_args(args, dims):
    return FixedConfig(**{k: dims[k] for k in ('card_vocab_size', 'ability_vocab_size',
        'public_scalar_size', 'entity_numeric_size', 'grid_channels')}, width=args.width,
        hidden_size=args.hidden_size, frame_window=args.frame_window, max_delay=args.max_delay,
        decision_period=args.decision_period)


def initialize_policy(config, *, checkpoint):
    saved = load_checkpoint(checkpoint)
    old = dict(saved['config']); new = asdict(config)
    if old.get('architecture') not in ('hokoff_cr_lstm_decisions_v1', config.architecture):
        raise ValueError('initial checkpoint must be a decision/fixed model')
    for key in ('architecture', 'decision_period'):
        old.pop(key, None); new.pop(key, None)
    if old != new:
        raise ValueError('initial checkpoint dimensions differ')
    model = FixedPolicy(config)
    model.load_state_dict(saved['model'])
    # A period action is a different target from an exact-tick action.
    nn.init.normal_(model.timing.weight, std=.01)
    nn.init.zeros_(model.timing.bias)
    return model


def parser():
    p = adapt_parser(base_parser())
    p.description = __doc__
    p.set_defaults(frame_window=17, targets=32, train_split='validation', val_split='train')
    p.add_argument('--hours', type=float, default=0.0, help='stop and save after this many training hours; 0 disables the time limit')
    p.add_argument('--decision-period', type=int, default=4)
    p.add_argument('--max-delay', type=int, default=8, help='pretrained time-feature normalization only')
    p.add_argument('--timing-positive-weight', type=float, default=32.0)
    p.add_argument('--init-from', type=Path, help='copy body/action weights; reset timing head and optimizer')
    return p


def run(args):
    if not math.isfinite(args.hours) or args.hours < 0:
        raise ValueError('hours must be finite and nonnegative')
    args.time_limit_seconds = args.hours*3600
    if not math.isfinite(args.timing_positive_weight) or args.timing_positive_weight <= 0:
        raise ValueError('timing-positive-weight must be finite and positive')
    if args.init_from is not None and args.resume is not None:
        raise ValueError('--init-from and --resume are mutually exclusive')
    if args.init_from is not None and args.run.exists() and any(args.run.iterdir()):
        raise FileExistsError('--init-from requires a new/empty run directory')
    factory = FixedPolicy if args.init_from is None else partial(initialize_policy, checkpoint=args.init_from)
    return base_run(args, model_factory=factory, config_factory=config_from_args,
        dataset_factory=partial(DecisionWindows, sampling='fixed', max_delay=args.max_delay,
                                decision_period=args.decision_period), collate_fn=collate_decisions,
        bc_loss=partial(bc_loss, timing_positive_weight=args.timing_positive_weight), summarize=summarize,
        console_formatter=format_console, contract_extra=dict(decision_contract=CONTRACT,
            decision_period=args.decision_period, decision_cache_sha256=digest(args.cache/'index.json'),
            timing_positive_weight=args.timing_positive_weight, training_only=True, delay_enabled=False))


if __name__ == '__main__':
    run(parser().parse_args())
