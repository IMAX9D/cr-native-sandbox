"""Train the CR pooled-entity LSTM using the existing offline data/checkpoint engine."""
import math
from functools import partial

from policy_v1.train import parser as base_parser, run as base_run
from .model import Policy, config_from_args
from .metrics import bc_loss, summarize


def adapt_parser(p):
    # These Transformer options have no meaning for this model. The reusable
    # engine still expects an event window for its shared data loader.
    for action in list(p._actions):
        if action.dest in ('heads','layers','event_window'):
            p._remove_action(action)
            for group in p._action_groups:
                if action in group._group_actions: group._group_actions.remove(action)
            for option in action.option_strings: p._option_string_actions.pop(option,None)
    p.set_defaults(width=256, heads=1, layers=1, event_window=1, batch_size=32)
    p.add_argument('--hidden-size', type=int, default=512)
    p.description = __doc__
    return p


def parser():
    p = adapt_parser(base_parser())
    p.add_argument("--timing-positive-weight", type=float, default=1.0)
    return p


def run(args):
    weight = args.timing_positive_weight
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("timing-positive-weight must be finite and positive")
    extra = {"timing_positive_weight": weight} if weight != 1.0 else None
    return base_run(args, model_factory=Policy, config_factory=config_from_args,
                    bc_loss=partial(bc_loss, timing_positive_weight=weight),
                    summarize=summarize, contract_extra=extra)


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
