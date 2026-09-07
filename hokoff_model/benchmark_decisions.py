"""Stage timing for the decision model; temporary weights, no checkpoint writes.

Run separately from active training to avoid contention. Defaults use ~/cr-data.
"""
from functools import partial
from pathlib import Path

from policy_v1.benchmark import parser as base_parser, run as base_run
from .train import adapt_parser
from .decision_data import collate_decisions, DecisionWindows
from .decision_model import DecisionPolicy, config_from_args
from .decision_loss import bc_loss
from .console_log import format_console


def parser():
    p = adapt_parser(base_parser())
    p.description = __doc__
    for action in p._actions:
        if action.dest in ('data', 'cache'):
            action.required = False
    base = Path.home()/'cr-data'
    p.set_defaults(data=base/'expert-dataset/native-bc-v1',
                   cache=base/'hokoff-decision-cache-k8', split='validation',
                   frame_window=17, targets=32, workers=4, warmup=10, steps=100,
                   log_every=100)
    p.add_argument('--max-delay', type=int, default=8)
    return p


def run(args):
    result = base_run(args, model_factory=DecisionPolicy, config_factory=config_from_args,
                    collate_fn=collate_decisions,
                      dataset_factory=partial(DecisionWindows, max_delay=args.max_delay),
                      loss_fn=bc_loss)
    print(format_console(result), flush=True)
    return result


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
