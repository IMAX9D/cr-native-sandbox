"""Standalone LSTM pipeline timing; no checkpoint reads or writes."""
from policy_v1.benchmark import parser as base_parser, run as base_run
from .train import adapt_parser
from .model import Policy, config_from_args


def main():
    args = adapt_parser(base_parser()).parse_args()
    base_run(args,model_factory=Policy,config_factory=config_from_args)


if __name__ == '__main__':
    main()
