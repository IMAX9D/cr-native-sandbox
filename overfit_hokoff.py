"""First diagnostic: python overfit_hokoff.py (fixed training windows only)."""
from hokoff_model.experiments import overfit, parser


if __name__ == '__main__':
    overfit(parser('overfit').parse_args())
