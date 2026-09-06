"""Second diagnostic: python compare_hokoff.py (baseline vs positive-weight BC)."""
from hokoff_model.experiments import compare, parser


if __name__ == '__main__':
    compare(parser('compare').parse_args())
