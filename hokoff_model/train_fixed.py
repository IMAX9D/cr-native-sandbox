"""Fixed-period action baseline. Reuses the recurrent engine without a delay loss."""
from dataclasses import dataclass, asdict
from functools import partial
import math
from pathlib import Path

import torch
from torch import nn
from .history import CONTRACT as HISTORY_CONTRACT
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
    spatial_type_dim: int = 0
    history_length: int = 0
    spatial_skip_channels: int = 0
    combat_features: list | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.combat_features is not None:
            if (len(self.combat_features) != self.card_vocab_size
                    or any(len(row) != 20 for row in self.combat_features)
                    or any(not math.isfinite(v) for row in self.combat_features for v in row)
                    or any(self.combat_features[0])):
                raise ValueError('invalid static combat feature table')
        if not 0 <= self.spatial_skip_channels <= 64:
            raise ValueError("spatial_skip_channels must be between 0 and 64")
        if not 0 <= self.history_length <= 16:
            raise ValueError("history_length must be between 0 and 16")
        if self.spatial_type_dim < 0:
            raise ValueError("spatial_type_dim must be nonnegative")
        if not 1 <= self.decision_period <= 32767:
            raise ValueError('invalid fixed decision period')


class FixedPolicy(DecisionPolicy):
    def __init__(self, config):
        super().__init__(config)
        # Retain keys for weight transfer, but never compute or optimize this head.
        self.delay.requires_grad_(False)
        nn.init.zeros_(self.timing.bias)
        if config.history_length:
            self.history_cards = nn.Embedding(config.card_vocab_size, 16, padding_idx=0)
            self.history_summary = nn.Sequential(
                nn.Linear(2*config.history_length*21, 64), nn.ReLU(),
                nn.Linear(64, config.hidden_size))

        if config.spatial_skip_channels:
            from .spatial_position import SpatialPositionHead
            self.position_skip = SpatialPositionHead(config.grid_channels+2*config.spatial_type_dim,
                                                     config.width, config.spatial_skip_channels)

    def encode(self, b):
        x = super().encode(b)
        if self.config.history_length:
            mask = b['history_mask']
            known = b['history_known'] & mask
            card = self.history_cards(b['history_card']) * known.unsqueeze(-1)
            pos = b['history_position']
            xy = torch.stack(((pos % 18).float()/17, (pos // 18).float()/31), -1)
            xy = xy * known.unsqueeze(-1)
            age = torch.log1p(b['history_age'].float()) / math.log(301)
            features = torch.cat((card, xy, (age*mask).unsqueeze(-1),
                                  mask.unsqueeze(-1), known.unsqueeze(-1)), -1)
            x = x + self.history_summary(features.flatten(-3))
        return x

    def heads(self, recurrent, b):
        out = Policy.heads(self, recurrent, b)
        if self.config.spatial_skip_channels:
            B, T = b['frame_mask'].shape
            valid = b['frame_mask']
            # BC burn-in and padding do not need position logits or spatial graphs.
            if 'loss_mask' in b: valid = valid & b['loss_mask']
            flat = valid.reshape(-1)
            if flat.any():
                grid = b['grid'].reshape(B*T, self.config.grid_channels, 32, 18)[flat]
                if self.config.spatial_type_dim:
                    selected = {k: b[k].reshape(B*T, *b[k].shape[2:])[flat].unsqueeze(1)
                                for k in ('entity_tokens', 'entity_positions', 'entity_relations', 'entity_mask')}
                    grid = torch.cat((grid, self.spatial_type_grid(selected)[:, 0].to(grid)), 1)
                context = out['context'].reshape(B*T, -1)[flat]
                hand = self.cards(b['hand_tokens'].reshape(B*T, 4)[flat])
                correction = self.position_skip(grid, context, hand)
                baseline = out['position'].reshape(B*T, 4, 576)
                out['position'] = baseline.index_add(0, flat.nonzero().flatten(),
                                                     correction.to(baseline.dtype)).reshape(B, T, 4, 576)
        return out

    def forward_stream(self, b, state=None, reset=None):
        if self.config.history_length:
            raise NotImplementedError('history checkpoints are BC-only; online history is not integrated')
        # Same encoder/time features and heads as BC; independent state per actor.
        return Policy.forward_stream(self, b, state=state, reset=reset)


def config_from_args(args, dims):
    combat = None
    if args.combat_features_file is not None:
        import json
        table = json.loads(args.combat_features_file.read_text())
        manifest = json.loads((args.data/'manifest.json').read_text())
        if (table.get('schema') != 'cr_nominal_static_combat_v1'
                or table.get('game_version') != '15.535.29'
                or table['card_vocabulary'] != manifest['card_vocabulary']):
            raise ValueError('static combat table schema/version/vocabulary mismatch')
        combat = table['features']
    return FixedConfig(**{k: dims[k] for k in ('card_vocab_size', 'ability_vocab_size',
        'public_scalar_size', 'entity_numeric_size', 'grid_channels')}, width=args.width,
        hidden_size=args.hidden_size, frame_window=args.frame_window, max_delay=args.max_delay,
        decision_period=args.decision_period, spatial_type_dim=args.spatial_type_dim, history_length=args.history_length, spatial_skip_channels=args.spatial_skip_channels, combat_features=combat)


def initialize_policy(config, *, checkpoint):
    saved = load_checkpoint(checkpoint)
    old = dict(saved['config']); new = asdict(config)
    if old.get('architecture') not in ('hokoff_cr_lstm_decisions_v1', config.architecture):
        raise ValueError('initial checkpoint must be a decision/fixed model')
    old.setdefault('spatial_type_dim', 0)
    old.setdefault('history_length', 0)
    old.setdefault('spatial_skip_channels', 0)
    old.setdefault('combat_features', None)
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
    p.add_argument('--combat-features-file', type=Path, help='versioned static combat table; use dedicated migration launcher for old weights')
    p.add_argument('--spatial-skip-channels', type=int, default=0, help='full-resolution position skip channels; 0 disables, 16 recommended')
    p.add_argument('--history-length', type=int, default=0, help='BC-only recent plays per side; 0 disables, 4 recommended')
    p.add_argument('--spatial-type-dim', type=int, default=0,
                   help='public card embedding channels per side in the grid; 0 disables, 8 recommended for comparison')
    p.add_argument('--decision-period', type=int, default=4)
    p.add_argument('--max-delay', type=int, default=8, help='pretrained time-feature normalization only')
    p.add_argument('--timing-positive-weight', type=float, default=32.0)
    p.add_argument('--init-weights', type=Path, help='retain all fixed-policy weights; start fresh optimizer and local step counter')
    p.add_argument('--weights-contract', type=Path, default=Path(__file__).with_name('match_encoder_contract.json'))
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
    from .weights_restart import prepare_restart, initialize_weights
    provenance = {}
    if args.init_weights is not None:
        if args.init_from is not None or args.resume is not None:
            raise ValueError('--init-weights, --init-from and --resume are mutually exclusive')
        if args.run.exists() and any(args.run.iterdir()):
            raise FileExistsError('--init-weights requires a new/empty run directory')
        saved, provenance = prepare_restart(args.init_weights, args.weights_contract, args.data)
        factory = partial(initialize_weights, saved=saved)
    else:
        factory = FixedPolicy if args.init_from is None else partial(initialize_policy, checkpoint=args.init_from)
        if args.resume is not None:
            provenance = load_checkpoint(args.resume)['contract'].get('weights_restart', {})
    return base_run(args, model_factory=factory, config_factory=config_from_args,
        dataset_factory=partial(DecisionWindows, sampling='fixed', max_delay=args.max_delay,
                                decision_period=args.decision_period, history_length=args.history_length), collate_fn=collate_decisions,
        bc_loss=partial(bc_loss, timing_positive_weight=args.timing_positive_weight), summarize=summarize,
        console_formatter=format_console, contract_extra=dict(decision_contract=CONTRACT,
            decision_period=args.decision_period, decision_cache_sha256=digest(args.cache/'index.json'),
            timing_positive_weight=args.timing_positive_weight, training_only=True, delay_enabled=False,
            **(dict(weights_restart=provenance) if provenance else {}),
            **(dict(history_contract=HISTORY_CONTRACT) if args.history_length else {})))


if __name__ == '__main__':
    run(parser().parse_args())
