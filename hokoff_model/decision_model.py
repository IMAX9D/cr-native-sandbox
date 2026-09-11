"""Training-only decision-sequence variant; existing runtime policy is unchanged."""
from dataclasses import dataclass

import torch
from torch import nn

from .model import Config, Policy, no_grad_burn_in


@dataclass
class DecisionConfig(Config):
    frame_window: int = 17
    max_delay: int = 8
    architecture: str = 'hokoff_cr_lstm_decisions_v1'

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.max_delay <= 32767:
            raise ValueError('invalid maximum delay')


def config_from_args(args, dims):
    return DecisionConfig(**{k: dims[k] for k in (
        'card_vocab_size', 'ability_vocab_size', 'public_scalar_size',
        'entity_numeric_size', 'grid_channels')}, width=args.width,
        hidden_size=args.hidden_size, frame_window=args.frame_window, max_delay=args.max_delay)


class DecisionPolicy(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.time_projection = nn.Linear(2, config.hidden_size)
        # Three conditional distributions: WAIT, DEPLOY, ABILITY. No labels enter the network.
        self.delay = nn.Linear(config.width, 3*config.max_delay)

    def encode(self, b):
        x = super().encode(b)
        time = torch.stack((b['frame_ticks'].float()/6000,
                            b['prev_elapsed_ticks'].float()/self.config.max_delay), dim=-1)
        return x+self.time_projection(time)

    def heads(self, recurrent, b):
        out = super().heads(recurrent, b)
        out['delay'] = self.delay(out['context']).reshape(*recurrent.shape[:2], 3, self.config.max_delay)
        return out

    def forward(self, b):
        B, T = b['frame_mask'].shape
        valid = b['frame_mask']
        burn_mask = valid & ~b['loss_mask']
        target_mask = valid & b['loss_mask']
        fields = ('entity_tokens', 'entity_positions', 'entity_relations', 'entity_numeric',
                  'entity_mask', 'hand_tokens', 'own_deck_tokens', 'revealed_enemy_tokens',
                  'next_card_token', 'grid', 'public_scalars', 'frame_ticks', 'prev_elapsed_ticks')
        if getattr(self.config, 'history_length', 0):
            from .history import HISTORY_FIELDS
            fields += HISTORY_FIELDS
        # Encode only selected, non-padding observations. Burn-in encoding has no autograd graph.
        def encode_rows(mask):
            selected = {k: b[k].reshape(B*T, *b[k].shape[2:])[mask.reshape(-1)].unsqueeze(1)
                        for k in fields}
            return self.encode(selected).squeeze(1)
        x = self.scene[0].weight.new_zeros(B*T, self.config.hidden_size)
        if burn_mask.any():
            with no_grad_burn_in():
                encoded = encode_rows(burn_mask)
            x = x.index_copy(0, burn_mask.reshape(-1).nonzero().flatten(), encoded.to(x.dtype))
        if target_mask.any():
            encoded = encode_rows(target_mask)
            x = x.index_copy(0, target_mask.reshape(-1).nonzero().flatten(), encoded.to(x.dtype))
        x = x.reshape(B, T, -1)
        burn = burn_mask.sum(-1)
        lengths = valid.sum(-1)
        with no_grad_burn_in():
            _, state = self.recurrent(x.detach(), burn)
        offsets = torch.arange(T, device=x.device)[None, :]+burn[:, None]
        target_x = x.gather(1, offsets.clamp_max(T-1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        target, _ = self.recurrent(target_x, lengths-burn, state)
        relative = torch.arange(T, device=x.device)[None, :]-burn[:, None]
        out = target.gather(1, relative.clamp_min(0).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        out = out*((relative >= 0) & valid).unsqueeze(-1)
        return self.heads(out, b)

    def forward_stream(self, *args, **kwargs):
        raise NotImplementedError('decision checkpoints are training-only; runtime integration is deferred')
