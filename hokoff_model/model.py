"""Public state -> entity pooling -> LSTM -> conditional CR action heads.

Independent implementation inspired by HoKoff 1v1 (see README). No HoK runtime
or upstream Python imports. Recurrent state order follows PyTorch: (hidden, cell).
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class Config:
    card_vocab_size: int
    ability_vocab_size: int
    public_scalar_size: int = 16
    entity_numeric_size: int = 3
    grid_channels: int = 8
    width: int = 256
    hidden_size: int = 512
    frame_window: int = 128
    architecture: str = "hokoff_cr_lstm_v1"

    def __post_init__(self):
        if min(self.width, self.hidden_size, self.frame_window) < 1:
            raise ValueError("positive model dimensions required")


def config_from_args(args, dims):
    return Config(**{k: dims[k] for k in (
        "card_vocab_size", "ability_vocab_size", "public_scalar_size",
        "entity_numeric_size", "grid_channels")}, width=args.width,
        hidden_size=args.hidden_size, frame_window=args.frame_window)


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        d = c.width
        self.cards = nn.Embedding(c.card_vocab_size, d, padding_idx=0)
        self.abilities = nn.Embedding(c.ability_vocab_size, d, padding_idx=0)
        self.positions = nn.Embedding(577, d, padding_idx=576)
        self.sides = nn.Embedding(2, d)
        self.entity = nn.Sequential(nn.Linear(3*d+c.entity_numeric_size, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.grid = nn.Sequential(nn.Conv2d(c.grid_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(16*8*5, d), nn.ReLU())
        # two side pools, ordered hand slots, deck, revealed cards, next card, grid
        self.scene = nn.Sequential(nn.Linear(10*d+c.public_scalar_size, c.hidden_size), nn.ReLU())
        self.lstm = nn.LSTM(c.hidden_size, c.hidden_size, batch_first=True)
        self.context = nn.Sequential(nn.Linear(c.hidden_size, d), nn.ReLU(), nn.LayerNorm(d))
        self.timing = nn.Linear(d, 1)
        self.kind = nn.Linear(d, 2)
        self.card_score = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
        self.ability_score = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
        self.position_head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 576))
        nn.init.constant_(self.timing.bias, math.log(0.02/0.98))

    def mean_cards(self, tokens):
        valid = tokens.ne(0).unsqueeze(-1)
        return (self.cards(tokens)*valid).sum(-2)/valid.sum(-2).clamp_min(1)

    def encode(self, b):
        B,T,N = b['entity_tokens'].shape
        units = self.entity(torch.cat((self.cards(b['entity_tokens']), self.positions(b['entity_positions']),
                                      self.sides(b['entity_relations']), b['entity_numeric']), -1))
        pools = []
        for side in (0,1):
            valid = b['entity_mask'] & (b['entity_relations'] == side)
            # A zero sentinel also handles N=0 and an entirely empty side.
            masked = units.masked_fill(~valid.unsqueeze(-1), torch.finfo(units.dtype).min)
            pools.append(torch.cat((masked, units.new_zeros(B,T,1,self.config.width)), -2).max(-2).values)
        return self.scene(torch.cat((*pools, self.cards(b['hand_tokens']).flatten(-2),
            self.mean_cards(b['own_deck_tokens']), self.mean_cards(b['revealed_enemy_tokens']),
            self.cards(b['next_card_token']), self.grid(b['grid'].reshape(B*T,self.config.grid_channels,32,18)).reshape(B,T,-1),
            b['public_scalars']), -1))

    def recurrent(self, x, lengths, state=None):
        """Right padding never advances hidden state, including zero-length rows."""
        B,T,_ = x.shape
        if state is None:
            state = (x.new_zeros(1,B,self.config.hidden_size), x.new_zeros(1,B,self.config.hidden_size))
        packed = pack_padded_sequence(x, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        out, new = self.lstm(packed, state)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        valid = torch.arange(T, device=x.device)[None,:] < lengths[:,None]
        out = out * valid.unsqueeze(-1)
        new = tuple(torch.where((lengths>0)[None,:,None], n, s) for n,s in zip(new,state))
        return out,new

    def heads(self, recurrent, b):
        context = self.context(recurrent)
        hand = self.cards(b['hand_tokens'])
        ability = self.abilities(b['ability_tokens'])
        card_input = torch.cat((context.unsqueeze(-2).expand_as(hand), hand), -1)
        ability_input = torch.cat((context.unsqueeze(-2).expand_as(ability), ability), -1)
        return dict(timing=self.timing(context).squeeze(-1), kind=self.kind(context),
                    card=self.card_score(card_input).squeeze(-1), ability=self.ability_score(ability_input).squeeze(-1),
                    position=self.position_head(card_input), ability_position=self.position_head(ability_input), context=context)

    def forward(self, b):
        x = self.encode(b)
        B,T,H = x.shape
        lengths = b['frame_mask'].sum(-1)
        # loss_mask is window metadata, not an expert action label.
        burn = (b['frame_mask'] & ~b['loss_mask']).sum(-1)
        with torch.no_grad():
            _, state = self.recurrent(x.detach(), burn)
        offsets = torch.arange(T,device=x.device)[None,:] + burn[:,None]
        target_x = x.gather(1, offsets.clamp_max(T-1).unsqueeze(-1).expand(-1,-1,H))
        target, _ = self.recurrent(target_x, lengths-burn, state)
        relative = torch.arange(T,device=x.device)[None,:] - burn[:,None]
        out = target.gather(1, relative.clamp_min(0).unsqueeze(-1).expand(-1,-1,H))
        out = out * ((relative>=0) & b['frame_mask']).unsqueeze(-1)
        return self.heads(out,b)

    def forward_stream(self, b, state=None, reset=None):
        """Inference chunks must remain within each actor's battle; reset at a new battle.

        This consumes observations only and ignores supervision/window metadata.
        Caller should use eval() and no_grad() for gameplay.
        """
        x = self.encode(b)
        if state is not None and reset is not None:
            state = tuple(s.masked_fill(reset[None,:,None],0) for s in state)
        out,state = self.recurrent(x,b['frame_mask'].sum(-1),state)
        return self.heads(out,b),state
