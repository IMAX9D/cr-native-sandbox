"""Factorized space / recent frames / public events Transformer policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict

import torch
from torch import Tensor, nn


@dataclass
class PolicyConfig:
    card_vocab_size: int
    ability_vocab_size: int
    public_scalar_size: int = 16
    entity_numeric_size: int = 3
    grid_channels: int = 8
    width: int = 128
    heads: int = 4
    layers: int = 2
    frame_window: int = 128
    event_window: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        if self.width < 4 or self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if min(self.layers, self.frame_window, self.event_window) < 1:
            raise ValueError("layers and history windows must be positive")

    def to_dict(self):
        return asdict(self)


class TimeEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000.0) * torch.arange(0, width, 2).float() / width),
        )

    def forward(self, seconds):
        x = seconds.float().unsqueeze(-1) * self.frequencies
        return torch.stack((x.sin(), x.cos()), -1).flatten(-2)


def encoder(c):
    layer = nn.TransformerEncoderLayer(
        c.width,
        c.heads,
        4 * c.width,
        dropout=c.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, c.layers, norm=nn.LayerNorm(c.width), enable_nested_tensor=False
    )


class Policy(nn.Module):
    """Forward consumes only public features; labels are used solely by loss.

    Empty scene/event streams have an always-valid summary/sentinel token.
    Temporal attention is causal and bounded; events at the decision tick are
    excluded. Each position distribution is conditioned on the chosen slot.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = c = config
        d = c.width
        self.cards = nn.Embedding(c.card_vocab_size, d, padding_idx=0)
        self.abilities = nn.Embedding(c.ability_vocab_size, d, padding_idx=0)
        self.positions = nn.Embedding(577, d, padding_idx=576)
        self.sides = nn.Embedding(2, d)
        self.entity = nn.Sequential(
            nn.Linear(3 * d + c.entity_numeric_size, d), nn.GELU(), nn.Linear(d, d)
        )
        self.scalars = nn.Sequential(
            nn.Linear(c.public_scalar_size, d), nn.GELU(), nn.Linear(d, d)
        )
        self.hand_fusion = nn.Linear(4 * d, d)
        self.grid = nn.Sequential(
            nn.Conv2d(c.grid_channels, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 5, d),
        )
        self.space = encoder(c)
        self.time = encoder(c)
        self.events = encoder(c)
        self.clock = TimeEncoding(d)
        self.event_type = nn.Embedding(2, d)
        self.event_fusion = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.empty_event = nn.Parameter(torch.zeros(1, 1, d))
        self.event_attention = nn.MultiheadAttention(
            d, c.heads, dropout=c.dropout, batch_first=True
        )
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.timing = nn.Linear(d, 1)
        self.kind = nn.Linear(d, 2)
        self.card_score = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.ability_score = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.position_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 576)
        )
        # Initial action probability ~0.02 per observed 50ms tick, not 0.5.
        nn.init.constant_(self.timing.bias, math.log(0.02 / 0.98))

    def _mean_cards(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        return (self.cards(tokens) * mask).sum(-2) / mask.sum(-2).clamp_min(1)

    def forward(self, b: Dict[str, Tensor]):
        B, T, N = b["entity_tokens"].shape
        d = self.config.width
        hand = self.cards(b["hand_tokens"])
        numeric = self.scalars(b["public_scalars"])
        summary = (
            numeric
            + self.hand_fusion(hand.flatten(-2))
            + self._mean_cards(b["own_deck_tokens"])
            + self._mean_cards(b["revealed_enemy_tokens"])
            + self.cards(b["next_card_token"])
            + self.grid(
                b["grid"].reshape(B * T, self.config.grid_channels, 32, 18)
            ).reshape(B, T, d)
        )
        units = self.entity(
            torch.cat(
                (
                    self.cards(b["entity_tokens"]),
                    self.positions(b["entity_positions"]),
                    self.sides(b["entity_relations"]),
                    b["entity_numeric"],
                ),
                -1,
            )
        )
        x = torch.cat((summary.unsqueeze(-2), units), -2).reshape(B * T, N + 1, d)
        padding = torch.cat(
            (
                torch.zeros(B, T, 1, dtype=torch.bool, device=x.device),
                ~b["entity_mask"],
            ),
            -1,
        ).reshape(B * T, N + 1)
        scene = self.space(x, src_key_padding_mask=padding)[:, 0].reshape(B, T, d)
        order = torch.arange(T, device=x.device)
        temporal_mask = (order[None, :] > order[:, None]) | (
            order[:, None] - order[None, :] >= self.config.frame_window
        )
        recent = self.time(
            scene + self.clock(b["frame_ticks"] / 20.0),
            mask=temporal_mask,
            src_key_padding_mask=~b["frame_mask"],
        )
        # Dedicated card/ability tables; token 0 is PAD in each vocabulary.
        event_identity = torch.where(
            b["event_kind"].bool().unsqueeze(-1),
            self.abilities(b["event_ability"]),
            self.cards(b["event_card"]),
        )
        ev = self.event_fusion(
            torch.cat(
                (
                    event_identity,
                    self.sides(b["event_side"]),
                    self.positions(b["event_position"]),
                    self.event_type(b["event_kind"]),
                ),
                -1,
            )
        )
        ev = ev + self.clock(b["event_ticks"] / 20.0)
        ev = torch.cat((self.empty_event.expand(B, -1, -1), ev), 1)
        E = ev.shape[1]
        epad = torch.cat(
            (torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~b["event_mask"]), 1
        )
        # Strictly earlier event times: simultaneous events do not see arbitrary
        # tie ordering, including deployments by the other player at that tick.
        times = torch.cat(
            (torch.full((B, 1), -1.0, device=x.device), b["event_ticks"].float()), 1
        )
        emask = times[:, None, :] >= times[:, :, None]
        emask[:, :, 0] = False
        diag = torch.arange(E, device=x.device)
        emask[:, diag, diag] = False
        emask = emask.repeat_interleave(self.config.heads, 0)
        memory = self.events(ev, mask=emask, src_key_padding_mask=epad)
        blocked = (times[:, None, :] >= b["frame_ticks"][:, :, None]) | epad[:, None, :]
        # Retain the latest event_window visible events at every query time.
        valid = ~blocked
        rank_from_end = valid.flip(-1).long().cumsum(-1).flip(-1)
        blocked = blocked | (rank_from_end > self.config.event_window)
        blocked[:, :, 0] = False
        past, _ = self.event_attention(
            recent,
            memory,
            memory,
            attn_mask=blocked.repeat_interleave(self.config.heads, 0),
            need_weights=False,
        )
        context = self.fusion(torch.cat((recent, past, scene), -1))
        cards = self.card_score(
            torch.cat((context.unsqueeze(-2).expand_as(hand), hand), -1)
        ).squeeze(-1)
        abilities = self.abilities(b["ability_tokens"])
        ability_logits = self.ability_score(
            torch.cat((context.unsqueeze(-2).expand_as(abilities), abilities), -1)
        ).squeeze(-1)
        return {
            "timing": self.timing(context).squeeze(-1),
            "kind": self.kind(context),
            "card": cards,
            "ability": ability_logits,
            "context": context,
            "position": self.position_logits(
                context.unsqueeze(-2).expand_as(hand), b["hand_tokens"]
            ),
            "ability_position": self.position_logits(
                context.unsqueeze(-2).expand_as(abilities),
                b["ability_tokens"],
                ability=True,
            ),
        }

    def position_logits(self, context, tokens, *, ability=False):
        embedding = self.abilities(tokens) if ability else self.cards(tokens)
        return self.position_head(torch.cat((context, embedding), -1))
