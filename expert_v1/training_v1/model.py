"""Variable-deck recurrent expert policy with conditional action heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .schema import (
    ARENA_COLUMNS,
    ARENA_ROWS,
    OBSERVATION_NATIVE,
    OBSERVATION_SEQUENCE,
    POSITION_COUNT,
)


@dataclass(frozen=True)
class ExpertPolicyConfig:
    grid_channels: int
    public_scalar_size: int
    card_vocab_size: int
    ability_vocab_size: int
    max_ability_slots: int
    card_embedding_size: int = 64
    spatial_size: int = 64
    hidden_size: int = 256
    lambda_max: float = 20.0
    lambda_initial: float = 0.30
    native_tick_seconds: float = 0.05
    observation_mode: str = OBSERVATION_NATIVE

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class ExpertPolicyOutput(NamedTuple):
    rate_logits: Tensor
    action_kind_logits: Tensor
    card_logits: Tensor
    position_logits: Tensor
    ability_logits: Tensor
    ability_position_logits: Tensor
    hidden: tuple[Tensor, Tensor]


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class RecurrentExpertPolicy(nn.Module):
    """Actor-only model; its API has no privileged-state argument."""

    def __init__(self, config: ExpertPolicyConfig) -> None:
        super().__init__()
        self.config = config
        embed = config.card_embedding_size
        spatial = config.spatial_size
        hidden = config.hidden_size
        if config.observation_mode not in (OBSERVATION_NATIVE, OBSERVATION_SEQUENCE):
            raise ValueError(f"unsupported observation mode: {config.observation_mode}")
        self.card_embedding = nn.Embedding(config.card_vocab_size, embed, padding_idx=0)
        self.ability_embedding = nn.Embedding(config.ability_vocab_size, embed, padding_idx=0)
        if config.observation_mode == OBSERVATION_NATIVE:
            if config.grid_channels <= 0:
                raise ValueError("native-state model requires grid channels")
            self.spatial = nn.Sequential(
                nn.Conv2d(config.grid_channels, 32, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, spatial, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(spatial, spatial, 3, padding=1),
                nn.SiLU(),
            )
            self.cell_features = nn.Conv2d(spatial, embed, 1)
            recurrent_spatial = spatial
        else:
            if config.grid_channels != 0:
                raise ValueError("sequence-only model cannot accept grid channels")
            self.spatial = None
            self.cell_features = None
            self.sequence_cell_embedding = nn.Embedding(POSITION_COUNT, embed)
            self.previous_position_embedding = nn.Embedding(
                POSITION_COUNT + 1, embed, padding_idx=POSITION_COUNT
            )
            self.previous_side_embedding = nn.Embedding(3, 8, padding_idx=0)
            self.event_context = nn.Sequential(
                nn.Linear(embed * 2 + 8, 96),
                nn.SiLU(),
                nn.Linear(96, spatial),
                nn.SiLU(),
            )
            recurrent_spatial = spatial
        self.scalar = nn.Sequential(
            nn.Linear(config.public_scalar_size, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.card_context = nn.Sequential(
            nn.Linear(embed * 7, 128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
        )
        self.recurrent = nn.LSTM(
            input_size=recurrent_spatial + 64 + 96 + 1,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.rate_head = nn.Linear(hidden, 1)
        self.action_kind_head = nn.Linear(hidden, 2)
        self.card_query = nn.Linear(hidden, embed, bias=False)
        self.card_key = nn.Linear(embed, embed, bias=False)
        self.card_bias = nn.Sequential(nn.Linear(embed, 32), nn.SiLU(), nn.Linear(32, 1))
        self.position_query = nn.Sequential(
            nn.Linear(hidden + embed, embed), nn.SiLU(), nn.Linear(embed, embed)
        )
        self.ability_query = nn.Linear(hidden, embed, bias=False)
        self.ability_key = nn.Linear(embed, embed, bias=False)
        self.ability_bias = nn.Sequential(nn.Linear(embed, 32), nn.SiLU(), nn.Linear(32, 1))
        self.ability_position_query = nn.Sequential(
            nn.Linear(hidden + embed, embed), nn.SiLU(), nn.Linear(embed, embed)
        )
        if not 0.0 < config.lambda_initial < config.lambda_max:
            raise ValueError("lambda_initial must be in (0, lambda_max)")
        with torch.no_grad():
            self.rate_head.weight.zero_()
            prior = config.lambda_initial / config.lambda_max
            self.rate_head.bias.fill_(math.log(prior / (1.0 - prior)))

    def initial_hidden(self, batch_size: int, *, device: torch.device | str) -> tuple[Tensor, Tensor]:
        shape = (1, batch_size, self.config.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def _masked_mean(values: Tensor, tokens: Tensor) -> Tensor:
        mask = (tokens != 0).unsqueeze(-1)
        numerator = (values * mask).sum(dim=-2)
        denominator = mask.sum(dim=-2).clamp_min(1)
        return numerator / denominator

    def forward_sequence(
        self,
        *,
        grid: Tensor | None,
        public_scalars: Tensor,
        own_deck_tokens: Tensor,
        hand_tokens: Tensor,
        next_card_token: Tensor,
        revealed_enemy_tokens: Tensor,
        ability_tokens: Tensor | None,
        delta_ticks: Tensor,
        previous_event_card_token: Tensor | None = None,
        previous_event_side: Tensor | None = None,
        previous_event_position: Tensor | None = None,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> ExpertPolicyOutput:
        batch, steps = public_scalars.shape[:2]
        if self.config.observation_mode == OBSERVATION_NATIVE:
            if grid is None or grid.ndim != 5:
                raise ValueError("grid must have shape [batch,time,channels,32,18]")
            if tuple(grid.shape[-2:]) != (ARENA_ROWS, ARENA_COLUMNS):
                raise ValueError("arena shape must be 32x18")
            assert self.spatial is not None
            flat_spatial = self.spatial(grid.flatten(0, 1))
            pooled = flat_spatial.mean(dim=(-2, -1)).reshape(batch, steps, -1)
        else:
            if grid is not None:
                raise ValueError("sequence-only model must not receive a native grid")
            if any(
                value is None
                for value in (
                    previous_event_card_token,
                    previous_event_side,
                    previous_event_position,
                )
            ):
                raise ValueError("sequence-only model requires previous public event fields")
            assert previous_event_card_token is not None
            assert previous_event_side is not None
            assert previous_event_position is not None
            pooled = self.event_context(
                torch.cat(
                    (
                        self.card_embedding(previous_event_card_token),
                        self.previous_position_embedding(previous_event_position),
                        self.previous_side_embedding(previous_event_side),
                    ),
                    dim=-1,
                )
            )
        hand = self.card_embedding(hand_tokens)
        own_deck = self._masked_mean(self.card_embedding(own_deck_tokens), own_deck_tokens)
        enemy = self._masked_mean(
            self.card_embedding(revealed_enemy_tokens), revealed_enemy_tokens
        )
        next_card = self.card_embedding(next_card_token)
        card_context = self.card_context(
            torch.cat(
                (hand.flatten(start_dim=-2), own_deck, enemy, next_card), dim=-1
            )
        )
        delta = torch.log1p(delta_ticks.clamp_min(0)).unsqueeze(-1) / math.log(121.0)
        recurrent_input = torch.cat(
            (pooled, self.scalar(public_scalars), card_context, delta), dim=-1
        )
        recurrent, next_hidden = self.recurrent(recurrent_input, hidden)

        card_logits = torch.einsum(
            "bte,btse->bts",
            self.card_query(recurrent),
            self.card_key(hand),
        ) / math.sqrt(hand.shape[-1])
        card_logits = card_logits + self.card_bias(hand).squeeze(-1)

        if self.config.observation_mode == OBSERVATION_NATIVE:
            assert self.cell_features is not None
            cells = self.cell_features(flat_spatial).flatten(2).transpose(1, 2)
            cells = cells.reshape(batch, steps, POSITION_COUNT, -1)
        else:
            cells = self.sequence_cell_embedding.weight.unsqueeze(0).unsqueeze(0)
            cells = cells.expand(batch, steps, -1, -1)
        position_queries = self.position_query(
            torch.cat((recurrent.unsqueeze(2).expand(-1, -1, 4, -1), hand), dim=-1)
        )
        position_logits = torch.einsum("btse,btpe->btsp", position_queries, cells)
        position_logits = position_logits / math.sqrt(cells.shape[-1])

        if ability_tokens is None:
            ability_logits = recurrent.new_zeros(
                batch, steps, self.config.max_ability_slots
            )
            ability_position_logits = recurrent.new_zeros(
                batch, steps, self.config.max_ability_slots, POSITION_COUNT
            )
        else:
            abilities = self.ability_embedding(ability_tokens)
            ability_logits = torch.einsum(
                "bte,btae->bta", self.ability_query(recurrent), self.ability_key(abilities)
            ) / math.sqrt(abilities.shape[-1])
            ability_logits = ability_logits + self.ability_bias(abilities).squeeze(-1)
            ability_position_queries = self.ability_position_query(
                torch.cat(
                    (
                        recurrent.unsqueeze(2).expand(
                            -1, -1, self.config.max_ability_slots, -1
                        ),
                        abilities,
                    ),
                    dim=-1,
                )
            )
            ability_position_logits = torch.einsum(
                "btae,btpe->btap", ability_position_queries, cells
            ) / math.sqrt(cells.shape[-1])
        return ExpertPolicyOutput(
            self.rate_head(recurrent).squeeze(-1),
            self.action_kind_head(recurrent),
            card_logits,
            position_logits,
            ability_logits,
            ability_position_logits,
            next_hidden,
        )

    def forward_batch(self, batch: dict[str, Tensor]) -> ExpertPolicyOutput:
        common = (
            "public_scalars",
            "own_deck_tokens",
            "hand_tokens",
            "next_card_token",
            "revealed_enemy_tokens",
            "delta_ticks",
        )
        values = {name: batch[name] for name in common}
        if self.config.observation_mode == OBSERVATION_SEQUENCE:
            values.update(
                grid=None,
                ability_tokens=None,
                previous_event_card_token=batch["previous_event_card_token"],
                previous_event_side=batch["previous_event_side"],
                previous_event_position=batch["previous_event_position"],
            )
        else:
            values.update(grid=batch["grid"], ability_tokens=batch["ability_tokens"])
        return self.forward_sequence(**values)
