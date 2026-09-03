"""Independent asymmetric Critic for the frozen expert Actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from expert_v1.training_v1.model import RecurrentExpertPolicy
from .actor_adapter import ExpertActorAdapter


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_first = nn.GroupNorm(8, channels)
        self.norm_second = nn.GroupNorm(8, channels)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = F.silu(self.norm_first(self.first(value)))
        value = self.norm_second(self.second(value))
        return F.silu(value + residual)


class _ResidualMlp(nn.Module):
    def __init__(self, width: int = 512, expansion: int = 2048) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, expansion)
        self.second = nn.Linear(expansion, width)

    def forward(self, value: Tensor) -> Tensor:
        hidden = F.silu(self.first(self.norm(value)))
        return value + self.second(hidden)


@dataclass(frozen=True)
class PrivilegedCriticConfig:
    actor_latent_size: int
    card_vocab_size: int
    public_grid_channels: int = 8
    entity_numeric_size: int = 3
    scalar_size: int = 32
    position_count: int = 32 * 18
    private_slot_count: int = 32


class CriticOutput(NamedTuple):
    values: Tensor
    wdl_logits: Tensor
    crown_difference: Tensor
    tower_hp_difference: Tensor
    future_damage: Tensor


class PrivilegedCritic(nn.Module):
    """Centralized training-only value network with no Actor parameter sharing."""

    def __init__(self, config: PrivilegedCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.actor_branch = nn.Sequential(
            nn.LayerNorm(config.actor_latent_size),
            nn.Linear(config.actor_latent_size, 256),
            nn.SiLU(),
        )
        self.grid_stem = nn.Sequential(
            nn.Conv2d(config.public_grid_channels, 32, 3, padding=1),
            nn.SiLU(),
            _ResidualConv(32),
            _ResidualConv(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            _ResidualConv(64),
            _ResidualConv(64),
        )
        self.grid_output = nn.Linear(128, 256)

        self.entity_card = nn.Embedding(config.card_vocab_size, 64, padding_idx=0)
        self.entity_position = nn.Embedding(config.position_count, 32)
        self.entity_relation = nn.Embedding(2, 8)
        self.entity_input = nn.Linear(64 + 32 + 8 + config.entity_numeric_size, 256)
        entity_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_transformer = nn.TransformerEncoder(
            entity_layer, num_layers=2, enable_nested_tensor=False
        )
        self.entity_output = nn.Linear(512, 256)

        self.private_card = nn.Embedding(config.card_vocab_size, 64, padding_idx=0)
        self.private_owner = nn.Embedding(2, 8)
        self.private_slot = nn.Embedding(config.private_slot_count, 16)
        self.private_input = nn.Linear(64 + 8 + 16, 128)
        private_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.private_transformer = nn.TransformerEncoder(
            private_layer, num_layers=2, enable_nested_tensor=False
        )
        self.private_output = nn.Linear(256, 256)
        self.scalar_branch = nn.Sequential(
            nn.Linear(config.scalar_size, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )

        self.fusion = nn.Linear(256 * 4 + 128, 512)
        self.fusion_blocks = nn.Sequential(*(_ResidualMlp() for _ in range(3)))
        self.value_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1))
        self.wdl_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 3))
        self.crown_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1))
        self.tower_hp_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1))
        self.future_damage_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 2))

    @staticmethod
    def _masked_pool(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        visible = mask.unsqueeze(-1)
        mean = (values * visible).sum(dim=1) / visible.sum(dim=1).clamp_min(1)
        minimum = torch.finfo(values.dtype).min
        maximum = values.masked_fill(~visible, minimum).max(dim=1).values
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)

    def forward(
        self,
        *,
        actor_latent: Tensor,
        grid: Tensor,
        entity_tokens: Tensor,
        entity_positions: Tensor,
        entity_relations: Tensor,
        entity_numeric: Tensor,
        entity_mask: Tensor,
        private_card_tokens: Tensor,
        private_card_owners: Tensor,
        private_card_slots: Tensor,
        private_card_mask: Tensor,
        scalars: Tensor,
    ) -> CriticOutput:
        batch, steps = actor_latent.shape[:2]
        prefix = (batch, steps)
        if grid.shape[:2] != prefix or scalars.shape[:2] != prefix:
            raise ValueError("Critic dense inputs have inconsistent prefixes")
        if entity_tokens.shape[:2] != prefix or private_card_tokens.shape[:2] != prefix:
            raise ValueError("Critic ragged inputs have inconsistent prefixes")
        flat = batch * steps
        actor_value = self.actor_branch(actor_latent.detach())

        spatial = self.grid_stem(grid.flatten(0, 1))
        spatial = torch.cat(
            (spatial.mean(dim=(-2, -1)), spatial.amax(dim=(-2, -1))), dim=-1
        )
        spatial = self.grid_output(spatial).reshape(batch, steps, -1)

        entity = self.entity_input(torch.cat((
            self.entity_card(entity_tokens),
            self.entity_position(entity_positions.clamp(0, self.config.position_count - 1)),
            self.entity_relation(entity_relations.clamp(0, 1)),
            entity_numeric,
        ), dim=-1)).reshape(flat, entity_tokens.shape[2], 256)
        entity_visible = entity_mask.reshape(flat, entity_mask.shape[2]).bool()
        entity_transform_mask = entity_visible.clone()
        entity_empty = ~entity_transform_mask.any(dim=1)
        if bool(entity_empty.any()):
            entity = entity.clone()
            entity[entity_empty, 0] = 0
            entity_transform_mask[entity_empty, 0] = True
        entity = self.entity_transformer(
            entity, src_key_padding_mask=~entity_transform_mask
        )
        entity = self.entity_output(self._masked_pool(entity, entity_visible))
        entity = entity.reshape(batch, steps, -1)

        private = self.private_input(torch.cat((
            self.private_card(private_card_tokens),
            self.private_owner(private_card_owners.clamp(0, 1)),
            self.private_slot(private_card_slots.clamp(0, self.config.private_slot_count - 1)),
        ), dim=-1)).reshape(flat, private_card_tokens.shape[2], 128)
        private_visible = private_card_mask.reshape(flat, private_card_mask.shape[2]).bool()
        private_transform_mask = private_visible.clone()
        private_empty = ~private_transform_mask.any(dim=1)
        if bool(private_empty.any()):
            private = private.clone()
            private[private_empty, 0] = 0
            private_transform_mask[private_empty, 0] = True
        private = self.private_transformer(
            private, src_key_padding_mask=~private_transform_mask
        )
        private = self.private_output(self._masked_pool(private, private_visible))
        private = private.reshape(batch, steps, -1)

        scalar = self.scalar_branch(scalars)
        fused = F.silu(self.fusion(torch.cat(
            (actor_value, spatial, entity, private, scalar), dim=-1
        )))
        fused = self.fusion_blocks(fused)
        return CriticOutput(
            values=self.value_head(fused).squeeze(-1),
            wdl_logits=self.wdl_head(fused),
            crown_difference=self.crown_head(fused).squeeze(-1),
            tower_hp_difference=self.tower_hp_head(fused).squeeze(-1),
            future_damage=self.future_damage_head(fused),
        )


class ExpertActorCritic(nn.Module):
    def __init__(
        self, actor: RecurrentExpertPolicy, critic: PrivilegedCritic
    ) -> None:
        super().__init__()
        self.actor_adapter = ExpertActorAdapter(actor)
        self.critic = critic

    @property
    def actor(self) -> RecurrentExpertPolicy:
        return self.actor_adapter.actor

    def forward(
        self,
        *,
        actor_inputs: dict[str, Tensor | tuple[Tensor, Tensor] | None],
        critic_inputs: dict[str, Tensor],
    ) -> tuple[object, CriticOutput]:
        actor = self.actor_adapter.forward_with_features(**actor_inputs)
        critic = self.critic(
            actor_latent=actor.pre_head_latent.detach(), **critic_inputs
        )
        return actor, critic
