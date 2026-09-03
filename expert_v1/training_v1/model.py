"""Variable-deck recurrent expert policy with conditional action heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    entity_numeric_size: int = 3
    card_embedding_size: int = 64
    spatial_size: int = 64
    hidden_size: int = 256
    lambda_max: float = 20.0
    lambda_initial: float = 0.30
    native_tick_seconds: float = 0.05
    observation_mode: str = OBSERVATION_NATIVE
    position_head_fp32: bool = False
    position_logit_softcap: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position_head_fp32, bool):
            raise ValueError("position_head_fp32 must be a boolean")
        if self.position_logit_softcap is not None:
            if not self.position_head_fp32:
                raise ValueError("position softcap requires the FP32 position head")
            if not math.isfinite(self.position_logit_softcap) or self.position_logit_softcap <= 0:
                raise ValueError("position softcap must be finite and positive")

    def to_dict(self) -> dict[str, int | float | str]:
        value = asdict(self)
        # Default/legacy checkpoints must retain their exact historical contract.
        if not self.position_head_fp32:
            value.pop("position_head_fp32")
        if self.position_logit_softcap is None:
            value.pop("position_logit_softcap")
        else:
            value["position_logit_softcap"] = float(self.position_logit_softcap)
        return value


class ExpertPolicyOutput(NamedTuple):
    rate_logits: Tensor
    action_kind_logits: Tensor
    card_logits: Tensor
    position_logits: Tensor
    ability_logits: Tensor
    ability_position_logits: Tensor
    hidden: tuple[Tensor, Tensor]


class ExpertPolicyWithFeatures(NamedTuple):
    output: ExpertPolicyOutput
    pre_head_latent: Tensor


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def configure_position_precision(config: ExpertPolicyConfig) -> None:
    """Opt-in numeric policy for the owning training/inference process."""
    if config.position_head_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


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
            self.entity_relation_embedding = nn.Embedding(2, 8)
            self.entity_encoder = nn.Sequential(
                nn.Linear(embed + 8 + config.entity_numeric_size, spatial),
                nn.SiLU(),
                nn.Linear(spatial, spatial),
            )
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

    def _query_fp32(self, value: Tensor) -> Tensor:
        first, _activation, last = self.position_query
        value = F.linear(value.float(), first.weight.float(), first.bias.float())
        value = F.silu(value)
        return F.linear(value, last.weight.float(), last.bias.float())

    def _cells_fp32(self, value: Tensor) -> Tensor:
        assert self.cell_features is not None
        layer = self.cell_features
        bias = layer.bias.float() if layer.bias is not None else None
        return F.conv2d(value.float(), layer.weight.float(), bias,
                        layer.stride, layer.padding, layer.dilation, layer.groups)

    def _cap_position_scores(self, scores: Tensor) -> Tensor:
        cap = self.config.position_logit_softcap
        if cap is None:
            return scores
        scores = scores - scores.mean(dim=-1, keepdim=True)
        return cap * torch.tanh(scores / cap)

    def _stable_position_logits(
        self, recurrent: Tensor, hand: Tensor, flat_spatial: Tensor | None,
        legacy_cells: Tensor, indices: tuple[Tensor, Tensor, Tensor] | None,
    ) -> Tensor:
        batch, steps = recurrent.shape[:2]
        if indices is not None and indices[0].numel() == 0:
            # No supervised positions: keep the original zero-gradient graph,
            # including AdamW's zero-grad vs None-grad behavior, without a large
            # dense FP32 allocation. These unused scores are not inference output.
            query = self.position_query(torch.cat(
                (recurrent.unsqueeze(2).expand(-1, -1, 4, -1), hand), dim=-1))
            return torch.einsum("btse,btpe->btsp", query, legacy_cells) / math.sqrt(legacy_cells.shape[-1])
        with torch.autocast(device_type=recurrent.device.type, enabled=False):
            if indices is not None:
                b, t, card = indices
                query = self._query_fp32(torch.cat((recurrent[b, t].float(), hand[b, t, card].float()), dim=-1))
                if flat_spatial is not None:
                    features = self._cells_fp32(flat_spatial.index_select(0, b * steps + t))
                    cells = features.flatten(2).transpose(1, 2)
                else:
                    cells = legacy_cells[b, t].float()
                scores = torch.einsum("ne,npe->np", query, cells) / math.sqrt(cells.shape[-1])
                scores = self._cap_position_scores(scores)
                output = scores.new_zeros(batch, steps, 4, POSITION_COUNT)
                output[b, t, card] = scores
                return output
            query = self._query_fp32(torch.cat(
                (recurrent.float().unsqueeze(2).expand(-1, -1, 4, -1), hand.float()), dim=-1))
            if flat_spatial is not None:
                cells = self._cells_fp32(flat_spatial).flatten(2).transpose(1, 2)
                cells = cells.reshape(batch, steps, POSITION_COUNT, -1)
            else:
                cells = legacy_cells.float()
            scores = torch.einsum("btse,btpe->btsp", query, cells) / math.sqrt(cells.shape[-1])
            return self._cap_position_scores(scores)

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
        entity_tokens: Tensor | None = None,
        entity_positions: Tensor | None = None,
        entity_relations: Tensor | None = None,
        entity_numeric: Tensor | None = None,
        entity_mask: Tensor | None = None,
        previous_event_card_token: Tensor | None = None,
        previous_event_side: Tensor | None = None,
        previous_event_position: Tensor | None = None,
        hidden: tuple[Tensor, Tensor] | None = None,
        _position_indices: tuple[Tensor, Tensor, Tensor] | None = None,
        _feature_sink: list[Tensor] | None = None,
    ) -> ExpertPolicyOutput:
        batch, steps = public_scalars.shape[:2]
        flat_spatial = None
        if self.config.observation_mode == OBSERVATION_NATIVE:
            if grid is None or grid.ndim != 5:
                raise ValueError("grid must have shape [batch,time,channels,32,18]")
            if tuple(grid.shape[-2:]) != (ARENA_ROWS, ARENA_COLUMNS):
                raise ValueError("arena shape must be 32x18")
            assert self.spatial is not None
            flat_spatial = self.spatial(grid.flatten(0, 1))
            if any(
                value is None
                for value in (
                    entity_tokens,
                    entity_positions,
                    entity_relations,
                    entity_numeric,
                    entity_mask,
                )
            ):
                raise ValueError("native-state model requires public ragged entity tokens")
            assert entity_tokens is not None
            assert entity_positions is not None
            assert entity_relations is not None
            assert entity_numeric is not None
            assert entity_mask is not None
            if entity_tokens.shape[:2] != (batch, steps):
                raise ValueError("entity token prefix must match [batch,time]")
            entities = self.entity_encoder(
                torch.cat(
                    (
                        self.card_embedding(entity_tokens),
                        self.entity_relation_embedding(entity_relations),
                        entity_numeric,
                    ),
                    dim=-1,
                )
            )
            entities = entities * entity_mask.unsqueeze(-1)
            flat_entities = entities.flatten(0, 1)
            positions = entity_positions.flatten(0, 1).clamp(0, POSITION_COUNT - 1)
            public_entity_cells = flat_entities.new_zeros(
                batch * steps, POSITION_COUNT, flat_entities.shape[-1]
            )
            public_entity_cells.scatter_add_(
                1,
                positions.unsqueeze(-1).expand_as(flat_entities),
                flat_entities,
            )
            flat_spatial = flat_spatial + public_entity_cells.transpose(1, 2).reshape(
                batch * steps,
                flat_spatial.shape[1],
                ARENA_ROWS,
                ARENA_COLUMNS,
            )
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
        if _feature_sink is not None:
            _feature_sink.append(recurrent)

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
        if self.config.position_head_fp32:
            position_logits = self._stable_position_logits(recurrent, hand, flat_spatial, cells, _position_indices)
        else:
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

    def forward_with_features(self, **values: Tensor | tuple[Tensor, Tensor] | None) -> ExpertPolicyWithFeatures:
        """Return the existing actor output plus its pre-head recurrent latent.

        This is a non-parameterized adapter surface for Actor/Critic training.
        The default forward API and ``state_dict`` remain byte-compatible with
        existing inference checkpoints.
        """
        sink: list[Tensor] = []
        output = self.forward_sequence(**values, _feature_sink=sink)
        if len(sink) != 1:
            raise RuntimeError("expert recurrent feature capture failed")
        return ExpertPolicyWithFeatures(output=output, pre_head_latent=sink[0])

    def forward_batch(self, batch: dict[str, Tensor], *, supervised_positions: bool = False) -> ExpertPolicyOutput:
        """Actor output by default; supervised mode is only a loss-computation optimization.

        The default never consults targets and returns every hand-slot position
        distribution. Training/evaluation may request only the rows whose
        position loss is supervised; their values/derivatives match dense mode.
        """
        common = (
            "public_scalars",
            "own_deck_tokens",
            "hand_tokens",
            "next_card_token",
            "revealed_enemy_tokens",
            "delta_ticks",
        )
        values = {name: batch[name] for name in common}
        if self.config.position_head_fp32 and supervised_positions:
            b, t = (batch["loss_mask"] & batch["position_label_mask"]).nonzero(as_tuple=True)
            values["_position_indices"] = (b, t, batch["card_slot"][b, t].clamp(0, 3))
        if self.config.observation_mode == OBSERVATION_SEQUENCE:
            values.update(
                grid=None,
                ability_tokens=None,
                entity_tokens=None,
                entity_positions=None,
                entity_relations=None,
                entity_numeric=None,
                entity_mask=None,
                previous_event_card_token=batch["previous_event_card_token"],
                previous_event_side=batch["previous_event_side"],
                previous_event_position=batch["previous_event_position"],
            )
        else:
            values.update(
                grid=batch["grid"],
                ability_tokens=batch["ability_tokens"],
                entity_tokens=batch["entity_tokens"],
                entity_positions=batch["entity_positions"],
                entity_relations=batch["entity_relations"],
                entity_numeric=batch["entity_numeric"],
                entity_mask=batch["entity_mask"],
            )
        return self.forward_sequence(**values)
