"""Public native-state encoder for batched Expert Actor inference.

This module is deliberately independent of the GUI and of the historical BC
dataset.  It accepts the live ``observe_train_v1`` payload, projects it to one
side's public view, and emits exactly the tensor arguments consumed by
``RecurrentExpertPolicy.forward_sequence``.

The encoder is fail-closed.  In particular it never silently drops a native
card entity because doing so turns a live battle into the all-empty scene that
the first expert deployment accidentally used.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from expert_v1.tick_store_v1.schema import (
    ActorTick,
    TickState,
    actor_projection,
    normalize_native_state,
)
from expert_v1.training_v1.schema import (
    ARENA_COLUMNS,
    ARENA_ROWS,
    DECK_SIZE,
    HAND_SIZE,
    OBSERVATION_NATIVE,
    POSITION_COUNT,
)
from native_core.card_catalog import metadata


GRID_CHANNELS = 8
PUBLIC_SCALAR_SIZE = 16
ENTITY_NUMERIC_SIZE = 3
MODEL_INPUT_KEYS = (
    "grid",
    "public_scalars",
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    "ability_tokens",
    "delta_ticks",
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
)


class NativeObservationContractError(RuntimeError):
    """A live native observation cannot be encoded without losing meaning."""


@dataclass(frozen=True, slots=True)
class NativeActorFrame:
    """One independently recurrent Actor row.

    ``own_deck`` must be the eight-card deck belonging to ``actor_side``.  A
    card may be an integer base ID, a normalized ``card_id/form_flags`` row,
    or a replay ``d/el`` row.  Revealed enemy information is explicit because
    dead units are no longer present in a native state and therefore cannot be
    reconstructed safely from the current frame.
    """

    state: Mapping[str, Any] | TickState
    actor_side: int
    own_deck: Sequence[int | Mapping[str, Any]]
    revealed_enemy_card_ids: Sequence[int] = ()
    revealed_enemy_tokens: Sequence[int] = ()
    delta_ticks: float = 1.0


@dataclass(frozen=True, slots=True)
class _DeckEncoding:
    tokens: tuple[int, ...]
    allowed_ability_base_ids: frozenset[int]
    allowed_ability_native_ids: frozenset[int]


@dataclass(frozen=True)
class EncodedNativeBatch(Mapping[str, Tensor]):
    """A Mapping that can be expanded directly into ``forward_sequence``.

    Example: ``output = actor.forward_sequence(**encoded, hidden=hidden)``.
    Audit/action-routing metadata is kept as attributes and is intentionally
    excluded from the Mapping keys.
    """

    model_inputs: Mapping[str, Tensor]
    ticks: tuple[int, ...]
    actor_sides: tuple[int, ...]
    native_entity_counts: tuple[int, ...]
    encoded_entity_counts: tuple[int, ...]
    ability_entity_keys: tuple[tuple[int, ...], ...]
    ability_mask: Tensor

    def __post_init__(self) -> None:
        if tuple(self.model_inputs) != MODEL_INPUT_KEYS:
            raise ValueError("encoded Actor input keys changed")
        batch = len(self.ticks)
        if not (
            len(self.actor_sides)
            == len(self.native_entity_counts)
            == len(self.encoded_entity_counts)
            == len(self.ability_entity_keys)
            == batch
        ):
            raise ValueError("encoded batch metadata lengths differ")

    def __getitem__(self, key: str) -> Tensor:
        return self.model_inputs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(MODEL_INPUT_KEYS)

    def __len__(self) -> int:
        return len(MODEL_INPUT_KEYS)

    @property
    def batch_size(self) -> int:
        return len(self.ticks)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "EncodedNativeBatch":
        return EncodedNativeBatch(
            model_inputs={
                key: value.to(device=device, non_blocking=non_blocking)
                for key, value in self.model_inputs.items()
            },
            ticks=self.ticks,
            actor_sides=self.actor_sides,
            native_entity_counts=self.native_entity_counts,
            encoded_entity_counts=self.encoded_entity_counts,
            ability_entity_keys=self.ability_entity_keys,
            ability_mask=self.ability_mask.to(
                device=device, non_blocking=non_blocking
            ),
        )


def native_id_token_map(vocabulary: Sequence[str], *, name: str) -> dict[int, int]:
    """Parse the immutable ``name@native_id`` vocabulary representation."""

    if not vocabulary or str(vocabulary[0]) != "<PAD>":
        raise NativeObservationContractError(f"{name} vocabulary lacks PAD at token 0")
    result: dict[int, int] = {}
    for token, raw in enumerate(vocabulary[1:], 1):
        value = str(raw)
        if "@" not in value:
            raise NativeObservationContractError(
                f"{name} vocabulary token has no native ID: {value!r}"
            )
        try:
            native_id = int(value.rsplit("@", 1)[1])
        except ValueError as error:
            raise NativeObservationContractError(
                f"{name} vocabulary token has an invalid native ID: {value!r}"
            ) from error
        if native_id <= 0 or native_id in result:
            raise NativeObservationContractError(
                f"{name} vocabulary contains an invalid/duplicate native ID: {native_id}"
            )
        result[native_id] = token
    return result


def _cell(x: int, y: int) -> int:
    if not 0 <= int(x) < 18_000 or not 0 <= int(y) < 32_000:
        raise NativeObservationContractError(
            f"native coordinate outside the 18x32 arena: {(x, y)}"
        )
    return min(31, int(y) // 1000) * ARENA_COLUMNS + min(17, int(x) // 1000)


def _tower_map(actor: ActorTick) -> dict[tuple[int, int, int], Any]:
    return {(tower.side, tower.role, tower.lane): tower for tower in actor.towers}


def _public_scalars(actor: ActorTick, state: TickState) -> np.ndarray:
    """Exact feature order/scaling used to train expert-v1.1/v1.2."""

    towers = _tower_map(actor)

    def hp(relation: int, role: int, lane: int) -> float:
        tower = towers.get((relation, role, lane))
        if tower is None or tower.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, tower.hp / tower.max_hp))

    own = [entity for entity in actor.entities if entity.relation == 0]
    enemy = [entity for entity in actor.entities if entity.relation == 1]
    return np.asarray(
        [
            min(1.0, actor.tick / 6000.0),
            actor.own_player.elixir_raw / 100_000.0,
            float(bool(state.episode.commands_allowed) and not actor.episode.terminated),
            float(actor.episode.terminated),
            actor.episode.own_crowns / 3.0,
            actor.episode.enemy_crowns / 3.0,
            hp(0, 0, -1),
            hp(0, 1, 0),
            hp(0, 1, 1),
            hp(1, 0, -1),
            hp(1, 1, 0),
            hp(1, 1, 1),
            math.log1p(len(own)) / math.log(257.0),
            math.log1p(len(enemy)) / math.log(257.0),
            math.log1p(sum(max(0, value.hp) for value in own))
            / math.log(1_000_001.0),
            math.log1p(sum(max(0, value.hp) for value in enemy))
            / math.log(1_000_001.0),
        ],
        dtype=np.float32,
    )


def _grid(actor: ActorTick) -> np.ndarray:
    """Exact quantized public 8x32x18 grid used by the Expert Actor."""

    result = np.zeros((GRID_CHANNELS, ARENA_ROWS, ARENA_COLUMNS), dtype=np.float32)
    for tower in actor.towers:
        row, column = divmod(_cell(tower.x, tower.y), ARENA_COLUMNS)
        offset = 0 if tower.side == 0 else 2
        result[offset, row, column] = 1.0
        result[offset + 1, row, column] = (
            max(0.0, min(1.0, tower.hp / tower.max_hp))
            if tower.max_hp > 0
            else 0.0
        )
    for entity in actor.entities:
        row, column = divmod(_cell(entity.x, entity.y), ARENA_COLUMNS)
        offset = 4 if entity.relation == 0 else 6
        result[offset, row, column] += 1.0 / 16.0
        if entity.max_hp > 0:
            result[offset + 1, row, column] += max(
                0.0, min(1.0, entity.hp / entity.max_hp)
            ) / 16.0
    return np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _deck_row(value: int | Mapping[str, Any]) -> tuple[int, int]:
    if isinstance(value, Mapping):
        raw_id = value.get("card_id", value.get("d"))
        if raw_id is None:
            raise NativeObservationContractError("deck card lacks card_id/d")
        card_id = int(raw_id)
        form_flags = int(value.get("form_flags", value.get("el", 0)))
    else:
        card_id = int(value)
        form_flags = 0
    if card_id <= 0 or form_flags not in (0, 1, 2, 3):
        raise NativeObservationContractError("invalid deck card identity/form flags")
    if form_flags == 3:
        raise NativeObservationContractError(
            "one Expert deck token cannot represent combined evolution+hero form"
        )
    return card_id, form_flags


class NativeObservationEncoder:
    """Encode live native states for one or many independent Actor streams."""

    def __init__(
        self,
        *,
        card_id_to_token: Mapping[int, int],
        ability_id_to_token: Mapping[int, int],
        max_ability_slots: int,
        card_vocab_size: int | None = None,
        ability_vocab_size: int | None = None,
        require_nonempty_public_scene: bool = True,
    ) -> None:
        self.card_id_to_token = {
            int(key): int(value) for key, value in card_id_to_token.items()
        }
        self.ability_id_to_token = {
            int(key): int(value) for key, value in ability_id_to_token.items()
        }
        if not self.card_id_to_token or any(
            key <= 0 or value <= 0 for key, value in self.card_id_to_token.items()
        ):
            raise NativeObservationContractError("card vocabulary mapping is invalid")
        if any(
            key <= 0 or value <= 0
            for key, value in self.ability_id_to_token.items()
        ):
            raise NativeObservationContractError("ability vocabulary mapping is invalid")
        if len(set(self.card_id_to_token.values())) != len(self.card_id_to_token):
            raise NativeObservationContractError("card vocabulary token IDs are duplicated")
        if len(set(self.ability_id_to_token.values())) != len(
            self.ability_id_to_token
        ):
            raise NativeObservationContractError("ability vocabulary token IDs are duplicated")
        self.max_ability_slots = int(max_ability_slots)
        if self.max_ability_slots < 1:
            raise NativeObservationContractError("max_ability_slots must be positive")
        inferred_cards = max(self.card_id_to_token.values(), default=0) + 1
        inferred_abilities = max(self.ability_id_to_token.values(), default=0) + 1
        self.card_vocab_size = int(card_vocab_size or inferred_cards)
        self.ability_vocab_size = int(ability_vocab_size or max(1, inferred_abilities))
        if self.card_vocab_size < inferred_cards or self.ability_vocab_size < inferred_abilities:
            raise NativeObservationContractError("vocabulary size excludes a mapped token")
        self.require_nonempty_public_scene = bool(require_nonempty_public_scene)
        self._deck_cache: dict[tuple[tuple[int, int], ...], _DeckEncoding] = {}

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        require_nonempty_public_scene: bool = True,
    ) -> "NativeObservationEncoder":
        dimensions = manifest.get("dimensions")
        if not isinstance(dimensions, Mapping):
            raise NativeObservationContractError("expert manifest lacks dimensions")
        card_vocabulary = manifest.get("card_vocabulary")
        ability_vocabulary = manifest.get("ability_vocabulary")
        if not isinstance(card_vocabulary, Sequence) or isinstance(
            card_vocabulary, (str, bytes)
        ):
            raise NativeObservationContractError("expert manifest lacks card vocabulary")
        if not isinstance(ability_vocabulary, Sequence) or isinstance(
            ability_vocabulary, (str, bytes)
        ):
            raise NativeObservationContractError("expert manifest lacks ability vocabulary")
        for name, expected in (
            ("grid_channels", GRID_CHANNELS),
            ("public_scalar_size", PUBLIC_SCALAR_SIZE),
            ("entity_numeric_size", ENTITY_NUMERIC_SIZE),
        ):
            if int(dimensions.get(name, -1)) != expected:
                raise NativeObservationContractError(
                    f"expert manifest {name} is incompatible with native encoder"
                )
        if int(dimensions.get("card_vocab_size", -1)) != len(card_vocabulary):
            raise NativeObservationContractError("manifest card vocabulary size mismatch")
        if int(dimensions.get("ability_vocab_size", -1)) != len(ability_vocabulary):
            raise NativeObservationContractError("manifest ability vocabulary size mismatch")
        return cls(
            card_id_to_token=native_id_token_map(
                [str(value) for value in card_vocabulary], name="card"
            ),
            ability_id_to_token=native_id_token_map(
                [str(value) for value in ability_vocabulary], name="ability"
            ),
            max_ability_slots=int(dimensions["max_ability_slots"]),
            card_vocab_size=len(card_vocabulary),
            ability_vocab_size=len(ability_vocabulary),
            require_nonempty_public_scene=require_nonempty_public_scene,
        )

    def assert_compatible(self, config: Any) -> None:
        expected = {
            "observation_mode": OBSERVATION_NATIVE,
            "grid_channels": GRID_CHANNELS,
            "public_scalar_size": PUBLIC_SCALAR_SIZE,
            "entity_numeric_size": ENTITY_NUMERIC_SIZE,
            "card_vocab_size": self.card_vocab_size,
            "ability_vocab_size": self.ability_vocab_size,
            "max_ability_slots": self.max_ability_slots,
        }
        differences = {
            name: (getattr(config, name, None), value)
            for name, value in expected.items()
            if getattr(config, name, None) != value
        }
        if differences:
            raise NativeObservationContractError(
                f"Expert Actor/encoder dimensions differ: {differences}"
            )

    def schema(self) -> dict[str, Any]:
        return {
            "kind": "cr_native_expert_actor_encoder_v1",
            "observation_mode": OBSERVATION_NATIVE,
            "grid_channels": GRID_CHANNELS,
            "public_scalar_size": PUBLIC_SCALAR_SIZE,
            "entity_numeric_size": ENTITY_NUMERIC_SIZE,
            "card_vocab_size": self.card_vocab_size,
            "ability_vocab_size": self.ability_vocab_size,
            "max_ability_slots": self.max_ability_slots,
            "card_id_to_token": sorted(self.card_id_to_token.items()),
            "ability_id_to_token": sorted(self.ability_id_to_token.items()),
            "side_projection": "actor_canonical_rotate_180_v1",
            "public_information_only": True,
            "unknown_entity_policy": "fail_closed",
        }

    def schema_sha256(self) -> str:
        encoded = json.dumps(
            self.schema(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _deck(self, raw: Sequence[int | Mapping[str, Any]]) -> _DeckEncoding:
        rows = tuple(_deck_row(value) for value in raw)
        if len(rows) != DECK_SIZE:
            raise NativeObservationContractError("own deck must contain exactly eight cards")
        if len({card_id for card_id, _flags in rows}) != DECK_SIZE:
            raise NativeObservationContractError("own deck contains duplicate base cards")
        cached = self._deck_cache.get(rows)
        if cached is not None:
            return cached
        tokens: list[int] = []
        allowed_base: set[int] = set()
        allowed_native: set[int] = set()
        for card_id, flags in rows:
            try:
                row = metadata(card_id)
            except KeyError as error:
                raise NativeObservationContractError(
                    f"deck card is absent from the native catalog: {card_id}"
                ) from error
            token_id = card_id
            if flags:
                field = "evolution_form_id" if flags == 1 else "hero_form_id"
                token_id = int(row.get(field) or 0)
                if token_id <= 0:
                    raise NativeObservationContractError(
                        f"deck card {card_id} lacks requested native form {flags}"
                    )
            token = self.card_id_to_token.get(token_id)
            if token is None:
                raise NativeObservationContractError(
                    f"deck native ID is outside frozen card vocabulary: {token_id}"
                )
            tokens.append(token)

            hero = bool(flags & 2)
            ability_name = row.get(
                "hero_active_ability" if hero else "active_ability"
            )
            if ability_name:
                ability_native_id = int(row.get("hero_form_id") or 0) if hero else card_id
                allowed_base.add(card_id)
                allowed_native.add(ability_native_id)
        result = _DeckEncoding(
            tuple(tokens), frozenset(allowed_base), frozenset(allowed_native)
        )
        self._deck_cache[rows] = result
        return result

    def _revealed(self, frame: NativeActorFrame) -> tuple[int, ...]:
        if frame.revealed_enemy_card_ids and frame.revealed_enemy_tokens:
            raise NativeObservationContractError(
                "supply revealed enemy native IDs or tokens, not both"
            )
        if frame.revealed_enemy_tokens:
            tokens = [int(value) for value in frame.revealed_enemy_tokens]
            if any(value <= 0 or value >= self.card_vocab_size for value in tokens):
                raise NativeObservationContractError("revealed enemy token is out of range")
        else:
            tokens = []
            for native_id in frame.revealed_enemy_card_ids:
                token = self.card_id_to_token.get(int(native_id))
                if token is None:
                    raise NativeObservationContractError(
                        f"revealed enemy native ID is outside vocabulary: {native_id}"
                    )
                tokens.append(token)
        unique = list(dict.fromkeys(tokens))
        if len(unique) > DECK_SIZE:
            raise NativeObservationContractError("more than eight enemy cards were revealed")
        return tuple(unique)

    @staticmethod
    def _normalize_state(value: Mapping[str, Any] | TickState) -> TickState:
        if isinstance(value, TickState):
            return value
        return normalize_native_state(value)

    def encode_one(
        self,
        state: Mapping[str, Any] | TickState,
        *,
        actor_side: int,
        own_deck: Sequence[int | Mapping[str, Any]],
        revealed_enemy_card_ids: Sequence[int] = (),
        revealed_enemy_tokens: Sequence[int] = (),
        delta_ticks: float = 1.0,
        device: torch.device | str | None = None,
    ) -> EncodedNativeBatch:
        return self.encode_batch(
            [
                NativeActorFrame(
                    state=state,
                    actor_side=actor_side,
                    own_deck=own_deck,
                    revealed_enemy_card_ids=revealed_enemy_card_ids,
                    revealed_enemy_tokens=revealed_enemy_tokens,
                    delta_ticks=delta_ticks,
                )
            ],
            device=device,
        )

    def encode_batch(
        self,
        frames: Sequence[NativeActorFrame],
        *,
        device: torch.device | str | None = None,
    ) -> EncodedNativeBatch:
        if not frames:
            raise NativeObservationContractError("cannot encode an empty Actor batch")

        normalized: list[TickState] = []
        actors: list[ActorTick] = []
        decks: list[_DeckEncoding] = []
        reveals: list[tuple[int, ...]] = []
        entity_rows: list[list[tuple[int, int, int, tuple[float, float, float]]]] = []
        ability_rows: list[list[tuple[int, int, bool]]] = []
        native_entity_counts: list[int] = []

        for batch_index, frame in enumerate(frames):
            if frame.actor_side not in (0, 1):
                raise NativeObservationContractError(
                    f"frame {batch_index} actor_side must be 0/1"
                )
            delta = float(frame.delta_ticks)
            if not math.isfinite(delta) or delta < 0:
                raise NativeObservationContractError(
                    f"frame {batch_index} delta_ticks must be finite and nonnegative"
                )
            state = self._normalize_state(frame.state)
            actor = actor_projection(state, actor_side=int(frame.actor_side))
            if self.require_nonempty_public_scene and not actor.towers and not actor.entities:
                raise NativeObservationContractError(
                    f"frame {batch_index}/tick {state.tick} has an empty public native scene"
                )
            if isinstance(frame.state, Mapping):
                reported = frame.state.get("entity_count")
                if reported is not None and int(reported) > 0 and not state.entities:
                    raise NativeObservationContractError(
                        f"frame {batch_index}/tick {state.tick} reports dynamic entities "
                        "but exposes none"
                    )
            deck = self._deck(frame.own_deck)

            encoded_entities: list[
                tuple[int, int, int, tuple[float, float, float]]
            ] = []
            eligible = 0
            for entity in actor.entities:
                if int(entity.card_id) < 0:
                    # Non-card effects remain present in the quantized public grid.
                    continue
                eligible += 1
                token = self.card_id_to_token.get(int(entity.card_id))
                if token is None:
                    raise NativeObservationContractError(
                        f"frame {batch_index}/tick {state.tick} native entity "
                        f"{entity.card_id} is outside frozen card vocabulary"
                    )
                encoded_entities.append(
                    (
                        token,
                        _cell(entity.x, entity.y),
                        int(entity.relation),
                        (
                            max(0.0, min(1.0, entity.level / 16.0)),
                            max(0.0, min(1.0, entity.hp / entity.max_hp))
                            if entity.max_hp > 0
                            else 0.0,
                            math.log1p(max(0, entity.max_hp))
                            / math.log(1_000_001.0),
                        ),
                    )
                )
            if eligible > 0 and not encoded_entities:
                raise NativeObservationContractError(
                    f"frame {batch_index}/tick {state.tick} native scene is nonempty "
                    "but categorical entity input is empty"
                )

            allowed_base = deck.allowed_ability_base_ids
            allowed_native = deck.allowed_ability_native_ids
            abilities: list[tuple[int, int, bool]] = []
            command_allowed = bool(state.episode.commands_allowed) and not bool(
                state.episode.terminated
            )
            for entity in sorted(
                (value for value in state.entities if value.side == frame.actor_side),
                key=lambda value: value.key,
            ):
                if entity.ability_slot <= 0:
                    continue
                native_id = int(entity.card_id)
                if native_id not in allowed_native and native_id not in allowed_base:
                    continue
                token = self.ability_id_to_token.get(native_id)
                if token is None:
                    raise NativeObservationContractError(
                        f"frame {batch_index}/tick {state.tick} ability native ID "
                        f"{native_id} is outside frozen ability vocabulary"
                    )
                abilities.append(
                    (int(entity.key), token, command_allowed and bool(entity.ability_available))
                )
            if len(abilities) > self.max_ability_slots:
                raise NativeObservationContractError(
                    f"frame {batch_index}/tick {state.tick} ability capacity exceeded: "
                    f"{len(abilities)} > {self.max_ability_slots}"
                )

            normalized.append(state)
            actors.append(actor)
            decks.append(deck)
            reveals.append(self._revealed(frame))
            entity_rows.append(encoded_entities)
            ability_rows.append(abilities)
            native_entity_counts.append(eligible)

        batch_size = len(frames)
        max_entities = max(1, *(len(row) for row in entity_rows))
        grid = np.empty(
            (batch_size, 1, GRID_CHANNELS, ARENA_ROWS, ARENA_COLUMNS),
            dtype=np.float32,
        )
        public = np.empty((batch_size, 1, PUBLIC_SCALAR_SIZE), dtype=np.float32)
        own_deck = np.empty((batch_size, 1, DECK_SIZE), dtype=np.int64)
        hand = np.zeros((batch_size, 1, HAND_SIZE), dtype=np.int64)
        next_card = np.empty((batch_size, 1), dtype=np.int64)
        revealed = np.zeros((batch_size, 1, DECK_SIZE), dtype=np.int64)
        ability = np.zeros(
            (batch_size, 1, self.max_ability_slots), dtype=np.int64
        )
        ability_mask = np.zeros_like(ability, dtype=np.bool_)
        deltas = np.empty((batch_size, 1), dtype=np.float32)
        entity_tokens = np.zeros((batch_size, 1, max_entities), dtype=np.int64)
        entity_positions = np.zeros_like(entity_tokens)
        entity_relations = np.zeros_like(entity_tokens)
        entity_numeric = np.zeros(
            (batch_size, 1, max_entities, ENTITY_NUMERIC_SIZE), dtype=np.float32
        )
        entity_mask = np.zeros_like(entity_tokens, dtype=np.bool_)
        ability_keys: list[tuple[int, ...]] = []

        for index, (frame, state, actor, deck) in enumerate(
            zip(frames, normalized, actors, decks, strict=True)
        ):
            grid[index, 0] = _grid(actor).astype(np.float32) / 255.0
            public[index, 0] = _public_scalars(actor, state)
            own_deck[index, 0] = deck.tokens
            for slot, deck_index in enumerate(actor.own_player.hand):
                hand[index, 0, slot] = 0 if deck_index < 0 else deck.tokens[deck_index]
            next_card[index, 0] = deck.tokens[actor.own_player.next_deck_index]
            revealed[index, 0, : len(reveals[index])] = reveals[index]
            for slot, (key, token, available) in enumerate(ability_rows[index]):
                ability[index, 0, slot] = token
                ability_mask[index, 0, slot] = available
            ability_keys.append(tuple(value[0] for value in ability_rows[index]))
            deltas[index, 0] = float(frame.delta_ticks)
            for slot, (token, position, relation, numeric) in enumerate(
                entity_rows[index]
            ):
                entity_tokens[index, 0, slot] = token
                entity_positions[index, 0, slot] = position
                entity_relations[index, 0, slot] = relation
                entity_numeric[index, 0, slot] = numeric
                entity_mask[index, 0, slot] = True

        inputs: dict[str, Tensor] = {
            "grid": torch.from_numpy(grid),
            "public_scalars": torch.from_numpy(public),
            "own_deck_tokens": torch.from_numpy(own_deck),
            "hand_tokens": torch.from_numpy(hand),
            "next_card_token": torch.from_numpy(next_card),
            "revealed_enemy_tokens": torch.from_numpy(revealed),
            "ability_tokens": torch.from_numpy(ability),
            "delta_ticks": torch.from_numpy(deltas),
            "entity_tokens": torch.from_numpy(entity_tokens),
            "entity_positions": torch.from_numpy(entity_positions),
            "entity_relations": torch.from_numpy(entity_relations),
            "entity_numeric": torch.from_numpy(entity_numeric),
            "entity_mask": torch.from_numpy(entity_mask),
        }
        result = EncodedNativeBatch(
            model_inputs=inputs,
            ticks=tuple(state.tick for state in normalized),
            actor_sides=tuple(int(frame.actor_side) for frame in frames),
            native_entity_counts=tuple(native_entity_counts),
            encoded_entity_counts=tuple(len(row) for row in entity_rows),
            ability_entity_keys=tuple(ability_keys),
            ability_mask=torch.from_numpy(ability_mask),
        )
        return result if device is None else result.to(device)


class RevealedEnemyTracker:
    """Per-episode public reveal history with no opponent-deck inspection."""

    def __init__(self, encoder: NativeObservationEncoder) -> None:
        self.encoder = encoder
        self._tokens: list[list[int]] = [[], []]

    def reset(self) -> None:
        self._tokens = [[], []]

    def record_play(
        self,
        *,
        played_side: int,
        card: int | Mapping[str, Any],
    ) -> None:
        if played_side not in (0, 1):
            raise NativeObservationContractError("played_side must be 0/1")
        card_id, flags = _deck_row(card)
        row = metadata(card_id)
        native_id = card_id
        if flags:
            field = "evolution_form_id" if flags == 1 else "hero_form_id"
            native_id = int(row.get(field) or 0)
        token = self.encoder.card_id_to_token.get(native_id)
        if token is None:
            raise NativeObservationContractError(
                f"played card is outside frozen vocabulary: {native_id}"
            )
        observer = 1 - played_side
        if token not in self._tokens[observer]:
            if len(self._tokens[observer]) >= DECK_SIZE:
                raise NativeObservationContractError("opponent reveal history exceeds eight cards")
            self._tokens[observer].append(token)

    def tokens_for(self, actor_side: int) -> tuple[int, ...]:
        if actor_side not in (0, 1):
            raise NativeObservationContractError("actor_side must be 0/1")
        return tuple(self._tokens[actor_side])
