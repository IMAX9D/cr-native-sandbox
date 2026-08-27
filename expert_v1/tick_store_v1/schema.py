"""Strict normalized schema for compact ``observe_train_v1`` states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


STORE_SCHEMA_VERSION = 2
NATIVE_TICK_HZ = 20
ENTITY_FIELDS = (
    "side",
    "x",
    "y",
    "card_id",
    "level",
    "hp",
    "max_hp",
    "behavior_state",
    "ability_slot",
    "ability_state_code",
    "ability_available",
    "ability_cooldown_remaining_ms",
    "ability_charges_remaining",
    "ability_pending_ms",
    "ability_mana_cost",
)
TOWER_FIELDS = ("side", "role", "lane", "x", "y", "hp", "max_hp")
PLAYER_FIELDS = (
    "elixir_raw", "hand0", "hand1", "hand2", "hand3",
    "next_deck_index", "refill_timer",
)
EPISODE_FIELDS = (
    "commands_allowed",
    "command_gate_code",
    "battle_phase",
    "logic_state",
    "logic_substate",
    "battle_flag",
    "terminated",
    "crowns0",
    "crowns1",
)


class TickStoreContractError(ValueError):
    """A native observation cannot be represented losslessly by this schema."""


@dataclass(frozen=True, slots=True)
class PlayerPrivate:
    side: int
    elixir_raw: int
    hand: tuple[int, int, int, int]
    next_deck_index: int
    refill_timer: int = 0

    def values(self) -> tuple[int, ...]:
        return (
            self.elixir_raw, *self.hand,
            self.next_deck_index, self.refill_timer,
        )


@dataclass(frozen=True, slots=True)
class TowerState:
    key: int
    side: int
    role: int  # 0 king, 1 princess
    lane: int  # -1 king/not-applicable, 0 left, 1 right
    x: int
    y: int
    hp: int
    max_hp: int

    def values(self) -> tuple[int, ...]:
        return (self.side, self.role, self.lane, self.x, self.y, self.hp, self.max_hp)


@dataclass(frozen=True, slots=True)
class EntityState:
    key: int  # native 5,000,000-series generation key
    side: int
    x: int
    y: int
    card_id: int
    level: int
    hp: int
    max_hp: int
    behavior_state: int
    ability_slot: int
    ability_state_code: int
    ability_available: int
    ability_cooldown_remaining_ms: int
    ability_charges_remaining: int
    ability_pending_ms: int
    ability_mana_cost: int

    def values(self) -> tuple[int, ...]:
        return (
            self.side,
            self.x,
            self.y,
            self.card_id,
            self.level,
            self.hp,
            self.max_hp,
            self.behavior_state,
            self.ability_slot,
            self.ability_state_code,
            self.ability_available,
            self.ability_cooldown_remaining_ms,
            self.ability_charges_remaining,
            self.ability_pending_ms,
            self.ability_mana_cost,
        )


@dataclass(frozen=True, slots=True)
class EpisodeState:
    commands_allowed: int
    command_gate_code: int
    battle_phase: int
    logic_state: int
    logic_substate: int
    battle_flag: int
    terminated: int
    crowns0: int
    crowns1: int

    def values(self) -> tuple[int, ...]:
        return (
            self.commands_allowed,
            self.command_gate_code,
            self.battle_phase,
            self.logic_state,
            self.logic_substate,
            self.battle_flag,
            self.terminated,
            self.crowns0,
            self.crowns1,
        )


@dataclass(frozen=True, slots=True)
class TickState:
    tick: int
    players: tuple[PlayerPrivate, PlayerPrivate]
    towers: tuple[TowerState, ...]
    entities: tuple[EntityState, ...]
    episode: EpisodeState


@dataclass(frozen=True, slots=True)
class ActorTick:
    """Public actor projection; it has no opponent-private player field."""

    tick: int
    actor_side: int
    own_player: PlayerPrivate
    towers: tuple[TowerState, ...]
    entities: tuple["ActorEntity", ...]
    episode: "ActorEpisode"


@dataclass(frozen=True, slots=True)
class ActorEntity:
    key: int
    relation: int  # 0 own, 1 enemy
    x: int
    y: int
    card_id: int
    level: int
    hp: int
    max_hp: int
    own_ability_slot: int
    own_ability_available: int


@dataclass(frozen=True, slots=True)
class ActorEpisode:
    terminated: int
    own_crowns: int
    enemy_crowns: int


def _integer(value: Any, name: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int):
        raise TickStoreContractError(f"{name} must be an integer")
    return int(value)


def _tower_key(side: int, role: int, lane: int) -> int:
    slot = 0 if role == 0 else (1 if lane == 0 else 2)
    return side * 3 + slot


def _normalize_players(state: Mapping[str, Any]) -> tuple[PlayerPrivate, PlayerPrivate]:
    result: list[PlayerPrivate] = []
    for raw in state.get("players", []):
        side = _integer(raw.get("side"), "player.side")
        if side not in (0, 1):
            raise TickStoreContractError("player side must be 0/1")
        hand = tuple(
            _integer(value, "hand_deck_index")
            for value in raw.get("hand_deck_indices", [])
        )
        if len(hand) != 4 or any(value < -1 or value > 7 for value in hand):
            raise TickStoreContractError("hand_deck_indices must contain four values in -1..7")
        visible = [value for value in hand if value >= 0]
        empty_slots = 4 - len(visible)
        next_deck_index = _integer(
            raw.get("next_deck_index"), "next_deck_index", default=-1
        )
        refill_timer = _integer(
            raw.get("refill_timer"), "refill_timer", default=0
        )
        if (
            len(set(visible)) != len(visible)
            or empty_slots not in (0, 1)
            or next_deck_index not in range(8)
            or next_deck_index in visible
            or not 0 <= refill_timer <= 10_000
            or (empty_slots == 1) != (refill_timer > 0)
        ):
            raise TickStoreContractError(
                "hand/next/refill native cycle state is inconsistent"
            )
        elixir_raw = raw.get("elixir_raw")
        if not isinstance(elixir_raw, int):
            elixir = _integer(raw.get("elixir"), "player.elixir")
            elixir_raw = elixir * 10_000
        if not 0 <= int(elixir_raw) <= 100_000:
            raise TickStoreContractError("elixir_raw outside 0..100000")
        result.append(
            PlayerPrivate(
                side=side,
                elixir_raw=int(elixir_raw),
                hand=hand,  # type: ignore[arg-type]
                next_deck_index=next_deck_index,
                refill_timer=refill_timer,
            )
        )
    result.sort(key=lambda item: item.side)
    if [item.side for item in result] != [0, 1]:
        raise TickStoreContractError("exactly two native players are required")
    return result[0], result[1]


def _normalize_towers(episode: Mapping[str, Any]) -> tuple[TowerState, ...]:
    result: list[TowerState] = []
    keys: set[int] = set()
    for raw in episode.get("crown_towers", []):
        side = _integer(raw.get("side"), "tower.side")
        role_name = str(raw.get("type", ""))
        role = 0 if role_name == "king" else 1 if role_name == "princess" else -1
        lane_name = raw.get("lane")
        lane = -1 if role == 0 else 0 if lane_name == "left" else 1 if lane_name == "right" else -2
        if side not in (0, 1) or role < 0 or lane < -1:
            raise TickStoreContractError("invalid crown tower identity")
        key = _tower_key(side, role, lane)
        if key in keys:
            raise TickStoreContractError("duplicate logical crown tower")
        keys.add(key)
        result.append(
            TowerState(
                key=key,
                side=side,
                role=role,
                lane=lane,
                x=_integer(raw.get("x"), "tower.x"),
                y=_integer(raw.get("y"), "tower.y"),
                hp=_integer(raw.get("hp"), "tower.hp"),
                max_hp=_integer(raw.get("max_hp"), "tower.max_hp"),
            )
        )
    return tuple(sorted(result, key=lambda item: item.key))


def _normalize_entities(state: Mapping[str, Any]) -> tuple[EntityState, ...]:
    result: list[EntityState] = []
    keys: set[int] = set()
    for raw in state.get("entities", []):
        key = _integer(raw.get("category"), "entity.category")
        if key in keys:
            raise TickStoreContractError("duplicate entity generation key")
        keys.add(key)
        side = _integer(raw.get("side"), "entity.side")
        x = _integer(raw.get("x"), "entity.x")
        y = _integer(raw.get("y"), "entity.y")
        if side not in (0, 1) or not 0 <= x <= 18_000 or not 0 <= y <= 32_000:
            raise TickStoreContractError("entity side/coordinates outside native arena")
        result.append(
            EntityState(
                key=key,
                side=side,
                x=x,
                y=y,
                card_id=_integer(raw.get("card_id"), "entity.card_id", default=-1),
                level=_integer(raw.get("level"), "entity.level", default=-1),
                hp=_integer(raw.get("hp"), "entity.hp", default=-1),
                max_hp=_integer(raw.get("max_hp"), "entity.max_hp", default=-1),
                behavior_state=_integer(
                    raw.get("behavior_state"), "entity.behavior_state", default=0
                ),
                ability_slot=_integer(raw.get("ability_slot"), "ability_slot", default=0),
                ability_state_code=_integer(
                    raw.get("ability_state_code"), "ability_state_code", default=-1
                ),
                ability_available=_integer(
                    raw.get("ability_available"), "ability_available", default=0
                ),
                ability_cooldown_remaining_ms=_integer(
                    raw.get("ability_cooldown_remaining_ms"),
                    "ability_cooldown_remaining_ms",
                    default=-1,
                ),
                ability_charges_remaining=_integer(
                    raw.get("ability_charges_remaining"),
                    "ability_charges_remaining",
                    default=-1,
                ),
                ability_pending_ms=_integer(
                    raw.get("ability_pending_ms"), "ability_pending_ms", default=-1
                ),
                ability_mana_cost=_integer(
                    raw.get("ability_mana_cost"), "ability_mana_cost", default=-1
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: item.key))


def _normalize_episode(raw: Mapping[str, Any]) -> EpisodeState:
    native_phase = raw.get("native_phase") or {}
    crowns = raw.get("crowns") or [0, 0]
    if len(crowns) != 2:
        raise TickStoreContractError("episode crowns must contain two values")
    return EpisodeState(
        commands_allowed=_integer(raw.get("commands_allowed"), "commands_allowed", default=1),
        command_gate_code=_integer(raw.get("command_gate_code"), "command_gate_code", default=0),
        battle_phase=_integer(native_phase.get("battle"), "native_phase.battle", default=-1),
        logic_state=_integer(native_phase.get("logic"), "native_phase.logic", default=-1),
        logic_substate=_integer(
            native_phase.get("logic_substate"), "native_phase.logic_substate", default=-1
        ),
        battle_flag=_integer(native_phase.get("flag_1e9"), "native_phase.flag", default=-1),
        terminated=_integer(raw.get("terminated"), "terminated", default=0),
        crowns0=_integer(crowns[0], "crowns0"),
        crowns1=_integer(crowns[1], "crowns1"),
    )


def normalize_native_state(state: Mapping[str, Any]) -> TickState:
    """Normalize one coherent compact/full native observation losslessly."""
    kind = state.get("kind")
    if kind not in ("libg_native_train_state_v1", "libg_native_state", None):
        raise TickStoreContractError(f"unsupported native observation kind: {kind}")
    if state.get("coherent") is False:
        raise TickStoreContractError("incoherent native observation")
    episode = state.get("episode")
    if not isinstance(episode, Mapping):
        raise TickStoreContractError("native observation lacks episode state")
    tick = _integer(state.get("tick"), "tick")
    if tick < 0:
        raise TickStoreContractError("tick cannot be negative")
    return TickState(
        tick=tick,
        players=_normalize_players(state),
        towers=_normalize_towers(episode),
        entities=_normalize_entities(state),
        episode=_normalize_episode(episode),
    )


def _rotate_entity(entity: EntityState, side: int) -> ActorEntity:
    relation = 0 if entity.side == side else 1
    x = entity.x if side == 0 else max(0, min(17_999, 17_999 - entity.x))
    y = entity.y if side == 0 else max(0, min(31_999, 31_999 - entity.y))
    return ActorEntity(
        entity.key,
        relation,
        x,
        y,
        entity.card_id,
        entity.level,
        entity.hp,
        entity.max_hp,
        entity.ability_slot if relation == 0 else 0,
        entity.ability_available if relation == 0 else 0,
    )


def _rotate_tower(tower: TowerState, side: int) -> TowerState:
    relation = 0 if tower.side == side else 1
    if side == 0:
        return TowerState(
            _tower_key(relation, tower.role, tower.lane), relation, tower.role, tower.lane,
            tower.x, tower.y, tower.hp, tower.max_hp,
        )
    lane = tower.lane if tower.lane < 0 else 1 - tower.lane
    return TowerState(
        _tower_key(relation, tower.role, lane),
        relation,
        tower.role,
        lane,
        max(0, min(17_999, 17_999 - tower.x)),
        max(0, min(31_999, 31_999 - tower.y)),
        tower.hp,
        tower.max_hp,
    )


def actor_projection(state: TickState, *, actor_side: int) -> ActorTick:
    """Return a canonical public view without opponent hand/elixir leakage."""
    if actor_side not in (0, 1):
        raise ValueError("actor_side must be 0 or 1")
    own = state.players[actor_side]
    projected_player = PlayerPrivate(
        side=0,
        elixir_raw=own.elixir_raw,
        hand=own.hand,
        next_deck_index=own.next_deck_index,
        refill_timer=own.refill_timer,
    )
    return ActorTick(
        tick=state.tick,
        actor_side=actor_side,
        own_player=projected_player,
        towers=tuple(sorted((_rotate_tower(item, actor_side) for item in state.towers), key=lambda item: item.key)),
        entities=tuple(sorted((_rotate_entity(item, actor_side) for item in state.entities), key=lambda item: item.key)),
        episode=ActorEpisode(
            terminated=state.episode.terminated,
            own_crowns=(state.episode.crowns0, state.episode.crowns1)[actor_side],
            enemy_crowns=(state.episode.crowns1, state.episode.crowns0)[actor_side],
        ),
    )


def require_consecutive(states: Iterable[TickState]) -> list[TickState]:
    result = list(states)
    if not result:
        raise TickStoreContractError("episode must contain at least one Tick")
    for previous, current in zip(result, result[1:]):
        if current.tick != previous.tick + 1:
            raise TickStoreContractError(
                f"Tick stream is not consecutive: {previous.tick}->{current.tick}"
            )
    return result
