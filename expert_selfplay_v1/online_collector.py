"""End-to-end native online collector for Expert Self-Play v1.

The collector owns no model and no native process.  It is the small synchronous
coordination layer between a set of :class:`NativeRoyaleEnv` workers and one
content-addressed :class:`BatchedPolicyService`:

* every ready side is encoded before the single policy-service call;
* the two sides of an episode may use the same or different Actor hashes;
* libg RPCs belonging to different workers run concurrently;
* only the learner side is committed to the immutable PPO episode; and
* a rollout is returned only after a normal, complete native terminal.

The default advances one native Tick per policy decision; an explicitly
configured 1..16-Tick macro step preserves its actual advance in GAE.  Process
level parallelism belongs in the runner, keeping transition contracts testable.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from native_core.env import NativeRoyaleEnv
from training.schema import ActionMaskCache, DefensiveTowerReward, build_action_masks

from .actions import ExpertActionMasks
from .batched_policy import BatchedPolicyService, PolicyRequest, SampledPolicyAction
from .decks import DeckFixture
from .native_observation import (
    EncodedNativeBatch,
    NativeActorFrame,
    NativeObservationEncoder,
    RevealedEnemyTracker,
)
from .rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from .rollout_storage import CriticPrivateObservation


ARENA_ROWS = 32
ARENA_COLUMNS = 18
POSITION_COUNT = ARENA_ROWS * ARENA_COLUMNS
STEP_PAYLOAD_KIND = "cr_native_expert_selfplay_step_payload_v1"


class OnlineCollectorContractError(RuntimeError):
    """A native episode cannot be represented by the v1 rollout contract."""


@dataclass(frozen=True)
class OnlineEpisodeSpec:
    """One native worker and its immutable episode assignment."""

    worker_id: Hashable
    env: NativeRoyaleEnv
    fixture: DeckFixture
    header: EpisodeHeader
    actor_hashes: Mapping[int, str]
    warmup_steps: int = 100


@dataclass(frozen=True)
class CollectedEpisode:
    """A complete learner episode plus shard-ready per-decision tensors."""

    episode: LearnerEpisodeBuffer
    step_payloads: tuple[dict[str, Any], ...]
    terminal_episode: Mapping[str, Any]
    attempted_actions: int
    accepted_actions: int
    native_ticks_advanced: int

    @property
    def episode_id(self) -> str:
        return self.episode.header.episode_id

    def frozen_episode(self) -> dict[str, Any]:
        return self.episode.freeze()


@dataclass
class _Cursor:
    spec: OnlineEpisodeSpec
    state: dict[str, Any]
    decks: list[list[dict[str, Any]]]
    tracker: RevealedEnemyTracker
    episode: LearnerEpisodeBuffer
    native_masks: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    mask_cache: ActionMaskCache = field(default_factory=ActionMaskCache)
    step_payloads: list[dict[str, Any]] = field(default_factory=list)
    target_states: list[dict[str, Any]] = field(default_factory=list)
    attempted_actions: int = 0
    accepted_actions: int = 0
    native_ticks_advanced: int = 0
    terminal_episode: dict[str, Any] | None = None
    final_rpc_profile: dict[str, float] | None = None


ValueFunction = Callable[
    [CriticPrivateObservation, Mapping[str, Tensor]], float
]


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _clone_tensor_mapping(value: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        str(name): tensor.detach().cpu().contiguous().clone()
        for name, tensor in value.items()
    }


def _mask_mapping(value: ExpertActionMasks) -> dict[str, Tensor]:
    return {
        name: getattr(value, name).detach().cpu().contiguous().clone()
        for name in value.__dataclass_fields__
    }


def _critic_mapping(value: CriticPrivateObservation) -> dict[str, Tensor]:
    return {
        name: torch.as_tensor(getattr(value, name)).detach().cpu().contiguous().clone()
        for name in value.__dataclass_fields__
    }


def _critic_inputs(value: CriticPrivateObservation) -> dict[str, Tensor]:
    """Return the singleton-time convention consumed by critic_training."""

    return {
        name: torch.as_tensor(getattr(value, name))
        .detach().cpu().contiguous().clone().unsqueeze(0)
        for name in value.__dataclass_fields__
    }


def _tower_target_state(state: Mapping[str, Any]) -> dict[str, Any]:
    towers = state.get("episode", {}).get("crown_towers", [])
    if not isinstance(towers, list):
        towers = []
    return {
        "tick": int(state.get("tick", 0)),
        "episode": {"crown_towers": deepcopy(towers)},
    }


def canonical_position_mask(mask: np.ndarray, side: int) -> np.ndarray:
    """Rotate an absolute libg placement mask into the Actor's side view."""

    value = np.asarray(mask, dtype=np.bool_)
    if value.shape != (4, POSITION_COUNT):
        raise OnlineCollectorContractError("card placement mask is not 4x576")
    grid = value.reshape(4, ARENA_ROWS, ARENA_COLUMNS)
    if side == 1:
        grid = grid[:, ::-1, ::-1]
    return np.ascontiguousarray(grid.reshape(4, POSITION_COUNT))


def canonical_position_to_native(position: int, side: int) -> tuple[int, int]:
    """Return the centre of a canonical Actor cell in absolute libg units."""

    if side not in (0, 1) or not 0 <= int(position) < POSITION_COUNT:
        raise OnlineCollectorContractError("invalid canonical placement cell")
    row, column = divmod(int(position), ARENA_COLUMNS)
    if side == 1:
        row, column = ARENA_ROWS - 1 - row, ARENA_COLUMNS - 1 - column
    return column * 1000 + 500, row * 1000 + 500


def _fixture_decks(fixture: DeckFixture) -> list[list[dict[str, Any]]]:
    try:
        battle = fixture.replay["battle"]
        result = []
        for side in (0, 1):
            cards = battle[f"deck{side}"]["sp"]
            if len(cards) != 8:
                raise ValueError
            result.append([
                {
                    "card_id": int(card["d"]),
                    "level": int(card.get("l", 0)) + 1,
                    "form_flags": int(card.get("el", 0)),
                }
                for card in cards
            ])
    except (KeyError, TypeError, ValueError) as error:
        raise OnlineCollectorContractError("fixture does not contain two eight-card decks") from error
    return result


def _mask_compatible_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Bridge compact ``elixir_raw`` to the legacy mask helper without mutation."""

    result = dict(state)
    players: list[dict[str, Any]] = []
    for raw in state.get("players", []):
        player = dict(raw)
        if "elixir" not in player:
            if "elixir_raw" not in player:
                raise OnlineCollectorContractError("native player has no elixir value")
            player["elixir"] = max(0, int(player["elixir_raw"]) // 10_000)
        players.append(player)
    result["players"] = players
    return result


def _terminal_rewards(episode: Mapping[str, Any]) -> dict[int, float]:
    value = episode.get("rewards_by_side")
    if isinstance(value, Mapping):
        try:
            return {side: float(value.get(side, value.get(str(side), 0.0))) for side in (0, 1)}
        except (TypeError, ValueError) as error:
            raise OnlineCollectorContractError("terminal rewards are invalid") from error
    raw = episode.get("rewards")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        return {0: float(raw[0]), 1: float(raw[1])}
    raise OnlineCollectorContractError("terminal episode has no two-sided reward vector")


def _reward_components(
    reward: DefensiveTowerReward,
    previous: Mapping[str, Any],
    current: Mapping[str, Any] | None,
    *,
    side: int,
    terminal_rewards: Mapping[int, float],
    done: bool,
) -> dict[str, float]:
    previous_hp, previous_alive = reward._tower_summary(previous)
    if current is None:
        current_hp, current_alive = previous_hp, previous_alive
    else:
        current_hp, current_alive = reward._tower_summary(current)
    enemy = 1 - side
    damage_received = [
        max(0, previous_hp[index] - current_hp[index]) for index in (0, 1)
    ]
    towers_lost = [
        max(0, previous_alive[index] - current_alive[index]) for index in (0, 1)
    ]
    terminal_raw = float(terminal_rewards.get(side, 0.0))
    components = {
        "damage_dealt": float(reward.damage_dealt_scale * damage_received[enemy]),
        "damage_received": float(-reward.damage_received_scale * damage_received[side]),
        "towers_dealt": float(reward.tower_destroyed_reward * towers_lost[enemy]),
        "towers_received": float(-reward.tower_destroyed_reward * towers_lost[side]),
        "terminal": float(
            reward.terminal_win_reward
            * (1.0 if terminal_raw > 0 else -1.0 if terminal_raw < 0 else 0.0)
            if done
            else 0.0
        ),
    }
    components["total"] = float(sum(components.values()))
    expected = reward.transition(
        previous,
        current,
        terminal_rewards=terminal_rewards,
        done=done,
    )[side]
    if not math.isfinite(components["total"]) or abs(components["total"] - expected) > 1e-6:
        raise OnlineCollectorContractError("reward audit differs from DefensiveTowerReward")
    return components


def _terminal_reward_state(episode: Mapping[str, Any]) -> dict[str, Any] | None:
    towers = episode.get("crown_towers")
    if isinstance(towers, list):
        return {"episode": {"crown_towers": deepcopy(towers)}}
    return None


class OnlineSelfPlayCollector:
    """Collect complete learner-only episodes from one or more native workers."""

    def __init__(
        self,
        encoder: NativeObservationEncoder,
        policy_service: BatchedPolicyService,
        *,
        reward: DefensiveTowerReward | None = None,
        max_decisions: int = 10_000,
        rpc_workers: int | None = None,
        step_ticks: int = 1,
        idle_step_ticks: int | None = None,
        critic_scalar_size: int = 32,
        critic_private_slot_count: int = 32,
        lean_step_payloads: bool = False,
    ) -> None:
        self.encoder = encoder
        self.policy_service = policy_service
        self.reward = reward or DefensiveTowerReward()
        self.max_decisions = int(max_decisions)
        self.rpc_workers = None if rpc_workers is None else int(rpc_workers)
        self.step_ticks = int(step_ticks)
        self.idle_step_ticks = (
            None if idle_step_ticks is None else int(idle_step_ticks)
        )
        self.critic_scalar_size = int(critic_scalar_size)
        self.critic_private_slot_count = int(critic_private_slot_count)
        self.lean_step_payloads = bool(lean_step_payloads)
        self.last_profile: dict[str, float] = {}
        if self.max_decisions < 1 or self.critic_scalar_size < 1:
            raise ValueError("collector limits must be positive")
        if not 1 <= self.step_ticks <= 16:
            raise ValueError("step_ticks must be in 1..16")
        if self.idle_step_ticks is not None and not (
            self.step_ticks <= self.idle_step_ticks <= 16
        ):
            raise ValueError("idle_step_ticks must be in step_ticks..16")
        if self.critic_private_slot_count < 26:
            raise ValueError("critic private state needs at least 26 card slots")

    @staticmethod
    def _validate_spec(spec: OnlineEpisodeSpec) -> dict[int, str]:
        spec.header.validate()
        try:
            hashes = {side: str(spec.actor_hashes[side]) for side in (0, 1)}
        except KeyError as error:
            raise OnlineCollectorContractError("both side Actor hashes are required") from error
        if any(not _valid_sha256(value) for value in hashes.values()):
            raise OnlineCollectorContractError("side Actor hash is not a lowercase SHA-256")
        if spec.fixture.learner_side != spec.header.learner_side:
            raise OnlineCollectorContractError("fixture/header learner side differs")
        if spec.fixture.learner_deck_sha256 != spec.header.learner_deck_sha256:
            raise OnlineCollectorContractError("fixture/header learner deck differs")
        if spec.fixture.opponent_deck_sha256 != spec.header.opponent_deck_sha256:
            raise OnlineCollectorContractError("fixture/header opponent deck differs")
        if hashes[spec.header.learner_side] != spec.header.behavior_actor_sha256:
            raise OnlineCollectorContractError("learner Actor hash differs from episode header")
        if hashes[1 - spec.header.learner_side] != spec.header.opponent_actor_sha256:
            raise OnlineCollectorContractError("opponent Actor hash differs from episode header")
        seed = int(spec.fixture.replay.get("rndSeed", -1))
        if seed != spec.header.seed:
            raise OnlineCollectorContractError("fixture/header seed differs")
        if int(spec.warmup_steps) < 0:
            raise OnlineCollectorContractError("warmup_steps cannot be negative")
        return hashes

    @staticmethod
    def _reset_one(spec: OnlineEpisodeSpec) -> dict[str, Any]:
        reset_profile = getattr(spec.env, "reset_rpc_profile", None)
        if callable(reset_profile):
            reset_profile()
        state = spec.env.reset(spec.fixture.replay, warmup_steps=int(spec.warmup_steps))
        if not isinstance(state, Mapping):
            raise OnlineCollectorContractError("native reset did not return a state")
        return dict(state)

    def _start(self, specs: Sequence[OnlineEpisodeSpec], executor: ThreadPoolExecutor) -> list[_Cursor]:
        seen_workers: set[Hashable] = set()
        for spec in specs:
            self._validate_spec(spec)
            if spec.worker_id in seen_workers:
                raise OnlineCollectorContractError("worker_id is duplicated in a collector batch")
            seen_workers.add(spec.worker_id)
            self.policy_service.reset_episode(spec.worker_id)
        futures = [executor.submit(self._reset_one, spec) for spec in specs]
        cursors: list[_Cursor] = []
        for spec, future in zip(specs, futures, strict=True):
            state = future.result()
            if bool(state.get("episode", {}).get("terminated")) or bool(
                state.get("episode", {}).get("truncated")
            ):
                raise OnlineCollectorContractError("native reset returned a terminal episode")
            fixture_decks = _fixture_decks(spec.fixture)
            raw_decks = getattr(spec.env, "decks", fixture_decks)
            decks = [[dict(card) for card in side] for side in raw_decks]
            if len(decks) != 2 or any(len(side) != 8 for side in decks):
                raise OnlineCollectorContractError("native environment has invalid decks")
            cursors.append(_Cursor(
                spec=spec,
                state=state,
                decks=decks,
                tracker=RevealedEnemyTracker(self.encoder),
                episode=LearnerEpisodeBuffer(spec.header),
            ))
        return cursors

    @staticmethod
    def _probe_cursor(cursor: _Cursor) -> None:
        for player in cursor.state.get("players", []):
            side = int(player["side"])
            for raw_index in player.get("hand_deck_indices", [])[:4]:
                deck_index = int(raw_index)
                key = (side, deck_index)
                if deck_index < 0 or key in cursor.native_masks:
                    continue
                result = cursor.spec.env.probe_grid(side=side, deck_index=deck_index)
                rows = result.get("rows") if isinstance(result, Mapping) else None
                if (
                    not isinstance(rows, list)
                    or len(rows) != ARENA_ROWS
                    or any(not isinstance(row, str) or len(row) != ARENA_COLUMNS for row in rows)
                ):
                    raise OnlineCollectorContractError("probe_grid did not return an 18x32 mask")
                cursor.native_masks[key] = list(rows)

    def _masks(
        self,
        cursor: _Cursor,
        encoded: EncodedNativeBatch,
        encoded_row: int,
        side: int,
    ) -> tuple[ExpertActionMasks, list[int]]:
        card_mask5, positions, hand = build_action_masks(
            _mask_compatible_state(cursor.state),
            side=side,
            native_masks=cursor.native_masks,
            decks=cursor.decks,
            cache=cursor.mask_cache,
        )
        cards = torch.from_numpy(np.ascontiguousarray(card_mask5[1:]))
        canonical = torch.from_numpy(canonical_position_mask(positions, side))
        abilities = encoded.ability_mask[encoded_row, 0].detach().cpu().bool().clone()
        # NativeRoyaleEnv's authoritative ability RPC identifies an entity and
        # has no target coordinates.  Targeted abilities therefore remain
        # fail-closed until libg exposes a targeted ability command.
        ability_requires_target = torch.zeros_like(abilities)
        ability_positions = torch.zeros(
            (abilities.numel(), POSITION_COUNT), dtype=torch.bool
        )
        action_kind = torch.tensor(
            [bool(cards.any()), bool(abilities.any())], dtype=torch.bool
        )
        return ExpertActionMasks(
            action_kind=action_kind,
            cards=cards.bool(),
            positions=canonical.bool(),
            abilities=abilities,
            ability_positions=ability_positions,
            ability_requires_target=ability_requires_target,
        ), hand

    @staticmethod
    def _actor_inputs(encoded: EncodedNativeBatch, row: int) -> dict[str, Tensor]:
        return {name: encoded[name][row] for name in encoded}

    def _critic_private(
        self,
        encoded: EncodedNativeBatch,
        *,
        row_by_side: Mapping[int, int],
        learner_side: int,
    ) -> CriticPrivateObservation:
        learner_row = row_by_side[learner_side]
        opponent_row = row_by_side[1 - learner_side]
        entity_count = int(encoded.encoded_entity_counts[learner_row])
        grid = encoded["grid"][learner_row, 0].detach().cpu().numpy().copy()
        entity_tokens = encoded["entity_tokens"][learner_row, 0, :entity_count].cpu().numpy().copy()
        entity_positions = encoded["entity_positions"][learner_row, 0, :entity_count].cpu().numpy().copy()
        entity_relations = encoded["entity_relations"][learner_row, 0, :entity_count].cpu().numpy().copy()
        entity_numeric = encoded["entity_numeric"][learner_row, 0, :entity_count].cpu().numpy().copy()
        entity_mask = encoded["entity_mask"][learner_row, 0, :entity_count].cpu().numpy().copy()

        tokens: list[int] = []
        owners: list[int] = []
        slots: list[int] = []
        for relation, row in ((0, learner_row), (1, opponent_row)):
            for token in encoded["own_deck_tokens"][row, 0].tolist():
                tokens.append(int(token)); owners.append(relation); slots.append(len(slots))
        for relation, row in ((0, learner_row), (1, opponent_row)):
            for token in encoded["hand_tokens"][row, 0].tolist():
                tokens.append(int(token)); owners.append(relation); slots.append(len(slots))
        for relation, row in ((0, learner_row), (1, opponent_row)):
            tokens.append(int(encoded["next_card_token"][row, 0]))
            owners.append(relation); slots.append(len(slots))
        if len(tokens) > self.critic_private_slot_count:
            raise OnlineCollectorContractError("critic private-card state exceeds configured slots")
        private_tokens = np.asarray(tokens, dtype=np.int64)
        private_owners = np.asarray(owners, dtype=np.int64)
        private_slots = np.asarray(slots, dtype=np.int64)
        private_mask = private_tokens > 0

        learner_public = encoded["public_scalars"][learner_row, 0].cpu().numpy()
        opponent_public = encoded["public_scalars"][opponent_row, 0].cpu().numpy()
        raw_scalars = np.concatenate((learner_public, opponent_public)).astype(np.float32)
        scalars = np.zeros(self.critic_scalar_size, dtype=np.float32)
        scalars[: min(len(raw_scalars), len(scalars))] = raw_scalars[: len(scalars)]
        return CriticPrivateObservation(
            grid=grid,
            entity_tokens=entity_tokens,
            entity_positions=entity_positions,
            entity_relations=entity_relations,
            entity_numeric=entity_numeric,
            entity_mask=entity_mask,
            private_card_tokens=private_tokens,
            private_card_owners=private_owners,
            private_card_slots=private_slots,
            private_card_mask=private_mask,
            scalars=scalars,
        )

    @staticmethod
    def _native_action(
        action: SampledPolicyAction,
        *,
        hand: Sequence[int],
        deck: Sequence[Mapping[str, Any]],
        ability_entity_keys: Sequence[int],
    ) -> dict[str, Any] | None:
        if not action.event_happened:
            return None
        if action.action_kind == 0:
            if not 0 <= action.card_slot < min(4, len(hand)):
                raise OnlineCollectorContractError("Actor chose an invalid hand slot")
            deck_index = int(hand[action.card_slot])
            if not 0 <= deck_index < len(deck):
                raise OnlineCollectorContractError("Actor chose an empty hand slot")
            x, y = canonical_position_to_native(action.position, action.side)
            return {
                "type": "play",
                "side": action.side,
                "deck_index": deck_index,
                "x": x,
                "y": y,
                "card_id": int(deck[deck_index]["card_id"]),
                "canonical_position": int(action.position),
            }
        if action.action_kind == 1:
            if action.ability_requires_target:
                raise OnlineCollectorContractError(
                    "targeted ability sampled without a native targeted-ability RPC"
                )
            if not 0 <= action.ability_slot < len(ability_entity_keys):
                raise OnlineCollectorContractError("Actor chose an invalid ability slot")
            return {
                "type": "ability",
                "side": action.side,
                "entity_id": int(ability_entity_keys[action.ability_slot]),
            }
        raise OnlineCollectorContractError("Actor emitted an unknown action kind")

    @staticmethod
    def _transition_one(
        cursor: _Cursor,
        actions: list[Mapping[str, Any]],
        *,
        steps: int,
    ) -> dict[str, Any]:
        result = cursor.spec.env.joint_training_transition(actions, steps=steps)
        if not isinstance(result, Mapping):
            raise OnlineCollectorContractError("native transition did not return an object")
        return dict(result)

    def _requested_step_ticks(self, cursor: _Cursor) -> int:
        """Use a wider hazard window outside the arena's tactical zone.

        A worker can keep long-lived troops/buildings in ``entities`` for most
        of a battle, so list emptiness alone is not a useful idle signal.  A
        side-0 entity moving upward becomes tactical near/over the river, as
        does a side-1 entity moving downward.  Back-line deployment and travel
        may use the wider window; river contact, pushes, and tower defence use
        the fixed fast cadence.  Both the Actor exposure and native transition
        receive this exact value, preserving variable-time rollout accounting.
        """

        if self.idle_step_ticks is None:
            return self.step_ticks
        entities = cursor.state.get("entities")
        if not isinstance(entities, list):
            return self.step_ticks
        for entity in entities:
            if not isinstance(entity, Mapping):
                return self.step_ticks
            side = entity.get("side")
            y = entity.get("y")
            if not isinstance(side, int) or not isinstance(y, int):
                return self.step_ticks
            if (side == 0 and y >= 13_000) or (side == 1 and y <= 19_000):
                return self.step_ticks
        return self.idle_step_ticks

    @staticmethod
    def _episode_from_transition(transition: Mapping[str, Any]) -> dict[str, Any]:
        episode = transition.get("step", {}).get("episode")
        if not isinstance(episode, Mapping):
            raise OnlineCollectorContractError("native transition has no episode metadata")
        return dict(episode)

    @staticmethod
    def _accepted_actions(
        transition: Mapping[str, Any], expected: Sequence[Mapping[str, Any]], *, allow_terminal_gate: bool
    ) -> tuple[int, set[int]]:
        joint = transition.get("joint_action")
        rows = joint.get("actions") if isinstance(joint, Mapping) else None
        if not isinstance(rows, list):
            raise OnlineCollectorContractError("native transition has no joint-action receipt")
        if len(rows) != len(expected):
            raise OnlineCollectorContractError("native joint-action receipt count differs")
        accepted_sides: set[int] = set()
        for row in rows:
            result = row.get("result") if isinstance(row, Mapping) else None
            if not isinstance(result, Mapping):
                raise OnlineCollectorContractError("native action receipt is malformed")
            if bool(result.get("accepted", False)):
                accepted_sides.add(int(row["side"]))
                continue
            code = int(result.get("result_code", -1))
            if not (allow_terminal_gate and code in (3, 4)):
                raise OnlineCollectorContractError(
                    f"action selected from its pre-action mask was rejected: code={code}"
                )
        return len(rows), accepted_sides

    @staticmethod
    def _validate_advance(
        state: Mapping[str, Any],
        transition: Mapping[str, Any],
        episode: Mapping[str, Any],
        *,
        requested_steps: int,
        allow_pending: bool = False,
    ) -> tuple[int, bool]:
        previous_tick = int(state["tick"])
        terminated = bool(episode.get("terminated", False))
        truncated = bool(episode.get("truncated", False))
        if terminated and truncated:
            raise OnlineCollectorContractError("native episode is both terminated and truncated")
        if truncated:
            raise OnlineCollectorContractError("time-truncated episode is not a PPO rollout")
        done = terminated
        if done:
            terminal_tick = episode.get("terminal_tick")
            if not isinstance(terminal_tick, int) or isinstance(terminal_tick, bool):
                raise OnlineCollectorContractError("terminal episode has no integer terminal_tick")
            advanced = int(terminal_tick) - previous_tick
            if episode.get("outcome") in (None, "ongoing"):
                raise OnlineCollectorContractError("terminal episode has no final outcome")
            crowns = episode.get("crowns")
            if not isinstance(crowns, list) or len(crowns) != 2:
                raise OnlineCollectorContractError("terminal episode has no crown vector")
            _terminal_rewards(episode)
        else:
            next_state = transition.get("state")
            if not isinstance(next_state, Mapping):
                raise OnlineCollectorContractError("nonterminal transition has no next state")
            advanced = int(next_state["tick"]) - previous_tick
        if advanced < 0 or advanced > requested_steps:
            raise OnlineCollectorContractError(
                f"native Tick advance is outside requested step range: "
                f"{previous_tick}+{advanced}/{requested_steps}"
            )
        if advanced == 0 and not done and not allow_pending:
            raise OnlineCollectorContractError("native Tick froze before terminal")
        return advanced, done

    def _resolve_pending_transition(
        self,
        cursor: _Cursor,
        initial: Mapping[str, Any],
        *,
        requested_steps: int,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int, bool]:
        """Resolve libg's one-RPC-late terminal without another Actor decision.

        ``joint_training_transition_v1`` can expose the final observable frame
        before its episode latch changes.  A bounded pair of empty-action steps
        lets that latch settle (or advances the originally sampled action) while
        retaining the receipt from the call that actually submitted the action.
        """

        first = dict(initial)
        first_receipt = first.get("joint_action")
        if not isinstance(first_receipt, Mapping):
            raise OnlineCollectorContractError("native transition has no first joint-action receipt")
        receipt = deepcopy(dict(first_receipt))
        resolution_receipts: list[dict[str, Any]] = []
        transition = first
        episode = self._episode_from_transition(transition)
        advanced, done = self._validate_advance(
            cursor.state,
            transition,
            episode,
            requested_steps=requested_steps,
            allow_pending=True,
        )
        for _attempt in range(2):
            if advanced or done:
                break
            transition = self._transition_one(cursor, [], steps=1)
            followup_receipt = transition.get("joint_action")
            if isinstance(followup_receipt, Mapping):
                resolution_receipts.append(deepcopy(dict(followup_receipt)))
            episode = self._episode_from_transition(transition)
            advanced, done = self._validate_advance(
                cursor.state,
                transition,
                episode,
                requested_steps=requested_steps,
                allow_pending=True,
            )
        if advanced == 0 and not done:
            raise OnlineCollectorContractError(
                "native Tick remained frozen after two empty pending-terminal steps"
            )
        resolved = dict(transition)
        resolved["joint_action"] = receipt
        if resolution_receipts:
            resolved["pending_resolution_joint_actions"] = resolution_receipts
        return resolved, receipt, resolution_receipts, advanced, done

    @staticmethod
    def _recorded_action_payload(action: SampledPolicyAction) -> dict[str, Tensor]:
        return {
            "event_happened": torch.tensor(action.event_happened, dtype=torch.bool),
            "action_kind": torch.tensor(action.action_kind, dtype=torch.long),
            "card_slot": torch.tensor(action.card_slot, dtype=torch.long),
            "position": torch.tensor(action.position, dtype=torch.long),
            "ability_slot": torch.tensor(action.ability_slot, dtype=torch.long),
            "ability_position": torch.tensor(action.ability_position, dtype=torch.long),
            "ability_requires_target": torch.tensor(
                action.ability_requires_target, dtype=torch.bool
            ),
            "old_logp_total": torch.tensor(action.old_logp_total, dtype=torch.float32),
            "old_logp_timing": torch.tensor(action.old_logp_timing, dtype=torch.float32),
            "old_logp_action_type": torch.tensor(
                action.old_logp_action_type, dtype=torch.float32
            ),
            "old_logp_slot": torch.tensor(action.old_logp_slot, dtype=torch.float32),
            "old_logp_position": torch.tensor(
                action.old_logp_position, dtype=torch.float32
            ),
        }

    @staticmethod
    def _needs_hidden_anchor(decision_index: int) -> bool:
        """Return whether v1 recurrent chunking retains this pre-action state."""

        # RecurrentChunkConfig is fixed at burn_in=16 / unroll=64.  Chunk loss
        # starts are 0,64,128,... and their sequence anchors are 0,48,112,... .
        return decision_index == 0 or (
            decision_index >= 48 and (decision_index - 48) % 64 == 0
        )

    def _append_decision(
        self,
        cursor: _Cursor,
        *,
        action: SampledPolicyAction,
        actor_inputs: Mapping[str, Tensor],
        pre_action_hidden: tuple[Tensor, Tensor] | None,
        masks: ExpertActionMasks,
        critic_private: CriticPrivateObservation,
        native_action: Mapping[str, Any] | None,
        native_receipt: Mapping[str, Any],
        components: Mapping[str, float],
        value: float,
        terminated: bool,
        advanced_ticks: int,
        encoded_row: int,
    ) -> None:
        if not math.isfinite(value):
            raise OnlineCollectorContractError("Critic value is NaN/Inf")
        side = cursor.spec.header.learner_side
        record = DecisionRecord(
            tick=int(cursor.state["tick"]),
            delta_ticks=int(advanced_ticks),
            side=side,
            event_happened=action.event_happened,
            action_kind=action.action_kind,
            card_slot=action.card_slot,
            position=action.position,
            ability_slot=action.ability_slot,
            ability_position=action.ability_position,
            old_logp_total=action.old_logp_total,
            old_logp_timing=action.old_logp_timing,
            old_logp_action_type=action.old_logp_action_type,
            old_logp_slot=action.old_logp_slot,
            old_logp_position=action.old_logp_position,
            reward_damage_dealt=float(components["damage_dealt"]),
            reward_damage_received=float(components["damage_received"]),
            reward_towers_dealt=float(components["towers_dealt"]),
            reward_towers_received=float(components["towers_received"]),
            reward_terminal=float(components["terminal"]),
            reward_total=float(components["total"]),
            value=float(value),
            terminated=terminated,
            truncated=False,
            native_entity_count=int(critic_private.entity_mask.size),
            encoded_entity_count=int(np.asarray(critic_private.entity_mask).sum()),
        )
        # Preserve the encoder's stricter native-vs-encoded audit counts.
        record = replace(
            record,
            native_entity_count=int(cursor._encoded.native_entity_counts[encoded_row]),  # type: ignore[attr-defined]
            encoded_entity_count=int(cursor._encoded.encoded_entity_counts[encoded_row]),  # type: ignore[attr-defined]
        )
        cursor.episode.append(record)
        cursor.target_states.append(_tower_target_state(cursor.state))
        stored_actor_inputs: dict[str, Any] = _clone_tensor_mapping(actor_inputs)
        if pre_action_hidden is not None:
            stored_actor_inputs["hidden"] = tuple(
                value.detach().cpu().contiguous().clone()
                for value in pre_action_hidden
            )
        payload = {
            "schema_version": 1,
            "kind": STEP_PAYLOAD_KIND,
            "tick": record.tick,
            "delta_ticks": record.delta_ticks,
            "actor_sha256": action.actor_sha256,
            "actor_inputs": stored_actor_inputs,
            "action_masks": _mask_mapping(masks),
            "recorded_action": self._recorded_action_payload(action),
            "critic_inputs": _critic_inputs(critic_private),
            "reward_components": {name: float(item) for name, item in components.items()},
            "value": torch.tensor(value, dtype=torch.float32),
            "terminated": bool(terminated),
            "terminal_merged_from_zero_delta": False,
        }
        if not self.lean_step_payloads:
            payload.update({
                "critic_private": _critic_mapping(critic_private),
                "native_action": (
                    None if native_action is None else deepcopy(dict(native_action))
                ),
                "native_receipt": deepcopy(dict(native_receipt)),
                "native_advanced_ticks": int(advanced_ticks),
            })
        cursor.step_payloads.append(payload)

    def _merge_zero_delta_terminal(
        self,
        cursor: _Cursor,
        *,
        episode: Mapping[str, Any],
        first_receipt: Mapping[str, Any],
        resolution_receipts: Sequence[Mapping[str, Any]],
    ) -> None:
        if not cursor.episode.decisions or not cursor.step_payloads:
            raise OnlineCollectorContractError(
                "zero-delta terminal arrived before any native Tick was collected"
            )
        terminal_rewards = _terminal_rewards(episode)
        terminal_state = _terminal_reward_state(episode)
        extra = _reward_components(
            self.reward,
            cursor.state,
            terminal_state,
            side=cursor.spec.header.learner_side,
            terminal_rewards=terminal_rewards,
            done=True,
        )
        previous = cursor.episode.decisions[-1]
        if previous.terminated:
            raise OnlineCollectorContractError("terminal was delivered twice")
        updated = replace(
            previous,
            reward_damage_dealt=previous.reward_damage_dealt + extra["damage_dealt"],
            reward_damage_received=previous.reward_damage_received + extra["damage_received"],
            reward_towers_dealt=previous.reward_towers_dealt + extra["towers_dealt"],
            reward_towers_received=previous.reward_towers_received + extra["towers_received"],
            reward_terminal=previous.reward_terminal + extra["terminal"],
            reward_total=previous.reward_total + extra["total"],
            terminated=True,
        )
        updated.validate(cursor.spec.header.learner_side)
        cursor.episode.decisions[-1] = updated
        payload = cursor.step_payloads[-1]
        rewards = dict(payload["reward_components"])
        for name in ("damage_dealt", "damage_received", "towers_dealt", "towers_received", "terminal", "total"):
            rewards[name] = float(rewards[name]) + float(extra[name])
        payload["reward_components"] = rewards
        payload["terminated"] = True
        payload["terminal_merged_from_zero_delta"] = True
        if not self.lean_step_payloads:
            payload["late_terminal_first_joint_action"] = deepcopy(dict(first_receipt))
            payload["late_terminal_resolution_joint_actions"] = [
                deepcopy(dict(value)) for value in resolution_receipts
            ]

    def _finalize_critic_targets(self, cursor: _Cursor) -> None:
        """Attach outcome and future-5-second labels after the full game exists."""

        episode = cursor.terminal_episode
        if episode is None or len(cursor.target_states) != len(cursor.step_payloads):
            raise OnlineCollectorContractError("critic target timeline is incomplete")
        learner = cursor.spec.header.learner_side
        enemy = 1 - learner
        rewards = _terminal_rewards(episode)
        outcome = 2 if rewards[learner] > 0 else 0 if rewards[learner] < 0 else 1
        crowns = episode.get("crowns")
        if not isinstance(crowns, list) or len(crowns) != 2:
            raise OnlineCollectorContractError("critic targets need final crowns")
        crown_difference = float(int(crowns[learner]) - int(crowns[enemy]))

        terminal_target = _terminal_reward_state(episode)
        if terminal_target is None:
            terminal_target = cursor.target_states[-1]
        terminal_target = dict(terminal_target)
        terminal_target["tick"] = int(episode["terminal_tick"])
        terminal_hp, _terminal_alive = self.reward._tower_summary(terminal_target)
        terminal_max = [0, 0]
        for tower in terminal_target.get("episode", {}).get("crown_towers", []):
            side = int(tower.get("side", -1))
            if side in (0, 1):
                terminal_max[side] += max(0, int(tower.get("max_hp", 0)))
        own_fraction = terminal_hp[learner] / max(1, terminal_max[learner])
        enemy_fraction = terminal_hp[enemy] / max(1, terminal_max[enemy])
        tower_hp_difference = float(own_fraction - enemy_fraction)

        timeline = [*cursor.target_states, terminal_target]
        for index, (state, payload) in enumerate(
            zip(cursor.target_states, cursor.step_payloads, strict=True)
        ):
            target_tick = int(state["tick"]) + 100  # native 20 Hz = five seconds
            future = terminal_target
            for candidate in timeline[index + 1 :]:
                future = candidate
                if int(candidate.get("tick", target_tick)) >= target_tick:
                    break
            current_hp, _ = self.reward._tower_summary(state)
            future_hp, _ = self.reward._tower_summary(future)
            payload["critic_targets"] = {
                "wdl_class": int(outcome),
                "crown_difference": crown_difference,
                "tower_hp_difference": tower_hp_difference,
                # The reward's native scale keeps this regression head near O(1).
                "future_damage": [
                    float(max(0, current_hp[enemy] - future_hp[enemy]) * 0.001),
                    float(max(0, current_hp[learner] - future_hp[learner]) * 0.001),
                ],
            }

    def collect_episode(
        self,
        *,
        env: NativeRoyaleEnv,
        fixture: DeckFixture,
        header: EpisodeHeader,
        actor_hashes: Mapping[int, str],
        worker_id: Hashable = 0,
        warmup_steps: int = 100,
        value_fn: ValueFunction | None = None,
    ) -> CollectedEpisode:
        """Collect one complete episode (a convenience wrapper around a batch)."""

        return self.collect_batch(
            [OnlineEpisodeSpec(worker_id, env, fixture, header, actor_hashes, warmup_steps)],
            value_fn=value_fn,
        )[0]

    def collect_batch(
        self,
        specs: Sequence[OnlineEpisodeSpec],
        *,
        value_fn: ValueFunction | None = None,
        rolling_replacement: bool = False,
    ) -> list[CollectedEpisode]:
        """Collect a barrier batch while batching all Actor rows each Tick."""

        rows = list(specs)
        if not rows:
            return []
        worker_count = self.rpc_workers or len(rows)
        if worker_count < 1:
            raise ValueError("rpc_workers must be positive")
        profile = {
            "start_seconds": 0.0,
            "probe_seconds": 0.0,
            "encode_seconds": 0.0,
            "request_build_seconds": 0.0,
            "policy_seconds": 0.0,
            "action_decode_seconds": 0.0,
            "transition_wait_seconds": 0.0,
            "record_seconds": 0.0,
            "transition_and_record_seconds": 0.0,
            "finalize_seconds": 0.0,
            "scheduler_turns": 0.0,
            "actor_rows": 0.0,
            "policy_forward_calls": 0.0,
            "active_step_requests": 0.0,
            "idle_step_requests": 0.0,
            "requested_native_ticks": 0.0,
            "total_seconds": 0.0,
        }
        batch_started = time.perf_counter()
        forward_calls_before = int(
            getattr(self.policy_service, "forward_calls", 0)
        )
        cursors: list[_Cursor] = []
        pending_by_worker: dict[Hashable, list[OnlineEpisodeSpec]] = {}
        initial_rows = rows
        if rolling_replacement:
            initial_rows = []
            env_by_worker: dict[Hashable, object] = {}
            for spec in rows:
                self._validate_spec(spec)
                previous_env = env_by_worker.setdefault(spec.worker_id, spec.env)
                if previous_env is not spec.env:
                    raise OnlineCollectorContractError(
                        "rolling worker_id maps to more than one native environment"
                    )
                queue = pending_by_worker.setdefault(spec.worker_id, [])
                if not queue:
                    initial_rows.append(spec)
                queue.append(spec)
            for spec in initial_rows:
                pending_by_worker[spec.worker_id].pop(0)
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                stage_started = time.perf_counter()
                active = self._start(initial_rows, executor)
                cursors.extend(active)
                profile["start_seconds"] += time.perf_counter() - stage_started
                while active:
                    profile["scheduler_turns"] += 1.0
                    requested_steps = {
                        id(cursor): self._requested_step_ticks(cursor)
                        for cursor in active
                    }
                    profile["active_step_requests"] += float(sum(
                        value == self.step_ticks for value in requested_steps.values()
                    ))
                    profile["idle_step_requests"] += float(sum(
                        value != self.step_ticks for value in requested_steps.values()
                    ))
                    profile["requested_native_ticks"] += float(
                        sum(requested_steps.values())
                    )
                    for cursor in active:
                        if len(cursor.episode.decisions) >= self.max_decisions:
                            raise OnlineCollectorContractError(
                                f"episode {cursor.spec.header.episode_id} exceeded max_decisions"
                            )

                    stage_started = time.perf_counter()
                    probe_futures = [executor.submit(self._probe_cursor, cursor) for cursor in active]
                    for future in probe_futures:
                        future.result()
                    profile["probe_seconds"] += time.perf_counter() - stage_started

                    stage_started = time.perf_counter()
                    frames: list[NativeActorFrame] = []
                    owners: list[tuple[_Cursor, int]] = []
                    for cursor in active:
                        for side in (0, 1):
                            frames.append(NativeActorFrame(
                                state=cursor.state,
                                actor_side=side,
                                own_deck=cursor.decks[side],
                                revealed_enemy_tokens=cursor.tracker.tokens_for(side),
                                delta_ticks=float(requested_steps[id(cursor)]),
                            ))
                            owners.append((cursor, side))
                    encoded = self.encoder.encode_batch(frames)
                    profile["encode_seconds"] += time.perf_counter() - stage_started
                    profile["actor_rows"] += float(len(owners))
                    for cursor in active:
                        cursor._encoded = encoded  # type: ignore[attr-defined]

                    stage_started = time.perf_counter()
                    policy_requests: list[PolicyRequest] = []
                    request_rows: list[dict[str, Any]] = []
                    row_by_cursor_side: dict[tuple[int, int], int] = {}
                    for index, (cursor, side) in enumerate(owners):
                        row_by_cursor_side[(id(cursor), side)] = index
                        masks, hand = self._masks(cursor, encoded, index, side)
                        inputs = self._actor_inputs(encoded, index)
                        hashes = self._validate_spec(cursor.spec)
                        policy_requests.append(PolicyRequest(
                            worker_id=cursor.spec.worker_id,
                            side=side,
                            actor_sha256=hashes[side],
                            actor_inputs=inputs,
                            masks=masks,
                            delta_ticks=requested_steps[id(cursor)],
                            reset_hidden=False,
                        ))
                        request_rows.append({
                            "cursor": cursor, "side": side, "masks": masks,
                            "hand": hand, "inputs": inputs, "encoded_row": index,
                        })
                    profile["request_build_seconds"] += (
                        time.perf_counter() - stage_started
                    )

                    # This is intentionally exactly one service call per scheduler
                    # turn.  The service itself groups equal content hashes.
                    stage_started = time.perf_counter()
                    sampled = self.policy_service.act(policy_requests)
                    profile["policy_seconds"] += time.perf_counter() - stage_started
                    if len(sampled) != len(policy_requests):
                        raise OnlineCollectorContractError("policy service dropped an Actor row")
                    anchor_indices = [
                        index
                        for index, (action, row) in enumerate(
                            zip(sampled, request_rows, strict=True)
                        )
                        if action.side == row["cursor"].spec.header.learner_side
                        and self._needs_hidden_anchor(
                            len(row["cursor"].episode.decisions)
                        )
                    ]
                    anchor_actions = [sampled[index] for index in anchor_indices]
                    hidden_batch = getattr(
                        self.policy_service, "last_pre_action_hidden_batch", None
                    )
                    if callable(hidden_batch):
                        anchor_hiddens = hidden_batch(anchor_actions)
                    else:
                        anchor_hiddens = [
                            self.policy_service.last_pre_action_hidden(
                                actor_sha256=action.actor_sha256,
                                worker_id=action.worker_id,
                                side=action.side,
                            )
                            for action in anchor_actions
                        ]
                    if len(anchor_hiddens) != len(anchor_indices):
                        raise OnlineCollectorContractError(
                            "policy service dropped a recurrent anchor"
                        )
                    pre_action_hiddens: list[
                        tuple[Tensor, Tensor] | None
                    ] = [None] * len(sampled)
                    for index, hidden in zip(
                        anchor_indices, anchor_hiddens, strict=True
                    ):
                        pre_action_hiddens[index] = hidden

                    stage_started = time.perf_counter()
                    per_cursor: dict[int, dict[str, Any]] = {
                        id(cursor): {"actions": [], "rows": {}} for cursor in active
                    }
                    for action, row, pre_action_hidden in zip(
                        sampled, request_rows, pre_action_hiddens, strict=True
                    ):
                        cursor = row["cursor"]
                        side = int(row["side"])
                        ability_keys = encoded.ability_entity_keys[row["encoded_row"]]
                        native = self._native_action(
                            action,
                            hand=row["hand"],
                            deck=cursor.decks[side],
                            ability_entity_keys=ability_keys,
                        )
                        row["sample"] = action
                        row["pre_action_hidden"] = pre_action_hidden
                        row["native_action"] = native
                        per_cursor[id(cursor)]["rows"][side] = row
                        if native is not None:
                            per_cursor[id(cursor)]["actions"].append(native)
                    profile["action_decode_seconds"] += (
                        time.perf_counter() - stage_started
                    )

                    stage_started = time.perf_counter()
                    transition_futures = {
                        id(cursor): executor.submit(
                            self._transition_one,
                            cursor,
                            per_cursor[id(cursor)]["actions"],
                            steps=requested_steps[id(cursor)],
                        )
                        for cursor in active
                    }
                    completed: list[_Cursor] = []
                    for cursor in active:
                        wait_started = time.perf_counter()
                        initial_transition = transition_futures[id(cursor)].result()
                        profile["transition_wait_seconds"] += (
                            time.perf_counter() - wait_started
                        )
                        record_started = time.perf_counter()
                        (
                            transition,
                            first_receipt,
                            resolution_receipts,
                            advanced,
                            done,
                        ) = self._resolve_pending_transition(
                            cursor,
                            initial_transition,
                            requested_steps=requested_steps[id(cursor)],
                        )
                        episode = self._episode_from_transition(transition)
                        attempted, accepted_sides = self._accepted_actions(
                            transition,
                            per_cursor[id(cursor)]["actions"],
                            allow_terminal_gate=done and advanced == 0,
                        )
                        cursor.attempted_actions += attempted
                        cursor.accepted_actions += len(accepted_sides)
                        cursor.native_ticks_advanced += advanced

                        if done and advanced == 0:
                            self._merge_zero_delta_terminal(
                                cursor,
                                episode=episode,
                                first_receipt=first_receipt,
                                resolution_receipts=resolution_receipts,
                            )
                            cursor.terminal_episode = deepcopy(episode)
                            completed.append(cursor)
                            continue

                        learner = cursor.spec.header.learner_side
                        row_map = {
                            side: row_by_cursor_side[(id(cursor), side)] for side in (0, 1)
                        }
                        critic_private = self._critic_private(
                            encoded, row_by_side=row_map, learner_side=learner
                        )
                        learner_row = per_cursor[id(cursor)]["rows"][learner]
                        terminal_rewards = (
                            _terminal_rewards(episode) if done else {0: 0.0, 1: 0.0}
                        )
                        next_state = (
                            _terminal_reward_state(episode)
                            if done
                            else dict(transition["state"])
                        )
                        components = _reward_components(
                            self.reward,
                            cursor.state,
                            next_state,
                            side=learner,
                            terminal_rewards=terminal_rewards,
                            done=done,
                        )
                        value = 0.0 if value_fn is None else float(
                            value_fn(critic_private, learner_row["inputs"])
                        )
                        joint_receipt = transition.get("joint_action", {})
                        self._append_decision(
                            cursor,
                            action=learner_row["sample"],
                            actor_inputs=learner_row["inputs"],
                            pre_action_hidden=learner_row["pre_action_hidden"],
                            masks=learner_row["masks"],
                            critic_private=critic_private,
                            native_action=learner_row["native_action"],
                            native_receipt=joint_receipt if isinstance(joint_receipt, Mapping) else {},
                            components=components,
                            value=value,
                            terminated=done,
                            advanced_ticks=advanced,
                            encoded_row=learner_row["encoded_row"],
                        )

                        for side in accepted_sides:
                            native = per_cursor[id(cursor)]["rows"][side]["native_action"]
                            if native is not None and native.get("type") == "play":
                                deck_index = int(native["deck_index"])
                                cursor.tracker.record_play(
                                    played_side=side, card=cursor.decks[side][deck_index]
                                )
                        if done:
                            cursor.terminal_episode = deepcopy(episode)
                            completed.append(cursor)
                        else:
                            cursor.state = dict(transition["state"])
                        profile["record_seconds"] += (
                            time.perf_counter() - record_started
                        )
                    if completed:
                        for cursor in completed:
                            cursor.final_rpc_profile = dict(
                                getattr(cursor.spec.env, "rpc_profile", {})
                            )
                        complete_ids = {id(cursor) for cursor in completed}
                        active = [cursor for cursor in active if id(cursor) not in complete_ids]
                        if rolling_replacement:
                            replacements = []
                            for cursor in completed:
                                queue = pending_by_worker.get(
                                    cursor.spec.worker_id, []
                                )
                                if queue:
                                    replacements.append(queue.pop(0))
                            if replacements:
                                replacement_started = time.perf_counter()
                                started = self._start(replacements, executor)
                                cursors.extend(started)
                                active.extend(started)
                                profile["start_seconds"] += (
                                    time.perf_counter() - replacement_started
                                )
                    profile["transition_and_record_seconds"] += (
                        time.perf_counter() - stage_started
                    )
        finally:
            for spec in rows:
                self.policy_service.reset_episode(spec.worker_id)

        finalize_started = time.perf_counter()
        results: list[CollectedEpisode] = []
        for cursor in cursors:
            if cursor.terminal_episode is None:
                raise OnlineCollectorContractError("collector returned an incomplete native episode")
            # freeze validates final-only termination and finite learner records.
            cursor.episode.freeze()
            self._finalize_critic_targets(cursor)
            if self.lean_step_payloads:
                cursor.step_payloads = [{
                    name: payload[name]
                    for name in (
                        "schema_version", "kind", "tick", "delta_ticks",
                        "actor_sha256", "actor_inputs", "action_masks",
                        "recorded_action", "critic_inputs", "critic_targets",
                    )
                } for payload in cursor.step_payloads]
            results.append(CollectedEpisode(
                episode=cursor.episode,
                step_payloads=tuple(cursor.step_payloads),
                terminal_episode=cursor.terminal_episode,
                attempted_actions=cursor.attempted_actions,
                accepted_actions=cursor.accepted_actions,
                native_ticks_advanced=cursor.native_ticks_advanced,
            ))
        profile["finalize_seconds"] = time.perf_counter() - finalize_started
        profile["policy_forward_calls"] = float(
            int(getattr(self.policy_service, "forward_calls", 0))
            - forward_calls_before
        )
        for key in (
            "rpc_calls", "rpc_failures", "rpc_request_bytes",
            "rpc_response_bytes", "rpc_serialize_seconds",
            "rpc_connect_seconds", "rpc_send_seconds",
            "rpc_receive_seconds", "rpc_parse_seconds", "rpc_total_seconds",
            "rpc_reconnects",
        ):
            profile[f"sum_{key}"] = sum(
                float(
                    (
                        cursor.final_rpc_profile
                        if cursor.final_rpc_profile is not None
                        else getattr(cursor.spec.env, "rpc_profile", {})
                    ).get(key, 0.0)
                )
                for cursor in cursors
            )
        profile["total_seconds"] = time.perf_counter() - batch_started
        self.last_profile = profile
        return results


__all__ = [
    "CollectedEpisode",
    "OnlineCollectorContractError",
    "OnlineEpisodeSpec",
    "OnlineSelfPlayCollector",
    "STEP_PAYLOAD_KIND",
    "canonical_position_mask",
    "canonical_position_to_native",
]
