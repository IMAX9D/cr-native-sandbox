"""Lossless per-Tick teacher-forced pilot helpers.

This module deliberately handles deployment-only battles.  Ability-bearing
sources are selected by the separate ability-aware runner; silently omitting a
button press would make every later native state wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback is supported
    orjson = None

from native_core.env import NativeRoyaleEnv
from native_core.card_catalog import catalog

from .native_freeze import (
    NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK,
    logic_freeze_audit,
    logic_freeze_failure,
)
from .native_profile import (
    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET,
    action_tick_provenance,
    native_teacher_forced_profile,
    validate_action_execution_tick_offset,
)
from .native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    ReplayPlanError,
    compile_battle,
    grouped_actions,
)
from .native_seed_search import (
    DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    resolve_native_seed,
)
from .tick_store_v1.schema import TickState, normalize_native_state, require_consecutive


@dataclass(frozen=True)
class PilotTask:
    battle_tag: str
    source_path: str
    source_sha256: str
    source_schema_version: int
    team_crowns: int
    opponent_crowns: int

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceReplay:
    audit: dict[str, Any]
    states: tuple[TickState, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def select_deployment_only_tasks(
    manifest_path: Path,
    *,
    limit: int,
    minimum_source_schema: int = 3,
    warmup_tick: int = 10,
) -> tuple[list[PilotTask], dict[str, Any]]:
    """Select deterministic, executable deployment-only native candidates."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected: list[PilotTask] = []
    reasons: Counter[str] = Counter()
    scanned = 0
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if len(selected) >= limit:
                break
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            if int(row.get("schema_version") or 1) < minimum_source_schema:
                reasons["source_schema_below_minimum"] += 1
                continue
            path = Path(str(row.get("source_path") or ""))
            if not path.is_file():
                reasons["source_missing"] += 1
                continue
            try:
                source = load_json(path)
                plan = compile_battle(
                    source,
                    terminal_crowns=(
                        int(row["team_crowns"]), int(row["opponent_crowns"])
                    ),
                )
            except (KeyError, TypeError, ValueError, ReplayPlanError):
                reasons["plan_compile_rejected"] += 1
                continue
            if source.get("schema_version") != row.get("schema_version"):
                reasons["manifest_schema_mismatch"] += 1
                continue
            if plan.battle_tag != str(row.get("battle_tag") or ""):
                reasons["manifest_battle_tag_mismatch"] += 1
                continue
            if not plan.native_replay_ready:
                reasons[f"tier_{plan.replay_tier}"] += 1
                continue
            if plan.ability_provenance != "source_reports_zero":
                reasons["ability_not_zero"] += 1
                continue
            if not plan.actions or plan.actions[0].tick < warmup_tick:
                reasons["first_action_precedes_warmup"] += 1
                continue
            selected.append(
                PilotTask(
                    battle_tag=plan.battle_tag,
                    source_path=str(path.resolve()),
                    source_sha256=sha256_file(path),
                    source_schema_version=plan.source_schema_version,
                    team_crowns=int(row["team_crowns"]),
                    opponent_crowns=int(row["opponent_crowns"]),
                )
            )
    if len(selected) != limit:
        raise RuntimeError(
            f"requested {limit} deployment-only candidates, found {len(selected)}"
        )
    return selected, {
        "manifest": str(manifest_path.resolve()),
        "requested": limit,
        "selected": len(selected),
        "rows_scanned": scanned,
        "minimum_source_schema": minimum_source_schema,
        "exclusions": dict(sorted(reasons.items())),
    }


def _logical_state_digest(
    states: Sequence[TickState], mappings: Sequence[Sequence[int]]
) -> str:
    """Hash normalized native states after removing seed-specific deck slots."""
    inverse: list[dict[int, int]] = []
    for mapping in mappings:
        current = {int(native): logical for logical, native in enumerate(mapping)}
        if set(current) != set(range(8)):
            raise ValueError("logical/native mapping is not bijective")
        inverse.append(current)
    digest = hashlib.sha256()
    for state in states:
        players = []
        for player in state.players:
            mapping = inverse[player.side]
            hand = tuple(mapping[value] if value >= 0 else value for value in player.hand)
            next_index = (
                mapping[player.next_deck_index]
                if player.next_deck_index >= 0 else player.next_deck_index
            )
            players.append((player.side, player.elixir_raw, hand, next_index))
        canonical = (
            state.tick,
            tuple(players),
            tuple((tower.key, *tower.values()) for tower in state.towers),
            tuple((entity.key, *entity.values()) for entity in state.entities),
            state.episode.values(),
        )
        digest.update(repr(canonical).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def action_execution_tick(source_tick: int, offset: int) -> int:
    """Map an immutable source label onto an explicit execution boundary.

    RoyaleAPI native teacher-forced profile v1 uses offset 1.  Explicit offset
    0 remains available only for reproducing the historical phase diagnostic;
    arbitrary offsets are rejected.
    """
    offset = validate_action_execution_tick_offset(offset)
    source_tick = int(source_tick)
    if source_tick < 0:
        raise ValueError("source Tick must be non-negative")
    return source_tick + offset


def execute_deployment_trace(
    env: NativeRoyaleEnv,
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]] | None = None,
    *,
    seed: int = DEFAULT_NATIVE_SEED,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    warmup_tick: int = 10,
    trace_batch_steps: int = 64,
    terminal_fence_ticks: int = 20,
    action_execution_tick_offset: int = (
        ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
    ),
) -> TraceReplay:
    """Replay deployments and retain every complete native Tick.

    Deployment acceptance and terminal agreement are intentionally separate.
    RoyaleAPI duration is rounded to seconds, and a teacher-forced trajectory
    remains usable when all source actions were accepted even if the coarse
    source fence does not expose libg's terminal object.
    """
    if not plan.native_replay_ready:
        raise ValueError(f"pilot requires native_replay_ready, got {plan.replay_tier}")
    if plan.ability_provenance != "source_reports_zero":
        raise ValueError("deployment-only pilot cannot omit source abilities")
    if not 1 <= trace_batch_steps <= 64:
        raise ValueError("trace_batch_steps must be in 1..64")
    # Validate even an empty plan before touching the Worker.
    action_execution_tick(0, action_execution_tick_offset)

    started = time.perf_counter()
    reset_seconds = trace_seconds = action_seconds = normalize_seconds = 0.0
    trace_rpc_count = trace_steps = accepted_actions = 0
    first_rejection: dict[str, Any] | None = None
    failure: str | None = None
    terminal_seen = False
    terminal_episode: Mapping[str, Any] | None = None
    previous_source_tick_by_side: list[int | None] = [None, None]
    previous_execution_tick_by_side: list[int | None] = [None, None]
    # ``calibration`` is a compatibility-only argument.  The old transform
    # moved source form slots and assumed card-independent shuffle; authoritative
    # libg evidence disproved that assumption.  Resolve the unobserved seed
    # directly while preserving source deck order and every action Tick.
    del calibration
    reset_started = time.perf_counter()
    seed_resolution = resolve_native_seed(
        env,
        plan,
        template,
        preferred_seed=seed,
        maximum_seeds_to_test=maximum_seeds_to_test,
        warmup_tick=warmup_tick,
    )
    reset_seconds += time.perf_counter() - reset_started
    mappings = seed_resolution.mappings
    latest_state: Mapping[str, Any] = seed_resolution.state

    current_tick = int(latest_state["tick"])
    normalization_started = time.perf_counter()
    states: list[TickState] = [normalize_native_state(latest_state)]
    normalize_seconds += time.perf_counter() - normalization_started
    complete_frames = 1
    incomplete_frames = 0
    logic_frozen_at_fence = False
    logic_freeze_before_execution: dict[str, Any] | None = None

    def advance_to(
        target_tick: int,
        *,
        source_tick: int | None = None,
        allow_nonterminal_freeze: bool = False,
    ) -> bool:
        nonlocal current_tick, latest_state, trace_seconds, trace_rpc_count
        nonlocal trace_steps, normalize_seconds, complete_frames, incomplete_frames
        nonlocal terminal_seen, terminal_episode, failure, logic_frozen_at_fence
        nonlocal logic_freeze_before_execution
        while current_tick < target_tick:
            requested = min(trace_batch_steps, target_tick - current_tick)
            trace_started = time.perf_counter()
            trace = env.trace_train(
                requested,
                # Decode a frozen suffix in both contexts.  Only the final
                # duration fence may accept it as diagnostics; between expert
                # actions it becomes a structured fail-closed replay failure.
                allow_nonterminal_freeze=True,
            )
            trace_seconds += time.perf_counter() - trace_started
            trace_rpc_count += 1
            initial = trace["initial_frame"]
            if int(initial["state"]["tick"]) != current_tick:
                failure = (
                    f"trace_initial_tick_{initial['state']['tick']}_expected_{current_tick}"
                )
                return False
            for frame in [initial, *trace["frames"]]:
                if frame.get("observation_complete") is True:
                    complete_frames += 1
                else:
                    incomplete_frames += 1
            frames = trace["frames"]
            stepped = int(trace["stepped"])
            if stepped != len(frames):
                failure = "trace_step_frame_count_mismatch"
                return False
            normalization_started = time.perf_counter()
            for frame in frames:
                if frame.get("observation_complete") is not True:
                    raw_episode = frame["state"].get("episode")
                    terminal_incomplete = (
                        trace.get("terminal") is True
                        and int(frame["state"].get("tick", -1)) == states[-1].tick
                    )
                    fence_freeze = (
                        allow_nonterminal_freeze
                        and trace.get("nonterminal_freeze") is True
                        and int(frame["state"].get("tick", -1)) == states[-1].tick
                    )
                    action_freeze = (
                        not allow_nonterminal_freeze
                        and trace.get("nonterminal_freeze") is True
                        and int(frame["state"].get("tick", -1)) == states[-1].tick
                    )
                    if (
                        not terminal_incomplete
                        and not fence_freeze
                        and not action_freeze
                    ):
                        failure = "incomplete_nonterminal_compact_trace_frame"
                        break
                    if terminal_incomplete:
                        if (
                            isinstance(raw_episode, Mapping)
                            and raw_episode.get("terminated")
                        ):
                            terminal_episode = raw_episode
                        terminal_seen = True
                    else:
                        if fence_freeze:
                            logic_frozen_at_fence = True
                        else:
                            assert source_tick is not None
                            logic_freeze_before_execution = logic_freeze_audit(
                                frame["state"],
                                fallback_state=latest_state,
                                source_tick=source_tick,
                                execution_tick=target_tick,
                                execution_tick_offset=(
                                    action_execution_tick_offset
                                ),
                                chosen_seed=seed_resolution.chosen_seed,
                                source_actions=len(plan.actions),
                                accepted_actions=accepted_actions,
                                collected_tick_count=len(states),
                                collected_tick_start=(
                                    states[0].tick if states else None
                                ),
                                collected_tick_stop=(
                                    states[-1].tick if states else None
                                ),
                                trace_requested_steps=requested,
                                trace_stepped_calls=stepped,
                            )
                            failure = logic_freeze_failure(
                                source_tick=source_tick,
                                execution_tick=target_tick,
                                last_native_tick=states[-1].tick,
                            )
                            break
                    continue
                normalized = normalize_native_state(frame["state"])
                if normalized.tick == states[-1].tick:
                    frozen_episode = frame["state"].get("episode", {})
                    if (
                        frozen_episode.get("terminated")
                        or frozen_episode.get("truncated")
                        or allow_nonterminal_freeze
                    ):
                        # libg may consume one final host step while its battle
                        # logic Tick is already frozen.  This is a terminal
                        # diagnostic frame, not a second sample for that Tick.
                        if allow_nonterminal_freeze and not (
                            frozen_episode.get("terminated")
                            or frozen_episode.get("truncated")
                        ):
                            logic_frozen_at_fence = True
                        continue
                    failure = f"duplicate_nonterminal_trace_tick_{normalized.tick}"
                    break
                if normalized.tick != states[-1].tick + 1:
                    failure = (
                        f"nonconsecutive_trace_{states[-1].tick}_{normalized.tick}"
                    )
                    break
                states.append(normalized)
                latest_state = frame["state"]
            normalize_seconds += time.perf_counter() - normalization_started
            if failure is not None:
                return False
            trace_steps += stepped
            current_tick = states[-1].tick
            episode = latest_state.get("episode", {})
            terminal_seen = bool(
                terminal_seen or trace.get("terminal")
                or episode.get("terminated")
                or episode.get("truncated")
            )
            if terminal_seen and terminal_episode is None:
                terminal_episode = episode
            if terminal_seen:
                return current_tick >= target_tick
            if logic_frozen_at_fence:
                return False
            if stepped != requested:
                failure = f"short_trace_{stepped}_requested_{requested}"
                return False
        return True

    if failure is None:
        for source_tick, actions in grouped_actions(plan, mappings):
            execution_tick = action_execution_tick(
                source_tick, action_execution_tick_offset
            )
            if execution_tick < current_tick:
                failure = (
                    f"source_tick_{source_tick}_precedes_native_tick_{current_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"execution_tick_{execution_tick}_source_tick_{source_tick}_"
                        f"precedes_native_tick_{current_tick}"
                    )
                )
                break
            if not advance_to(execution_tick, source_tick=source_tick):
                if failure is None and terminal_seen:
                    failure = (
                        f"native_terminal_before_source_tick_{source_tick}"
                        if action_execution_tick_offset == 0
                        else (
                            f"native_terminal_before_execution_tick_{execution_tick}_"
                            f"source_tick_{source_tick}"
                        )
                    )
                break
            if terminal_seen:
                failure = (
                    f"native_terminal_before_source_tick_{source_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"native_terminal_before_execution_tick_{execution_tick}_"
                        f"source_tick_{source_tick}"
                    )
                )
                break
            by_side = {
                int(player["side"]): player for player in latest_state["players"]
            }
            mismatch = next(
                (
                    action for action in actions
                    if int(action["deck_index"]) not in {
                        int(value)
                        for value in by_side[int(action["side"])]["hand_deck_indices"]
                    }
                ),
                None,
            )
            if mismatch is not None:
                failure = f"hand_mismatch_event_{mismatch['source_event_index']}"
                break
            native_actions = [
                {
                    "type": "play",
                    "side": int(action["side"]),
                    "deck_index": int(action["deck_index"]),
                    "x": int(action["x"]),
                    "y": int(action["y"]),
                }
                for action in actions
            ]
            action_started = time.perf_counter()
            action_result = env.joint_act(native_actions)
            action_seconds += time.perf_counter() - action_started
            results = action_result.get("actions", [])
            if len(results) != len(actions):
                failure = (
                    f"native_action_count_mismatch_tick_{source_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"native_action_count_mismatch_execution_tick_{execution_tick}_"
                        f"source_tick_{source_tick}"
                    )
                )
                break
            rejected = [
                result for result in results
                if not bool(result.get("result", {}).get("accepted", False))
            ]
            accepted_actions += len(results) - len(rejected)
            if rejected:
                plan_by_event = {
                    item.source_event_index: item for item in plan.actions
                }
                event_evidence = []
                for action, native_result in zip(actions, results, strict=True):
                    result_value = native_result.get("result", {})
                    if bool(result_value.get("accepted", False)):
                        continue
                    source_action = plan_by_event[int(action["source_event_index"])]
                    player = by_side[int(action["side"])]
                    side = int(action["side"])
                    result_code = int(result_value.get("result_code", -1))
                    card_spec = plan.sides[side].deck[
                        source_action.logical_card_index
                    ]
                    catalog_card_cost_raw = int(
                        catalog()[card_spec.card_id]["elixir"]
                    ) * 10_000
                    native_resource = result_value.get("resource_before")
                    card_cost_raw = (
                        int(native_resource["card_cost"]) * 10_000
                        if isinstance(native_resource, Mapping)
                        and isinstance(native_resource.get("card_cost"), int)
                        else catalog_card_cost_raw
                    )
                    previous_source_tick = previous_source_tick_by_side[side]
                    previous_execution_tick = previous_execution_tick_by_side[side]
                    episode_snapshot = latest_state.get("episode", {})
                    event_evidence.append({
                        "source_event_index": int(action["source_event_index"]),
                        "source_tick": int(source_tick),
                        "execution_tick": int(execution_tick),
                        "execution_tick_offset": int(
                            action_execution_tick_offset
                        ),
                        "side": side,
                        "base_token": source_action.base_token,
                        "logical_card_index": source_action.logical_card_index,
                        "native_deck_index": int(action["deck_index"]),
                        "x": int(action["x"]),
                        "y": int(action["y"]),
                        "pre_action_elixir_raw": int(player.get("elixir_raw", -1)),
                        "native_card_cost_raw": card_cost_raw,
                        "catalog_wrapper_card_cost_raw": catalog_card_cost_raw,
                        "native_elixir_margin_raw": (
                            int(player.get("elixir_raw", -1)) - card_cost_raw
                        ),
                        "pre_action_hand_deck_indices": [
                            int(value) for value in player["hand_deck_indices"]
                        ],
                        "pre_action_next_deck_index": int(
                            player.get("next_deck_index", -1)
                        ),
                        "pre_action_refill_timer": int(
                            player.get("refill_timer", -1)
                        ),
                        "source_ticks_since_previous_side_action": (
                            None
                            if previous_source_tick is None
                            else source_tick - previous_source_tick
                        ),
                        "ticks_since_previous_side_action": (
                            None
                            if previous_source_tick is None
                            else source_tick - previous_source_tick
                        ),
                        "execution_ticks_since_previous_side_action": (
                            None
                            if previous_execution_tick is None
                            else execution_tick - previous_execution_tick
                        ),
                        "pre_action_entity_count": len(
                            latest_state.get("entities", [])
                        ),
                        "pre_action_episode": {
                            key: episode_snapshot.get(key)
                            for key in (
                                "terminated",
                                "truncated",
                                "outcome",
                                "winner",
                                "crowns",
                                "commands_allowed",
                                "command_gate_code",
                                "native_phase",
                                "terminal_tick",
                            )
                        },
                        "pre_action_crown_towers": [
                            {
                                key: tower.get(key)
                                for key in (
                                    "side", "type", "lane", "x", "y", "hp", "max_hp"
                                )
                            }
                            for tower in episode_snapshot.get("crown_towers", [])
                            if isinstance(tower, Mapping)
                        ],
                        "native_result": dict(result_value),
                        "result_code_classification": (
                            "insufficient_elixir_native_exact_tick"
                            if result_code == 13
                            else "battle_logic_predicate_d503d0"
                            if result_code == 4
                            else "unclassified_native_execute_code"
                        ),
                    })
                first_rejection = {
                    # ``tick`` remains the actual native execution Tick for
                    # existing diagnostics.  Source provenance is explicit.
                    "tick": execution_tick,
                    "source_tick": source_tick,
                    "execution_tick": execution_tick,
                    "execution_tick_offset": action_execution_tick_offset,
                    "events": event_evidence,
                    "result_codes": [
                        int(item.get("result", {}).get("result_code", -1))
                        for item in rejected
                    ],
                }
                failure = (
                    f"native_rejected_tick_{execution_tick}_codes_"
                    f"{first_rejection['result_codes']}"
                )
                break
            for action in actions:
                side = int(action["side"])
                previous_source_tick_by_side[side] = source_tick
                previous_execution_tick_by_side[side] = execution_tick

    teacher_forced_success = (
        failure is None and accepted_actions == len(plan.actions)
    )
    if teacher_forced_success:
        target_tick = max(current_tick, plan.duration_ticks + terminal_fence_ticks)
        advance_to(target_tick, allow_nonterminal_freeze=True)
        # A terminal after the final expert action is diagnostic, not a reason
        # to discard a trajectory whose actions and Tick stream are complete.
        if failure and failure.startswith("short_trace_") and terminal_seen:
            failure = None

    consecutive = False
    if states:
        try:
            require_consecutive(states)
            consecutive = True
        except ValueError:
            consecutive = False
    tick_count_expected = states[-1].tick - states[0].tick + 1 if states else 0
    every_tick_present = consecutive and len(states) == tick_count_expected
    episode = (
        terminal_episode
        if terminal_episode is not None
        else latest_state.get("episode", {})
    )
    observed_crowns: tuple[int, int] | None = None
    crowns = episode.get("crowns") if isinstance(episode, Mapping) else None
    if terminal_seen and isinstance(crowns, list) and len(crowns) == 2:
        observed_crowns = (int(crowns[0]), int(crowns[1]))
    if logic_freeze_before_execution is not None:
        terminal_status = NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK
    elif not teacher_forced_success:
        terminal_status = "not_evaluated_teacher_forced_failure"
    elif logic_frozen_at_fence:
        terminal_status = "logic_frozen_at_source_duration_fence"
    elif not terminal_seen or observed_crowns is None:
        terminal_status = "missing_at_source_duration_fence"
    elif plan.terminal_crowns is None:
        terminal_status = "observed_source_unknown"
    elif observed_crowns == plan.terminal_crowns:
        terminal_status = "match"
    else:
        terminal_status = "mismatch"

    # A normalization/trace defect invalidates the usable trajectory even if
    # every deployment happened to be accepted.
    usable = (
        teacher_forced_success
        and every_tick_present
        and failure is None
    )
    audit = {
        "schema_version": 1,
        "kind": "expert_native_deployment_trace_pilot_v1",
        "battle_tag": plan.battle_tag,
        "source_schema_version": plan.source_schema_version,
        "seed": seed_resolution.chosen_seed,
        **seed_resolution.audit(),
        "teacher_forced_success": teacher_forced_success,
        "usable_tick_trajectory": usable,
        "failure": failure,
        "failure_class": (
            NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK
            if logic_freeze_before_execution is not None
            else None
        ),
        "logic_freeze_diagnostic": logic_freeze_before_execution,
        "source_deployment_actions": len(plan.actions),
        "accepted_deployment_actions": accepted_actions,
        "first_rejection": first_rejection,
        "action_execution_tick_offset": action_execution_tick_offset,
        "action_tick_provenance": action_tick_provenance(
            action_execution_tick_offset
        ),
        "native_teacher_forced_profile": native_teacher_forced_profile(
            action_execution_tick_offset
        ),
        "reset_tick": states[0].tick if states else None,
        "last_source_action_tick": plan.actions[-1].tick if plan.actions else None,
        "last_execution_action_tick": (
            action_execution_tick(
                plan.actions[-1].tick, action_execution_tick_offset
            )
            if plan.actions else None
        ),
        "final_tick": states[-1].tick if states else None,
        "stored_tick_count": len(states),
        "expected_tick_count_from_bounds": tick_count_expected,
        "every_native_tick_present": every_tick_present,
        "complete_observation_frames": complete_frames,
        "incomplete_observation_frames": incomplete_frames,
        "incomplete_diagnostic_frames_not_stored": incomplete_frames,
        "invalid_incomplete_observation_frames": 0,
        "trace_rpc_count": trace_rpc_count,
        "native_ticks_advanced": (
            states[-1].tick - states[0].tick if states else 0
        ),
        "native_step_calls_advanced": trace_steps,
        "terminal_status": terminal_status,
        "terminal_seen": terminal_seen,
        "logic_frozen_at_source_duration_fence": logic_frozen_at_fence,
        "source_crowns": plan.terminal_crowns,
        "observed_crowns": observed_crowns,
        "seed_shuffle_layout_calibrated": True,
        "layout_calibration_attempts": 0,
        "logical_training_state_sha256": _logical_state_digest(states, mappings),
        "state_provenance": "native_teacher_forced_from_observed_actions",
        "action_provenance": plan.action_provenance,
        "ability_provenance": plan.ability_provenance,
        "timing_seconds": {
            "reset": reset_seconds,
            "trace": trace_seconds,
            "action": action_seconds,
            "normalize": normalize_seconds,
            "wall": time.perf_counter() - started,
        },
    }
    return TraceReplay(audit=audit, states=tuple(states))


__all__ = [
    "PilotTask",
    "TraceReplay",
    "action_execution_tick",
    "execute_deployment_trace",
    "load_json",
    "select_deployment_only_tasks",
    "sha256_file",
]
