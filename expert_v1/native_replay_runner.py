"""Execute compiled expert action plans against persistent native Workers.

The fast path records compact states only at expert decision ticks and stores
the exact wait interval as a label.  This avoids turning a 100k-battle corpus
into hundreds of millions of full-observation JSON objects.  A plan is always
fail-closed: the first hand mismatch, native rejection, premature terminal, or
tick mismatch ends that replay and preserves an audit record.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from native_core.env import NativeRoyaleEnv

from .native_capabilities import ability_cards, resolve_live_ability
from .native_freeze import (
    NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK,
    logic_freeze_audit,
    logic_freeze_failure,
)
from .native_profile import (
    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET,
    action_tick_provenance,
    native_teacher_forced_profile,
)
from .native_pilot import action_execution_tick
from .native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    grouped_replay_events,
)
from .native_seed_search import (
    DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    layouts_accept_plan,
    resolve_fixed_native_seed,
    resolve_native_seed,
)
from .tick_store_v1 import (
    NativeDeploymentMaskCapture,
    TickState,
    TickTraceAccumulator,
)
from .tick_store_v1.deployment_masks import EPISODE_METADATA_KEY


@dataclass(frozen=True)
class NativeReplayResult:
    battle_tag: str
    accepted: bool
    teacher_forced_success: bool
    failure: str | None
    source_actions: int
    accepted_actions: int
    source_deploy_actions: int
    accepted_deploy_actions: int
    source_ability_events: int
    accepted_ability_actions: int
    missing_ability_event_count: int
    ability_log_tier: str
    ability_replay_complete: bool
    numeric_game_mode_id: int | None
    numeric_game_mode_provenance: str
    native_execution_game_mode_id: int | None
    native_execution_game_mode_provenance: str
    king_tower_levels: tuple[int | None, int | None]
    king_tower_level_provenance: tuple[str, str]
    ability_resolution_counts: dict[str, int]
    ability_resolutions: tuple[dict[str, Any], ...]
    action_execution_tick_offset: int
    action_tick_provenance: str
    coordinate_provenance: str
    coordinate_audit: dict[str, Any]
    preferred_seed: int
    chosen_seed: int
    seeds_tested: int
    maximum_seeds_to_test: int
    seed_search_cache_hit: bool
    seed_search_native_resets: int
    source_seed_recovered: bool
    layout_resolution_mode: str
    final_tick: int
    native_ticks_advanced: int
    reset_seconds: float
    step_seconds: float
    observe_seconds: float
    action_seconds: float
    deployment_mask_probe_seconds: float
    deployment_mask_probe_rpc_count: int
    deployment_mask_base_probe_rpc_count: int
    deployment_mask_dynamic_label_probe_rpc_count: int
    deployment_mask_slots_captured: int
    deployment_mask_capture_complete: bool
    deployment_mask_metadata: dict[str, Any] | None
    deployment_mask_label_checks: int
    deployment_mask_label_rejections: int
    deployment_mask_first_label_rejection: dict[str, Any] | None
    wall_seconds: float
    terminal_validated: bool
    terminal_match: bool | None
    terminal_diagnostic_status: str
    source_crowns: tuple[int, int] | None
    observed_crowns: tuple[int, int] | None
    terminal_tower_hp_validated: bool
    terminal_tower_hp_match: bool | None
    terminal_tower_hp_diagnostic_status: str
    source_final_tower_hp: tuple[dict[str, Any], dict[str, Any]] | None
    observed_final_tower_hp: tuple[dict[str, Any], dict[str, Any]] | None
    decision_records: tuple[dict[str, Any], ...]
    action_acceptance_sequence: tuple[dict[str, Any], ...]
    tick_trace_batches: int
    tick_trace_complete_frames: int
    tick_trace_incomplete_terminal_frames: int
    tick_trace_incomplete_nonterminal_freeze_frames: int
    collected_tick_states: tuple[TickState, ...]
    logic_freeze_diagnostic: dict[str, Any] | None
    tick_store_entry: dict[str, Any] | None

    def json(self) -> dict[str, Any]:
        fields = dict(self.__dict__)
        collected_states = fields.pop("collected_tick_states")
        return {
            "schema_version": 1,
            "kind": "expert_native_replay_result_v1",
            **fields,
            "collected_tick_state_count": len(collected_states),
            "collected_tick_start": (
                None if not collected_states else collected_states[0].tick
            ),
            "collected_tick_stop": (
                None if not collected_states else collected_states[-1].tick
            ),
            "native_teacher_forced_profile": native_teacher_forced_profile(
                self.action_execution_tick_offset
            ),
            "decision_records": list(self.decision_records),
        }


def calibrated_players(
    env: NativeRoyaleEnv,
    template: Mapping[str, Any],
    *,
    seed: int = DEFAULT_NATIVE_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the libg shuffle permutation for one fixed seed/Worker."""
    replay = json.loads(json.dumps(template))
    replay["rndSeed"] = int(seed)
    state = env.reset(replay, warmup_steps=10)
    players = sorted(state["players"], key=lambda item: int(item["side"]))
    if len(players) != 2:
        raise RuntimeError("native calibration did not expose two players")
    for player in players:
        native_layout_order(player)
    return dict(players[0]), dict(players[1])


def _compact_decision_state(
    state: Mapping[str, Any],
    *,
    actor_side: int,
    source_tick: int,
    wait_ticks: int,
    expert_action: Mapping[str, int] | None,
) -> dict[str, Any]:
    """Return one public-information actor view.

    ``observe_train_v1`` exposes both players because self-play also builds a
    privileged critic.  Expert BC must never persist the opponent hand or
    exact opponent elixir, so this projection happens before any record is
    returned to a caller.
    """
    own = next(
        player for player in state.get("players", [])
        if int(player["side"]) == actor_side
    )
    own_player = {
        "side": actor_side,
        "elixir": int(own["elixir"]),
        "elixir_raw": int(own["elixir_raw"]),
        "hand_deck_indices": [int(value) for value in own["hand_deck_indices"]],
        # Older deployed hosts omitted these two fields from the compact
        # observation even though the full observation exposed them.  Keep
        # the runner compatible while the upgraded host is rolled out; -1 is
        # explicit "not observed", never a guessed cycle value.
        "next_deck_index": int(own.get("next_deck_index", -1)),
        "refill_timer": int(own.get("refill_timer", -1)),
    }
    entities = []
    for entity in state.get("entities", []):
        entities.append({
            "entity_id": int(entity.get("entity_id", entity.get("generation_key", -1))),
            "side": int(entity.get("side", -1)),
            "card_id": int(entity.get("card_id", 0)),
            "x": int(entity.get("x", 0)),
            "y": int(entity.get("y", 0)),
            "hp": int(entity.get("hp", 0)),
            "max_hp": int(entity.get("max_hp", 0)),
            "behavior_state": int(entity.get("behavior_state", 0)),
            "ability_available": bool(entity.get("ability_available", False)),
        })
    episode = state.get("episode") if isinstance(state.get("episode"), Mapping) else {}
    towers = [
        {
            "side": int(tower["side"]),
            "type": str(tower["type"]),
            "x": int(tower["x"]),
            "y": int(tower["y"]),
            "hp": int(tower["hp"]),
            "max_hp": int(tower["max_hp"]),
        }
        for tower in episode.get("crown_towers", [])
    ]
    return {
        "tick": int(state["tick"]),
        "actor_side": actor_side,
        "source_tick": int(source_tick),
        "wait_ticks_before": int(wait_ticks),
        # Hash is audit-only and is never part of the BC tensor whitelist.
        "audit_state_hash": str(state.get("state_hash", "")),
        "own_player": own_player,
        "entities": entities,
        "crown_towers": towers,
        "expert_action": None if expert_action is None else dict(expert_action),
    }


def _source_final_tower_hp(
    plan: BattlePlan,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    values = [side.final_tower_hp for side in plan.sides]
    if any(value is None for value in values):
        return None
    return tuple(asdict(value) for value in values)  # type: ignore[return-value]


def _observed_final_tower_hp(
    episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Project native tower HP without assigning Princess slots to lanes."""
    raw_towers = episode.get("crown_towers")
    if not isinstance(raw_towers, list):
        return None
    result: list[dict[str, Any]] = []
    for side in (0, 1):
        towers = [
            item for item in raw_towers
            if isinstance(item, Mapping) and int(item.get("side", -1)) == side
        ]
        kings = [
            item for item in towers
            if "king" in str(item.get("type") or "").lower()
        ]
        princesses = [
            item for item in towers
            if "princess" in str(item.get("type") or "").lower()
        ]
        if len(kings) != 1 or len(princesses) != 2:
            return None
        king = max(0, int(kings[0].get("hp", 0)))
        princess_hp = sorted(max(0, int(item.get("hp", 0))) for item in princesses)
        result.append({
            "king": king,
            "princess_multiset": princess_hp,
            "total": king + sum(princess_hp),
            "slot_mapping_provenance": "native_princess_lanes_compared_as_multiset",
        })
    return (result[0], result[1])


def _tower_hp_matches(
    source: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> bool:
    if len(source) != 2 or len(observed) != 2:
        return False
    for expected, actual in zip(source, observed, strict=True):
        expected_princess = sorted((
            int(expected["princess0"]), int(expected["princess1"]),
        ))
        if (
            int(expected["king"]) != int(actual["king"])
            or expected_princess
            != [int(item) for item in actual["princess_multiset"]]
            or int(expected["total"]) != int(actual["total"])
        ):
            return False
    return True


def _advance_native(
    env: NativeRoyaleEnv,
    steps: int,
    accumulator: TickTraceAccumulator | None,
    *,
    trace_batch_steps: int,
    allow_nonterminal_freeze: bool = False,
) -> dict[str, Any]:
    """Advance normally or capture every Tick through compact batch RPCs."""
    if steps <= 0:
        raise ValueError("native advance must be positive")
    if accumulator is None:
        return env.step(steps)
    remaining = steps
    stepped_total = 0
    final_state: Mapping[str, Any] | None = None
    last_complete_state: Mapping[str, Any] | None = None
    terminal = False
    nonterminal_freeze = False
    while remaining > 0 and not terminal:
        requested = min(remaining, trace_batch_steps)
        trace = (
            env.trace_train(requested, allow_nonterminal_freeze=True)
            if allow_nonterminal_freeze
            else env.trace_train(requested)
        )
        accumulator.extend(
            trace, allow_nonterminal_freeze=allow_nonterminal_freeze
        )
        stepped = int(trace["stepped"])
        if stepped <= 0 and not trace.get("terminal", False):
            raise RuntimeError("compact native Tick trace made no progress")
        stepped_total += stepped
        remaining -= stepped
        terminal = bool(trace.get("terminal", False))
        nonterminal_freeze = bool(trace.get("nonterminal_freeze", False))
        final_frame = (
            trace["frames"][-1]
            if trace["frames"]
            else trace["initial_frame"]
        )
        final_state = final_frame["state"]
        for frame in reversed([trace["initial_frame"], *trace["frames"]]):
            if frame.get("observation_complete") is True:
                last_complete_state = frame["state"]
                break
        if nonterminal_freeze:
            break
    if final_state is None:
        raise RuntimeError("compact native Tick trace returned no state")
    return {
        "requested_steps": steps,
        "stepped": stepped_total,
        "battle_active": not terminal,
        "nonterminal_freeze": nonterminal_freeze,
        "fixed_dt": 0.05,
        "tick_after": int(final_state["tick"]),
        "episode": dict(final_state["episode"]),
        "state": dict(final_state),
        "last_complete_state": (
            None if last_complete_state is None else dict(last_complete_state)
        ),
    }


def execute_plan(
    env: NativeRoyaleEnv,
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]] | None = None,
    *,
    seed: int = DEFAULT_NATIVE_SEED,
    fixed_seed: int | None = None,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    warmup_tick: int = 10,
    capture_decisions: bool = True,
    ability_branch_choices: Mapping[int, int] | None = None,
    tick_sink: Any | None = None,
    tick_store_metadata: Mapping[str, Any] | None = None,
    trace_batch_steps: int = 64,
    capture_deployment_masks: bool = False,
    action_execution_tick_offset: int = (
        ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
    ),
) -> NativeReplayResult:
    """Replay one battle with gap-batched native stepping."""
    if tick_sink is not None and not 1 <= trace_batch_steps <= 64:
        raise ValueError("trace_batch_steps must be in 1..64")
    # Validate the experimental execution boundary even when the source plan
    # happens to be empty.  Source labels are immutable RoyaleAPI ``time_raw``
    # values; only the native command boundary may move by the audited offset.
    action_execution_tick(0, action_execution_tick_offset)
    action_tick_provenance_value = action_tick_provenance(
        action_execution_tick_offset
    )
    coordinate_audit = asdict(plan.coordinate_audit)
    started = time.perf_counter()
    reset_seconds = step_seconds = observe_seconds = action_seconds = 0.0
    deployment_mask_probe_seconds = 0.0
    native_ticks = accepted_actions = 0
    accepted_deploy_actions = accepted_ability_actions = 0
    ability_resolution_counts: Counter[str] = Counter()
    ability_resolutions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    action_acceptance_sequence: list[dict[str, Any]] = []
    failure: str | None = None
    final_tick = 0
    terminal_validated = False
    terminal_match: bool | None = None
    observed_crowns: tuple[int, int] | None = None
    source_final_tower_hp = _source_final_tower_hp(plan)
    observed_final_tower_hp: tuple[dict[str, Any], dict[str, Any]] | None = None
    terminal_tower_hp_validated = False
    terminal_tower_hp_match: bool | None = None
    tick_store_entry: dict[str, Any] | None = None
    deployment_mask_capture: NativeDeploymentMaskCapture | None = None
    deployment_mask_metadata: dict[str, Any] | None = None
    deployment_mask_label_checks = 0
    deployment_mask_label_rejections = 0
    deployment_mask_first_label_rejection: dict[str, Any] | None = None
    logic_freeze_diagnostic: dict[str, Any] | None = None
    tick_accumulator = TickTraceAccumulator() if tick_sink is not None else None
    del calibration
    allowed_abilities = [ability_cards(side.deck) for side in plan.sides]
    missing_ability_events = sum(
        side.missing_ability_event_count for side in plan.sides
    )
    first_exact_ticks: list[int | None] = []
    for side, side_plan in enumerate(plan.sides):
        exact_index = side_plan.cycle.first_exact_action_index
        side_actions = [action for action in plan.actions if action.side == side]
        first_exact_ticks.append(
            None if exact_index is None else side_actions[exact_index].tick
        )
    reset_started = time.perf_counter()
    seed_resolution = (
        resolve_native_seed(
            env,
            plan,
            template,
            preferred_seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            warmup_tick=warmup_tick,
        )
        if fixed_seed is None
        else resolve_fixed_native_seed(
            env,
            plan,
            template,
            chosen_seed=fixed_seed,
            warmup_tick=warmup_tick,
        )
    )
    mappings = seed_resolution.mappings
    state = seed_resolution.state
    reset_seconds += time.perf_counter() - reset_started
    final_tick = int(state["tick"])

    # Revalidate the authoritative source-order layout at the execution
    # boundary.  No deck slot or action Tick may be rewritten to make it pass.
    players = sorted(state["players"], key=lambda item: int(item["side"]))
    compatible_layouts = layouts_accept_plan(plan, players)
    if not all(compatible_layouts):
        mismatched = [
            side for side, compatible in enumerate(compatible_layouts)
            if not compatible
        ]
        failure = "native_seed_search_layout_revalidation_failed_sides_" + "_".join(
            str(side) for side in mismatched
        )
    if failure is None and missing_ability_events:
        failure = f"source_ability_ticks_missing_count_{missing_ability_events}"

    # Compile-time validation owns source semantics: a side may issue only one
    # deployment/ability command at a source Tick.  Re-check both source and
    # mapped execution keys here so the optional shift can never manufacture
    # a same-side native-Tick collision.
    replay_groups: list[tuple[int, int, list[dict[str, Any]]]] = []
    source_occupancy: set[tuple[int, int]] = set()
    execution_occupancy: set[tuple[int, int]] = set()
    for source_tick, events in grouped_replay_events(plan, mappings):
        execution_tick = action_execution_tick(
            source_tick, action_execution_tick_offset
        )
        for event in events:
            side = int(event["side"])
            source_key = (int(source_tick), side)
            execution_key = (execution_tick, side)
            if source_key in source_occupancy:
                raise ValueError(
                    "source plan contains a same-side deploy/ability conflict "
                    f"at Tick {source_tick}"
                )
            if execution_key in execution_occupancy:
                raise ValueError(
                    "action execution offset created a same-side conflict at "
                    f"Tick {execution_tick}"
                )
            source_occupancy.add(source_key)
            execution_occupancy.add(execution_key)
        replay_groups.append((int(source_tick), execution_tick, events))

    # Do not pay trace or validator-probe costs until the fixed reset has
    # passed layout/source preconditions.  In particular, a nondeterministic
    # fixed-seed layout mismatch must leave no partial trace/mask capture.
    if failure is None and capture_deployment_masks:
        deployment_mask_capture = NativeDeploymentMaskCapture([
            {
                "side": side_index,
                "deck_index": deck_index,
                "card_id": card.card_id,
                "level": card.level,
                "form_flags": card.form_flags,
                "source_token": card.source_token,
                "base_token": card.base_token,
            }
            for side_index, side_plan in enumerate(plan.sides)
            for deck_index, card in enumerate(side_plan.deck)
        ])
        mask_started = time.perf_counter()
        deployment_mask_capture.capture_available(env, state)
        deployment_mask_probe_seconds += time.perf_counter() - mask_started
    if failure is None and tick_accumulator is not None:
        observe_started = time.perf_counter()
        tick_accumulator.start(env.observe_train())
        observe_seconds += time.perf_counter() - observe_started

    previous_source_tick = final_tick
    previous_execution_tick = final_tick
    if failure is None:
        for source_tick, execution_tick, events in replay_groups:
            if execution_tick < final_tick:
                failure = (
                    f"source_tick_{source_tick}_precedes_native_tick_{final_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"execution_tick_{execution_tick}_source_tick_{source_tick}_"
                        f"precedes_native_tick_{final_tick}"
                    )
                )
                break
            gap = execution_tick - final_tick
            if gap:
                tick_before_advance = final_tick
                step_started = time.perf_counter()
                native_step = _advance_native(
                    env, gap, tick_accumulator,
                    trace_batch_steps=trace_batch_steps,
                    allow_nonterminal_freeze=(tick_accumulator is not None),
                )
                step_seconds += time.perf_counter() - step_started
                episode = native_step.get("episode", {})
                final_tick = int(native_step.get("tick_after", execution_tick))
                native_ticks += max(0, final_tick - tick_before_advance)
                nonterminal_freeze = bool(
                    native_step.get("nonterminal_freeze", False)
                ) or (
                    final_tick < execution_tick
                    and not episode.get("terminated")
                    and not episode.get("truncated")
                )
                if nonterminal_freeze:
                    frozen_state = native_step.get("state")
                    if not isinstance(frozen_state, Mapping):
                        frozen_state = {
                            "tick": final_tick,
                            "episode": episode,
                        }
                    collected_states = (
                        ()
                        if tick_accumulator is None
                        else tuple(tick_accumulator.states)
                    )
                    logic_freeze_diagnostic = logic_freeze_audit(
                        frozen_state,
                        fallback_state=(
                            native_step.get("last_complete_state")
                            if isinstance(
                                native_step.get("last_complete_state"), Mapping
                            )
                            else None
                        ),
                        source_tick=source_tick,
                        execution_tick=execution_tick,
                        execution_tick_offset=action_execution_tick_offset,
                        chosen_seed=seed_resolution.chosen_seed,
                        source_actions=(
                            len(plan.actions) + len(plan.ability_events)
                        ),
                        accepted_actions=accepted_actions,
                        collected_tick_count=len(collected_states),
                        collected_tick_start=(
                            None
                            if not collected_states
                            else collected_states[0].tick
                        ),
                        collected_tick_stop=(
                            None
                            if not collected_states
                            else collected_states[-1].tick
                        ),
                        trace_requested_steps=gap,
                        trace_stepped_calls=int(
                            native_step.get("stepped", 0)
                        ),
                    )
                    failure = logic_freeze_failure(
                        source_tick=source_tick,
                        execution_tick=execution_tick,
                        last_native_tick=final_tick,
                    )
                    break
                if final_tick != execution_tick:
                    failure = (
                        f"native_tick_mismatch_{final_tick}_expected_{execution_tick}"
                        if action_execution_tick_offset == 0
                        else (
                            f"native_tick_mismatch_{final_tick}_expected_execution_"
                            f"tick_{execution_tick}_source_tick_{source_tick}"
                        )
                    )
                    break
                if episode.get("terminated") or episode.get("truncated"):
                    failure = (
                        f"native_terminal_before_source_tick_{source_tick}"
                        if action_execution_tick_offset == 0
                        else (
                            f"native_terminal_before_execution_tick_{execution_tick}_"
                            f"source_tick_{source_tick}"
                        )
                    )
                    break

            observe_started = time.perf_counter()
            state = env.observe_train()
            observe_seconds += time.perf_counter() - observe_started
            if int(state["tick"]) != execution_tick:
                failure = (
                    f"observation_tick_{state['tick']}_expected_{execution_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"observation_tick_{state['tick']}_expected_execution_tick_"
                        f"{execution_tick}_source_tick_{source_tick}"
                    )
                )
                break
            if deployment_mask_capture is not None:
                mask_started = time.perf_counter()
                deployment_mask_capture.capture_available(env, state)
                deployment_mask_capture.capture_label_variants(
                    env,
                    state,
                    (event for event in events if event["type"] == "play"),
                )
                for event in events:
                    if event["type"] != "play":
                        continue
                    label_audit = deployment_mask_capture.audit_label_position(
                        state, event
                    )
                    deployment_mask_label_checks += 1
                    if not label_audit["legal"]:
                        deployment_mask_label_rejections += 1
                        deployment_mask_first_label_rejection = label_audit
                        failure = (
                            "derived_deployment_mask_rejected_source_event_"
                            f"{event['source_event_index']}"
                        )
                        break
                deployment_mask_probe_seconds += time.perf_counter() - mask_started
            if failure is not None:
                break
            by_side = {int(player["side"]): player for player in state["players"]}
            for event in events:
                if event["type"] != "play":
                    continue
                if int(event["deck_index"]) not in {
                    int(value)
                    for value in by_side[int(event["side"])]["hand_deck_indices"]
                }:
                    failure = (
                        f"hand_mismatch_event_{event['source_event_index']}"
                    )
                    break
            if failure is not None:
                break
            native_actions: list[dict[str, Any]] = []
            resolved_events: list[dict[str, Any]] = []
            for event in events:
                if event["type"] == "play":
                    native_action = {
                        key: int(event[key])
                        for key in ("side", "deck_index", "x", "y")
                    } | {"type": "play"}
                    native_actions.append(native_action)
                    resolved_events.append(event)
                    continue
                side = int(event["side"])
                marker = int(event["source_marker_index"])
                resolution = resolve_live_ability(
                    state, side=side, tick=execution_tick,
                    allowed_cards=allowed_abilities[side],
                )
                resolution_row = {
                    **resolution.json(),
                    "source_tick": int(source_tick),
                    "execution_tick": int(execution_tick),
                    "execution_tick_offset": int(action_execution_tick_offset),
                    "action_tick_provenance": action_tick_provenance_value,
                    "source_event_index": int(event["source_event_index"]),
                    "source_marker_index": marker,
                    "source_ability_id": event.get("source_ability_id"),
                    "selected_entity_id": None,
                    "execution": "not_executed",
                }
                ability_resolution_counts[resolution.status] += 1
                selected_entity: int | None = None
                if resolution.status == "unique":
                    selected_entity = resolution.candidate_entity_ids[0]
                    resolution_row["execution"] = "unique_executed"
                elif resolution.status == "branch_required":
                    explicit = (
                        None if ability_branch_choices is None
                        else ability_branch_choices.get(marker)
                    )
                    if explicit is None:
                        resolution_row["execution"] = "branch_required_unselected"
                        ability_resolutions.append(resolution_row)
                        failure = (
                            f"ability_branch_required_marker_{marker}_candidates_"
                            f"{list(resolution.candidate_entity_ids)}"
                        )
                        break
                    selected_entity = int(explicit)
                    if selected_entity not in resolution.candidate_entity_ids:
                        resolution_row["selected_entity_id"] = selected_entity
                        resolution_row["execution"] = "invalid_explicit_branch"
                        ability_resolutions.append(resolution_row)
                        failure = (
                            f"ability_branch_choice_invalid_marker_{marker}_entity_"
                            f"{selected_entity}"
                        )
                        break
                    resolution_row["execution"] = "explicit_branch_executed"
                    ability_resolution_counts["explicit_branch_selected"] += 1
                else:
                    resolution_row["execution"] = resolution.status
                    ability_resolutions.append(resolution_row)
                    failure = f"ability_{resolution.status}_marker_{marker}"
                    break
                resolution_row["selected_entity_id"] = selected_entity
                ability_resolutions.append(resolution_row)
                native_actions.append({
                    "type": "ability", "side": side,
                    "entity_id": int(selected_entity),
                })
                resolved_events.append(event | {
                    "type": "ability",
                    "entity_id": int(selected_entity),
                    "ability_resolution": resolution.status,
                    "source_tick": int(source_tick),
                    "execution_tick": int(execution_tick),
                    "execution_tick_offset": int(action_execution_tick_offset),
                })
            if failure is not None:
                break
            if capture_decisions:
                action_by_side = {
                    int(event["side"]): event for event in resolved_events
                }
                for actor_side in (0, 1):
                    record = _compact_decision_state(
                        state,
                        actor_side=actor_side,
                        source_tick=source_tick,
                        wait_ticks=source_tick - previous_source_tick,
                        expert_action=action_by_side.get(actor_side),
                    )
                    record.update({
                        "execution_tick": int(execution_tick),
                        "execution_tick_offset": int(action_execution_tick_offset),
                        "action_tick_provenance": action_tick_provenance_value,
                        "source_wait_ticks_before": source_tick - previous_source_tick,
                        "execution_wait_ticks_before": (
                            execution_tick - previous_execution_tick
                        ),
                        "state_provenance": plan.state_provenance,
                        "action_provenance": plan.action_provenance,
                        "coordinate_provenance": plan.coordinate_provenance,
                        "coordinate_audit": coordinate_audit,
                        "hand_provenance": (
                            "inferred_exact"
                            if first_exact_ticks[actor_side] is not None
                            and source_tick >= int(first_exact_ticks[actor_side])
                            else "inferred_ambiguous_prefix"
                        ),
                        "ability_provenance": plan.ability_provenance,
                        "terminal_provenance": plan.terminal_provenance,
                        "numeric_game_mode_id": plan.numeric_game_mode_id,
                        "numeric_game_mode_provenance": (
                            plan.numeric_game_mode_provenance
                        ),
                        "native_execution_game_mode_id": (
                            plan.native_execution_game_mode_id
                        ),
                        "native_execution_game_mode_provenance": (
                            plan.native_execution_game_mode_provenance
                        ),
                        "king_tower_levels": tuple(
                            side.king_tower_level for side in plan.sides
                        ),
                        "king_tower_level_provenance": tuple(
                            side.king_tower_level_provenance for side in plan.sides
                        ),
                        "battle_index": plan.battle_index,
                        "battle_index_provenance": plan.battle_index_provenance,
                        "authoritative_contract_sha256": (
                            plan.authoritative_contract_sha256
                        ),
                    })
                    records.append(record)
            action_started = time.perf_counter()
            action_result = env.joint_act(native_actions)
            action_seconds += time.perf_counter() - action_started
            results = action_result.get("actions", [])
            # Keep a compact, phase-independent acceptance transcript.  The
            # two-stage dataset generator compares this byte-for-byte between
            # the cheap preflight and the fully instrumented replay.
            for index, (native_action, source_event) in enumerate(zip(
                native_actions, resolved_events, strict=True
            )):
                item = results[index] if index < len(results) else None
                native_result = (
                    item.get("result", {})
                    if isinstance(item, Mapping)
                    and isinstance(item.get("result"), Mapping)
                    else {}
                )
                action_acceptance_sequence.append({
                    "source_tick": int(source_tick),
                    "execution_tick": int(execution_tick),
                    "source_event_index": int(source_event["source_event_index"]),
                    "type": str(native_action["type"]),
                    "side": int(native_action["side"]),
                    "accepted": (
                        None if item is None
                        else bool(native_result.get("accepted", False))
                    ),
                    "result_code": (
                        None
                        if item is None or "result_code" not in native_result
                        else int(native_result["result_code"])
                    ),
                })
            if len(results) != len(native_actions):
                failure = (
                    f"native_action_count_mismatch_tick_{source_tick}"
                    if action_execution_tick_offset == 0
                    else (
                        f"native_action_count_mismatch_execution_tick_"
                        f"{execution_tick}_source_tick_{source_tick}"
                    )
                )
                break
            rejected = [
                item for item in results
                if not bool(item.get("result", {}).get("accepted", False))
            ]
            for source_event, item in zip(
                resolved_events, results, strict=True
            ):
                if bool(item.get("result", {}).get("accepted", False)):
                    accepted_actions += 1
                    if source_event["type"] == "ability":
                        accepted_ability_actions += 1
                    else:
                        accepted_deploy_actions += 1
            if rejected:
                codes = [
                    int(item.get("result", {}).get("result_code", -1))
                    for item in rejected
                ]
                failure = (
                    f"native_rejected_tick_{source_tick}_codes_{codes}"
                    if action_execution_tick_offset == 0
                    else (
                        f"native_rejected_tick_{execution_tick}_source_tick_"
                        f"{source_tick}_codes_{codes}"
                    )
                )
                break
            previous_source_tick = source_tick
            previous_execution_tick = execution_tick

    # The last accepted deployment can reveal one new hand slot.  One final
    # compact observation is enough to collect it; all earlier slots were
    # captured at their normal decision observations.  Never poll per Tick.
    if (
        failure is None
        and deployment_mask_capture is not None
        and not deployment_mask_capture.complete
    ):
        observe_started = time.perf_counter()
        state = env.observe_train()
        observe_seconds += time.perf_counter() - observe_started
        mask_started = time.perf_counter()
        deployment_mask_capture.capture_available(env, state)
        deployment_mask_probe_seconds += time.perf_counter() - mask_started
        if not deployment_mask_capture.complete:
            failure = (
                "native_deployment_mask_capture_incomplete_slots_"
                + "_".join(
                    f"{side}-{deck}"
                    for side, deck in deployment_mask_capture.missing_slots
                )
            )

    if failure is None and plan.terminal_crowns is not None:
        # Duration is stored at one-second resolution.  A 20-Tick fence lets
        # libg emit the terminal object without accepting an unbounded run.
        remaining = max(1, plan.duration_ticks + 20 - final_tick)
        step_started = time.perf_counter()
        final_step = _advance_native(
            env, remaining, tick_accumulator,
            trace_batch_steps=trace_batch_steps,
        )
        step_seconds += time.perf_counter() - step_started
        native_ticks += max(0, int(final_step.get("stepped", remaining)))
        final_tick = int(final_step.get("tick_after", final_tick + remaining))
        episode = final_step.get("episode", {})
        crowns = episode.get("crowns")
        if (
            (episode.get("terminated") or episode.get("truncated"))
            and isinstance(crowns, list)
            and len(crowns) == 2
        ):
            observed_crowns = (int(crowns[0]), int(crowns[1]))
            terminal_validated = True
            terminal_match = observed_crowns == plan.terminal_crowns
            if source_final_tower_hp is not None:
                observed_final_tower_hp = _observed_final_tower_hp(episode)
                terminal_tower_hp_validated = observed_final_tower_hp is not None
                terminal_tower_hp_match = (
                    False
                    if observed_final_tower_hp is None
                    else _tower_hp_matches(
                        source_final_tower_hp, observed_final_tower_hp
                    )
                )
        else:
            terminal_match = False

    teacher_forced_success = (
        failure is None
        and accepted_actions == len(plan.actions) + len(plan.ability_events)
        and missing_ability_events == 0
    )
    if teacher_forced_success and tick_sink is not None:
        assert tick_accumulator is not None
        if deployment_mask_capture is not None:
            deployment_mask_metadata = deployment_mask_capture.metadata()
        metadata = {
            **tick_accumulator.metadata(),
            **dict(tick_store_metadata or {}),
            **seed_resolution.audit(),
            "seed": seed_resolution.chosen_seed,
            "state_provenance": plan.state_provenance,
            "action_provenance": plan.action_provenance,
            "action_execution_tick_offset": action_execution_tick_offset,
            "action_tick_provenance": action_tick_provenance_value,
            "native_teacher_forced_profile": native_teacher_forced_profile(
                action_execution_tick_offset
            ),
            "coordinate_provenance": plan.coordinate_provenance,
            "coordinate_audit": coordinate_audit,
            "ability_provenance": plan.ability_provenance,
            "terminal_provenance": plan.terminal_provenance,
            "numeric_game_mode_id": plan.numeric_game_mode_id,
            "numeric_game_mode_provenance": plan.numeric_game_mode_provenance,
            "native_execution_game_mode_id": (
                plan.native_execution_game_mode_id
            ),
            "native_execution_game_mode_provenance": (
                plan.native_execution_game_mode_provenance
            ),
            "king_tower_levels": tuple(
                side.king_tower_level for side in plan.sides
            ),
            "king_tower_level_provenance": tuple(
                side.king_tower_level_provenance for side in plan.sides
            ),
            "battle_index": plan.battle_index,
            "battle_index_provenance": plan.battle_index_provenance,
            "authoritative_contract_game_version": (
                plan.authoritative_contract_game_version
            ),
            "authoritative_contract_sha256": plan.authoritative_contract_sha256,
            "authoritative_contract_file_sha256": (
                plan.authoritative_contract_file_sha256
            ),
            "source_final_tower_hp": source_final_tower_hp,
            "source_actions": len(plan.actions) + len(plan.ability_events),
            "deployment_mask_label_checks": deployment_mask_label_checks,
            "deployment_mask_label_rejections": (
                deployment_mask_label_rejections
            ),
            "deployment_mask_label_audit": (
                "captured_native_base_plus_tick_tower_projection_v1"
            ),
        }
        if deployment_mask_metadata is not None:
            metadata[EPISODE_METADATA_KEY] = deployment_mask_metadata
        try:
            if deployment_mask_capture is not None:
                stage_masks = getattr(
                    tick_sink, "stage_deployment_masks", None
                )
                if not callable(stage_masks):
                    raise RuntimeError(
                        "Tick sink cannot stage native deployment masks"
                    )
                stage_masks(deployment_mask_capture)
            tick_store_entry = dict(tick_sink.append(
                plan.battle_tag, tick_accumulator.states, metadata
            ))
        except Exception as error:
            failure = f"tick_store_write_{type(error).__name__}:{error}"
            teacher_forced_success = False
    if (
        deployment_mask_capture is not None
        and deployment_mask_metadata is None
    ):
        deployment_mask_metadata = deployment_mask_capture.metadata(
            require_complete=False
        )
    if logic_freeze_diagnostic is not None:
        terminal_diagnostic_status = (
            NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK
        )
    elif plan.terminal_crowns is None:
        terminal_diagnostic_status = "not_requested"
    elif terminal_validated and terminal_match:
        terminal_diagnostic_status = "match"
    elif terminal_validated:
        terminal_diagnostic_status = "crowns_mismatch"
    else:
        terminal_diagnostic_status = "native_terminal_missing"
    if source_final_tower_hp is None:
        terminal_tower_hp_diagnostic_status = "not_requested"
    elif terminal_tower_hp_validated and terminal_tower_hp_match:
        terminal_tower_hp_diagnostic_status = "match_unmapped_princess_slots"
    elif terminal_tower_hp_validated:
        terminal_tower_hp_diagnostic_status = "tower_hp_mismatch"
    else:
        terminal_tower_hp_diagnostic_status = "native_terminal_tower_hp_missing"

    return NativeReplayResult(
        battle_tag=plan.battle_tag,
        accepted=teacher_forced_success,
        teacher_forced_success=teacher_forced_success,
        failure=failure,
        source_actions=len(plan.actions) + len(plan.ability_events),
        accepted_actions=accepted_actions,
        source_deploy_actions=len(plan.actions),
        accepted_deploy_actions=accepted_deploy_actions,
        source_ability_events=len(plan.ability_events),
        accepted_ability_actions=accepted_ability_actions,
        missing_ability_event_count=missing_ability_events,
        ability_log_tier=plan.ability_log_tier,
        ability_replay_complete=(
            missing_ability_events == 0
            and accepted_ability_actions == len(plan.ability_events)
        ),
        numeric_game_mode_id=plan.numeric_game_mode_id,
        numeric_game_mode_provenance=plan.numeric_game_mode_provenance,
        native_execution_game_mode_id=plan.native_execution_game_mode_id,
        native_execution_game_mode_provenance=(
            plan.native_execution_game_mode_provenance
        ),
        king_tower_levels=(
            plan.sides[0].king_tower_level,
            plan.sides[1].king_tower_level,
        ),
        king_tower_level_provenance=(
            plan.sides[0].king_tower_level_provenance,
            plan.sides[1].king_tower_level_provenance,
        ),
        ability_resolution_counts=dict(ability_resolution_counts),
        ability_resolutions=tuple(ability_resolutions),
        action_execution_tick_offset=action_execution_tick_offset,
        action_tick_provenance=action_tick_provenance_value,
        coordinate_provenance=plan.coordinate_provenance,
        coordinate_audit=coordinate_audit,
        preferred_seed=seed_resolution.preferred_seed,
        chosen_seed=seed_resolution.chosen_seed,
        seeds_tested=seed_resolution.seeds_tested,
        maximum_seeds_to_test=seed_resolution.maximum_seeds_to_test,
        seed_search_cache_hit=seed_resolution.cache_hit,
        seed_search_native_resets=seed_resolution.native_resets,
        source_seed_recovered=seed_resolution.source_seed_recovered,
        layout_resolution_mode=seed_resolution.resolution_mode,
        final_tick=final_tick,
        native_ticks_advanced=native_ticks,
        reset_seconds=reset_seconds,
        step_seconds=step_seconds,
        observe_seconds=observe_seconds,
        action_seconds=action_seconds,
        deployment_mask_probe_seconds=deployment_mask_probe_seconds,
        deployment_mask_probe_rpc_count=(
            0
            if deployment_mask_capture is None
            else deployment_mask_capture.probe_rpc_count
        ),
        deployment_mask_base_probe_rpc_count=(
            0
            if deployment_mask_capture is None
            else deployment_mask_capture.captured_slots
        ),
        deployment_mask_dynamic_label_probe_rpc_count=(
            0
            if deployment_mask_capture is None
            else deployment_mask_capture.dynamic_label_probe_rpc_count
        ),
        deployment_mask_slots_captured=(
            0
            if deployment_mask_capture is None
            else deployment_mask_capture.captured_slots
        ),
        deployment_mask_capture_complete=(
            False
            if deployment_mask_capture is None
            else deployment_mask_capture.complete
        ),
        deployment_mask_metadata=deployment_mask_metadata,
        deployment_mask_label_checks=deployment_mask_label_checks,
        deployment_mask_label_rejections=deployment_mask_label_rejections,
        deployment_mask_first_label_rejection=(
            deployment_mask_first_label_rejection
        ),
        wall_seconds=time.perf_counter() - started,
        terminal_validated=terminal_validated,
        terminal_match=terminal_match,
        terminal_diagnostic_status=terminal_diagnostic_status,
        source_crowns=plan.terminal_crowns,
        observed_crowns=observed_crowns,
        terminal_tower_hp_validated=terminal_tower_hp_validated,
        terminal_tower_hp_match=terminal_tower_hp_match,
        terminal_tower_hp_diagnostic_status=(
            terminal_tower_hp_diagnostic_status
        ),
        source_final_tower_hp=source_final_tower_hp,
        observed_final_tower_hp=observed_final_tower_hp,
        decision_records=tuple(records),
        action_acceptance_sequence=tuple(action_acceptance_sequence),
        tick_trace_batches=(
            0 if tick_accumulator is None else tick_accumulator.batches
        ),
        tick_trace_complete_frames=(
            0 if tick_accumulator is None
            else tick_accumulator.complete_frames
        ),
        tick_trace_incomplete_terminal_frames=(
            0 if tick_accumulator is None
            else tick_accumulator.incomplete_terminal_frames
        ),
        tick_trace_incomplete_nonterminal_freeze_frames=(
            0
            if tick_accumulator is None
            else tick_accumulator.incomplete_nonterminal_freeze_frames
        ),
        collected_tick_states=(
            () if tick_accumulator is None else tuple(tick_accumulator.states)
        ),
        logic_freeze_diagnostic=logic_freeze_diagnostic,
        tick_store_entry=tick_store_entry,
    )


def load_template(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("battle"), dict):
        raise TypeError("native template must contain battle")
    return value
