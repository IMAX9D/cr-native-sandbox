"""Execute compiled expert action plans against persistent native Workers.

The fast path records compact states only at expert decision ticks and stores
the exact wait interval as a label.  This avoids turning a 100k-battle corpus
into hundreds of millions of full-observation JSON objects.  A plan is always
fail-closed: the first hand mismatch, native rejection, premature terminal, or
tick mismatch ends that replay and preserves an audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from native_core.env import NativeRoyaleEnv

from .native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    grouped_actions,
    materialize_replay,
    native_layout_order,
)


@dataclass(frozen=True)
class NativeReplayResult:
    battle_tag: str
    accepted: bool
    failure: str | None
    source_actions: int
    accepted_actions: int
    final_tick: int
    native_ticks_advanced: int
    reset_seconds: float
    step_seconds: float
    observe_seconds: float
    action_seconds: float
    wall_seconds: float
    terminal_validated: bool
    terminal_match: bool | None
    source_crowns: tuple[int, int] | None
    observed_crowns: tuple[int, int] | None
    decision_records: tuple[dict[str, Any], ...]

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "expert_native_replay_result_v1",
            **self.__dict__,
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
        "next_deck_index": int(own["next_deck_index"]),
        "refill_timer": int(own["refill_timer"]),
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


def execute_plan(
    env: NativeRoyaleEnv,
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_NATIVE_SEED,
    capture_decisions: bool = True,
) -> NativeReplayResult:
    """Replay one battle with gap-batched native stepping."""
    started = time.perf_counter()
    reset_seconds = step_seconds = observe_seconds = action_seconds = 0.0
    native_ticks = accepted_actions = 0
    records: list[dict[str, Any]] = []
    failure: str | None = None
    final_tick = 0
    terminal_validated = False
    terminal_match: bool | None = None
    observed_crowns: tuple[int, int] | None = None
    replay, mappings = materialize_replay(
        plan, template, calibration, seed=seed
    )
    reset_started = time.perf_counter()
    state = env.reset(replay, warmup_steps=10)
    reset_seconds += time.perf_counter() - reset_started
    final_tick = int(state["tick"])

    # Verify that card identities did not perturb the calibrated shuffle.
    players = sorted(state["players"], key=lambda item: int(item["side"]))
    for side, player in enumerate(players):
        desired_native_order = tuple(
            mappings[side][logical]
            for logical in (
                plan.sides[side].cycle.initial_hand
                + plan.sides[side].cycle.initial_queue
            )
        )
        if native_layout_order(player) != desired_native_order:
            failure = f"native_shuffle_layout_changed_side_{side}"
            break

    previous_source_tick = final_tick
    if failure is None:
        for source_tick, actions in grouped_actions(plan, mappings):
            if source_tick < final_tick:
                failure = (
                    f"source_tick_{source_tick}_precedes_native_tick_{final_tick}"
                )
                break
            gap = source_tick - final_tick
            if gap:
                step_started = time.perf_counter()
                native_step = env.step(gap)
                step_seconds += time.perf_counter() - step_started
                native_ticks += gap
                episode = native_step.get("episode", {})
                final_tick = int(native_step.get("tick_after", source_tick))
                if final_tick != source_tick:
                    failure = f"native_tick_mismatch_{final_tick}_expected_{source_tick}"
                    break
                if episode.get("terminated") or episode.get("truncated"):
                    failure = f"native_terminal_before_source_tick_{source_tick}"
                    break

            observe_started = time.perf_counter()
            state = env.observe_train()
            observe_seconds += time.perf_counter() - observe_started
            if int(state["tick"]) != source_tick:
                failure = f"observation_tick_{state['tick']}_expected_{source_tick}"
                break
            by_side = {int(player["side"]): player for player in state["players"]}
            for action in actions:
                if int(action["deck_index"]) not in {
                    int(value) for value in by_side[int(action["side"])]["hand_deck_indices"]
                }:
                    failure = (
                        f"hand_mismatch_event_{action['source_event_index']}"
                    )
                    break
            if failure is not None:
                break
            if capture_decisions:
                action_by_side = {int(action["side"]): action for action in actions}
                for actor_side in (0, 1):
                    record = _compact_decision_state(
                        state,
                        actor_side=actor_side,
                        source_tick=source_tick,
                        wait_ticks=source_tick - previous_source_tick,
                        expert_action=action_by_side.get(actor_side),
                    )
                    record.update({
                        "state_provenance": plan.state_provenance,
                        "action_provenance": plan.action_provenance,
                        "hand_provenance": plan.hand_provenance,
                        "ability_provenance": plan.ability_provenance,
                        "terminal_provenance": plan.terminal_provenance,
                    })
                    records.append(record)
            native_actions = [
                {key: int(action[key]) for key in ("side", "deck_index", "x", "y")}
                | {"type": "play"}
                for action in actions
            ]
            action_started = time.perf_counter()
            action_result = env.joint_act(native_actions)
            action_seconds += time.perf_counter() - action_started
            results = action_result.get("actions", [])
            if len(results) != len(actions):
                failure = f"native_action_count_mismatch_tick_{source_tick}"
                break
            rejected = [
                item for item in results
                if not bool(item.get("result", {}).get("accepted", False))
            ]
            if rejected:
                codes = [
                    int(item.get("result", {}).get("result_code", -1))
                    for item in rejected
                ]
                failure = f"native_rejected_tick_{source_tick}_codes_{codes}"
                break
            accepted_actions += len(results)
            previous_source_tick = source_tick

    if failure is None and plan.terminal_crowns is not None:
        # Duration is stored at one-second resolution.  A 20-Tick fence lets
        # libg emit the terminal object without accepting an unbounded run.
        remaining = max(1, plan.duration_ticks + 20 - final_tick)
        step_started = time.perf_counter()
        final_step = env.step(remaining)
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
            if not terminal_match:
                failure = (
                    f"terminal_crowns_{observed_crowns}_expected_"
                    f"{plan.terminal_crowns}"
                )
        else:
            terminal_match = False
            failure = "native_terminal_missing_at_source_end"

    return NativeReplayResult(
        battle_tag=plan.battle_tag,
        accepted=failure is None and accepted_actions == len(plan.actions),
        failure=failure,
        source_actions=len(plan.actions),
        accepted_actions=accepted_actions,
        final_tick=final_tick,
        native_ticks_advanced=native_ticks,
        reset_seconds=reset_seconds,
        step_seconds=step_seconds,
        observe_seconds=observe_seconds,
        action_seconds=action_seconds,
        wall_seconds=time.perf_counter() - started,
        terminal_validated=terminal_validated,
        terminal_match=terminal_match,
        source_crowns=plan.terminal_crowns,
        observed_crowns=observed_crowns,
        decision_records=tuple(records),
    )


def load_template(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("battle"), dict):
        raise TypeError("native template must contain battle")
    return value
