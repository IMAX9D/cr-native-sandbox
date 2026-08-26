"""Structured fail-closed evidence for native logic that stops between actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK = (
    "native_logic_frozen_before_execution_tick"
)


def logic_freeze_failure(
    *, source_tick: int, execution_tick: int, last_native_tick: int
) -> str:
    return (
        f"{NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK}_{execution_tick}_"
        f"source_tick_{source_tick}_last_tick_{last_native_tick}"
    )


def logic_freeze_audit(
    state: Mapping[str, Any],
    *,
    fallback_state: Mapping[str, Any] | None = None,
    source_tick: int,
    execution_tick: int,
    execution_tick_offset: int,
    chosen_seed: int,
    source_actions: int,
    accepted_actions: int,
    collected_tick_count: int,
    collected_tick_start: int | None,
    collected_tick_stop: int | None,
    trace_requested_steps: int,
    trace_stepped_calls: int,
) -> dict[str, Any]:
    """Capture enough native truth to audit a frozen, unusable prefix."""
    effective_state = dict(fallback_state or {})
    effective_state.update(state)
    fallback_episode = (
        fallback_state.get("episode")
        if isinstance(fallback_state, Mapping)
        else None
    )
    state_episode = state.get("episode")
    if isinstance(fallback_episode, Mapping) or isinstance(state_episode, Mapping):
        merged_episode = dict(
            fallback_episode if isinstance(fallback_episode, Mapping) else {}
        )
        merged_episode.update(
            state_episode if isinstance(state_episode, Mapping) else {}
        )
        effective_state["episode"] = merged_episode
    episode_value = effective_state.get("episode")
    episode = episode_value if isinstance(episode_value, Mapping) else {}
    last_native_tick = int(
        effective_state.get("tick", collected_tick_stop or -1)
    )
    crowns = episode.get("crowns")
    crown_towers = episode.get("crown_towers")
    native_phase = episode.get("native_phase")
    return {
        "schema_version": 1,
        "kind": "native_logic_freeze_before_execution_tick_audit_v1",
        "failure_class": NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK,
        "source_tick": int(source_tick),
        "execution_tick": int(execution_tick),
        "execution_tick_offset": int(execution_tick_offset),
        "last_native_tick": last_native_tick,
        "missing_native_ticks_to_execution": max(
            0, int(execution_tick) - last_native_tick
        ),
        "chosen_seed": int(chosen_seed),
        "source_actions": int(source_actions),
        "accepted_actions_before_freeze": int(accepted_actions),
        "collected_tick_count": int(collected_tick_count),
        "collected_tick_start": (
            None if collected_tick_start is None else int(collected_tick_start)
        ),
        "collected_tick_stop": (
            None if collected_tick_stop is None else int(collected_tick_stop)
        ),
        "trace_requested_steps": int(trace_requested_steps),
        "trace_stepped_calls": int(trace_stepped_calls),
        "state_hash": effective_state.get("state_hash"),
        "entity_count": effective_state.get("entity_count"),
        "episode": {
            "terminated": bool(episode.get("terminated", False)),
            "truncated": bool(episode.get("truncated", False)),
            "outcome": episode.get("outcome"),
            "winner": episode.get("winner"),
            "crowns": deepcopy(crowns) if isinstance(crowns, list) else crowns,
            "crown_towers": (
                deepcopy(crown_towers)
                if isinstance(crown_towers, list)
                else crown_towers
            ),
            "commands_allowed": episode.get("commands_allowed"),
            "command_gate_code": episode.get("command_gate_code"),
            "native_phase": (
                deepcopy(dict(native_phase))
                if isinstance(native_phase, Mapping)
                else native_phase
            ),
            "terminal_tick": episode.get("terminal_tick"),
        },
        "training_usable": False,
        "teacher_forced_success": False,
    }
