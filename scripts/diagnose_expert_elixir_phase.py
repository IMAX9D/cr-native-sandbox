"""Run read-only-in-purpose T..T+n reset A/Bs for native code-13 cases.

Every offset starts from a fresh native reset and replays all earlier expert
deployments at their original ticks.  Only the rejected target deployment is
delayed.  This is a diagnostic experiment; it never rewrites source actions
or emits a corrected training trajectory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.elixir_tick_diagnostics import (
    Code13Case,
    code13_cases,
    load_result_rows,
    lower_bound_regen_ticks,
    source_prefix,
    source_resource_flags,
)
from expert_v1.native_pilot import load_json
from expert_v1.native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    grouped_actions,
    materialize_replay,
    native_layout_order,
)
from expert_v1.native_replay_runner import calibrated_players, load_template
from expert_v1.native_replay_plan import compile_battle
from native_core.env import NativeRoyaleEnv


DEFAULT_RESULTS = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-compact-v6"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-code13-tick-ab-v1.json"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reset_with_layout(
    env: NativeRoyaleEnv,
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], tuple[tuple[int, ...], tuple[int, ...]]]:
    layout_calibration: Sequence[Mapping[str, Any]] = calibration
    for _attempt in range(3):
        replay, mappings = materialize_replay(
            plan, template, layout_calibration, seed=seed
        )
        state = env.reset(replay, warmup_steps=10)
        players = tuple(sorted(state["players"], key=lambda item: int(item["side"])))
        matches = True
        for side, player in enumerate(players):
            desired = tuple(
                mappings[side][logical]
                for logical in (
                    plan.sides[side].cycle.initial_hand
                    + plan.sides[side].cycle.initial_queue
                )
            )
            if native_layout_order(player) != desired:
                matches = False
        if matches:
            return (dict(layout_calibration[0]), dict(layout_calibration[1])), mappings
        layout_calibration = tuple(dict(player) for player in players)
    raise RuntimeError("native shuffle layout did not converge")


def _advance(env: NativeRoyaleEnv, current_tick: int, target_tick: int) -> tuple[int, dict[str, Any] | None]:
    if target_tick < current_tick:
        raise RuntimeError(f"target tick {target_tick} precedes {current_tick}")
    if target_tick == current_tick:
        return current_tick, None
    result = env.step(target_tick - current_tick)
    return int(result.get("tick_after", current_tick)), dict(result.get("episode", {}))


def _trial(
    env: NativeRoyaleEnv,
    plan: BattlePlan,
    case: Code13Case,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    offset: int,
    capture_prefix: bool = False,
) -> dict[str, Any]:
    replay, mappings = materialize_replay(plan, template, calibration, seed=seed)
    state = env.reset(replay, warmup_steps=10)
    players = sorted(state["players"], key=lambda item: int(item["side"]))
    for side, player in enumerate(players):
        desired = tuple(
            mappings[side][logical]
            for logical in (
                plan.sides[side].cycle.initial_hand
                + plan.sides[side].cycle.initial_queue
            )
        )
        if native_layout_order(player) != desired:
            raise RuntimeError(f"layout changed during offset {offset}, side {side}")

    current_tick = int(state["tick"])
    prior_actions = 0
    prefix_native_actions: list[dict[str, Any]] = []
    plan_actions = {
        int(action.source_event_index): action for action in plan.actions
    }
    for source_tick, actions in grouped_actions(plan, mappings):
        targets = [
            action for action in actions
            if int(action["source_event_index"]) == case.source_event_index
        ]
        if targets:
            if len(targets) != 1 or len(actions) != 1:
                raise RuntimeError(
                    "target shares its tick with another deployment; isolated delay "
                    "would change joint-action semantics"
                )
            target_tick = int(source_tick) + int(offset)
            current_tick, episode = _advance(env, current_tick, target_tick)
            if current_tick != target_tick:
                return {
                    "offset": offset,
                    "requested_tick": target_tick,
                    "observed_tick": current_tick,
                    "accepted": False,
                    "failure": "native_stopped_before_target",
                    "episode": episode,
                    "prior_actions_accepted": prior_actions,
                }
            pre = env.observe_train()
            player = next(
                item for item in pre["players"] if int(item["side"]) == case.side
            )
            result = env.joint_act([{key: value for key, value in targets[0].items()
                                    if key != "source_event_index"}])
            native_result = dict(result["actions"][0]["result"])
            return {
                "offset": offset,
                "requested_tick": target_tick,
                "observed_tick": int(pre["tick"]),
                "accepted": bool(native_result.get("accepted")),
                "result_code": int(native_result.get("result_code", -1)),
                "result_reason": native_result.get("result_reason"),
                "pre_action_elixir_raw": int(player["elixir_raw"]),
                "pre_action_hand_deck_indices": [
                    int(value) for value in player["hand_deck_indices"]
                ],
                "pre_action_next_deck_index": int(player.get("next_deck_index", -1)),
                "pre_action_refill_timer": int(player.get("refill_timer", -1)),
                "pre_action_episode": {
                    key: pre.get("episode", {}).get(key)
                    for key in (
                        "commands_allowed", "command_gate_code", "terminated",
                        "truncated", "crowns", "native_phase",
                    )
                },
                "native_resource_before": native_result.get("resource_before"),
                "native_guard_before": native_result.get("guard_before"),
                "native_resolved_data_id": int(
                    native_result.get("resolved_data_id", -1)
                ),
                "native_packed_selection": int(
                    native_result.get("packed_selection", -1)
                ),
                "prior_actions_accepted": prior_actions,
                "prefix_native_actions": prefix_native_actions,
            }

        current_tick, episode = _advance(env, current_tick, int(source_tick))
        if current_tick != int(source_tick):
            raise RuntimeError(
                f"native stopped at {current_tick} before source tick {source_tick}: {episode}"
            )
        pre = env.observe_train() if capture_prefix else None
        by_side = (
            {} if pre is None else {
                int(item["side"]): item for item in pre["players"]
            }
        )
        result = env.joint_act([
            {key: value for key, value in action.items() if key != "source_event_index"}
            for action in actions
        ])
        rejected = [
            item for item in result["actions"]
            if not bool(item.get("result", {}).get("accepted"))
        ]
        if rejected:
            codes = [item["result"].get("result_code") for item in rejected]
            raise RuntimeError(f"earlier native rejection at {source_tick}: {codes}")
        if capture_prefix:
            for action, item in zip(actions, result["actions"], strict=True):
                event_index = int(action["source_event_index"])
                source_action = plan_actions[event_index]
                player = by_side[int(action["side"])]
                native_result = item["result"]
                prefix_native_actions.append({
                    "source_event_index": event_index,
                    "source_tick": int(source_tick),
                    "side": int(action["side"]),
                    "card": source_action.base_token,
                    "logical_card_index": int(source_action.logical_card_index),
                    "native_deck_index": int(action["deck_index"]),
                    "pre_action_elixir_raw": int(player["elixir_raw"]),
                    "pre_action_hand_deck_indices": [
                        int(value) for value in player["hand_deck_indices"]
                    ],
                    "pre_action_next_deck_index": int(
                        player.get("next_deck_index", -1)
                    ),
                    "pre_action_refill_timer": int(player.get("refill_timer", -1)),
                    "accepted": bool(native_result.get("accepted")),
                    "result_code": int(native_result.get("result_code", -1)),
                    "native_resource_before": native_result.get("resource_before"),
                    "native_resolved_data_id": int(
                        native_result.get("resolved_data_id", -1)
                    ),
                    "native_packed_selection": int(
                        native_result.get("packed_selection", -1)
                    ),
                })
        prior_actions += len(actions)
    raise RuntimeError(f"target event {case.source_event_index} was not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=38034)
    parser.add_argument("--seed", type=int, default=DEFAULT_NATIVE_SEED)
    parser.add_argument("--max-offset", type=int, default=160)
    parser.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    args = parser.parse_args()

    rows = load_result_rows(args.results_root.resolve(strict=True))
    cases = code13_cases(rows)
    if not cases:
        raise RuntimeError("no result-code-13 cases found")
    template = load_template(args.template.resolve(strict=True))
    reports: list[dict[str, Any]] = []
    with NativeRoyaleEnv(port=args.port, timeout=60.0) as env:
        bootstrap = calibrated_players(env, template, seed=args.seed)
        for case in cases:
            source = load_json(Path(case.source_path))
            plan = compile_battle(source)
            stable_calibration, _mappings = _reset_with_layout(
                env, plan, template, bootstrap, seed=args.seed
            )
            trials: list[dict[str, Any]] = []
            minimum: int | None = None
            for offset in range(args.max_offset + 1):
                trial = _trial(
                    env, plan, case, template, stable_calibration,
                    seed=args.seed, offset=offset, capture_prefix=offset == 0,
                )
                trials.append(trial)
                if trial.get("accepted") is True:
                    minimum = offset
                    break
                if trial.get("failure") == "native_stopped_before_target":
                    break
            same_tick_actions = [
                action.source_event_index for action in plan.actions
                if action.tick == case.tick
            ]
            reports.append({
                **case.json(),
                "source_schema_version": plan.source_schema_version,
                "source_duration_ticks": plan.duration_ticks,
                "source_same_tick_action_indices": same_tick_actions,
                "source_resource_changing_cards": source_resource_flags(plan),
                "source_side_elixir_stats": source.get("elixir_stats", {}).get(
                    "team" if case.side == 0 else "opponent"
                ),
                "source_prefix": source_prefix(plan, case),
                "passive_regen_lower_bound_ticks": lower_bound_regen_ticks(
                    case.deficit_raw, case.tick
                ),
                "minimum_accepted_offset": minimum,
                "trials": trials,
            })
            print(
                f"{case.battle_tag} T={case.tick} {case.base_token}: "
                f"minimum_offset={minimum} trials={len(trials)}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "kind": "expert_native_code13_tick_ab_v1",
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "semantics": (
            "diagnostic only: prior actions at source ticks; fresh reset per offset; "
            "only rejected target delayed; never a training-data correction"
        ),
        "results_root": str(args.results_root.resolve()),
        "port": args.port,
        "seed": args.seed,
        "case_count": len(reports),
        "cases": reports,
    }
    _atomic_json(args.output.resolve(), output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
