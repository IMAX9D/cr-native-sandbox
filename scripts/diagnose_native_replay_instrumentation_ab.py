"""Run four fixed-seed native replay instrumentation lanes for diagnosis.

This tool is deliberately outside the production generator.  It never starts
or stops an Android Worker and never publishes Tick/Mask training artifacts.
The caller must provide a fresh output directory and an already-running native
Worker port.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_dataset_generator import (  # noqa: E402
    RecordingCountingEnv,
    _semantic_digest,
    native_replay_semantics,
)
from expert_v1.native_ingest_contract import (  # noqa: E402
    load_native_ingest_contract,
)
from expert_v1.native_replay_plan import compile_battle  # noqa: E402
from expert_v1.native_replay_runner import (  # noqa: E402
    NativeReplayResult,
    execute_plan,
    load_template,
)
from native_core.env import NativeRoyaleEnv  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\smoke-current-v3-full-prefix-gate-v2\native-authoritative-ticks-v1"
)
DEFAULT_SELECTION = DEFAULT_RUN_ROOT / "selection.jsonl"
DEFAULT_RUN_CONTRACT = DEFAULT_RUN_ROOT / "run-contract.json"
DEFAULT_CONTRACT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\contracts"
    r"\native-ingest-v150535029.json"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_CASES = (
    ("020YP2RLL9UC", 14),
    ("08GPVLU0J90G", 16),
    ("009YLPVVQ29U", 61),
    ("09LP9R0PLGQ9", 8),
    ("028YPJQUYL0P", 53),
)
LANES = (
    ("preflight", False, False),
    ("trace_only", True, False),
    ("mask_only", False, True),
    ("trace_and_mask", True, True),
)


class DiagnosticError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    temporary.write_bytes(payload)
    temporary.replace(path)


def parse_cases(values: Sequence[str] | None) -> tuple[tuple[str, int], ...]:
    if not values:
        return DEFAULT_CASES
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw in values:
        tag, separator, seed_text = str(raw).partition("=")
        tag = tag.strip()
        if not separator or not tag or tag in seen:
            raise DiagnosticError("--case must be one unique TAG=SEED value")
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise DiagnosticError("--case seed must be an integer") from error
        if seed <= 0:
            raise DiagnosticError("--case seed must be positive")
        seen.add(tag)
        result.append((tag, seed))
    return tuple(result)


def _strict_manifest_rows(
    manifest: Path, cases: Sequence[tuple[str, int]]
) -> list[dict[str, Any]]:
    requested = {tag: seed for tag, seed in cases}
    found: dict[str, dict[str, Any]] = {}
    with manifest.resolve(strict=True).open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise DiagnosticError(
                    f"manifest row {line_number} is not an object"
                )
            tag = str(row.get("battle_tag") or "")
            if tag not in requested:
                continue
            if tag in found:
                raise DiagnosticError(f"manifest tag is duplicated: {tag}")
            source_path = Path(str(row.get("source_path") or "")).resolve(
                strict=True
            )
            source_sha = str(row.get("source_sha256") or "")
            if _sha256_file(source_path) != source_sha:
                raise DiagnosticError(f"Schema5 source SHA changed: {tag}")
            found[tag] = {
                **dict(row),
                "source_path": str(source_path),
                "fixed_seed": requested[tag],
            }
    missing = sorted(set(requested) - set(found))
    if missing:
        raise DiagnosticError("manifest lacks requested tags: " + ",".join(missing))
    return [found[tag] for tag, _seed in cases]


def validate_run_contract(
    run_contract_path: Path,
    *,
    selection: Path,
    native_contract: Path,
    template: Path,
) -> dict[str, Any]:
    value = json.loads(run_contract_path.resolve(strict=True).read_text(
        encoding="utf-8-sig"
    ))
    if not isinstance(value, Mapping):
        raise DiagnosticError("run contract root is not an object")
    if (
        value.get("kind") != "expert_authoritative_native_tick_generator_v1"
        or Path(str(value.get("selection_manifest") or "")).resolve()
        != selection.resolve()
        or str(value.get("selection_manifest_sha256") or "")
        != _sha256_file(selection)
        or Path(str((value.get("native_ingest_contract") or {}).get("path") or ""))
        .resolve()
        != native_contract.resolve()
        or str((value.get("native_ingest_contract") or {}).get("file_sha256") or "")
        != _sha256_file(native_contract)
        or Path(str(value.get("template") or "")).resolve() != template.resolve()
        or str(value.get("template_sha256") or "") != _sha256_file(template)
    ):
        raise DiagnosticError("run contract selection/contract/template binding changed")
    return dict(value)


def _control_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    episode = state.get("episode")
    episode = episode if isinstance(episode, Mapping) else {}
    players = []
    for raw in state.get("players") or []:
        if not isinstance(raw, Mapping):
            continue
        players.append({
            "side": int(raw.get("side", -1)),
            "elixir_raw": int(raw.get("elixir_raw", -1)),
            "hand_deck_indices": [
                int(value) for value in raw.get("hand_deck_indices") or []
            ],
            "next_deck_index": int(raw.get("next_deck_index", -1)),
            "refill_timer": int(raw.get("refill_timer", -1)),
        })
    towers = []
    for raw in episode.get("crown_towers") or []:
        if not isinstance(raw, Mapping):
            continue
        towers.append({
            "side": int(raw.get("side", -1)),
            "type": str(raw.get("type") or ""),
            "x": int(raw.get("x", 0)),
            "y": int(raw.get("y", 0)),
            "hp": int(raw.get("hp", 0)),
            "max_hp": int(raw.get("max_hp", 0)),
            "destroyed": bool(raw.get("destroyed", False)),
        })
    entities = []
    for raw in state.get("entities") or []:
        if not isinstance(raw, Mapping):
            continue
        entities.append({
            "side": int(raw.get("side", -1)),
            "card_id": int(raw.get("card_id", -1)),
            "x": int(raw.get("x", 0)),
            "y": int(raw.get("y", 0)),
            "hp": int(raw.get("hp", 0)),
            "max_hp": int(raw.get("max_hp", 0)),
            "behavior_state": int(raw.get("behavior_state", 0)),
            "ability_slot": int(raw.get("ability_slot", 0)),
            "ability_available": bool(raw.get("ability_available", False)),
        })
    return {
        "tick": int(state.get("tick", -1)),
        "coherent": bool(state.get("coherent", True)),
        "players": sorted(players, key=lambda row: row["side"]),
        "entities": sorted(
            entities,
            key=lambda row: (
                row["side"], row["card_id"], row["x"], row["y"],
                row["hp"], row["behavior_state"],
            ),
        ),
        "episode": {
            "terminated": bool(episode.get("terminated", False)),
            "truncated": bool(episode.get("truncated", False)),
            "winner": episode.get("winner"),
            "crowns": list(episode.get("crowns") or []),
            "crown_towers": sorted(
                towers, key=lambda row: (row["side"], row["type"], row["x"])
            ),
        },
    }


def control_state_sha256(state: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical(_control_projection(state)))


@dataclass
class InMemoryTickSink:
    appended_ticks: int = 0
    staged_mask_metadata: dict[str, Any] | None = None

    def stage_deployment_masks(self, capture: Any) -> None:
        self.staged_mask_metadata = capture.metadata(require_complete=False)

    def append(
        self,
        battle_tag: str,
        states: Iterable[Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        rows = tuple(states)
        self.appended_ticks = len(rows)
        return {
            "kind": "diagnostic_in_memory_tick_sink_v1",
            "battle_tag": battle_tag,
            "ticks": len(rows),
            "metadata_sha256": _sha256_bytes(_canonical(dict(metadata))),
        }


@dataclass
class InstrumentedEnv:
    """Transparent env proxy that hashes observation/probe boundaries."""

    env: Any
    state_observations: list[dict[str, Any]] = field(default_factory=list)
    probe_checks: list[dict[str, Any]] = field(default_factory=list)
    _pending_probes: list[int] = field(default_factory=list)
    _latest_state: Mapping[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _record_state(self, state: Mapping[str, Any], source: str) -> str:
        control = _control_projection(state)
        digest = _sha256_bytes(_canonical(control))
        self._latest_state = dict(state)
        self.state_observations.append({
            "tick": int(state.get("tick", -1)),
            "source": source,
            "control_sha256": digest,
            "tower_hp": [
                [
                    int(tower["side"]), str(tower["type"]),
                    int(tower["x"]), int(tower["y"]), int(tower["hp"]),
                ]
                for tower in control["episode"]["crown_towers"]
            ],
            "player_cycle": [
                [
                    int(player["side"]), int(player["elixir_raw"]),
                    list(player["hand_deck_indices"]),
                    int(player["next_deck_index"]), int(player["refill_timer"]),
                ]
                for player in control["players"]
            ],
        })
        return digest

    def reset(self, replay: Mapping[str, Any], *, warmup_steps: int) -> Any:
        state = self.env.reset(replay, warmup_steps=warmup_steps)
        if isinstance(state, Mapping):
            self._record_state(state, "reset")
        self._pending_probes.clear()
        return state

    def observe_train(self) -> Any:
        state = self.env.observe_train()
        if isinstance(state, Mapping):
            self._record_state(state, "observe_train")
        return state

    def probe_grid(self, *, side: int, deck_index: int) -> Any:
        before = self.env.observe_train()
        before_sha = self._record_state(before, "probe_before")
        value = self.env.probe_grid(side=side, deck_index=deck_index)
        after = self.env.observe_train()
        after_sha = self._record_state(after, "probe_after")
        row = {
            "side": int(side),
            "deck_index": int(deck_index),
            "tick_before": int(before.get("tick", -1)),
            "tick_after": int(after.get("tick", -1)),
            "control_before_sha256": before_sha,
            "control_after_sha256": after_sha,
            "same_tick": before.get("tick") == after.get("tick"),
            "same_control": before_sha == after_sha,
            "probe_result_sha256": _sha256_bytes(_canonical(value)),
            "post_action_control_sha256": None,
            "post_step_control_sha256": None,
        }
        self.probe_checks.append(row)
        self._pending_probes.append(len(self.probe_checks) - 1)
        return value

    def joint_act(self, actions: list[Mapping[str, Any]]) -> Any:
        result = self.env.joint_act(actions)
        if self._pending_probes:
            state = self.env.observe_train()
            digest = self._record_state(state, "post_action")
            for index in self._pending_probes:
                self.probe_checks[index]["post_action_control_sha256"] = digest
        return result

    def _complete_pending_step(self, state: Mapping[str, Any]) -> None:
        digest = self._record_state(state, "post_step")
        for index in self._pending_probes:
            self.probe_checks[index]["post_step_control_sha256"] = digest
        self._pending_probes.clear()

    def step(self, steps: int = 1) -> Any:
        result = self.env.step(steps)
        if self._pending_probes:
            episode = result.get("episode") if isinstance(result, Mapping) else {}
            if not isinstance(episode, Mapping) or not (
                episode.get("terminated") or episode.get("truncated")
            ):
                self._complete_pending_step(self.env.observe_train())
            else:
                self._pending_probes.clear()
        return result

    def trace_train(self, steps: int = 1, **kwargs: Any) -> Any:
        result = self.env.trace_train(steps, **kwargs)
        frames = []
        if isinstance(result, Mapping):
            initial = result.get("initial_frame")
            if isinstance(initial, Mapping):
                frames.append(initial)
            frames.extend(
                frame for frame in result.get("frames") or []
                if isinstance(frame, Mapping)
            )
        for frame in frames:
            state = frame.get("state")
            if isinstance(state, Mapping):
                self._record_state(state, "trace_frame")
        if self._pending_probes and frames:
            state = frames[-1].get("state")
            if isinstance(state, Mapping):
                self._complete_pending_step(state)
        return result

    def diagnostics(self) -> dict[str, Any]:
        latest = self._latest_state
        return {
            "state_observations": self.state_observations,
            "probe_checks": self.probe_checks,
            "all_probes_same_tick": all(
                row["same_tick"] for row in self.probe_checks
            ),
            "all_probes_same_control": all(
                row["same_control"] for row in self.probe_checks
            ),
            "final_control_sha256": (
                None if latest is None else control_state_sha256(latest)
            ),
            "final_state": (
                None if latest is None else _control_projection(latest)
            ),
        }


def _lane_summary(
    result: NativeReplayResult,
    instrumented: InstrumentedEnv,
    sink: InMemoryTickSink | None,
    recorder: RecordingCountingEnv,
) -> dict[str, Any]:
    return {
        "semantic_sha256": _semantic_digest(result),
        "semantics": native_replay_semantics(result),
        "action_acceptance_sequence": [
            dict(row) for row in result.action_acceptance_sequence
        ],
        "action_prefix_length": len(result.action_acceptance_sequence),
        "failure": result.failure,
        "teacher_forced_success": bool(result.teacher_forced_success),
        "chosen_seed": int(result.chosen_seed),
        "layout_resolution_mode": result.layout_resolution_mode,
        "final_tick": int(result.final_tick),
        "terminal": {
            "validated": bool(result.terminal_validated),
            "match": result.terminal_match,
            "diagnostic_status": result.terminal_diagnostic_status,
            "tower_hp_validated": bool(result.terminal_tower_hp_validated),
            "tower_hp_match": result.terminal_tower_hp_match,
            "tower_hp_diagnostic_status": (
                result.terminal_tower_hp_diagnostic_status
            ),
        },
        "trace": {
            "batches": int(result.tick_trace_batches),
            "complete_frames": int(result.tick_trace_complete_frames),
            "incomplete_terminal_frames": int(
                result.tick_trace_incomplete_terminal_frames
            ),
            "incomplete_nonterminal_freeze_frames": int(
                result.tick_trace_incomplete_nonterminal_freeze_frames
            ),
            "in_memory_sink_ticks": 0 if sink is None else sink.appended_ticks,
        },
        "mask": {
            "probe_rpcs": int(result.deployment_mask_probe_rpc_count),
            "base_probe_rpcs": int(result.deployment_mask_base_probe_rpc_count),
            "dynamic_label_probe_rpcs": int(
                result.deployment_mask_dynamic_label_probe_rpc_count
            ),
            "captured_slots": int(result.deployment_mask_slots_captured),
            "capture_complete": bool(result.deployment_mask_capture_complete),
            "label_checks": int(result.deployment_mask_label_checks),
            "label_rejections": int(result.deployment_mask_label_rejections),
            "first_label_rejection": (
                result.deployment_mask_first_label_rejection
            ),
        },
        "instrumentation": instrumented.diagnostics(),
        "production_recorder": recorder.snapshot(),
        "native_result": result.json(),
    }


def _common_action_prefix(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    count = 0
    for left_row, right_row in zip(left, right):
        if dict(left_row) != dict(right_row):
            break
        count += 1
    return count


def _observed_action_tick_hashes(lane: Mapping[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    rows = (lane.get("instrumentation") or {}).get("state_observations") or []
    for row in rows:
        if isinstance(row, Mapping) and row.get("source") == "observe_train":
            result[int(row["tick"])] = str(row["control_sha256"])
    return result


def compare_lanes(lanes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = lanes["preflight"]
    comparisons: dict[str, Any] = {}
    for name, lane in lanes.items():
        left_states = _observed_action_tick_hashes(baseline)
        right_states = _observed_action_tick_hashes(lane)
        shared_ticks = sorted(set(left_states) & set(right_states))
        mismatches = [
            tick for tick in shared_ticks
            if left_states[tick] != right_states[tick]
        ]
        comparisons[name] = {
            "semantic_equal_to_preflight": (
                lane["semantic_sha256"] == baseline["semantic_sha256"]
            ),
            "common_action_prefix_with_preflight": _common_action_prefix(
                baseline["action_acceptance_sequence"],
                lane["action_acceptance_sequence"],
            ),
            "shared_action_observation_ticks": len(shared_ticks),
            "first_action_state_mismatch_tick": (
                None if not mismatches else mismatches[0]
            ),
            "action_state_mismatch_count": len(mismatches),
        }
    return comparisons


def run_four_lanes(
    env: Any,
    plan: Any,
    template: Mapping[str, Any],
    *,
    fixed_seed: int,
    execute: Callable[..., NativeReplayResult] = execute_plan,
) -> dict[str, Any]:
    lanes: dict[str, dict[str, Any]] = {}
    for name, trace_enabled, mask_enabled in LANES:
        history_size = (
            len(getattr(plan, "actions", ()))
            + len(getattr(plan, "ability_events", ()))
            + 8
        )
        recorder = RecordingCountingEnv(env, history_size=max(8, history_size))
        instrumented = InstrumentedEnv(recorder)
        sink = InMemoryTickSink() if trace_enabled else None
        result = execute(
            instrumented,
            plan,
            template,
            seed=fixed_seed,
            fixed_seed=fixed_seed,
            capture_decisions=False,
            tick_sink=sink,
            capture_deployment_masks=mask_enabled,
            collect_tick_states_on_failure=False,
            action_execution_tick_offset=1,
        )
        if (
            int(result.chosen_seed) != int(fixed_seed)
            or int(result.seeds_tested) != 0
            or int(result.seed_search_native_resets) != 1
            or result.layout_resolution_mode != "fixed_preflight_seed_replay"
        ):
            raise DiagnosticError(
                f"{name} did not execute one exact fixed-seed reset"
            )
        lanes[name] = _lane_summary(result, instrumented, sink, recorder)
    return {
        "fixed_seed": int(fixed_seed),
        "lanes": lanes,
        "comparisons": compare_lanes(lanes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--run-contract", type=Path, default=DEFAULT_RUN_CONTRACT
    )
    parser.add_argument("--native-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--case", action="append", help="TAG=FIXED_SEED")
    parser.add_argument("--port", type=int, default=38_031)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = parse_cases(args.case)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise DiagnosticError("--output-root must be a new directory")
    selection = args.selection.resolve(strict=True)
    run_contract_path = args.run_contract.resolve(strict=True)
    contract_path = args.native_contract.resolve(strict=True)
    template_path = args.template.resolve(strict=True)
    if any(
        output_root == path or output_root.is_relative_to(path)
        for path in (selection, run_contract_path, contract_path, template_path)
    ):
        raise DiagnosticError("diagnostic output overlaps an immutable input")
    run_contract = validate_run_contract(
        run_contract_path,
        selection=selection,
        native_contract=contract_path,
        template=template_path,
    )
    rows = _strict_manifest_rows(selection, cases)
    contract = load_native_ingest_contract(contract_path)
    template = load_template(template_path)
    output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    execution_errors = 0
    with NativeRoyaleEnv(port=args.port) as env:
        for row in rows:
            tag = str(row["battle_tag"])
            source_path = Path(str(row["source_path"]))
            source = json.loads(source_path.read_text(encoding="utf-8-sig"))
            plan = compile_battle(source, native_ingest_contract=contract)
            record: dict[str, Any] = {
                "schema_version": 1,
                "kind": "cr_native_fixed_seed_four_lane_ab_case_v1",
                "battle_tag": tag,
                "source_path": str(source_path),
                "source_sha256": str(row["source_sha256"]),
                "fixed_seed": int(row["fixed_seed"]),
                "plan": plan.json(),
            }
            try:
                record.update(run_four_lanes(
                    env, plan, template, fixed_seed=int(row["fixed_seed"])
                ))
            except Exception as error:
                execution_errors += 1
                record["execution_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            body = dict(record)
            record["canonical_sha256"] = _sha256_bytes(_canonical(body))
            _atomic_json(output_root / "cases" / f"{tag}.json", record)
            results.append({
                "battle_tag": tag,
                "fixed_seed": int(row["fixed_seed"]),
                "case_path": str((output_root / "cases" / f"{tag}.json").resolve()),
                "case_sha256": _sha256_file(
                    output_root / "cases" / f"{tag}.json"
                ),
                "execution_error": record.get("execution_error"),
                "comparisons": record.get("comparisons"),
            })
    summary = {
        "schema_version": 1,
        "kind": "cr_native_fixed_seed_four_lane_ab_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "path": str(selection),
            "sha256": _sha256_file(selection),
        },
        "run_contract": {
            "path": str(run_contract_path),
            "sha256": _sha256_file(run_contract_path),
            "run_contract_version": run_contract.get("run_contract_version"),
            "component_sha256": run_contract.get("component_sha256"),
        },
        "native_contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
        },
        "template": {
            "path": str(template_path),
            "sha256": _sha256_file(template_path),
        },
        "execute_plan_sha256": _sha256_file(
            PROJECT_ROOT / "expert_v1" / "native_replay_runner.py"
        ),
        "diagnostic_tool_sha256": _sha256_file(Path(__file__)),
        "port": int(args.port),
        "case_count": len(results),
        "execution_errors": execution_errors,
        "results": results,
    }
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if execution_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
