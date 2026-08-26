"""Deterministic schema-v3 ability-positive native replay pilot helpers.

This module deliberately stays separate from the deployment-only pilot.  It
selects battles whose source contains exact ability-button Ticks and executes
them through :func:`native_replay_runner.execute_plan`, which performs live
entity resolution and submits the ability in the same native joint command
path as deployments.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback is tested implicitly
    orjson = None

from .native_profile import (
    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET,
    action_tick_provenance,
    native_teacher_forced_profile,
)
from .native_replay_plan import (
    BattlePlan,
    ReplayPlanError,
    compile_battle,
)
from .native_replay_runner import NativeReplayResult, execute_plan
from .native_seed_search import (
    DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    NativeSeedSearchError,
)


TASK_KIND = "expert_native_ability_pilot_task_v1"
RESULT_KIND = "expert_native_ability_pilot_result_v1"
DIAGNOSTIC_KIND = "expert_native_ability_failure_diagnostic_v1"
EXACT_ABILITY_TIER = "observed_ticks_identity_runtime_resolved"


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AbilityPilotTask:
    selection_index: int
    selection_digest: str
    battle_tag: str
    source_path: str
    source_sha256: str
    source_schema_version: int
    team_crowns: int
    opponent_crowns: int
    deploy_action_count: int
    ability_event_count: int
    duration_ticks: int

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": TASK_KIND,
            **asdict(self),
        }


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tags: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            value = orjson.loads(raw) if orjson is not None else json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"manifest line {line_number} is not an object")
            tag = str(value.get("battle_tag") or "")
            if not tag:
                raise ValueError(f"manifest line {line_number} has no battle_tag")
            if tag in tags:
                raise ValueError(f"duplicate battle_tag in manifest: {tag}")
            tags.add(tag)
            rows.append(value)
    return rows


def _selection_digest(seed: str, battle_tag: str) -> str:
    return hashlib.sha256(f"{seed}\0{battle_tag}".encode("utf-8")).hexdigest()


def _terminal_crowns(row: Mapping[str, Any]) -> tuple[int, int]:
    crowns = (int(row["team_crowns"]), int(row["opponent_crowns"]))
    if any(not 0 <= value <= 3 for value in crowns):
        raise ValueError(f"invalid terminal crowns: {crowns}")
    return crowns


def require_exact_ability_plan(plan: BattlePlan) -> None:
    """Reject anything that is not an executable exact-Tick ability source."""
    if plan.source_schema_version != 3:
        raise ValueError("ability pilot requires source schema version 3")
    if not plan.native_replay_ready:
        raise ValueError(f"plan is not native-replay-ready: {plan.replay_tier}")
    if plan.ability_log_tier != EXACT_ABILITY_TIER:
        raise ValueError(f"ability log is not exact: {plan.ability_log_tier}")
    if not plan.ability_events:
        raise ValueError("ability pilot requires at least one ability event")
    missing = sum(side.missing_ability_event_count for side in plan.sides)
    if missing:
        raise ValueError(f"ability pilot source is missing {missing} ability Ticks")


def select_ability_positive_tasks(
    manifest_path: Path,
    *,
    limit: int = 100,
    selection_seed: str = "schema3-ability-positive-v1",
) -> tuple[list[AbilityPilotTask], dict[str, Any]]:
    """Select a stable sample independent of manifest line ordering.

    Schema-v3 rows are ranked by a SHA-256 digest of ``seed + battle_tag``.
    Ranked sources are then compiled until exactly ``limit`` executable,
    ability-positive plans have been found.  No source ability identity or
    Tick is synthesized during selection.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    manifest_path = manifest_path.resolve(strict=True)
    rows = _manifest_rows(manifest_path)
    schema_three = [row for row in rows if int(row.get("schema_version") or 1) == 3]
    ranked = sorted(
        schema_three,
        key=lambda row: (
            _selection_digest(selection_seed, str(row["battle_tag"])),
            str(row["battle_tag"]),
        ),
    )
    reasons: Counter[str] = Counter()
    tasks: list[AbilityPilotTask] = []
    examined = 0
    for row in ranked:
        examined += 1
        tag = str(row["battle_tag"])
        source_path = Path(str(row.get("source_path") or ""))
        try:
            source_path = source_path.resolve(strict=True)
            source = load_json(source_path)
            if int(source.get("schema_version") or 1) != 3:
                raise ValueError("source schema does not match manifest schema3")
            raw_abilities = source.get("ability_plays")
            if not isinstance(raw_abilities, list) or not raw_abilities:
                reasons["source_has_no_ability_events"] += 1
                continue
            crowns = _terminal_crowns(row)
            plan = compile_battle(source, terminal_crowns=crowns)
            require_exact_ability_plan(plan)
            if plan.battle_tag != tag:
                raise ValueError(
                    f"source battle_tag mismatch: {plan.battle_tag} != {tag}"
                )
        except (OSError, TypeError, ValueError, ReplayPlanError) as error:
            reasons[f"{type(error).__name__}:{error}"] += 1
            continue
        tasks.append(AbilityPilotTask(
            selection_index=len(tasks),
            selection_digest=_selection_digest(selection_seed, tag),
            battle_tag=tag,
            source_path=str(source_path),
            source_sha256=sha256_file(source_path),
            source_schema_version=plan.source_schema_version,
            team_crowns=crowns[0],
            opponent_crowns=crowns[1],
            deploy_action_count=len(plan.actions),
            ability_event_count=len(plan.ability_events),
            duration_ticks=plan.duration_ticks,
        ))
        if len(tasks) == limit:
            break
    if len(tasks) != limit:
        raise RuntimeError(
            f"requested {limit} exact ability-positive battles, found {len(tasks)}"
        )
    summary = {
        "schema_version": 1,
        "kind": "expert_native_ability_pilot_selection_v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "selection_seed": selection_seed,
        "selection_algorithm": "sha256(seed\\0battle_tag), ascending",
        "manifest_rows": len(rows),
        "schema3_rows": len(schema_three),
        "ranked_rows_examined": examined,
        "selected_battles": len(tasks),
        "selected_deploy_actions": sum(task.deploy_action_count for task in tasks),
        "selected_ability_events": sum(task.ability_event_count for task in tasks),
        "selected_duration_ticks": sum(task.duration_ticks for task in tasks),
        "rejection_counts": dict(reasons.most_common()),
        "first_selection_digest": tasks[0].selection_digest,
        "last_selection_digest": tasks[-1].selection_digest,
    }
    return tasks, summary


def write_task_manifest(
    path: Path,
    tasks: Sequence[AbilityPilotTask],
    summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for task in tasks:
            output.write(
                json.dumps(
                    task.json(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
            )
    temporary.replace(path)
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    temporary_summary = summary_path.with_name(summary_path.name + ".tmp")
    temporary_summary.write_text(
        json.dumps(
            {**dict(summary), "task_manifest_sha256": sha256_file(path)},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)


def load_task_manifest(path: Path) -> list[AbilityPilotTask]:
    tasks: list[AbilityPilotTask] = []
    with path.resolve(strict=True).open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("kind") != TASK_KIND or value.get("schema_version") != 1:
                raise ValueError(f"invalid task manifest row {line_number}")
            fields = {
                key: value[key]
                for key in AbilityPilotTask.__dataclass_fields__
            }
            tasks.append(AbilityPilotTask(**fields))
    if [task.selection_index for task in tasks] != list(range(len(tasks))):
        raise ValueError("task manifest selection indices are not consecutive")
    if len({task.battle_tag for task in tasks}) != len(tasks):
        raise ValueError("task manifest contains duplicate battle tags")
    return tasks


class RecordingNativeEnv:
    """Transparent native environment proxy retaining failure-boundary state."""

    def __init__(self, env: Any, *, history_size: int = 8) -> None:
        self.env = env
        self.latest_state: dict[str, Any] | None = None
        self.reset_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.action_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.trace_history: deque[dict[str, Any]] = deque(maxlen=history_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, replay: Mapping[str, Any], *, warmup_steps: int) -> dict[str, Any]:
        state = self.env.reset(replay, warmup_steps=warmup_steps)
        self.latest_state = deepcopy(state)
        self.reset_history.append({
            "tick": int(state.get("tick", -1)),
            "state_hash": str(state.get("state_hash", "")),
            "players": deepcopy(state.get("players", [])),
        })
        return state

    def observe_train(self) -> dict[str, Any]:
        state = self.env.observe_train()
        self.latest_state = deepcopy(state)
        return state

    def trace_train(self, steps: int) -> dict[str, Any]:
        response = self.env.trace_train(steps)
        frames = response.get("frames") or []
        final = frames[-1] if frames else response.get("initial_frame")
        if isinstance(final, Mapping) and isinstance(final.get("state"), Mapping):
            self.latest_state = deepcopy(dict(final["state"]))
        self.trace_history.append({
            "requested_steps": int(steps),
            "stepped": int(response.get("stepped", -1)),
            "initial_tick": int(
                response.get("initial_frame", {}).get("state", {}).get("tick", -1)
            ),
            "final_tick": int(response.get("final_tick", -1)),
            "terminal": bool(response.get("terminal", False)),
        })
        return response

    def joint_act(self, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        audit = {
            "pre_action_state": deepcopy(self.latest_state),
            "request": deepcopy(list(actions)),
            "response": None,
            "exception": None,
        }
        try:
            response = self.env.joint_act(list(actions))
            audit["response"] = deepcopy(response)
            return response
        except Exception as error:
            audit["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.action_history.append(audit)

    def snapshot(self) -> dict[str, Any]:
        return {
            "latest_state": deepcopy(self.latest_state),
            "reset_history": list(deepcopy(self.reset_history)),
            "recent_action_history": list(deepcopy(self.action_history)),
            "recent_trace_history": list(deepcopy(self.trace_history)),
        }


def _failure_class(result: NativeReplayResult | None, error: Exception | None) -> str:
    if error is not None:
        if isinstance(error, NativeSeedSearchError):
            return "native_seed_search_exhausted"
        return "exception"
    assert result is not None
    failure = str(result.failure or "unknown")
    if "ability_branch_required" in failure:
        return "ability_branch_required"
    if "ability_no_legal" in failure:
        return "ability_entity_missing"
    if "native_rejected" in failure:
        return "native_action_rejected"
    if "tick_store_write" in failure:
        return "tick_store_write_failed"
    return "teacher_forced_failure"


def execute_ability_task(
    env: Any,
    task: AbilityPilotTask,
    template: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]] | None,
    tick_sink: Any,
    *,
    seed: int,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    trace_batch_steps: int = 64,
    action_execution_tick_offset: int = (
        ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
    ),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute one task and return a compact result plus full failure evidence."""
    recorder = RecordingNativeEnv(env)
    source: dict[str, Any] | None = None
    plan: BattlePlan | None = None
    result: NativeReplayResult | None = None
    layout_reports: tuple[dict[str, Any], ...] = ()
    error: Exception | None = None
    error_traceback: str | None = None
    try:
        source_path = Path(task.source_path).resolve(strict=True)
        actual_sha = sha256_file(source_path)
        if actual_sha != task.source_sha256:
            raise RuntimeError(
                f"source SHA changed: {actual_sha} != {task.source_sha256}"
            )
        source = load_json(source_path)
        plan = compile_battle(
            source,
            terminal_crowns=(task.team_crowns, task.opponent_crowns),
        )
        require_exact_ability_plan(plan)
        if (
            plan.battle_tag != task.battle_tag
            or len(plan.actions) != task.deploy_action_count
            or len(plan.ability_events) != task.ability_event_count
        ):
            raise RuntimeError("compiled source no longer matches selected task")
        result = execute_plan(
            recorder,
            plan,
            template,
            calibration,
            seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            capture_decisions=False,
            # Source markers do not identify one of several live copies.  The
            # pilot must branch or fail; this non-branching acceptance run
            # intentionally never supplies an arbitrary choice.
            ability_branch_choices=None,
            tick_sink=tick_sink,
            tick_store_metadata={
                "source_path": task.source_path,
                "source_sha256": task.source_sha256,
                "source_schema_version": 3,
                "exact_tick_ability_events": task.ability_event_count,
                "every_native_tick_present": True,
                "selection_digest": task.selection_digest,
                "seed": int(seed),
                "action_execution_tick_offset": int(
                    action_execution_tick_offset
                ),
                "native_teacher_forced_profile": native_teacher_forced_profile(
                    action_execution_tick_offset
                ),
            },
            trace_batch_steps=trace_batch_steps,
            action_execution_tick_offset=action_execution_tick_offset,
        )
        layout_reports = ({
            "mode": "source_order_bounded_native_seed_search",
            "preferred_seed": result.preferred_seed,
            "chosen_seed": result.chosen_seed,
            "seeds_tested": result.seeds_tested,
            "cache_hit": result.seed_search_cache_hit,
            "source_seed_recovered": result.source_seed_recovered,
            "action_execution_tick_offset": result.action_execution_tick_offset,
            "action_tick_provenance": result.action_tick_provenance,
        },)
    except Exception as caught:
        error = caught
        error_traceback = traceback.format_exc()
        if isinstance(caught, NativeSeedSearchError):
            layout_reports = ({
                "mode": "source_order_bounded_native_seed_search",
                "failure": str(caught),
                "preferred_seed": caught.preferred_seed,
                "seeds_tested": caught.seeds_tested,
                "maximum_seeds_to_test": caught.maximum_seeds_to_test,
                "source_seed_recovered": False,
            },)

    success = bool(result is not None and result.teacher_forced_success)
    entry = None if result is None else result.tick_store_entry
    store_integrity = None
    if success:
        assert entry is not None and result is not None
        store_integrity = bool(
            int(entry["ticks"]) == result.tick_trace_complete_frames
            # Tick-store ``tick_stop`` is the conventional exclusive bound.
            and int(entry["tick_stop"]) - int(entry["tick_start"])
            == int(entry["ticks"])
            and result.accepted_ability_actions == result.source_ability_events
        )
        if not store_integrity:
            success = False
            error = RuntimeError("successful replay violated per-Tick store integrity")
            error_traceback = None

    action_tick_provenance_value = (
        action_tick_provenance(action_execution_tick_offset)
        if result is None else result.action_tick_provenance
    )
    if result is not None:
        coordinate_provenance = result.coordinate_provenance
        coordinate_audit = result.coordinate_audit
    elif plan is not None:
        coordinate_provenance = plan.coordinate_provenance
        coordinate_audit = plan.json()["coordinate_audit"]
    else:
        coordinate_provenance = None
        coordinate_audit = None

    record = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "battle_tag": task.battle_tag,
        "selection_index": task.selection_index,
        "source_path": task.source_path,
        "source_sha256": task.source_sha256,
        "teacher_forced_success": success,
        "failure_class": None if success else _failure_class(result, error),
        "failure": (
            None if success
            else f"{type(error).__name__}: {error}" if error is not None
            else result.failure
        ),
        "layout_calibration_attempts": 0,
        "chosen_seed": None if result is None else result.chosen_seed,
        "seeds_tested": 0 if result is None else result.seeds_tested,
        "seed_search_cache_hit": (
            False if result is None else result.seed_search_cache_hit
        ),
        "source_seed_recovered": (
            False if result is None else result.source_seed_recovered
        ),
        "action_execution_tick_offset": int(action_execution_tick_offset),
        "action_tick_provenance": action_tick_provenance_value,
        "native_teacher_forced_profile": native_teacher_forced_profile(
            action_execution_tick_offset
        ),
        "coordinate_provenance": coordinate_provenance,
        "coordinate_audit": coordinate_audit,
        "source_deploy_actions": task.deploy_action_count,
        "accepted_deploy_actions": (
            0 if result is None else result.accepted_deploy_actions
        ),
        "source_ability_events": task.ability_event_count,
        "accepted_ability_actions": (
            0 if result is None else result.accepted_ability_actions
        ),
        "ability_resolution_counts": (
            {} if result is None else result.ability_resolution_counts
        ),
        "ability_resolutions": (
            [] if result is None else list(result.ability_resolutions)
        ),
        "terminal_diagnostic_status": (
            "not_reached" if result is None else result.terminal_diagnostic_status
        ),
        "native_ticks_advanced": 0 if result is None else result.native_ticks_advanced,
        "tick_trace_batches": 0 if result is None else result.tick_trace_batches,
        "tick_trace_complete_frames": (
            0 if result is None else result.tick_trace_complete_frames
        ),
        "tick_store_integrity": store_integrity,
        "tick_store_entry": entry,
        "wall_seconds": 0.0 if result is None else result.wall_seconds,
    }
    if success:
        return record, None
    diagnostic = {
        "schema_version": 1,
        "kind": DIAGNOSTIC_KIND,
        "battle_tag": task.battle_tag,
        "failure_class": record["failure_class"],
        "failure": record["failure"],
        "action_execution_tick_offset": int(action_execution_tick_offset),
        "action_tick_provenance": record["action_tick_provenance"],
        "native_teacher_forced_profile": record[
            "native_teacher_forced_profile"
        ],
        "coordinate_provenance": record["coordinate_provenance"],
        "coordinate_audit": record["coordinate_audit"],
        "task": task.json(),
        "source": source,
        "plan": None if plan is None else plan.json(),
        "native_result": None if result is None else result.json(),
        "layout_calibration": list(layout_reports),
        "native_boundary_snapshot": recorder.snapshot(),
        "exception_traceback": error_traceback,
    }
    return record, diagnostic
