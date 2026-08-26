"""Prepare or run the isolated 100-battle native ability acceptance pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_profile import (
    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET,
    action_tick_provenance,
    native_teacher_forced_profile,
)
from expert_v1.native_ability_pilot import (
    AbilityPilotTask,
    execute_ability_task,
    load_task_manifest,
    select_ability_positive_tasks,
    sha256_file,
    write_task_manifest,
)
from expert_v1.native_replay_plan import DEFAULT_NATIVE_SEED
from expert_v1.native_replay_runner import load_template
from expert_v1.tick_store_v1.shard import WorkerShardSink, build_store_manifest
from native_core.env import NativeRoyaleEnv


DEFAULT_MANIFEST = Path(
    r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
    r"\version-window-20260804\accepted-cycle-clean.jsonl"
)
DEFAULT_PLAN = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-ability-pilot-100-plan\selected.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-ability-pilot-100-execution"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(handle: Any, value: Any) -> None:
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    handle.flush()


def prepare(args: argparse.Namespace) -> int:
    tasks, summary = select_ability_positive_tasks(
        args.manifest,
        limit=args.limit,
        selection_seed=args.selection_seed,
    )
    write_task_manifest(args.output, tasks, summary)
    print(json.dumps({
        **summary,
        "task_manifest": str(args.output.resolve()),
        "task_manifest_sha256": sha256_file(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


def _worker(
    worker_index: int,
    port: int,
    tasks: Queue[AbilityPilotTask],
    *,
    output_root: Path,
    template_path: Path,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    action_execution_tick_offset: int,
) -> dict[str, Any]:
    worker_id = f"ability-worker-{worker_index:02d}"
    result_path = output_root / f"{worker_id}.results.jsonl"
    sink: WorkerShardSink | None = None
    completed = successes = stored_ticks = 0
    worker_error: str | None = None
    started = time.perf_counter()
    try:
        template = load_template(template_path)
        sink = WorkerShardSink(
            output_root / "shards",
            worker_id,
            episodes_per_shard=100,
            anchor_interval=256,
            compression_level=1,
        )
        with NativeRoyaleEnv(port=port, timeout=60.0) as env:
            with result_path.open("a", encoding="utf-8", newline="\n") as output:
                while True:
                    try:
                        task = tasks.get_nowait()
                    except Empty:
                        break
                    try:
                        record, diagnostic = execute_ability_task(
                            env,
                            task,
                            template,
                            None,
                            sink,
                            seed=seed,
                            maximum_seeds_to_test=maximum_seeds_to_test,
                            trace_batch_steps=trace_batch_steps,
                            action_execution_tick_offset=(
                                action_execution_tick_offset
                            ),
                        )
                        record.update({
                            "worker_id": worker_id,
                            "port": port,
                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        if diagnostic is not None:
                            diagnostic.update({"worker_id": worker_id, "port": port})
                            diagnostic_path = (
                                output_root / "diagnostics" / f"{task.battle_tag}.json"
                            )
                            _atomic_json(diagnostic_path, diagnostic)
                            record["diagnostic_path"] = str(diagnostic_path.resolve())
                        else:
                            record["diagnostic_path"] = None
                        _append_jsonl(output, record)
                        completed += 1
                        if record["teacher_forced_success"]:
                            successes += 1
                            stored_ticks += int(record["tick_store_entry"]["ticks"])
                    finally:
                        tasks.task_done()
    except Exception as error:
        worker_error = f"{type(error).__name__}: {error}"
    finally:
        manifests = [] if sink is None else sink.finalize()
    return {
        "worker_id": worker_id,
        "port": port,
        "completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "stored_ticks": stored_ticks,
        "action_execution_tick_offset": action_execution_tick_offset,
        "native_teacher_forced_profile": native_teacher_forced_profile(
            action_execution_tick_offset
        ),
        "wall_seconds": time.perf_counter() - started,
        "worker_error": worker_error,
        "shards": manifests,
    }


def _read_results(root: Path) -> list[dict[str, Any]]:
    by_tag: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("ability-worker-*.results.jsonl")):
        with path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                value = json.loads(line)
                tag = str(value["battle_tag"])
                if tag in by_tag:
                    raise RuntimeError(f"duplicate final result for {tag}")
                by_tag[tag] = value
    results = [by_tag[tag] for tag in sorted(by_tag)]
    with (root / "results.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for value in results:
            _append_jsonl(out, value)
    return results


def run(args: argparse.Namespace) -> int:
    task_manifest = args.tasks.resolve(strict=True)
    tasks = load_task_manifest(task_manifest)
    if not tasks:
        raise ValueError("task manifest is empty")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"ability pilot output must be a fresh directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostics").mkdir(exist_ok=True)
    (output_root / "shards").mkdir(exist_ok=True)
    template_path = args.template.resolve(strict=True)
    queue: Queue[AbilityPilotTask] = Queue()
    for task in tasks:
        queue.put(task)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(args.ports)) as executor:
        futures = [
            executor.submit(
                _worker,
                index,
                port,
                queue,
                output_root=output_root,
                template_path=template_path,
                seed=args.seed,
                maximum_seeds_to_test=args.maximum_seeds,
                trace_batch_steps=args.trace_batch_steps,
                action_execution_tick_offset=args.action_execution_tick_offset,
            )
            for index, port in enumerate(args.ports)
        ]
        worker_reports = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    results = _read_results(output_root)
    success = [row for row in results if row["teacher_forced_success"]]
    stored = [row for row in results if row.get("tick_store_entry") is not None]
    total_store_ticks = sum(int(row["tick_store_entry"]["ticks"]) for row in stored)
    store_manifest = build_store_manifest(
        output_root / "shards",
        source_manifest=task_manifest,
        # A post-write integrity failure is still a physical append-only
        # frame.  Count it here so the store audit reports the evidence rather
        # than crashing while trying to pretend that frame does not exist.
        expected_episodes=len(stored),
        expected_ticks=total_store_ticks,
        store_metadata={
            "native_teacher_forced_profile": native_teacher_forced_profile(
                args.action_execution_tick_offset
            )
        },
    )
    failures: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    terminal: Counter[str] = Counter()
    coordinates: Counter[str] = Counter()
    coordinate_totals: Counter[str] = Counter()
    for row in results:
        if not row["teacher_forced_success"]:
            failures[str(row.get("failure_class") or "unknown")] += 1
        resolutions.update({
            str(key): int(value)
            for key, value in row.get("ability_resolution_counts", {}).items()
        })
        terminal[str(row.get("terminal_diagnostic_status") or "unknown")] += 1
        coordinates[str(row.get("coordinate_provenance") or "unknown")] += 1
        coordinate_audit = row.get("coordinate_audit")
        if isinstance(coordinate_audit, dict):
            for key in (
                "raw_data_i_events",
                "data_i_zero_events",
                "data_i_one_events",
                "legacy_xy_fallback_events",
            ):
                coordinate_totals[key] += int(coordinate_audit.get(key, 0))
    processed_tags = {str(row["battle_tag"]) for row in results}
    expected_tags = {task.battle_tag for task in tasks}
    missing_tags = sorted(expected_tags - processed_tags)
    unexpected_tags = sorted(processed_tags - expected_tags)
    per_tick_verified = bool(success) and all(
        row.get("tick_store_integrity") is True for row in success
    )
    action_boundary_verified = bool(results) and all(
        int(row.get("action_execution_tick_offset", -1))
        == int(args.action_execution_tick_offset)
        and f"source_tick+{args.action_execution_tick_offset}"
        in str(row.get("action_tick_provenance") or "")
        and row.get("native_teacher_forced_profile")
        == native_teacher_forced_profile(args.action_execution_tick_offset)
        for row in results
    )
    zero_worker_errors = all(row["worker_error"] is None for row in worker_reports)
    acceptance_pass = bool(
        len(results) == len(tasks)
        and len(success) == len(tasks)
        and per_tick_verified
        and action_boundary_verified
        and zero_worker_errors
        and not missing_tags
        and not unexpected_tags
    )
    summary = {
        "schema_version": 1,
        "kind": "expert_native_ability_pilot_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_teacher_forced_profile": native_teacher_forced_profile(
            args.action_execution_tick_offset
        ),
        "task_manifest": str(task_manifest),
        "task_manifest_sha256": sha256_file(task_manifest),
        "template": str(template_path),
        "ports": args.ports,
        "seed": args.seed,
        "seed_role": "legacy_preferred_only; chosen seed is searched per battle",
        "maximum_seeds_to_test": args.maximum_seeds,
        "trace_batch_steps": args.trace_batch_steps,
        "action_execution_tick_offset": args.action_execution_tick_offset,
        "action_tick_provenance": action_tick_provenance(
            args.action_execution_tick_offset
        ),
        "selected_battles": len(tasks),
        "processed_battles": len(results),
        "teacher_forced_successes": len(success),
        "teacher_forced_failures": len(results) - len(success),
        "missing_result_tags": missing_tags,
        "unexpected_result_tags": unexpected_tags,
        "source_deploy_actions": sum(task.deploy_action_count for task in tasks),
        "accepted_deploy_actions": sum(
            int(row["accepted_deploy_actions"]) for row in results
        ),
        "source_ability_events": sum(task.ability_event_count for task in tasks),
        "accepted_ability_actions": sum(
            int(row["accepted_ability_actions"]) for row in results
        ),
        "ability_resolution_counts": dict(resolutions),
        "coordinate_provenance_counts": dict(coordinates),
        "coordinate_audit_totals": dict(coordinate_totals),
        "failure_class_counts": dict(failures),
        "terminal_diagnostic_counts": dict(terminal),
        "stored_episodes": int(store_manifest["episode_count"]),
        "stored_ticks": int(store_manifest["tick_count"]),
        "stored_bytes": int(store_manifest["total_bytes"]),
        "bytes_per_tick": (
            0.0 if not total_store_ticks
            else int(store_manifest["total_bytes"]) / total_store_ticks
        ),
        "wall_seconds": wall_seconds,
        "stored_ticks_per_second": total_store_ticks / wall_seconds,
        "per_tick_store_verified": per_tick_verified,
        "action_boundary_audit_verified": action_boundary_verified,
        "zero_worker_errors": zero_worker_errors,
        "acceptance_pass": acceptance_pass,
        "branch_policy": "multiple live candidates fail closed; no guessed entity",
        "worker_reports": worker_reports,
    }
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if acceptance_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    prepare_parser.add_argument("--limit", type=int, default=100)
    prepare_parser.add_argument(
        "--selection-seed", default="schema3-ability-positive-v1"
    )
    prepare_parser.set_defaults(function=prepare)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--tasks", type=Path, default=DEFAULT_PLAN)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    run_parser.add_argument(
        "--ports", type=int, nargs="+", default=[38031, 38032, 38033, 38034]
    )
    run_parser.add_argument("--seed", type=int, default=DEFAULT_NATIVE_SEED)
    run_parser.add_argument("--maximum-seeds", type=int, default=4096)
    run_parser.add_argument("--trace-batch-steps", type=int, default=64)
    run_parser.add_argument(
        "--action-execution-tick-offset",
        type=int,
        choices=(0, 1),
        default=(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        help=(
            "execute deploy and ability commands at time_raw+offset while "
            "preserving every source Tick; RoyaleAPI native teacher-forced "
            "profile v1 defaults to 1; pass 0 only for the historical phase "
            "diagnostic"
        ),
    )
    run_parser.set_defaults(function=run)
    args = parser.parse_args()
    if getattr(args, "trace_batch_steps", 1) not in range(1, 65):
        raise ValueError("trace_batch_steps must be in 1..64")
    if len(set(getattr(args, "ports", []))) != len(getattr(args, "ports", [])):
        raise ValueError("native Worker ports must be unique")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
