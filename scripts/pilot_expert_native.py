"""Run a 4-Worker, work-stealing, per-Tick native expert pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import threading
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
from expert_v1.native_pilot import (
    PilotTask,
    execute_deployment_trace,
    load_json,
    select_deployment_only_tasks,
    sha256_file,
)
from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import load_template
from expert_v1.tick_store_v1.shard import WorkerShardSink, build_store_manifest
from expert_v1.tick_store_v1.work_queue import TickStoreWorkQueue
from native_core.env import NativeRoyaleEnv


DEFAULT_MANIFEST = Path(
    r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
    r"\version-window-20260804\accepted-cycle-clean.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-v2"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    handle.flush()


def _task_payload(task: PilotTask) -> dict[str, Any]:
    return {
        "source_schema_version": task.source_schema_version,
        "team_crowns": task.team_crowns,
        "opponent_crowns": task.opponent_crowns,
    }


def _load_fixed_selection(
    path: Path, *, expected_episodes: int
) -> tuple[list[PilotTask], dict[str, Any]]:
    """Replay an immutable prior selection instead of rescanning a manifest."""
    tasks: list[PilotTask] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            tasks.append(PilotTask(
                battle_tag=str(value["battle_tag"]),
                source_path=str(value["source_path"]),
                source_sha256=str(value["source_sha256"]),
                source_schema_version=int(value["source_schema_version"]),
                team_crowns=int(value["team_crowns"]),
                opponent_crowns=int(value["opponent_crowns"]),
            ))
    if len(tasks) != expected_episodes:
        raise ValueError(
            f"fixed selection contains {len(tasks)} tasks, expected {expected_episodes}"
        )
    tags = [task.battle_tag for task in tasks]
    if len(tags) != len(set(tags)):
        raise ValueError("fixed selection contains duplicate battle tags")
    return tasks, {
        "kind": "fixed_prior_selection_v1",
        "requested": expected_episodes,
        "selected": len(tasks),
        "source_selection": str(path.resolve()),
        "source_selection_sha256": sha256_file(path),
    }


def _worker_loop(
    *,
    worker_index: int,
    port: int,
    queue_path: Path,
    output_root: Path,
    template_path: Path,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    action_execution_tick_offset: int,
) -> dict[str, Any]:
    worker_id = f"worker-{worker_index:02d}"
    result_path = output_root / f"{worker_id}.results.jsonl"
    template = load_template(template_path)
    completed = failed = ticks = actions = 0
    started = time.perf_counter()
    with NativeRoyaleEnv(port=port, timeout=60.0) as env:
        with TickStoreWorkQueue(queue_path) as queue:
            sink = WorkerShardSink(
                output_root / "shards",
                worker_id,
                episodes_per_shard=256,
                anchor_interval=256,
                compression_level=1,
            )
            with result_path.open("a", encoding="utf-8", newline="\n") as output:
                while True:
                    claimed = queue.claim(
                        worker_id, limit=1, lease_seconds=900.0, maximum_attempts=2
                    )
                    if not claimed:
                        break
                    task = claimed[0]
                    final_attempt = True
                    try:
                        source_path = Path(task.source_path)
                        actual_sha = sha256_file(source_path)
                        if actual_sha != task.source_sha256:
                            raise RuntimeError(
                                f"source SHA changed: {actual_sha} != {task.source_sha256}"
                            )
                        source = load_json(source_path)
                        plan = compile_battle(
                            source,
                            terminal_crowns=(
                                int(task.payload["team_crowns"]),
                                int(task.payload["opponent_crowns"]),
                            ),
                        )
                        replay = execute_deployment_trace(
                            env,
                            plan,
                            template,
                            seed=seed,
                            maximum_seeds_to_test=maximum_seeds_to_test,
                            trace_batch_steps=trace_batch_steps,
                            action_execution_tick_offset=(
                                action_execution_tick_offset
                            ),
                        )
                        audit = {
                            **replay.audit,
                            "worker_id": worker_id,
                            "port": port,
                            "source_path": str(source_path.resolve()),
                            "source_sha256": actual_sha,
                            "attempt": task.attempts,
                            "final_attempt": True,
                        }
                        if replay.audit["usable_tick_trajectory"]:
                            entry = sink.append(
                                task.battle_tag,
                                replay.states,
                                {
                                    "source_path": str(source_path.resolve()),
                                    "source_sha256": actual_sha,
                                    "source_schema_version": plan.source_schema_version,
                                    "seed": replay.audit["chosen_seed"],
                                    "preferred_seed": replay.audit["preferred_seed"],
                                    "seeds_tested": replay.audit["seeds_tested"],
                                    "source_seed_recovered": False,
                                    "action_execution_tick_offset": (
                                        action_execution_tick_offset
                                    ),
                                    "action_tick_provenance": replay.audit[
                                        "action_tick_provenance"
                                    ],
                                    "native_teacher_forced_profile": replay.audit[
                                        "native_teacher_forced_profile"
                                    ],
                                    "teacher_forced_success": True,
                                    "every_native_tick_present": True,
                                    "terminal_status": replay.audit["terminal_status"],
                                    "logical_training_state_sha256": replay.audit[
                                        "logical_training_state_sha256"
                                    ],
                                },
                            )
                            queue.complete(
                                worker_id,
                                task.battle_tag,
                                output_shard=str(entry["shard"]),
                                frame_offset=int(entry["offset"]),
                                frame_size=int(entry["frame_size"]),
                                episode_sha256=str(entry["payload_sha256"]),
                            )
                            audit["store_entry"] = entry
                            completed += 1
                            ticks += int(replay.audit["stored_tick_count"])
                        else:
                            queue.fail(
                                worker_id,
                                task.battle_tag,
                                str(replay.audit.get("failure") or "unusable trace"),
                                retry=False,
                            )
                            failed += 1
                        actions += int(replay.audit["accepted_deployment_actions"])
                        _append_jsonl(output, audit)
                    except Exception as error:
                        retry = task.attempts < 2
                        final_attempt = not retry
                        queue.fail(
                            worker_id,
                            task.battle_tag,
                            f"{type(error).__name__}: {error}",
                            retry=retry,
                        )
                        _append_jsonl(
                            output,
                            {
                                "schema_version": 1,
                                "kind": "expert_native_deployment_trace_pilot_error_v1",
                                "battle_tag": task.battle_tag,
                                "worker_id": worker_id,
                                "port": port,
                                "attempt": task.attempts,
                                "final_attempt": final_attempt,
                                "teacher_forced_success": False,
                                "usable_tick_trajectory": False,
                                "action_execution_tick_offset": (
                                    action_execution_tick_offset
                                ),
                                "native_teacher_forced_profile": (
                                    native_teacher_forced_profile(
                                        action_execution_tick_offset
                                    )
                                ),
                                "failure": f"{type(error).__name__}: {error}",
                            },
                        )
                        if final_attempt:
                            failed += 1
            manifests = sink.finalize()
    return {
        "worker_id": worker_id,
        "port": port,
        "completed": completed,
        "failed": failed,
        "stored_ticks": ticks,
        "accepted_actions": actions,
        "wall_seconds": time.perf_counter() - started,
        "shards": manifests,
    }


def _load_final_results(output_root: Path) -> list[dict[str, Any]]:
    by_tag: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(output_root.glob("worker-*.results.jsonl")):
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                tag = str(value["battle_tag"])
                attempts[tag].append(value)
                if value.get("final_attempt") is True:
                    by_tag[tag] = value
    final = [by_tag[tag] for tag in sorted(by_tag)]
    with (output_root / "results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for value in final:
            _append_jsonl(handle, value)
    return final


def _seed_probe(
    tasks: list[PilotTask],
    main_results: dict[str, dict[str, Any]],
    *,
    ports: list[int],
    template_path: Path,
    alternate_seed: int,
    trace_batch_steps: int,
    action_execution_tick_offset: int,
) -> list[dict[str, Any]]:
    template = load_template(template_path)

    def probe(index: int, task: PilotTask) -> dict[str, Any]:
        port = ports[index % len(ports)]
        source = load_json(Path(task.source_path))
        plan = compile_battle(
            source, terminal_crowns=(task.team_crowns, task.opponent_crowns)
        )
        with NativeRoyaleEnv(port=port, timeout=60.0) as env:
            replay = execute_deployment_trace(
                env,
                plan,
                template,
                seed=alternate_seed,
                trace_batch_steps=trace_batch_steps,
                action_execution_tick_offset=action_execution_tick_offset,
            )
        main = main_results[task.battle_tag]
        return {
            "battle_tag": task.battle_tag,
            "port": port,
            "alternate_seed": alternate_seed,
            "action_execution_tick_offset": action_execution_tick_offset,
            "alternate_chosen_seed": replay.audit.get("chosen_seed"),
            "alternate_teacher_forced_success": replay.audit[
                "teacher_forced_success"
            ],
            "alternate_failure": replay.audit.get("failure"),
            "alternate_layout_calibration_attempts": replay.audit.get(
                "layout_calibration_attempts"
            ),
            "main_logical_training_state_sha256": main.get(
                "logical_training_state_sha256"
            ),
            "alternate_logical_training_state_sha256": replay.audit[
                "logical_training_state_sha256"
            ],
            "logical_training_state_equal": (
                replay.audit["teacher_forced_success"]
                and main.get("logical_training_state_sha256")
                == replay.audit["logical_training_state_sha256"]
            ),
            "main_tick_count": main.get("stored_tick_count"),
            "alternate_tick_count": replay.audit["stored_tick_count"],
        }

    # One probe per Worker avoids concurrent access to the same native port.
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(probe, index, task) for index, task in enumerate(tasks)]
        return [future.result() for future in futures]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--selection-from",
        type=Path,
        default=None,
        help="reuse an exact prior selection.jsonl instead of scanning manifest",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--template",
        type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=38031)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--maximum-seeds", type=int, default=4096)
    parser.add_argument("--alternate-seed", type=int, default=1)
    parser.add_argument("--seed-probe-count", type=int, default=4)
    parser.add_argument("--trace-batch-steps", type=int, default=64)
    parser.add_argument(
        "--action-execution-tick-offset",
        type=int,
        choices=(0, 1),
        default=(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        help=(
            "execute at time_raw+offset while preserving the source label; "
            "RoyaleAPI native teacher-forced profile v1 defaults to 1; pass "
            "0 only to reproduce the historical phase diagnostic"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.episodes <= 0:
        raise ValueError("workers and episodes must be positive")
    if not 0 <= args.seed_probe_count <= args.workers:
        raise ValueError("seed-probe-count must be in 0..workers")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    ports = [args.base_port + index for index in range(args.workers)]

    if args.selection_from is None:
        tasks, selection = select_deployment_only_tasks(
            args.manifest.resolve(strict=True), limit=args.episodes
        )
    else:
        tasks, selection = _load_fixed_selection(
            args.selection_from.resolve(strict=True),
            expected_episodes=args.episodes,
        )
    selection_path = output_root / "selection.jsonl"
    with selection_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            _append_jsonl(handle, task.json())
    if args.selection_from is not None:
        source_selection_sha256 = sha256_file(
            args.selection_from.resolve(strict=True)
        )
        emitted_selection_sha256 = sha256_file(selection_path)
        if emitted_selection_sha256 != source_selection_sha256:
            raise RuntimeError(
                "fixed selection serialization changed: "
                f"{emitted_selection_sha256} != {source_selection_sha256}"
            )
    _atomic_json(output_root / "selection-summary.json", selection)

    queue_path = output_root / "work-queue.sqlite3"
    with TickStoreWorkQueue(queue_path) as queue:
        inserted = queue.add_tasks(
            {
                "battle_tag": task.battle_tag,
                "source_path": task.source_path,
                "source_sha256": task.source_sha256,
                "payload": _task_payload(task),
            }
            for task in tasks
        )
    if inserted != len(tasks):
        raise RuntimeError(f"expected {len(tasks)} new tasks, inserted {inserted}")

    run_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _worker_loop,
                worker_index=index,
                port=port,
                queue_path=queue_path,
                output_root=output_root,
                template_path=args.template.resolve(strict=True),
                seed=args.seed,
                maximum_seeds_to_test=args.maximum_seeds,
                trace_batch_steps=args.trace_batch_steps,
                action_execution_tick_offset=args.action_execution_tick_offset,
            )
            for index, port in enumerate(ports)
        ]
        workers = [future.result() for future in futures]
    run_wall_seconds = time.perf_counter() - run_started

    results = _load_final_results(output_root)
    with TickStoreWorkQueue(queue_path) as queue:
        queue_counts = queue.counts()
    successful = [value for value in results if value.get("usable_tick_trajectory")]
    total_ticks = sum(int(value.get("stored_tick_count") or 0) for value in successful)
    total_source_actions = sum(
        int(value.get("source_deployment_actions") or 0) for value in results
    )
    total_accepted_actions = sum(
        int(value.get("accepted_deployment_actions") or 0) for value in results
    )
    store = build_store_manifest(
        output_root / "shards",
        source_manifest=selection_path,
        expected_episodes=len(successful),
        expected_ticks=total_ticks,
        store_metadata={
            "native_teacher_forced_profile": native_teacher_forced_profile(
                args.action_execution_tick_offset
            )
        },
    )

    by_tag = {str(value["battle_tag"]): value for value in successful}
    probe_tasks = [
        task for task in tasks if task.battle_tag in by_tag
    ][: args.seed_probe_count]
    seed_probe = _seed_probe(
        probe_tasks,
        by_tag,
        ports=ports,
        template_path=args.template.resolve(strict=True),
        alternate_seed=args.alternate_seed,
        trace_batch_steps=args.trace_batch_steps,
        action_execution_tick_offset=args.action_execution_tick_offset,
    ) if probe_tasks else []
    with (output_root / "seed-probe.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for value in seed_probe:
            _append_jsonl(handle, value)

    terminal_counts = Counter(
        str(value.get("terminal_status") or "unknown") for value in results
    )
    failures = [
        {
            "battle_tag": value.get("battle_tag"),
            "failure": value.get("failure"),
            "first_rejection": value.get("first_rejection"),
        }
        for value in results
        if not value.get("usable_tick_trajectory")
    ]
    rejection_codes: Counter[str] = Counter()
    rejection_classes: Counter[str] = Counter()
    for value in results:
        rejection = value.get("first_rejection")
        if not isinstance(rejection, dict):
            continue
        for code in rejection.get("result_codes", []):
            rejection_codes[str(code)] += 1
        for event in rejection.get("events", []):
            if isinstance(event, dict):
                rejection_classes[
                    str(event.get("result_code_classification") or "unknown")
                ] += 1
    rejected_action_events = sum(rejection_codes.values())
    attempted_action_events = total_accepted_actions + rejected_action_events
    summary = {
        "schema_version": 1,
        "kind": "expert_native_deployment_trace_pilot_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_teacher_forced_profile": native_teacher_forced_profile(
            args.action_execution_tick_offset
        ),
        "configuration": {
            "episodes": args.episodes,
            "workers": args.workers,
            "ports": ports,
            "seed": args.seed,
            "seed_role": "legacy_preferred_only; chosen seed is searched per battle",
            "maximum_seeds_to_test": args.maximum_seeds,
            "alternate_seed": args.alternate_seed,
            "trace_batch_steps": args.trace_batch_steps,
            "tick_hz": 20,
            "source_filter": "schema>=3/native-ready/zero-ability",
            "selection_mode": (
                "manifest_scan"
                if args.selection_from is None
                else "fixed_prior_selection"
            ),
            "action_execution_tick_offset": (
                args.action_execution_tick_offset
            ),
            "action_tick_provenance": action_tick_provenance(
                args.action_execution_tick_offset
            ),
        },
        "selection": selection,
        "queue_counts": queue_counts,
        "teacher_forced": {
            "successful_episodes": len(successful),
            "failed_episodes": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "source_deployment_actions": total_source_actions,
            "accepted_deployment_actions": total_accepted_actions,
            # Fail-closed replay does not attempt actions after the first
            # divergence.  This ratio is coverage of the source action corpus,
            # not acceptance among actions that reached native execute.
            "source_action_coverage_before_first_failure": (
                total_accepted_actions / total_source_actions
                if total_source_actions else 0.0
            ),
            "attempted_deployment_actions": attempted_action_events,
            "rejected_deployment_actions": rejected_action_events,
            "attempted_deployment_acceptance_rate": (
                total_accepted_actions / attempted_action_events
                if attempted_action_events else 0.0
            ),
            "first_failures": failures,
            "native_rejection_code_counts": dict(sorted(rejection_codes.items())),
            "native_rejection_class_counts": dict(
                sorted(rejection_classes.items())
            ),
        },
        "tick_trace": {
            "stored_ticks": total_ticks,
            "all_successful_episodes_consecutive": all(
                value.get("every_native_tick_present") is True
                for value in successful
            ),
            "incomplete_observation_frames": sum(
                int(value.get("incomplete_observation_frames") or 0)
                for value in results
            ),
            "trace_rpc_count": sum(
                int(value.get("trace_rpc_count") or 0) for value in results
            ),
            "ticks_per_wall_second": (
                total_ticks / run_wall_seconds if run_wall_seconds else 0.0
            ),
            "store_bytes": store["total_bytes"],
            "store_bytes_per_tick": (
                store["total_bytes"] / total_ticks if total_ticks else 0.0
            ),
        },
        "throughput": {
            "pilot_wall_seconds": run_wall_seconds,
            "episodes_per_hour": (
                len(successful) * 3600.0 / run_wall_seconds
                if run_wall_seconds else 0.0
            ),
            "accepted_actions_per_second": (
                total_accepted_actions / run_wall_seconds
                if run_wall_seconds else 0.0
            ),
        },
        "terminal_diagnostic": dict(sorted(terminal_counts.items())),
        "seed_diagnostic": {
            "probed": len(seed_probe),
            "comparable": sum(
                value["alternate_teacher_forced_success"] for value in seed_probe
            ),
            "logical_training_state_equal": sum(
                value["logical_training_state_equal"] for value in seed_probe
            ),
            "all_probes_equal": bool(seed_probe) and all(
                value["logical_training_state_equal"] for value in seed_probe
            ),
            "all_comparable_equal": all(
                value["logical_training_state_equal"]
                for value in seed_probe
                if value["alternate_teacher_forced_success"]
            ),
        },
        "workers": workers,
        "tick_store": store,
    }
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(successful) == args.episodes else 2


if __name__ == "__main__":
    raise SystemExit(main())
