"""One-command strict-version Stage-2 native self-play PPO loop."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.mps_runtime import ManagedMPS

COLLECT_SCRIPT = PROJECT_ROOT / "scripts" / "run_expert_selfplay_v1.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_expert_selfplay_stage2.py"
ENSURE_WORKERS_SCRIPT = PROJECT_ROOT / "scripts" / "ensure_bionic_workers.py"
KIND = "cr_native_expert_selfplay_stage2_loop_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(dict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def evict_file_cache(path: Path) -> bool:
    """Release pages for an immutable rollout after its learner consumed it."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
    return True


def evict_rollout_cache(update_dir: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for payload in update_dir.glob("collect-p*/rollouts/shard-*/rollout.pt"):
        length = payload.stat().st_size
        if evict_file_cache(payload):
            count += 1
            size += length
    return count, size


def parse_ports(value: str) -> list[int]:
    ports: list[int] = []
    seen = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            raise ValueError("empty Worker port item")
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError("descending Worker port range")
            candidates = range(start, stop + 1)
        else:
            candidates = (int(token),)
        for port in candidates:
            if not 1 <= port <= 65535 or port in seen:
                raise ValueError(f"invalid or duplicate Worker port: {port}")
            seen.add(port)
            ports.append(port)
    return ports


def split_ports(ports: Sequence[int], collectors: int) -> list[list[int]]:
    if collectors < 1 or len(ports) % collectors:
        raise ValueError("collector count must divide Worker port count")
    width = len(ports) // collectors
    return [list(ports[index * width:(index + 1) * width]) for index in range(collectors)]


def continuation_identity(path: Path) -> tuple[int, int]:
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    kind = value.get("kind") if isinstance(value, Mapping) else None
    if kind == "cr_native_expert_selfplay_checkpoint_v1":
        return int(value["global_update"]), 0
    if kind == "cr_native_expert_selfplay_stage2_checkpoint_v1":
        return int(value["global_update"]), int(value["policy_version"])
    raise RuntimeError("unsupported Stage-2 continuation checkpoint")


def collector_command(
    args: argparse.Namespace,
    *,
    ports: Sequence[int],
    run_dir: Path,
    seed: int,
    policy_version: int,
    behavior_checkpoint: Path,
) -> list[str]:
    command = [
        str(args.python), str(args.collect_script.resolve()),
        "--checkpoint", str(behavior_checkpoint),
        "--opponent-checkpoint", str(args.base_opponent_checkpoint.resolve()),
        "--expert-manifest", str(args.expert_manifest.resolve()),
        "--ports", ",".join(str(port) for port in ports),
        "--host", str(args.host),
        "--run-dir", str(run_dir),
        "--learner-deck", str(args.learner_deck.resolve()),
        "--opponent-deck-root", str(args.opponent_deck_root.resolve()),
        "--episodes", str(len(ports)),
        "--updates", "1",
        "--collect-only",
        "--policy-version", str(policy_version),
        "--curriculum-stage", "stage2_reaction",
        "--opponent-policy-id", "BASE",
        "--step-ticks", str(args.step_ticks),
        "--max-decisions", str(args.max_decisions),
        "--timeout", str(args.timeout),
        "--seed", str(seed),
        "--device", str(args.device),
        "--cpu-threads", str(args.collector_cpu_threads),
    ]
    idle_step_ticks = getattr(args, "idle_step_ticks", None)
    if idle_step_ticks is not None:
        command.extend(("--idle-step-ticks", str(idle_step_ticks)))
    collection_waves = int(getattr(args, "collection_waves", 1))
    if collection_waves != 1:
        command.extend(("--collection-waves", str(collection_waves)))
    if bool(getattr(args, "async_shard_writes", False)):
        command.append("--async-shard-writes")
    if bool(getattr(args, "rolling_collection", False)):
        command.append("--rolling-collection")
    if bool(getattr(args, "compile_actor", False)):
        command.extend((
            "--compile-actor",
            "--compile-batch-size", str(args.compile_batch_size),
            "--compile-entity-slots", str(args.compile_entity_slots),
        ))
    if bool(getattr(args, "dense_policy_sampling", False)):
        command.append("--dense-policy-sampling")
    policy_server_address = getattr(args, "policy_server_address", None)
    if policy_server_address:
        command.extend(("--policy-server-address", str(policy_server_address)))
    return command


def trainer_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    continuation: Path,
    shards: Sequence[Path],
) -> list[str]:
    command = [
        str(args.python), str(args.train_script.resolve()),
        "--base-checkpoint", str(args.base_checkpoint.resolve()),
        "--continuation-checkpoint", str(continuation),
        "--expert-manifest", str(args.expert_manifest.resolve()),
        "--run-dir", str(run_dir),
        "--device", str(args.device),
        "--cpu-threads", str(args.trainer_cpu_threads),
        "--ppo-epochs", str(args.ppo_epochs),
        "--chunk-batch-size", str(args.chunk_batch_size),
        "--preprocess-window-size", str(
            getattr(args, "preprocess_window_size", 256)
        ),
        "--preprocess-batch-size", str(
            getattr(args, "preprocess_batch_size", 3)
        ),
        "--retain-checkpoints", str(args.retain_checkpoints),
    ]
    for shard in shards:
        command.extend(("--shard", str(shard)))
    if getattr(args, "prepared_cache_gib", 4.0) != 4.0:
        command.extend(("--prepared-cache-gib", str(args.prepared_cache_gib)))
    if getattr(args, "training_precision", "float32") != "float32":
        command.extend(("--training-precision", str(args.training_precision)))
    if bool(getattr(args, "fused_optimizer", False)):
        command.append("--fused-optimizer")
    if int(getattr(args, "chunk_padding_multiple", 0)):
        command.extend(("--chunk-padding-multiple", str(args.chunk_padding_multiple)))
    return command


def ensure_workers_command(
    args: argparse.Namespace,
    *,
    ports: Sequence[int],
) -> list[str] | None:
    if bool(getattr(args, "skip_worker_restore", False)):
        return None
    script = getattr(args, "ensure_workers_script", None)
    if script is None:
        return None
    expected = list(range(min(ports), min(ports) + len(ports)))
    if list(ports) != expected:
        raise ValueError("automatic Worker restore requires one contiguous port range")
    runtime_root = getattr(args, "worker_runtime_root", None)
    if runtime_root is None:
        raise ValueError("automatic Worker restore requires --worker-runtime-root")
    return [
        str(args.python), str(Path(script).resolve()),
        "--runtime-root", str(Path(runtime_root).resolve()),
        "--base-port", str(min(ports)),
        "--count", str(len(ports)),
        "--execution-mode", str(
            getattr(args, "worker_execution_mode", "interpreter")
        ),
        "--ready-timeout", str(getattr(args, "worker_ready_timeout", 120.0)),
    ]


def validate_collection(run_dir: Path) -> Path:
    return validate_collection_shards(run_dir)[0]


def wait_collectors(
    processes: Sequence[subprocess.Popen],
    collection_dirs: Sequence[Path],
    *,
    prepare_shard: Callable[[Path], None] | None = None,
    timeout_seconds: float = 900.0,
) -> int:
    """Prepare atomically published shards while other native games continue."""
    deadline = time.monotonic() + timeout_seconds
    queued: set[Path] = set()
    futures = []
    pool = None if prepare_shard is None else ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="rollout-preparation"
    )
    try:
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                details = {}
                for index, code in enumerate(codes):
                    if code not in (None, 0):
                        log = collection_dirs[index].with_suffix(".log")
                        if log.is_file():
                            details[str(log)] = log.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"Stage-2 collector failure codes: {codes}; logs={details}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Stage-2 collection/preparation exceeded its time limit")
            if pool is not None:
                for directory in collection_dirs:
                    for shard in sorted(directory.glob("rollouts/shard-*")):
                        if shard not in queued and (shard / "manifest.json").is_file():
                            queued.add(shard)
                            futures.append(pool.submit(prepare_shard, shard))
                for future in futures:
                    if future.done():
                        future.result()
            if all(code == 0 for code in codes):
                break
            time.sleep(0.25)
        for future in futures:
            future.result(timeout=max(0.01, deadline - time.monotonic()))
        return len(queued)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def validate_collection_shards(run_dir: Path) -> list[Path]:
    result = read_object(run_dir / "collection-result.json")
    if result.get("status") != "collected" or result.get("ledger_state") != "CLOSED":
        raise RuntimeError(f"Stage-2 collector did not close safely: {run_dir}")
    raw_shards = result.get("shards")
    paths = (
        [Path(str(row.get("directory", ""))).resolve() for row in raw_shards]
        if isinstance(raw_shards, list) and raw_shards
        else [Path(str(result.get("shard", ""))).resolve()]
    )
    if any(not path.is_dir() for path in paths):
        raise RuntimeError(f"Stage-2 collector shard is missing: {run_dir}")
    return paths


def validate_update(run_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    result = read_object(run_dir / "result.json")
    if result.get("status") != "completed" or result.get("guard", {}).get("action") != "accept":
        raise RuntimeError(f"Stage-2 guarded update did not commit: {run_dir}")
    if any(state != "COMMITTED" for state in result.get("ledger_states", [])):
        raise RuntimeError(f"Stage-2 update has an open rollout ledger: {run_dir}")
    if any(
        not math.isfinite(float(value))
        for value in result.get("metrics", {}).values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError(f"Stage-2 update contains non-finite metrics: {run_dir}")
    checkpoint = Path(str(result["checkpoint"])).resolve()
    export = Path(str(result["behavior_export"])).resolve()
    if not checkpoint.is_file() or not export.is_file():
        raise RuntimeError(f"Stage-2 update artifacts are missing: {run_dir}")
    return result, checkpoint, export


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "overlap_preparation", False)) and not bool(
        getattr(args, "persistent_learner", False)
    ):
        raise ValueError("overlap preparation requires a persistent learner")
    ports = parse_ports(args.ports)
    groups = split_ports(ports, args.collectors)
    if args.updates < 1 or not 1 <= args.step_ticks <= 16:
        raise ValueError("Stage-2 update/tick limits are invalid")
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"Stage-2 loop directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    progress: dict[str, Any] = {
        "kind": KIND,
        "status": "running",
        "created_utc": utc_now(),
        "target_updates": args.updates,
        "completed_updates": [],
        "active_update": None,
        "latest_checkpoint": str(args.initial_continuation.resolve(strict=True)),
        "latest_behavior_export": str(
            (
                args.base_checkpoint
                if getattr(args, "initial_behavior_export", None) is None
                else args.initial_behavior_export
            ).resolve(strict=True)
        ),
    }
    atomic_json(run_root / "progress.json", progress)
    events = run_root / "events.jsonl"
    mps_runtime = ManagedMPS(
        enabled=bool(getattr(args, "enable_mps", False)),
        root=(
            getattr(args, "mps_root", None)
            or (run_root / "mps-runtime")
        ),
    )
    process_environment = os.environ.copy()
    resident_learner = None

    def event(name: str, **fields: Any) -> None:
        with events.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(
                {"at_utc": utc_now(), "event": name, **fields},
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
            output.flush()
            os.fsync(output.fileno())

    try:
        process_environment = mps_runtime.start()
        event(
            "mps_runtime_ready",
            enabled=bool(getattr(args, "enable_mps", False)),
            root=str(mps_runtime.root),
        )
        if bool(getattr(args, "persistent_learner", False)):
            if args.train_script.resolve() != TRAIN_SCRIPT.resolve():
                raise ValueError("persistent learner requires the production Stage-2 trainer")
            from scripts.train_expert_selfplay_stage2 import PersistentStage2Learner

            resident_learner = PersistentStage2Learner()
        for update_index in range(1, args.updates + 1):
            update_started_at = time.monotonic()
            free_gib = shutil.disk_usage(run_root).free / 1024**3
            if free_gib < args.minimum_free_gb:
                raise RuntimeError(
                    f"disk safety gate: {free_gib:.2f} GiB free is below "
                    f"{args.minimum_free_gb:.2f} GiB"
                )
            continuation = Path(progress["latest_checkpoint"]).resolve(strict=True)
            behavior = Path(progress["latest_behavior_export"]).resolve(strict=True)
            global_update, policy_version = continuation_identity(continuation)
            update_dir = run_root / f"update-{update_index:06d}-policy-{policy_version:06d}"
            update_dir.mkdir()
            progress["active_update"] = {
                "index": update_index,
                "policy_version": policy_version,
                "global_update": global_update,
                "state": "restoring_workers",
                "run_dir": str(update_dir),
            }
            atomic_json(run_root / "progress.json", progress)
            event("policy_batch_started", update=update_index, policy_version=policy_version)

            restore_command = ensure_workers_command(args, ports=ports)
            if restore_command is not None:
                with (update_dir / "workers.log").open("w", encoding="utf-8") as log:
                    restored = subprocess.run(
                        restore_command,
                        cwd=str(PROJECT_ROOT),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=process_environment,
                    )
                if restored.returncode != 0:
                    raise RuntimeError(
                        f"Stage-2 Worker restore failed with exit code "
                        f"{restored.returncode}: {update_dir / 'workers.log'}"
                    )
                event("worker_fleet_ready", update=update_index, workers=len(ports))
            progress["active_update"]["state"] = "collecting"
            atomic_json(run_root / "progress.json", progress)
            overlap_preparation = bool(getattr(args, "overlap_preparation", False))
            if overlap_preparation:
                resident_learner.initialize(argparse.Namespace(**{
                    **vars(args), "continuation_checkpoint": continuation,
                }))
                event("resident_learner_ready_for_preparation", update=update_index)

            processes = []
            logs = []
            collection_dirs = []
            try:
                for collector_index, group in enumerate(groups):
                    collection_dir = update_dir / f"collect-p{collector_index:02d}"
                    collection_dirs.append(collection_dir)
                    log = (update_dir / f"collect-p{collector_index:02d}.log").open(
                        "w", encoding="utf-8"
                    )
                    logs.append(log)
                    process = subprocess.Popen(
                        collector_command(
                            args,
                            ports=group,
                            run_dir=collection_dir,
                            seed=args.seed + (update_index - 1) * len(groups) + collector_index,
                            policy_version=policy_version,
                            behavior_checkpoint=behavior,
                        ),
                        cwd=str(PROJECT_ROOT),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=process_environment,
                    )
                    processes.append(process)
                prepared_count = wait_collectors(
                    processes, collection_dirs,
                    prepare_shard=resident_learner.prepare if overlap_preparation else None,
                    timeout_seconds=float(getattr(args, "collector_timeout_seconds", 900.0)),
                )
            finally:
                for log in logs:
                    log.close()
            shards = [
                shard
                for path in collection_dirs
                for shard in validate_collection_shards(path)
            ]
            progress["active_update"]["state"] = "training"
            progress["active_update"]["shards"] = [str(path) for path in shards]
            atomic_json(run_root / "progress.json", progress)
            event("policy_batch_closed", update=update_index, shards=len(shards))
            event("overlapped_preparation_complete", update=update_index, shards=prepared_count)

            train_dir = update_dir / "learner"
            with (update_dir / "learner.log").open("w", encoding="utf-8") as log:
                command = trainer_command(
                    args, run_dir=train_dir,
                    continuation=continuation, shards=shards,
                )
                if resident_learner is None:
                    trained = subprocess.run(
                        command, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, check=False,
                        env=process_environment,
                        timeout=float(getattr(args, "learner_timeout_seconds", 900.0)),
                    )
                    training_returncode = trained.returncode
                else:
                    from scripts.train_expert_selfplay_stage2 import build_parser as learner_parser

                    with redirect_stdout(log), redirect_stderr(log):
                        trained_result = resident_learner.run(learner_parser().parse_args(command[2:]))
                        print(json.dumps(trained_result, ensure_ascii=False, allow_nan=False))
                    training_returncode = 0
            if training_returncode != 0:
                raise RuntimeError(
                    f"Stage-2 learner failed with exit code {training_returncode}: {train_dir}"
                )
            result, checkpoint, export = validate_update(train_dir)
            if int(result["policy_version"]) != policy_version + 1:
                raise RuntimeError("Stage-2 policy version did not advance exactly once")
            record = {
                "index": update_index,
                "pre_policy_version": policy_version,
                "policy_version": int(result["policy_version"]),
                "global_update": int(result["global_update"]),
                "checkpoint": str(checkpoint),
                "behavior_export": str(export),
                "metrics": result["metrics"],
                "retry_attempt": int(result["retry_attempt"]),
                "completed_utc": utc_now(),
                "episodes": int(result.get("rollout", {}).get("episodes", 0)),
                "decisions": int(result.get("rollout", {}).get("decisions", 0)),
                "elapsed_seconds": time.monotonic() - update_started_at,
                "overlap_preparation": overlap_preparation,
            }
            record["completed_games_per_day_estimate"] = (
                record["episodes"] * 86400 / max(record["elapsed_seconds"], 1e-9)
            )
            progress["completed_updates"].append(record)
            progress["latest_checkpoint"] = str(checkpoint)
            progress["latest_behavior_export"] = str(export)
            progress["active_update"] = None
            atomic_json(run_root / "progress.json", progress)
            event("guarded_policy_update_committed", **record)

            evicted_files, evicted_bytes = evict_rollout_cache(update_dir)
            event(
                "rollout_file_cache_evicted",
                update=update_index,
                files=evicted_files,
                bytes=evicted_bytes,
            )

            # Keep exact hashes/manifests but prune old heavy rollout payloads.
            completed = progress["completed_updates"]
            if len(completed) > args.retain_rollout_updates:
                stale_index = len(completed) - args.retain_rollout_updates - 1
                stale_dir = run_root / (
                    f"update-{stale_index + 1:06d}-policy-"
                    f"{completed[stale_index]['pre_policy_version']:06d}"
                )
                for payload in stale_dir.glob("collect-p*/rollouts/shard-*/rollout.pt"):
                    payload.resolve(strict=True).relative_to(stale_dir.resolve())
                    payload.unlink()
            if len(completed) > args.retain_artifact_updates:
                stale_index = len(completed) - args.retain_artifact_updates - 1
                stale_dir = run_root / (
                    f"update-{stale_index + 1:06d}-policy-"
                    f"{completed[stale_index]['pre_policy_version']:06d}"
                ) / "learner"
                for pattern in ("checkpoints/*.pt", "exports/*.pt", "pre-update/*.pt"):
                    for payload in stale_dir.glob(pattern):
                        payload.resolve(strict=True).relative_to(stale_dir.resolve())
                        payload.unlink()

        progress["status"] = "completed"
        progress["completion_reason"] = "requested_updates_committed"
        atomic_json(run_root / "progress.json", progress)
        event("stage2_loop_completed", updates=len(progress["completed_updates"]))
        return progress
    except BaseException as error:
        progress["status"] = "failed"
        progress["last_error"] = {
            "at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(run_root / "progress.json", progress)
        event(
            "stage2_loop_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
        )
        raise
    finally:
        if resident_learner is not None:
            resident_learner.close()
        stopped = mps_runtime.stop()
        if stopped["requested"]:
            event("mps_runtime_stopped", **stopped)


def run_isolated(args: argparse.Namespace) -> dict[str, Any]:
    """Recycle the CUDA context after each committed version, keeping native Workers."""
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    progress = {
        "kind": KIND, "status": "running", "created_utc": utc_now(),
        "target_updates": args.updates, "completed_updates": [], "active_update": None,
        "latest_checkpoint": str(args.initial_continuation.resolve(strict=True)),
        "latest_behavior_export": str((
            getattr(args, "initial_behavior_export", None) or args.base_checkpoint
        ).resolve(strict=True)),
        "cuda_process_per_update": True,
    }
    runtime = ManagedMPS(
        enabled=bool(getattr(args, "enable_mps", False)),
        root=getattr(args, "mps_root", None) or root / "mps-runtime",
    )

    def event(name, **values):
        with (root / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"at_utc": utc_now(), "event": name, **values},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        atomic_json(root / "progress.json", progress)

    event("isolated_pipeline_started")
    try:
        environment = runtime.start()
        for index in range(1, args.updates + 1):
            child_root = root / f"cycle-{index:06d}"
            values = {
                **vars(args), "run_root": child_root, "updates": 1,
                "initial_continuation": Path(progress["latest_checkpoint"]),
                "initial_behavior_export": Path(progress["latest_behavior_export"]),
                "isolate_updates": False, "enable_mps": False, "mps_root": None,
                "seed": args.seed + (index - 1) * args.collectors,
            }
            if getattr(args, "ensure_workers_script", None) is None:
                values["skip_worker_restore"] = True
            command = [str(args.python), str(Path(__file__).resolve())]
            for name, value in values.items():
                if value is None or value is False:
                    continue
                option = "--" + name.replace("_", "-")
                if value is True:
                    command.append(option)
                else:
                    command.extend((option, str(value)))
            progress["active_update"] = {"index": index, "run_dir": str(child_root)}
            event("isolated_update_started", update=index, run_dir=str(child_root))
            started = time.monotonic()
            with (root / f"cycle-{index:06d}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                    timeout=float(getattr(args, "collector_timeout_seconds", 900.0))
                    + float(getattr(args, "learner_timeout_seconds", 900.0)),
                )
            if result.returncode:
                child_progress = child_root / "progress.json"
                details = read_object(child_progress).get("last_error") if child_progress.is_file() else None
                raise RuntimeError(f"isolated update {index} failed: {details or result.returncode}")
            child = read_object(child_root / "progress.json")
            if child.get("status") != "completed" or len(child.get("completed_updates", [])) != 1:
                raise RuntimeError("isolated update did not publish exactly one accepted policy")
            record = dict(child["completed_updates"][0])
            record.update(index=index, elapsed_seconds=time.monotonic() - started)
            record["completed_games_per_day_estimate"] = (
                int(record.get("episodes", 0)) * 86400 / max(record["elapsed_seconds"], 1e-9)
            )
            progress["completed_updates"].append(record)
            progress["latest_checkpoint"] = record["checkpoint"]
            progress["latest_behavior_export"] = record["behavior_export"]
            progress["active_update"] = None
            event("guarded_policy_update_committed", **record)
            completed = progress["completed_updates"]
            for stale in completed[-args.retain_rollout_updates - 1:-args.retain_rollout_updates]:
                directory = Path(stale["checkpoint"]).parents[2].resolve()
                directory.relative_to(root)
                for path in directory.glob("collect-p*/rollouts/shard-*/rollout.pt"):
                    path.resolve(strict=True).relative_to(root)
                    path.unlink()
            for stale in completed[-args.retain_artifact_updates - 1:-args.retain_artifact_updates]:
                directory = Path(stale["checkpoint"]).parents[1].resolve()
                directory.relative_to(root)
                for pattern in ("checkpoints/*.pt", "exports/*.pt", "pre-update/*.pt"):
                    for path in directory.glob(pattern):
                        path.resolve(strict=True).relative_to(root)
                        path.unlink()
            if index < args.updates:
                runtime.stop()
                environment = runtime.start()
                event("cuda_context_recycled", update=index)
        progress.update(status="completed", completion_reason="requested_updates_committed")
        event("isolated_pipeline_completed")
        return progress
    except BaseException as error:
        progress.update(status="failed", last_error={"at_utc": utc_now(),
                        "error_type": type(error).__name__, "error": str(error)})
        event("isolated_pipeline_failed", **progress["last_error"])
        raise
    finally:
        runtime.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-continuation", type=Path, required=True)
    parser.add_argument("--initial-behavior-export", type=Path)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--collectors", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--step-ticks", type=int, default=4)
    parser.add_argument("--idle-step-ticks", type=int)
    parser.add_argument("--collection-waves", type=int, default=1)
    parser.add_argument("--async-shard-writes", action="store_true")
    parser.add_argument("--rolling-collection", action="store_true")
    parser.add_argument("--compile-actor", action="store_true")
    parser.add_argument("--compile-batch-size", type=int)
    parser.add_argument("--compile-entity-slots", type=int)
    parser.add_argument("--dense-policy-sampling", action="store_true")
    parser.add_argument("--enable-mps", action="store_true")
    parser.add_argument("--mps-root", type=Path)
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20500000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--collector-cpu-threads", type=int, default=2)
    parser.add_argument("--trainer-cpu-threads", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--preprocess-window-size", type=int, default=256)
    parser.add_argument("--preprocess-batch-size", type=int, default=3)
    parser.add_argument("--prepared-cache-gib", type=float, default=4.0)
    parser.add_argument("--training-precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--chunk-padding-multiple", type=int, default=0)
    parser.add_argument("--persistent-learner", action="store_true")
    parser.add_argument("--isolate-updates", action="store_true")
    parser.add_argument("--skip-worker-restore", action="store_true")
    parser.add_argument("--overlap-preparation", action="store_true")
    parser.add_argument("--collector-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--learner-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument("--retain-rollout-updates", type=int, default=2)
    parser.add_argument("--retain-artifact-updates", type=int, default=3)
    parser.add_argument("--minimum-free-gb", type=float, default=25.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--collect-script", type=Path, default=COLLECT_SCRIPT)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--ensure-workers-script", type=Path, default=ENSURE_WORKERS_SCRIPT)
    parser.add_argument(
        "--worker-runtime-root",
        type=Path,
        default=PROJECT_ROOT / "bionic-runtime",
    )
    parser.add_argument("--worker-ready-timeout", type=float, default=120.0)
    parser.add_argument(
        "--worker-execution-mode",
        choices=("interpreter", "jit"),
        default="interpreter",
        help="Bionic Java host mode used when restoring missing Workers",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(
        args.collectors, args.updates, args.collector_cpu_threads,
        args.trainer_cpu_threads, args.ppo_epochs, args.chunk_batch_size,
        args.preprocess_window_size,
        args.preprocess_batch_size,
        args.collection_waves,
        args.retain_checkpoints, args.retain_rollout_updates,
        args.retain_artifact_updates,
    ) < 1:
        raise ValueError("Stage-2 loop values must be positive")
    runner = run_isolated if args.isolate_updates else run
    print(json.dumps(runner(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
