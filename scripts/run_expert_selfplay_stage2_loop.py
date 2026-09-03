"""One-command strict-version Stage-2 native self-play PPO loop."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    return [
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
        "--retain-checkpoints", str(args.retain_checkpoints),
    ]
    for shard in shards:
        command.extend(("--shard", str(shard)))
    return command


def ensure_workers_command(
    args: argparse.Namespace,
    *,
    ports: Sequence[int],
) -> list[str] | None:
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
        "--ready-timeout", str(getattr(args, "worker_ready_timeout", 120.0)),
    ]


def validate_collection(run_dir: Path) -> Path:
    result = read_object(run_dir / "collection-result.json")
    if result.get("status") != "collected" or result.get("ledger_state") != "CLOSED":
        raise RuntimeError(f"Stage-2 collector did not close safely: {run_dir}")
    shard = Path(str(result.get("shard", ""))).resolve()
    if not shard.is_dir():
        raise RuntimeError(f"Stage-2 collector shard is missing: {run_dir}")
    return shard


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

    def event(name: str, **fields: Any) -> None:
        with events.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(
                {"at_utc": utc_now(), "event": name, **fields},
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
            output.flush()
            os.fsync(output.fileno())

    try:
        for update_index in range(1, args.updates + 1):
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
                    )
                if restored.returncode != 0:
                    raise RuntimeError(
                        f"Stage-2 Worker restore failed with exit code "
                        f"{restored.returncode}: {update_dir / 'workers.log'}"
                    )
                event("worker_fleet_ready", update=update_index, workers=len(ports))
            progress["active_update"]["state"] = "collecting"
            atomic_json(run_root / "progress.json", progress)

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
                    )
                    processes.append(process)
                codes = [process.wait() for process in processes]
            finally:
                for log in logs:
                    log.close()
            if any(code != 0 for code in codes):
                raise RuntimeError(f"Stage-2 collector failure codes: {codes}")
            shards = [validate_collection(path) for path in collection_dirs]
            progress["active_update"]["state"] = "training"
            progress["active_update"]["shards"] = [str(path) for path in shards]
            atomic_json(run_root / "progress.json", progress)
            event("policy_batch_closed", update=update_index, shards=len(shards))

            train_dir = update_dir / "learner"
            with (update_dir / "learner.log").open("w", encoding="utf-8") as log:
                trained = subprocess.run(
                    trainer_command(
                        args, run_dir=train_dir,
                        continuation=continuation, shards=shards,
                    ),
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if trained.returncode != 0:
                raise RuntimeError(
                    f"Stage-2 learner failed with exit code {trained.returncode}: {train_dir}"
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
            }
            progress["completed_updates"].append(record)
            progress["latest_checkpoint"] = str(checkpoint)
            progress["latest_behavior_export"] = str(export)
            progress["active_update"] = None
            atomic_json(run_root / "progress.json", progress)
            event("guarded_policy_update_committed", **record)

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
                for pattern in ("checkpoints/*.pt", "exports/*.pt"):
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
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20500000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--collector-cpu-threads", type=int, default=2)
    parser.add_argument("--trainer-cpu-threads", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(
        args.collectors, args.updates, args.collector_cpu_threads,
        args.trainer_cpu_threads, args.ppo_epochs, args.chunk_batch_size,
        args.retain_checkpoints, args.retain_rollout_updates,
        args.retain_artifact_updates,
    ) < 1:
        raise ValueError("Stage-2 loop values must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
