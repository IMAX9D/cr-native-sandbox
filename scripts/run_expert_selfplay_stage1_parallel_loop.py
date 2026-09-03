"""Unattended Stage-1 loop with multi-process native collection.

Each wave starts several collect-only processes on disjoint Worker ports.  Once
all immutable shards are CLOSED, one GPU learner update consumes their union
and publishes the sole continuation checkpoint.  Failures are fail-closed:
the loop never recollects an uncertain wave and never advances past a missing
or uncommitted result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = PROJECT_ROOT / "scripts" / "run_expert_selfplay_v1.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_collected_expert_selfplay_stage1.py"
LOOP_KIND = "cr_native_expert_selfplay_stage1_parallel_loop_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ports(value: str) -> list[int]:
    ports: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("--ports contains an empty item")
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"descending port range: {token}")
            values = range(start, stop + 1)
        else:
            values = (int(token),)
        for port in values:
            if not 1 <= port <= 65535 or port in seen:
                raise ValueError(f"invalid or duplicate port: {port}")
            ports.append(port)
            seen.add(port)
    return ports


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


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking OS lock for the lifetime of the loop."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.tell() == 0 and handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"another loop owns {path}") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another loop owns {path}") from error
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Journal:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.progress_path = root / "progress.json"
        self.events_path = root / "events.jsonl"
        self.started = time.monotonic()

    def write(self, value: Mapping[str, Any]) -> None:
        row = dict(value)
        row["kind"] = LOOP_KIND
        row["updated_utc"] = utc_now()
        row["invocation_elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.progress_path, row)

    def event(self, event: str, **fields: Any) -> None:
        row = {"at_utc": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())


def _configuration(args: argparse.Namespace, ports: Sequence[int]) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "expert_manifest": str(args.expert_manifest.resolve()),
        "ports": list(ports),
        "collectors": int(args.collectors),
        "step_ticks": int(args.step_ticks),
        "host": str(args.host),
        "learner_deck": str(args.learner_deck.resolve()),
        "opponent_deck_root": str(args.opponent_deck_root.resolve()),
        "max_decisions": int(args.max_decisions),
        "timeout": float(args.timeout),
        "base_seed": int(args.seed),
        "device": str(args.device),
        "collector_cpu_threads": int(args.collector_cpu_threads),
        "trainer_cpu_threads": int(args.trainer_cpu_threads),
        "retain_checkpoints": int(args.retain_checkpoints),
        "python": str(args.python),
        "collect_script": str(args.collect_script.resolve()),
        "train_script": str(args.train_script.resolve()),
        "start_resume": str(args.start_resume.resolve()),
        "minimum_free_gb": float(args.minimum_free_gb),
        "target_validation_ev": float(args.target_validation_ev),
        "validation_gate_consecutive": int(args.validation_gate_consecutive),
        "retain_rollout_waves": int(args.retain_rollout_waves),
        "retain_checkpoint_waves": int(args.retain_checkpoint_waves),
    }


def _split_ports(ports: Sequence[int], collectors: int) -> list[list[int]]:
    if collectors < 1 or len(ports) % collectors:
        raise ValueError("--collectors must divide the explicit Worker port count")
    width = len(ports) // collectors
    return [list(ports[index * width:(index + 1) * width]) for index in range(collectors)]


def _port_text(ports: Sequence[int]) -> str:
    return ",".join(str(port) for port in ports)


def _ensure_disk(root: Path, minimum_free_gb: float) -> None:
    free = shutil.disk_usage(root).free
    if free < minimum_free_gb * 1024**3:
        raise RuntimeError(
            f"disk safety gate: {free / 1024**3:.2f} GiB free is below "
            f"{minimum_free_gb:.2f} GiB"
        )


def _prune_committed_history(
    *,
    run_root: Path,
    completed: list[dict[str, Any]],
    retain_rollout_waves: int,
    retain_checkpoint_waves: int,
    journal: Journal,
) -> None:
    """Prune only payloads whose wave result and all child ledgers committed."""

    rollout_cutoff = max(0, len(completed) - retain_rollout_waves)
    checkpoint_cutoff = max(0, len(completed) - retain_checkpoint_waves)
    for index, record in enumerate(completed):
        wave_dir = Path(str(record["wave_dir"])).resolve(strict=True)
        try:
            wave_dir.relative_to(run_root)
        except ValueError as error:
            raise RuntimeError(f"refusing to prune wave outside run root: {wave_dir}") from error
        result = _object(wave_dir / "result.json")
        if result.get("status") != "completed" or not result.get("actor_unchanged"):
            raise RuntimeError(f"refusing to prune uncommitted wave: {wave_dir}")
        if any(state != "COMMITTED" for state in result.get("ledger_states", [])):
            raise RuntimeError(f"refusing to prune wave with open ledgers: {wave_dir}")

        removed_rollouts = 0
        if index < rollout_cutoff and not record.get("rollout_payloads_pruned"):
            for payload in sorted(wave_dir.glob("collect-p*/rollouts/shard-*/rollout.pt")):
                payload.resolve(strict=True).relative_to(wave_dir)
                payload.unlink()
                removed_rollouts += 1
            record["rollout_payloads_pruned"] = True
            record["rollout_payloads_removed"] = removed_rollouts

        removed_checkpoints = 0
        if index < checkpoint_cutoff and not record.get("checkpoint_payloads_pruned"):
            checkpoint_dir = wave_dir / "checkpoints"
            if checkpoint_dir.is_dir():
                for payload in sorted(checkpoint_dir.glob("*.pt")):
                    payload.resolve(strict=True).relative_to(checkpoint_dir.resolve())
                    payload.unlink()
                    removed_checkpoints += 1
            record["checkpoint_payloads_pruned"] = True
            record["checkpoint_payloads_removed"] = removed_checkpoints

        if removed_rollouts or removed_checkpoints:
            record["payloads_pruned_utc"] = utc_now()
            journal.event(
                "committed_wave_payloads_pruned",
                wave=int(record["wave"]),
                rollout_payloads=removed_rollouts,
                checkpoint_payloads=removed_checkpoints,
            )


def _collection_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    ports: Sequence[int],
    seed: int,
) -> list[str]:
    return [
        str(args.python), str(args.collect_script.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--expert-manifest", str(args.expert_manifest.resolve()),
        "--ports", _port_text(ports),
        "--host", str(args.host),
        "--run-dir", str(run_dir),
        "--learner-deck", str(args.learner_deck.resolve()),
        "--opponent-deck-root", str(args.opponent_deck_root.resolve()),
        "--episodes", str(len(ports)),
        "--updates", "1",
        "--collect-only",
        "--step-ticks", str(args.step_ticks),
        "--max-decisions", str(args.max_decisions),
        "--timeout", str(args.timeout),
        "--seed", str(seed),
        "--device", str(args.device),
        "--cpu-threads", str(args.collector_cpu_threads),
        "--retain-checkpoints", str(args.retain_checkpoints),
    ]


def _train_command(
    args: argparse.Namespace,
    *,
    wave_dir: Path,
    collection_dirs: Sequence[Path],
    resume_checkpoint: Path,
) -> list[str]:
    command = [
        str(args.python), str(args.train_script.resolve()),
        "--run-dir", str(wave_dir),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--expert-manifest", str(args.expert_manifest.resolve()),
        "--resume-checkpoint", str(resume_checkpoint),
        "--device", str(args.device),
        "--cpu-threads", str(args.trainer_cpu_threads),
        "--retain-checkpoints", str(args.retain_checkpoints),
    ]
    for collection_dir in collection_dirs:
        command.extend(("--collection-run", str(collection_dir)))
    return command


def _validated_collection(run_dir: Path) -> dict[str, Any]:
    value = _object(run_dir / "collection-result.json")
    if value.get("status") != "collected" or value.get("ledger_state") != "CLOSED":
        raise RuntimeError(f"collection did not publish a CLOSED result: {run_dir}")
    return value


def _validated_wave(wave_dir: Path) -> tuple[dict[str, Any], Path]:
    value = _object(wave_dir / "result.json")
    if value.get("kind") != "cr_native_expert_selfplay_stage1_parallel_update_v1":
        raise RuntimeError(f"unexpected wave result kind: {wave_dir}")
    if value.get("status") != "completed" or not value.get("actor_unchanged"):
        raise RuntimeError(f"wave result failed its commit contract: {wave_dir}")
    if any(state != "COMMITTED" for state in value.get("ledger_states", [])):
        raise RuntimeError(f"wave collection ledgers are not COMMITTED: {wave_dir}")
    checkpoint = Path(str(value.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"wave checkpoint is missing: {checkpoint}")
    return value, checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    ports = parse_ports(args.ports)
    groups = _split_ports(ports, args.collectors)
    if not 1 <= args.step_ticks <= 16 or args.waves < 1:
        raise ValueError("step ticks and waves must be positive and valid")
    if min(args.collector_cpu_threads, args.trainer_cpu_threads, args.retain_checkpoints) < 1:
        raise ValueError("thread and retention values must be positive")
    if args.validation_gate_consecutive < 1:
        raise ValueError("validation gate count must be positive")
    if min(args.retain_rollout_waves, args.retain_checkpoint_waves) < 1:
        raise ValueError("wave retention values must be positive")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args, ports)
    journal = Journal(run_root)

    with exclusive_lock(run_root / ".loop.lock"):
        if journal.progress_path.is_file():
            progress = _object(journal.progress_path)
            if progress.get("kind") != LOOP_KIND or progress.get("configuration") != configuration:
                raise RuntimeError("parallel loop configuration differs from existing progress")
        else:
            progress = {
                "kind": LOOP_KIND,
                "status": "ready",
                "created_utc": utc_now(),
                "configuration": configuration,
                "target_waves": args.waves,
                "completed_waves": [],
                "latest_checkpoint": str(args.start_resume.resolve(strict=True)),
                "active_wave": None,
                "validation_gate_streak": 0,
            }
            journal.write(progress)
            journal.event("loop_created", collectors=args.collectors, workers=len(ports))

        completed = list(progress.get("completed_waves", []))
        if args.waves < len(completed):
            raise ValueError("--waves is below the number of committed waves")
        if progress.get("active_wave") is not None:
            active = dict(progress["active_wave"])
            wave_dir = Path(str(active["wave_dir"])).resolve()
            if (wave_dir / "result.json").is_file():
                result, checkpoint = _validated_wave(wave_dir)
                completed.append({
                    "wave": int(active["wave"]),
                    "wave_dir": str(wave_dir),
                    "checkpoint": str(checkpoint),
                    "global_update": int(result["global_update"]),
                    "episodes": int(result["episodes"]),
                    "decisions": int(result["decisions"]),
                    "fresh_validation_explained_variance": float(
                        result["fresh_validation_before_update"]["explained_variance"]
                    ),
                })
                progress["completed_waves"] = completed
                progress["latest_checkpoint"] = str(checkpoint)
                progress["active_wave"] = None
                _prune_committed_history(
                    run_root=run_root,
                    completed=completed,
                    retain_rollout_waves=args.retain_rollout_waves,
                    retain_checkpoint_waves=args.retain_checkpoint_waves,
                    journal=journal,
                )
                journal.write(progress)
            else:
                raise RuntimeError(
                    f"uncertain active wave preserved; refusing automatic recollection: {wave_dir}"
                )

        progress["status"] = "running"
        progress["target_waves"] = args.waves
        progress.pop("last_error", None)
        journal.write(progress)
        try:
            validation_gate_streak = 0
            for record in reversed(completed):
                if float(record.get("fresh_validation_explained_variance", -1.0)) < args.target_validation_ev:
                    break
                validation_gate_streak += 1
            progress["validation_gate_streak"] = validation_gate_streak
            gate_met = validation_gate_streak >= args.validation_gate_consecutive
            while len(completed) < args.waves and not gate_met:
                _ensure_disk(run_root, args.minimum_free_gb)
                wave_number = len(completed) + 1
                wave_dir = run_root / f"wave-{wave_number:06d}"
                if wave_dir.exists():
                    raise RuntimeError(f"unclaimed wave directory already exists: {wave_dir}")
                wave_dir.mkdir()
                collection_dirs = [
                    wave_dir / f"collect-p{index:02d}"
                    for index in range(len(groups))
                ]
                resume_checkpoint = Path(str(progress["latest_checkpoint"])).resolve(strict=True)
                active = {
                    "wave": wave_number,
                    "wave_dir": str(wave_dir),
                    "state": "collecting",
                    "resume_checkpoint": str(resume_checkpoint),
                    "started_utc": utc_now(),
                }
                progress["active_wave"] = active
                journal.write(progress)
                journal.event("wave_collection_started", wave=wave_number)

                processes: list[subprocess.Popen[Any]] = []
                logs = []
                try:
                    for index, (group, collection_dir) in enumerate(zip(groups, collection_dirs, strict=True)):
                        seed = args.seed + (wave_number - 1) * len(groups) + index
                        command = _collection_command(
                            args, run_dir=collection_dir, ports=group, seed=seed
                        )
                        log = (wave_dir / f"collect-p{index:02d}.log").open("w", encoding="utf-8")
                        logs.append(log)
                        process = subprocess.Popen(
                            command,
                            cwd=str(PROJECT_ROOT),
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        processes.append(process)
                    atomic_json(
                        wave_dir / "collector-processes.json",
                        {
                            "started_utc": utc_now(),
                            "pids": [process.pid for process in processes],
                            "collection_runs": [str(path) for path in collection_dirs],
                        },
                    )
                    returncodes = [process.wait() for process in processes]
                finally:
                    for log in logs:
                        log.close()
                if any(code != 0 for code in returncodes):
                    raise RuntimeError(f"parallel collector failure codes: {returncodes}")
                collection_results = [_validated_collection(path) for path in collection_dirs]
                active["state"] = "training"
                active["collection_episodes"] = sum(int(row["episodes"]) for row in collection_results)
                active["collection_decisions"] = sum(int(row["decisions"]) for row in collection_results)
                progress["active_wave"] = active
                journal.write(progress)
                journal.event(
                    "wave_collection_completed",
                    wave=wave_number,
                    episodes=active["collection_episodes"],
                    decisions=active["collection_decisions"],
                )

                train_log_path = wave_dir / "train.log"
                with train_log_path.open("w", encoding="utf-8") as train_log:
                    training = subprocess.run(
                        _train_command(
                            args,
                            wave_dir=wave_dir,
                            collection_dirs=collection_dirs,
                            resume_checkpoint=resume_checkpoint,
                        ),
                        cwd=str(PROJECT_ROOT),
                        stdin=subprocess.DEVNULL,
                        stdout=train_log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if training.returncode != 0:
                    raise RuntimeError(
                        f"parallel trainer failed with exit code {training.returncode}: {train_log_path}"
                    )
                result, checkpoint = _validated_wave(wave_dir)
                record = {
                    "wave": wave_number,
                    "wave_dir": str(wave_dir),
                    "checkpoint": str(checkpoint),
                    "global_update": int(result["global_update"]),
                    "episodes": int(result["episodes"]),
                    "decisions": int(result["decisions"]),
                    "fresh_validation_explained_variance": float(
                        result["fresh_validation_before_update"]["explained_variance"]
                    ),
                    "completed_utc": utc_now(),
                }
                completed.append(record)
                if record["fresh_validation_explained_variance"] >= args.target_validation_ev:
                    progress["validation_gate_streak"] = int(
                        progress.get("validation_gate_streak", 0)
                    ) + 1
                else:
                    progress["validation_gate_streak"] = 0
                gate_met = (
                    int(progress["validation_gate_streak"])
                    >= args.validation_gate_consecutive
                )
                progress["completed_waves"] = completed
                progress["latest_checkpoint"] = str(checkpoint)
                progress["active_wave"] = None
                journal.write(progress)
                journal.event("wave_completed", **record)

            progress["status"] = "completed"
            progress["completion_reason"] = (
                "stage1_validation_gate_met" if gate_met else "maximum_waves_reached"
            )
            progress["active_wave"] = None
            journal.write(progress)
            journal.event(
                "loop_completed",
                waves=len(completed),
                latest_checkpoint=progress["latest_checkpoint"],
            )
            return progress
        except BaseException as error:
            progress["status"] = "failed"
            progress["last_error"] = {
                "at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            journal.write(progress)
            journal.event(
                "loop_failed",
                error_type=type(error).__name__,
                error=str(error),
                traceback="".join(traceback.format_exception(error)),
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--collectors", type=int, required=True)
    parser.add_argument("--step-ticks", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--waves", type=int, required=True)
    parser.add_argument("--start-resume", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=3_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20270000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--collector-cpu-threads", type=int, default=2)
    parser.add_argument("--trainer-cpu-threads", type=int, default=16)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument("--minimum-free-gb", type=float, default=25.0)
    parser.add_argument("--target-validation-ev", type=float, default=0.20)
    parser.add_argument("--validation-gate-consecutive", type=int, default=3)
    parser.add_argument("--retain-rollout-waves", type=int, default=2)
    parser.add_argument("--retain-checkpoint-waves", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--collect-script", type=Path, default=COLLECT_SCRIPT)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
