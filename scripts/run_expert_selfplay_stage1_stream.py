"""Continuous producer-consumer Stage-1 Critic warm-up.

Native collectors run in independent lanes and immediately enqueue each CLOSED
rollout shard.  A single Critic learner consumes one shard at a time while the
other lanes keep producing.  This is valid only while the behavior Actor is
frozen; Actor PPO stages must restore a strict policy-version barrier.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = PROJECT_ROOT / "scripts" / "run_expert_selfplay_v1.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_collected_expert_selfplay_stage1.py"
KIND = "cr_native_expert_selfplay_stage1_stream_v1"


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
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


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
            candidates = range(start, stop + 1)
        else:
            candidates = (int(token),)
        for port in candidates:
            if not 1 <= port <= 65535 or port in seen:
                raise ValueError(f"invalid or duplicate port: {port}")
            ports.append(port)
            seen.add(port)
    return ports


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"another stream owns {path}") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another stream owns {path}") from error
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
        self.progress_path = root / "progress.json"
        self.events_path = root / "events.jsonl"
        self.started = time.monotonic()

    def write(self, progress: Mapping[str, Any]) -> None:
        value = dict(progress)
        value["kind"] = KIND
        value["updated_utc"] = utc_now()
        value["invocation_elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.progress_path, value)

    def event(self, event: str, **fields: Any) -> None:
        row = {"at_utc": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())


def split_ports(ports: Sequence[int], lanes: int) -> list[list[int]]:
    if lanes < 1 or len(ports) % lanes:
        raise ValueError("--lanes must exactly divide the Worker port count")
    width = len(ports) // lanes
    return [list(ports[index * width:(index + 1) * width]) for index in range(lanes)]


def configuration(args: argparse.Namespace, ports: Sequence[int]) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "expert_manifest": str(args.expert_manifest.resolve()),
        "ports": list(ports),
        "lanes": int(args.lanes),
        "step_ticks": int(args.step_ticks),
        "start_resume": str(args.start_resume.resolve()),
        "learner_deck": str(args.learner_deck.resolve()),
        "opponent_deck_root": str(args.opponent_deck_root.resolve()),
        "host": str(args.host),
        "max_decisions": int(args.max_decisions),
        "timeout": float(args.timeout),
        "base_seed": int(args.seed),
        "device": str(args.device),
        "collector_cpu_threads": int(args.collector_cpu_threads),
        "trainer_cpu_threads": int(args.trainer_cpu_threads),
        "maximum_queue": int(args.maximum_queue),
        "target_validation_ev": float(args.target_validation_ev),
        "validation_gate_consecutive": int(args.validation_gate_consecutive),
        "minimum_free_gb": float(args.minimum_free_gb),
        "retain_rollouts": int(args.retain_rollouts),
        "retain_checkpoints": int(args.retain_checkpoints),
        "python": str(args.python),
        "collect_script": str(args.collect_script.resolve()),
        "train_script": str(args.train_script.resolve()),
    }


def collect_command(
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
        "--ports", ",".join(str(port) for port in ports),
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
    ]


def train_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    collection_run: Path,
    resume_checkpoint: Path,
) -> list[str]:
    return [
        str(args.python), str(args.train_script.resolve()),
        "--run-dir", str(run_dir),
        "--collection-run", str(collection_run),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--expert-manifest", str(args.expert_manifest.resolve()),
        "--resume-checkpoint", str(resume_checkpoint),
        "--device", str(args.device),
        "--cpu-threads", str(args.trainer_cpu_threads),
        "--retain-checkpoints", "1",
        "--validation-chunks", "64",
    ]


def validate_collection(run_dir: Path) -> dict[str, Any]:
    result = read_object(run_dir / "collection-result.json")
    if result.get("status") != "collected" or result.get("ledger_state") != "CLOSED":
        raise RuntimeError(f"collector did not publish a CLOSED shard: {run_dir}")
    if not Path(str(result.get("shard", ""))).is_dir():
        raise RuntimeError(f"collector result shard is missing: {run_dir}")
    return result


def validate_update(run_dir: Path) -> tuple[dict[str, Any], Path]:
    result = read_object(run_dir / "result.json")
    if result.get("status") != "completed" or not result.get("actor_unchanged"):
        raise RuntimeError(f"learner update did not complete safely: {run_dir}")
    if result.get("ledger_states") != ["COMMITTED"]:
        raise RuntimeError(f"learner update did not commit its shard: {run_dir}")
    metrics = result.get("metrics")
    validation = result.get("fresh_validation_before_update")
    for label, values in (("training", metrics), ("validation", validation)):
        if not isinstance(values, Mapping) or any(
            not math.isfinite(float(value))
            for value in values.values()
            if isinstance(value, (int, float))
        ):
            raise RuntimeError(f"{label} metrics are missing or non-finite: {run_dir}")
    checkpoint = Path(str(result.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"learner checkpoint is missing: {checkpoint}")
    return result, checkpoint


def ensure_disk(root: Path, minimum_free_gb: float) -> None:
    free = shutil.disk_usage(root).free / 1024**3
    if free < minimum_free_gb:
        raise RuntimeError(
            f"disk safety gate: {free:.2f} GiB free is below {minimum_free_gb:.2f} GiB"
        )


def prune_payloads(
    *,
    run_root: Path,
    completed: list[dict[str, Any]],
    retain_rollouts: int,
    retain_checkpoints: int,
    journal: Journal,
) -> None:
    rollout_cutoff = max(0, len(completed) - retain_rollouts)
    checkpoint_cutoff = max(0, len(completed) - retain_checkpoints)
    for index, record in enumerate(completed):
        collection_dir = Path(str(record["collection_run"])).resolve(strict=True)
        update_dir = Path(str(record["update_run"])).resolve(strict=True)
        for directory in (collection_dir, update_dir):
            directory.relative_to(run_root)
        if index < rollout_cutoff and not record.get("rollout_pruned"):
            result = validate_collection(collection_dir)
            update_result = read_object(update_dir / "result.json")
            if update_result.get("ledger_states") != ["COMMITTED"]:
                raise RuntimeError(f"refusing to prune an open rollout: {collection_dir}")
            payload = Path(str(result["shard"])) / "rollout.pt"
            if payload.exists():
                payload.resolve(strict=True).relative_to(collection_dir)
                payload.unlink()
            record["rollout_pruned"] = True
            record["rollout_pruned_utc"] = utc_now()
            journal.event("consumed_rollout_pruned", update=record["update"])
        if index < checkpoint_cutoff and not record.get("checkpoint_pruned"):
            checkpoint_dir = update_dir / "checkpoints"
            if checkpoint_dir.is_dir():
                for payload in checkpoint_dir.glob("*.pt"):
                    payload.resolve(strict=True).relative_to(checkpoint_dir.resolve())
                    payload.unlink()
            record["checkpoint_pruned"] = True
            record["checkpoint_pruned_utc"] = utc_now()
            journal.event("old_checkpoint_pruned", update=record["update"])


def terminate(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()


def run(args: argparse.Namespace) -> dict[str, Any]:
    ports = parse_ports(args.ports)
    groups = split_ports(ports, args.lanes)
    if not 1 <= args.step_ticks <= 16:
        raise ValueError("--step-ticks must be in 1..16")
    if min(
        args.max_updates, args.maximum_queue, args.collector_cpu_threads,
        args.trainer_cpu_threads, args.validation_gate_consecutive,
        args.retain_rollouts, args.retain_checkpoints,
    ) < 1:
        raise ValueError("stream limits must be positive")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    journal = Journal(run_root)
    config = configuration(args, ports)

    with exclusive_lock(run_root / ".stream.lock"):
        if journal.progress_path.exists():
            raise RuntimeError(
                "automatic recovery of an interrupted stream is intentionally fail-closed"
            )
        progress: dict[str, Any] = {
            "kind": KIND,
            "status": "running",
            "created_utc": utc_now(),
            "configuration": config,
            "max_updates": int(args.max_updates),
            "latest_checkpoint": str(args.start_resume.resolve(strict=True)),
            "next_collection": 1,
            "next_update": 1,
            "active_collectors": {},
            "queue": [],
            "active_training": None,
            "completed_updates": [],
            "validation_gate_streak": 0,
        }
        journal.write(progress)
        journal.event("stream_started", lanes=len(groups), workers=len(ports))

        collectors: dict[int, tuple[subprocess.Popen[Any], Any, Path]] = {}
        trainer: tuple[subprocess.Popen[Any], Any, Path, Path] | None = None
        completed: list[dict[str, Any]] = []
        queue: list[str] = []

        def launch_collector(lane: int) -> None:
            sequence = int(progress["next_collection"])
            progress["next_collection"] = sequence + 1
            run_dir = run_root / "collections" / f"collection-{sequence:08d}-lane-{lane:02d}"
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            seed = args.seed + sequence - 1
            log = run_dir.with_suffix(".log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                collect_command(args, run_dir=run_dir, ports=groups[lane], seed=seed),
                cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            collectors[lane] = (process, log, run_dir)
            progress["active_collectors"][str(lane)] = {
                "pid": process.pid, "run_dir": str(run_dir), "seed": seed,
                "started_utc": utc_now(),
            }
            journal.event(
                "collector_started", lane=lane, pid=process.pid,
                run_dir=str(run_dir), seed=seed,
            )

        try:
            for lane in range(len(groups)):
                launch_collector(lane)
            journal.write(progress)

            while len(completed) < args.max_updates:
                ensure_disk(run_root, args.minimum_free_gb)

                for lane, (process, log, run_dir) in list(collectors.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del collectors[lane]
                    progress["active_collectors"].pop(str(lane), None)
                    if code != 0:
                        raise RuntimeError(
                            f"collector lane {lane} failed with exit code {code}: {run_dir}"
                        )
                    result = validate_collection(run_dir)
                    queue.append(str(run_dir))
                    progress["queue"] = list(queue)
                    journal.event(
                        "collection_enqueued", lane=lane, run_dir=str(run_dir),
                        episodes=result["episodes"], decisions=result["decisions"],
                        queue_depth=len(queue),
                    )

                if trainer is not None:
                    process, log, update_dir, collection_dir = trainer
                    code = process.poll()
                    if code is not None:
                        log.close()
                        trainer = None
                        if code != 0:
                            raise RuntimeError(
                                f"Critic trainer failed with exit code {code}: {update_dir}"
                            )
                        result, checkpoint = validate_update(update_dir)
                        validation_ev = float(
                            result["fresh_validation_before_update"]["explained_variance"]
                        )
                        if validation_ev >= args.target_validation_ev:
                            progress["validation_gate_streak"] += 1
                        else:
                            progress["validation_gate_streak"] = 0
                        record = {
                            "update": int(progress["next_update"]),
                            "global_update": int(result["global_update"]),
                            "collection_run": str(collection_dir),
                            "update_run": str(update_dir),
                            "checkpoint": str(checkpoint),
                            "episodes": int(result["episodes"]),
                            "decisions": int(result["decisions"]),
                            "validation_loss": float(
                                result["fresh_validation_before_update"]["loss"]
                            ),
                            "validation_explained_variance": validation_ev,
                            "training_loss": float(result["metrics"]["loss"]),
                            "training_explained_variance": float(
                                result["metrics"]["explained_variance"]
                            ),
                            "completed_utc": utc_now(),
                        }
                        progress["next_update"] += 1
                        completed.append(record)
                        progress["completed_updates"] = completed
                        progress["latest_checkpoint"] = str(checkpoint)
                        progress["active_training"] = None
                        prune_payloads(
                            run_root=run_root, completed=completed,
                            retain_rollouts=args.retain_rollouts,
                            retain_checkpoints=args.retain_checkpoints,
                            journal=journal,
                        )
                        journal.event("critic_update_committed", **record)
                        if progress["validation_gate_streak"] >= args.validation_gate_consecutive:
                            progress["status"] = "completed"
                            progress["completion_reason"] = "stage1_validation_gate_met"
                            break
                        if len(completed) >= args.max_updates:
                            progress["status"] = "completed"
                            progress["completion_reason"] = "maximum_updates_reached"
                            break

                if progress.get("status") == "completed":
                    break

                if trainer is None and queue:
                    collection_dir = Path(queue.pop(0)).resolve(strict=True)
                    update_index = int(progress["next_update"])
                    update_dir = run_root / "updates" / f"update-{update_index:08d}"
                    update_dir.parent.mkdir(parents=True, exist_ok=True)
                    resume = Path(str(progress["latest_checkpoint"])).resolve(strict=True)
                    log = update_dir.with_suffix(".log").open("w", encoding="utf-8")
                    process = subprocess.Popen(
                        train_command(
                            args, run_dir=update_dir,
                            collection_run=collection_dir,
                            resume_checkpoint=resume,
                        ),
                        cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    trainer = (process, log, update_dir, collection_dir)
                    progress["queue"] = list(queue)
                    progress["active_training"] = {
                        "pid": process.pid, "update": update_index,
                        "run_dir": str(update_dir),
                        "collection_run": str(collection_dir),
                        "resume_checkpoint": str(resume),
                        "started_utc": utc_now(),
                    }
                    journal.event(
                        "critic_update_started", update=update_index,
                        run_dir=str(update_dir), collection_run=str(collection_dir),
                    )

                for lane in range(len(groups)):
                    if lane not in collectors and len(queue) + len(collectors) < args.maximum_queue:
                        launch_collector(lane)

                journal.write(progress)
                time.sleep(args.poll_seconds)

            if progress.get("status") != "completed":
                progress["status"] = "completed"
                progress["completion_reason"] = "maximum_updates_reached"

            # A normal stop must not terminate a Worker RPC mid-flight.  Let
            # every producer finish its current immutable shard, preserve those
            # surplus shards as CLOSED, and only then exit.  Failure paths below
            # still terminate promptly and retain their partial scene.
            progress["status"] = "draining_collectors"
            journal.write(progress)
            for lane, (process, log, collection_dir) in list(collectors.items()):
                code = process.wait()
                log.close()
                if code != 0:
                    raise RuntimeError(
                        f"collector lane {lane} failed while draining with exit code "
                        f"{code}: {collection_dir}"
                    )
                result = validate_collection(collection_dir)
                queue.append(str(collection_dir))
                journal.event(
                    "surplus_collection_preserved",
                    lane=lane,
                    run_dir=str(collection_dir),
                    episodes=result["episodes"],
                    decisions=result["decisions"],
                )
            collectors.clear()
            progress["active_collectors"] = {}
            progress["queue"] = list(queue)
            progress["active_training"] = None
            progress["status"] = "completed"
            journal.write(progress)
            journal.event(
                "stream_completed", reason=progress["completion_reason"],
                updates=len(completed), latest_checkpoint=progress["latest_checkpoint"],
            )
            return progress
        except BaseException as error:
            progress["status"] = "failed"
            progress["last_error"] = {
                "at_utc": utc_now(), "error_type": type(error).__name__,
                "error": str(error),
            }
            journal.write(progress)
            journal.event(
                "stream_failed", error_type=type(error).__name__,
                error=str(error), traceback="".join(traceback.format_exception(error)),
            )
            raise
        finally:
            running = [value[0] for value in collectors.values()]
            if trainer is not None:
                running.append(trainer[0])
            terminate(running)
            for _process, log, _path in collectors.values():
                log.close()
            if trainer is not None:
                trainer[1].close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--step-ticks", type=int, default=4)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--start-resume", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--maximum-queue", type=int, default=12)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=3_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20300000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--collector-cpu-threads", type=int, default=2)
    parser.add_argument("--trainer-cpu-threads", type=int, default=12)
    parser.add_argument("--target-validation-ev", type=float, default=0.20)
    parser.add_argument("--validation-gate-consecutive", type=int, default=3)
    parser.add_argument("--minimum-free-gb", type=float, default=25.0)
    parser.add_argument("--retain-rollouts", type=int, default=2)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
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
