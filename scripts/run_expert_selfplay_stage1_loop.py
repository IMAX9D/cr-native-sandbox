"""Run sequential fresh-rollout Stage-1 self-play batches.

This is intentionally a thin, fail-closed orchestrator around
``scripts/run_expert_selfplay_v1.py``.  It never collects or trains in-process:
every Critic update gets a new native rollout directory and the checkpoint
committed by one batch is the only continuation checkpoint accepted by the
next batch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_RUNNER = PROJECT_ROOT / "scripts" / "run_expert_selfplay_v1.py"
LOOP_KIND = "cr_native_expert_selfplay_stage1_loop_v1"
LOCK_FILENAME = ".stage1-loop.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON object and reject non-finite JSON values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        dict(value), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_ports(value: str) -> list[int]:
    """Parse an ordered explicit port list such as ``39031,39033-39035``."""

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
            if not 1 <= port <= 65535:
                raise ValueError(f"port outside 1..65535: {port}")
            if port in seen:
                raise ValueError(f"duplicate port: {port}")
            ports.append(port)
            seen.add(port)
    if not ports:
        raise ValueError("--ports requires at least one Worker")
    return ports


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object: {path}")
    return value


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def _result_path(value: object, *, run_dir: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _tail(value: object, limit: int = 8_000) -> str:
    text = "" if value is None else str(value)
    return text[-limit:]


class RunRootLock:
    """Hold one crash-safe, non-blocking OS lock for a loop run root.

    The lock file is intentionally persistent for diagnostics, while the OS
    lock itself is released automatically if the owning process exits.  This
    avoids both concurrent loop writers and stale PID-file deadlocks.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("Stage-1 loop lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            # Windows byte-range locks require the byte to exist.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            handle.close()
            try:
                owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                owner = ""
            detail = f"; owner={owner}" if owner else ""
            raise RuntimeError(
                f"another Stage-1 loop is already active for {self.path.parent}{detail}"
            ) from error

        self._handle = handle
        try:
            owner = json.dumps(
                {"pid": os.getpid(), "acquired_utc": utc_now()},
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            handle.seek(0)
            handle.truncate()
            handle.write(owner)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "RunRootLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class LoopJournal:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.progress_path = root / "progress.json"
        self.events_path = root / "events.jsonl"
        self.started = time.monotonic()

    def event(self, event: str, **fields: Any) -> None:
        row = {"at_utc": utc_now(), "event": event, **fields}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with self.events_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())

    def write(self, progress: Mapping[str, Any]) -> None:
        value = dict(progress)
        value["kind"] = LOOP_KIND
        value["updated_utc"] = utc_now()
        value["invocation_elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.progress_path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--step-ticks", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--start-resume", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--learner-deck", type=Path)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--max-decisions", type=int, default=12_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner-script", type=Path, default=STAGE1_RUNNER)
    return parser


def _configuration(args: argparse.Namespace, ports: Sequence[int]) -> dict[str, Any]:
    """Return immutable loop settings; ``batches`` is extendable by design."""

    return {
        "checkpoint": _resolved(args.checkpoint),
        "expert_manifest": _resolved(args.expert_manifest),
        "ports": list(ports),
        "worker_count": int(args.worker_count),
        "step_ticks": int(args.step_ticks),
        "start_resume": (
            None if args.start_resume is None else _resolved(args.start_resume)
        ),
        "host": str(args.host),
        "learner_deck": (
            None if args.learner_deck is None else _resolved(args.learner_deck)
        ),
        "opponent_deck_root": _resolved(args.opponent_deck_root),
        "runtime_manifest": (
            None
            if args.runtime_manifest is None
            else _resolved(args.runtime_manifest)
        ),
        "max_decisions": int(args.max_decisions),
        "timeout": float(args.timeout),
        "base_seed": int(args.seed),
        "device": str(args.device),
        "cpu_threads": int(args.cpu_threads),
        "retain_checkpoints": int(args.retain_checkpoints),
        "python": str(args.python),
        "runner_script": _resolved(args.runner_script),
    }


def _validate_args(args: argparse.Namespace) -> list[int]:
    ports = parse_ports(args.ports)
    if args.worker_count < 1 or args.worker_count > len(ports):
        raise ValueError("--worker-count must be in 1..len(--ports)")
    if args.step_ticks < 1:
        raise ValueError("--step-ticks must be positive")
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    if args.max_decisions < 1 or args.timeout <= 0 or args.cpu_threads < 1:
        raise ValueError("invalid runtime limit")
    if args.retain_checkpoints < 1:
        raise ValueError("--retain-checkpoints must be positive")
    return ports


def _new_progress(
    *, run_root: Path, configuration: Mapping[str, Any], target_batches: int
) -> dict[str, Any]:
    return {
        "kind": LOOP_KIND,
        "status": "ready",
        "created_utc": utc_now(),
        "run_root": str(run_root),
        "configuration": dict(configuration),
        "target_batches": target_batches,
        "completed_count": 0,
        "next_batch_index": 1,
        "active_batch": None,
        "latest_checkpoint": configuration.get("start_resume"),
        "completed_batches": [],
    }


def _validated_child_result(run_dir: Path) -> tuple[dict[str, Any], Path]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    result = _read_object(result_path, label="child result")
    if result.get("status") != "completed":
        raise RuntimeError(f"child result is not completed: {result_path}")
    if result.get("ledger_state") != "COMMITTED":
        raise RuntimeError(f"child result ledger is not COMMITTED: {result_path}")
    if result.get("run_id") != run_dir.name:
        raise RuntimeError(f"child result run_id differs from directory: {result_path}")
    if int(result.get("updates", -1)) != 1:
        raise RuntimeError(f"child result did not commit exactly one update: {result_path}")
    checkpoint = _result_path(result.get("checkpoint", ""), run_dir=run_dir)
    checkpoint_root = (run_dir / "checkpoints").resolve()
    try:
        checkpoint.relative_to(checkpoint_root)
    except ValueError as error:
        raise RuntimeError(
            f"child checkpoint is outside its checkpoints directory: {checkpoint}"
        ) from error
    if not checkpoint.is_file():
        raise RuntimeError(f"child checkpoint is missing: {checkpoint}")
    return result, checkpoint


def _completed_record(
    *, batch_index: int, seed: int, run_dir: Path, result: Mapping[str, Any], checkpoint: Path
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "seed": seed,
        "run_dir": str(run_dir),
        "result_path": str(run_dir / "result.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "episodes": int(result.get("episodes", 0)),
        "decisions": int(result.get("decisions", 0)),
        "chunks": int(result.get("chunks", 0)),
        "completed_utc": utc_now(),
    }


def _verify_completed_history(progress: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = progress.get("completed_batches", [])
    if not isinstance(raw, list):
        raise RuntimeError("loop progress completed_batches must be a list")
    completed: list[dict[str, Any]] = []
    for expected_index, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            raise RuntimeError("loop progress contains a malformed completed batch")
        if int(value.get("batch_index", -1)) != expected_index:
            raise RuntimeError("completed batch indices must be contiguous from one")
        run_dir = Path(str(value.get("run_dir", ""))).resolve()
        result, checkpoint = _validated_child_result(run_dir)
        recorded_checkpoint = Path(str(value.get("checkpoint", ""))).resolve()
        if recorded_checkpoint != checkpoint:
            raise RuntimeError(
                f"recorded checkpoint differs from committed child result: {run_dir}"
            )
        if int(value.get("seed", -1)) < 0:
            raise RuntimeError(f"completed batch lacks a valid seed: {run_dir}")
        # Retain the durable historical record, while checking the child result
        # rather than trusting progress.json alone.
        completed.append(dict(value))
    return completed


def _child_is_known_failed(run_dir: Path) -> bool:
    child_progress = run_dir / "progress.json"
    if not child_progress.is_file():
        return False
    try:
        return _read_object(child_progress, label="child progress").get("status") == "failed"
    except RuntimeError:
        return False


def _reconcile_active(
    progress: dict[str, Any], journal: LoopJournal
) -> list[dict[str, Any]]:
    """Adopt a committed child after parent interruption without recollecting it."""

    completed = _verify_completed_history(progress)
    active = progress.get("active_batch")
    if active is None:
        return completed
    if not isinstance(active, dict):
        raise RuntimeError("loop progress active_batch must be an object or null")
    batch_index = int(active.get("batch_index", -1))
    if batch_index != len(completed) + 1:
        raise RuntimeError("active batch index does not follow completed history")
    run_dir = Path(str(active.get("run_dir", ""))).resolve()
    if (run_dir / "result.json").is_file():
        result, checkpoint = _validated_child_result(run_dir)
        record = _completed_record(
            batch_index=batch_index,
            seed=int(active["seed"]),
            run_dir=run_dir,
            result=result,
            checkpoint=checkpoint,
        )
        completed.append(record)
        progress["completed_batches"] = completed
        progress["completed_count"] = len(completed)
        progress["next_batch_index"] = len(completed) + 1
        progress["latest_checkpoint"] = str(checkpoint)
        progress["active_batch"] = None
        journal.event(
            "committed_batch_recovered",
            batch_index=batch_index,
            run_dir=str(run_dir),
            checkpoint=str(checkpoint),
        )
        journal.write(progress)
        return completed

    # The parent can be marked failed by Ctrl-C or a runner/wait exception
    # while the child outcome remains unknown.  Only child-specific evidence
    # authorizes a fresh attempt; otherwise fail closed to avoid recollection.
    known_failed = (
        active.get("state") == "failed" or _child_is_known_failed(run_dir)
    )
    if not known_failed:
        raise RuntimeError(
            "active batch outcome is unknown; refusing to duplicate native "
            f"collection: {run_dir}"
        )
    journal.event(
        "failed_batch_preserved_for_manual_retry",
        batch_index=batch_index,
        run_dir=str(run_dir),
    )
    progress["active_batch"] = None
    journal.write(progress)
    return completed


def _unique_run_dir(run_root: Path, *, batch_index: int, seed: int) -> Path:
    base = run_root / f"batch-{batch_index:06d}-seed-{seed}"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = run_root / (
            f"batch-{batch_index:06d}-seed-{seed}-attempt-{attempt:03d}"
        )
        if not candidate.exists():
            return candidate
        attempt += 1


def _child_command(
    args: argparse.Namespace,
    *, run_dir: Path,
    seed: int,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        str(args.python),
        _resolved(args.runner_script),
        "--checkpoint", _resolved(args.checkpoint),
        "--expert-manifest", _resolved(args.expert_manifest),
        "--ports", str(args.ports),
        "--host", str(args.host),
        "--run-dir", str(run_dir),
        "--opponent-deck-root", _resolved(args.opponent_deck_root),
        "--episodes", str(args.worker_count),
        "--updates", "1",
        "--step-ticks", str(args.step_ticks),
        "--max-decisions", str(args.max_decisions),
        "--timeout", str(args.timeout),
        "--seed", str(seed),
        "--device", str(args.device),
        "--cpu-threads", str(args.cpu_threads),
        "--retain-checkpoints", str(args.retain_checkpoints),
    ]
    if args.learner_deck is not None:
        command.extend(("--learner-deck", _resolved(args.learner_deck)))
    if args.runtime_manifest is not None:
        command.extend(("--runtime-manifest", _resolved(args.runtime_manifest)))
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(resume_checkpoint.resolve())))
    return command


def _invoke_runner(runner: Callable[..., Any], command: Sequence[str]) -> Any:
    return runner(
        list(command),
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _run_locked(
    args: argparse.Namespace,
    *,
    ports: Sequence[int],
    run_root: Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    journal = LoopJournal(run_root)
    configuration = _configuration(args, ports)

    if journal.progress_path.exists():
        progress = _read_object(journal.progress_path, label="loop progress")
        if progress.get("kind") != LOOP_KIND:
            raise RuntimeError("run root contains incompatible progress.json")
        if progress.get("configuration") != configuration:
            raise RuntimeError("loop configuration differs from existing progress")
        journal.event(
            "loop_resumed",
            requested_batches=args.batches,
            prior_status=progress.get("status"),
        )
    else:
        unrelated = [
            path for path in run_root.iterdir()
            if path.name not in ("events.jsonl", "progress.json", LOCK_FILENAME)
        ]
        if unrelated:
            raise RuntimeError(
                "run root is non-empty but has no compatible progress.json"
            )
        progress = _new_progress(
            run_root=run_root,
            configuration=configuration,
            target_batches=args.batches,
        )
        journal.write(progress)
        journal.event("loop_created", target_batches=args.batches)

    completed = _reconcile_active(progress, journal)
    if args.batches < len(completed):
        raise ValueError(
            f"--batches {args.batches} is below {len(completed)} committed batches"
        )
    progress["target_batches"] = args.batches
    progress["completed_count"] = len(completed)
    progress["next_batch_index"] = len(completed) + 1
    progress["status"] = "running"
    progress.pop("last_error", None)
    journal.write(progress)

    try:
        while len(completed) < args.batches:
            batch_index = len(completed) + 1
            seed = int(args.seed) + batch_index - 1
            run_dir = _unique_run_dir(
                run_root, batch_index=batch_index, seed=seed
            )
            resume_value = progress.get("latest_checkpoint")
            resume_checkpoint = (
                None if resume_value is None else Path(str(resume_value)).resolve()
            )
            if resume_checkpoint is not None and not resume_checkpoint.is_file():
                raise RuntimeError(
                    f"continuation checkpoint is missing: {resume_checkpoint}"
                )
            command = _child_command(
                args,
                run_dir=run_dir,
                seed=seed,
                resume_checkpoint=resume_checkpoint,
            )
            progress["active_batch"] = {
                "batch_index": batch_index,
                "seed": seed,
                "run_dir": str(run_dir),
                "resume_checkpoint": (
                    None if resume_checkpoint is None else str(resume_checkpoint)
                ),
                "state": "running",
                "started_utc": utc_now(),
                "command": command,
            }
            progress["next_batch_index"] = batch_index
            journal.write(progress)
            journal.event(
                "batch_started",
                batch_index=batch_index,
                seed=seed,
                run_dir=str(run_dir),
                resume_checkpoint=(
                    None if resume_checkpoint is None else str(resume_checkpoint)
                ),
            )

            completed_process = _invoke_runner(runner, command)
            returncode = int(getattr(completed_process, "returncode", 0))
            if returncode != 0:
                active = dict(progress["active_batch"])
                active["state"] = "failed"
                active["returncode"] = returncode
                progress["active_batch"] = active
                raise RuntimeError(
                    f"Stage-1 batch {batch_index} failed with exit code {returncode}; "
                    f"run directory preserved at {run_dir}"
                )

            result, checkpoint = _validated_child_result(run_dir)
            record = _completed_record(
                batch_index=batch_index,
                seed=seed,
                run_dir=run_dir,
                result=result,
                checkpoint=checkpoint,
            )
            completed.append(record)
            progress["completed_batches"] = completed
            progress["completed_count"] = len(completed)
            progress["next_batch_index"] = len(completed) + 1
            progress["latest_checkpoint"] = str(checkpoint)
            progress["active_batch"] = None
            journal.write(progress)
            journal.event(
                "batch_completed",
                batch_index=batch_index,
                seed=seed,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                episodes=record["episodes"],
                decisions=record["decisions"],
            )

        progress["status"] = "completed"
        progress["active_batch"] = None
        progress["completed_count"] = len(completed)
        progress["next_batch_index"] = len(completed) + 1
        journal.write(progress)
        journal.event(
            "loop_completed",
            completed_batches=len(completed),
            latest_checkpoint=progress.get("latest_checkpoint"),
        )
        return dict(progress)
    except BaseException as error:
        progress["status"] = "failed"
        progress["last_error"] = {
            "at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        journal.write(progress)
        active = progress.get("active_batch")
        journal.event(
            "loop_failed",
            batch_index=(active or {}).get("batch_index") if isinstance(active, dict) else None,
            run_dir=(active or {}).get("run_dir") if isinstance(active, dict) else None,
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
            child_stdout=_tail(
                locals().get("completed_process", None)
                and getattr(locals()["completed_process"], "stdout", "")
            ),
            child_stderr=_tail(
                locals().get("completed_process", None)
                and getattr(locals()["completed_process"], "stderr", "")
            ),
        )
        raise


def run(
    args: argparse.Namespace,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    ports = _validate_args(args)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with RunRootLock(run_root / LOCK_FILENAME):
        return _run_locked(
            args,
            ports=ports,
            run_root=run_root,
            runner=runner,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
