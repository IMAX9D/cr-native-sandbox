"""Resumable, fail-closed Schema5/contract-v3 training orchestrator.

The default command is intentionally a long-running foreground supervisor.  It
keeps the authoritative crawler alive until the frozen 100k contract is met,
then advances through native replay, immutable dataset compilation, a real-data
training smoke, and finally the resumable expert-v1 training run.

This module never accepts the historical Schema3 eligibility queue.  Every
native-generator candidate is regenerated from the frozen Schema5 contract-v3
corpus and is checked row-by-row before an Android worker can be started.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterable, Mapping, Sequence

from expert_v1.native_ingest_contract import (
    CONTRACT_KIND as NATIVE_CONTRACT_KIND,
    CONTRACT_SCHEMA_VERSION as NATIVE_CONTRACT_SCHEMA_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3"
)
DEFAULT_CRAWLER_ROOT = Path(r"D:\皇室战争数据集")
DEFAULT_CRAWLER_PYTHON = Path(
    r"D:\Deepseek\cr_re\tools\Python312\python.exe"
)
DEFAULT_TRAINING_PYTHON = Path(
    r"D:\AI_data\runtime\venv\Scripts\python.exe"
)
DEFAULT_CRAWLER_CONFIG = DEFAULT_CRAWLER_ROOT / "config.authoritative.toml"
DEFAULT_CONTRACT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\contracts"
    r"\native-ingest-v150535029.json"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_TARGET = 100_000
DEFAULT_PORTS = tuple(range(38_031, 38_039))
DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_COUNT = 1
DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_RATE = 0.10
DEFAULT_NATIVE_HARDWARE_LOCK = Path(
    r"D:\AI_data\cr-native-core\locks\native-hardware-v1.lock"
)
TWO_AVD_MIN_AVAILABLE_RAM_BYTES = 16 * 1024**3
EXPECTED_AUTHORITATIVE_ROOT_NAME = "authoritative-schema5-v3"
LEGACY_DATA_ROOT_NAMES = frozenset({"one-click-schema5-v2"})
STATE_KIND = "cr_expert_one_click_state_v1"
STATE_SCHEMA_VERSION = 2
STATE_CONTRACT_GENERATION = "schema5-contract-v3"

STAGES = (
    "collect_schema5_v3",
    "freeze_schema5_v3",
    "audit_schema5_v3",
    "generate_native_ticks",
    "stop_native_workers",
    "validate_tick_store_and_masks",
    "compile_native_bc",
    "real_data_training_smoke",
    "formal_training",
)


class OneClickError(RuntimeError):
    """A fail-closed orchestration or artifact-contract violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def native_contract_binding(path: Path) -> dict[str, Any]:
    """Authenticate and return the two immutable native-contract identities."""

    source = path.resolve(strict=True)
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise OneClickError("native contract root is not an object")
    identity = (value.get("schema_version"), value.get("kind"))
    expected_identity = (
        NATIVE_CONTRACT_SCHEMA_VERSION,
        NATIVE_CONTRACT_KIND,
    )
    if identity != expected_identity:
        raise OneClickError(
            "one-click requires native ingest contract v3; "
            f"found identity={identity!r}, expected={expected_identity!r}"
        )
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    canonical_sha = hashlib.sha256(_canonical(payload)).hexdigest()
    if str(value.get("contract_sha256") or "") != canonical_sha:
        raise OneClickError("native contract canonical SHA-256 mismatch")
    file_sha = hashlib.sha256(raw).hexdigest()
    sidecar = source.with_suffix(source.suffix + ".sha256")
    try:
        sidecar_sha = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as error:
        raise OneClickError("native contract file SHA-256 sidecar is missing") from error
    if sidecar_sha != file_sha:
        raise OneClickError("native contract file SHA-256 mismatch")
    return {
        "path": str(source),
        "schema_version": NATIVE_CONTRACT_SCHEMA_VERSION,
        "kind": NATIVE_CONTRACT_KIND,
        "canonical_sha256": canonical_sha,
        "file_sha256": file_sha,
    }


def available_physical_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OneClickError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    if hasattr(os, "sysconf"):
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
    raise OneClickError("cannot determine available physical RAM")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2
            ).encode("utf-8")
            + b"\n"
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise OneClickError(f"JSON root is not an object: {path}")
    return value


def _validate_state_generation(value: Mapping[str, Any], path: Path) -> None:
    if not value:
        return
    if (
        value.get("kind") != STATE_KIND
        or int(value.get("schema_version", -1)) != STATE_SCHEMA_VERSION
        or value.get("contract_generation") != STATE_CONTRACT_GENERATION
    ):
        raise OneClickError(
            "one-click state belongs to another contract generation; "
            f"v2/v3 state mixing is forbidden: {path}"
        )


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise OneClickError(f"required file is missing: {resolved}")
    return {
        "kind": "file_sha256_v1",
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def value_fingerprint(name: str, value: Any) -> dict[str, Any]:
    raw = _canonical(value)
    return {
        "kind": "canonical_json_sha256_v1",
        "name": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def fingerprint_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [file_fingerprint(path) for path in paths]


def component_fingerprints(
    config: "OneClickConfig", *relative_paths: str
) -> list[dict[str, Any]]:
    """Treat executable implementation bytes as immutable stage inputs."""

    return fingerprint_files(
        config.project_root / relative for relative in relative_paths
    )


def _verify_fingerprints(values: Sequence[Mapping[str, Any]]) -> None:
    for item in values:
        kind = item.get("kind")
        if kind == "file_sha256_v1":
            path = Path(str(item.get("path") or ""))
            actual = file_fingerprint(path)
            if dict(item) != actual:
                raise OneClickError(
                    f"persisted artifact SHA changed: {path} "
                    f"({item.get('sha256')} -> {actual['sha256']})"
                )
        elif kind == "canonical_json_sha256_v1":
            # Value fingerprints are immutable declarations; their current
            # value is checked by exact comparison with the newly built input
            # set in StageJournal.begin().
            if not item.get("name") or len(str(item.get("sha256") or "")) != 64:
                raise OneClickError("malformed persisted value fingerprint")
        else:
            raise OneClickError(f"unknown fingerprint kind: {kind}")


class OneClickLock(AbstractContextManager["OneClickLock"]):
    """Non-blocking OS lock; a PID file alone is not used as authority."""

    def __init__(
        self,
        path: Path,
        *,
        conflict_message: str = (
            "another START_EXPERT_ONE_CLICK_V1 instance is already running"
        ),
    ) -> None:
        self.path = path
        self.conflict_message = conflict_message
        self.handle: Any | None = None

    def __enter__(self) -> "OneClickLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production is Windows
                import fcntl

                fcntl.flock(
                    self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise OneClickError(self.conflict_message) from error
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


class StageJournal:
    """Durable stage journal with input and output SHA fences."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.value = _read_json(path)
        if self.value:
            _validate_state_generation(self.value, path)
        else:
            self.value = {
                "schema_version": STATE_SCHEMA_VERSION,
                "kind": STATE_KIND,
                "contract_generation": STATE_CONTRACT_GENERATION,
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "active_stage": None,
                "stages": {},
                "last_error": None,
            }
            self.save()

    def save(self) -> None:
        self.value["updated_utc"] = utc_now()
        _atomic_json(self.path, self.value)

    def begin(
        self, stage: str, inputs: Sequence[Mapping[str, Any]]
    ) -> bool:
        if stage not in STAGES:
            raise OneClickError(f"unknown one-click stage: {stage}")
        normalized = [dict(item) for item in inputs]
        _verify_fingerprints(normalized)
        existing = (self.value.get("stages") or {}).get(stage)
        if (
            isinstance(existing, Mapping)
            and existing.get("inputs")
            and existing.get("inputs") != normalized
        ):
            raise OneClickError(
                f"persisted stage input SHA changed; use a new data root: {stage}"
            )
        if isinstance(existing, Mapping) and existing.get("status") == "completed":
            outputs = existing.get("outputs")
            if not isinstance(outputs, list):
                raise OneClickError(f"completed stage has no output SHA set: {stage}")
            _verify_fingerprints(outputs)
            return False
        stages = self.value.setdefault("stages", {})
        previous_started = (
            existing.get("started_utc") if isinstance(existing, Mapping) else None
        )
        stages[stage] = {
            "status": "running",
            "started_utc": previous_started or utc_now(),
            "resumed_utc": utc_now() if previous_started else None,
            "inputs": normalized,
            "outputs": [],
            "details": {},
        }
        self.value["active_stage"] = stage
        self.value["last_error"] = None
        self.save()
        return True

    def progress(self, stage: str, details: Mapping[str, Any]) -> None:
        record = self.value["stages"][stage]
        if record.get("status") != "running":
            raise OneClickError(f"cannot update non-running stage: {stage}")
        record["details"] = dict(details)
        record["progress_utc"] = utc_now()
        self.save()

    def complete(
        self,
        stage: str,
        outputs: Sequence[Mapping[str, Any]],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = [dict(item) for item in outputs]
        _verify_fingerprints(normalized)
        record = self.value["stages"][stage]
        record.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "outputs": normalized,
                "details": dict(details or {}),
            }
        )
        self.value["active_stage"] = None
        self.value["last_error"] = None
        self.save()

    def fail(self, stage: str, error: BaseException) -> None:
        stages = self.value.setdefault("stages", {})
        record = stages.setdefault(stage, {})
        record["status"] = "failed"
        record["failed_utc"] = utc_now()
        record["error"] = f"{type(error).__name__}: {error}"
        self.value["active_stage"] = None
        self.value["last_error"] = {
            "stage": stage,
            "error": record["error"],
            "utc": utc_now(),
        }
        self.save()


@dataclass(frozen=True)
class OneClickConfig:
    project_root: Path
    data_root: Path
    crawler_root: Path
    crawler_python: Path
    training_python: Path
    crawler_config: Path
    authoritative_db: Path
    authoritative_root: Path
    native_contract: Path
    template: Path
    target: int = DEFAULT_TARGET
    workers: int = 4
    avds: int = 1
    workers_per_avd: int = 4
    ports: tuple[int, ...] = DEFAULT_PORTS
    minimum_native_success_rate: float = 0.50
    minimum_ability_positive_success_count: int = (
        DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_COUNT
    )
    minimum_ability_positive_success_rate: float = (
        DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_RATE
    )
    waive_ability_positive_coverage: bool = False
    ability_positive_waiver_reason: str | None = None
    native_layout_reason: str = "explicit_or_test"
    available_ram_at_selection: int = 0
    requested_workers: int | None = None
    poll_seconds: float = 30.0
    audit_workers: int = max(8, os.cpu_count() or 8)
    compile_io_workers: int = min(32, max(4, (os.cpu_count() or 4) * 2))
    compile_process_workers: int = max(1, (os.cpu_count() or 2) // 2)

    @property
    def state_path(self) -> Path:
        return self.data_root / "control" / "state.json"

    @property
    def lock_path(self) -> Path:
        return self.data_root / "control" / "run.lock"

    @property
    def native_hardware_lock_path(self) -> Path:
        # Fixed across all --data-root values: Android emulator/ADB/direct TCP
        # resources are machine-global, not run-local.
        return DEFAULT_NATIVE_HARDWARE_LOCK

    @property
    def logs_root(self) -> Path:
        return self.data_root / "logs"

    @property
    def frozen_manifest(self) -> Path:
        return self.data_root / "manifests" / "schema5-v3-100k.jsonl"

    @property
    def frozen_metadata(self) -> Path:
        return self.frozen_manifest.with_suffix(
            self.frozen_manifest.suffix + ".manifest.json"
        )

    @property
    def eligibility_root(self) -> Path:
        # audit_native_eligibility deliberately only replaces a derived tree
        # carrying this exact leaf name.
        return self.data_root / "eligibility" / "native-eligibility-v1"

    @property
    def candidate_queue(self) -> Path:
        return (
            self.eligibility_root
            / "queues"
            / "authoritative-native-full.jsonl"
        )

    @property
    def native_root(self) -> Path:
        return self.data_root / "native-authoritative-ticks-v1"

    @property
    def tick_store_root(self) -> Path:
        return self.native_root / "shards"

    @property
    def audit_prefix_store_root(self) -> Path:
        return self.native_root / "audit-prefix-shards"

    @property
    def tick_validation_receipt(self) -> Path:
        return self.data_root / "receipts" / "tick-store-validation.json"

    @property
    def native_generation_receipt(self) -> Path:
        return self.data_root / "receipts" / "native-generation-coverage.json"

    @property
    def compiled_root(self) -> Path:
        return self.data_root / "compiled" / "native-bc-v1"

    @property
    def worker_stop_receipt(self) -> Path:
        return self.data_root / "receipts" / "native-workers-stopped.json"

    @property
    def smoke_output_root(self) -> Path:
        return self.data_root / "runs-smoke-real"

    @property
    def training_output_root(self) -> Path:
        return self.data_root / "runs"

    @property
    def training_run_id(self) -> str:
        return "expert-v1-schema5-v3-100k"

    def declared(self, *, include_native_layout: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result.pop("requested_workers", None)
        if not include_native_layout:
            for key in (
                "workers",
                "avds",
                "workers_per_avd",
                "ports",
                "native_layout_reason",
                "available_ram_at_selection",
            ):
                result.pop(key, None)
        return {
            key: ([str(item) for item in value] if key == "ports" else str(value))
            if isinstance(value, Path)
            else value
            for key, value in result.items()
        }


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str


class CommandRunner:
    """Run a command in the foreground while teeing durable stage logs."""

    def __init__(self, logs_root: Path) -> None:
        self.logs_root = logs_root

    def run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path,
        log_name: str,
        check: bool = True,
    ) -> CommandResult:
        normalized = tuple(str(item) for item in command)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / f"{log_name}.log"
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        lines: list[str] = []
        with log_path.open("a", encoding="utf-8", newline="") as log:
            header = f"\n[{utc_now()}] {json.dumps(normalized, ensure_ascii=False)}\n"
            log.write(header)
            log.flush()
            process = subprocess.Popen(
                normalized,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        result = CommandResult(normalized, returncode, "".join(lines))
        if check and returncode:
            raise OneClickError(
                f"command failed ({returncode}); see {log_path}: "
                + " ".join(normalized)
            )
        return result


def crawler_command(config: OneClickConfig, action: str) -> tuple[str, ...]:
    if action not in {"start", "stop", "status"}:
        raise ValueError(action)
    return (
        str(config.crawler_python),
        "-m",
        "crawler.authoritative_production",
        action,
        "--config",
        str(config.crawler_config),
    )


def native_worker_command(
    config: OneClickConfig, action: str
) -> tuple[str, ...]:
    if action not in {"start", "stop", "status"}:
        raise ValueError(action)
    command = [
        str(config.training_python),
        "-m",
        "native_core.worker",
        action,
        "--workers",
        str(config.workers),
        "--avds",
        str(config.avds),
        "--workers-per-avd",
        str(config.workers_per_avd),
        "--transport",
        "direct",
    ]
    if action == "stop":
        command.append("--stop-vm")
    return tuple(command)


def native_generation_command(config: OneClickConfig) -> tuple[str, ...]:
    return (
        str(config.training_python),
        str(config.project_root / "scripts" / "generate_expert_native_ticks.py"),
        "run",
        "--queue",
        str(config.candidate_queue),
        "--output-root",
        str(config.native_root),
        "--template",
        str(config.template),
        "--native-contract",
        str(config.native_contract),
        "--workers",
        str(config.workers),
        "--ports",
        *(str(port) for port in config.ports[: config.workers]),
        "--selection-seed",
        "authoritative-schema5-v3-100k-v1",
    )


def compile_command(config: OneClickConfig) -> tuple[str, ...]:
    return (
        str(config.training_python),
        "-m",
        "expert_v1.compile_native_bc_dataset",
        "--tick-store-root",
        str(config.tick_store_root),
        "--schema5-manifest",
        str(config.frozen_manifest),
        "--output-root",
        str(config.compiled_root),
        "--native-contract",
        str(config.native_contract),
        "--native-generation-receipt",
        str(config.native_generation_receipt),
        "--io-workers",
        str(config.compile_io_workers),
        "--process-workers",
        str(config.compile_process_workers),
    )


def training_smoke_command(config: OneClickConfig) -> tuple[str, ...]:
    dataset_sha = sha256_file(config.compiled_root / "manifest.json")
    return (
        str(config.training_python),
        "-m",
        "expert_v1.training_v1.train",
        "--smoke",
        "--resume",
        "--dataset-root",
        str(config.compiled_root),
        "--expected-source-manifest",
        str(config.frozen_manifest),
        "--output-root",
        str(config.smoke_output_root),
        "--run-id",
        f"real-schema5-v3-{dataset_sha[:16]}",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--sequence-length",
        "16",
        "--burn-in",
        "4",
        "--workers",
        "0",
        "--max-train-batches",
        "2",
        "--max-eval-batches",
        "1",
        "--hidden-size",
        "64",
        "--card-embedding-size",
        "32",
        "--device",
        "auto",
        "--allow-unanchored-native-states",
    )


def formal_training_command(config: OneClickConfig) -> tuple[str, ...]:
    return (
        str(config.training_python),
        "-m",
        "expert_v1.training_v1.train",
        "--resume",
        "--dataset-root",
        str(config.compiled_root),
        "--expected-source-manifest",
        str(config.frozen_manifest),
        "--output-root",
        str(config.training_output_root),
        "--run-id",
        config.training_run_id,
        "--allow-unanchored-native-states",
    )


def _authoritative_settings(config: OneClickConfig) -> dict[str, Any]:
    raw = config.crawler_config.resolve(strict=True).read_bytes()
    value = tomllib.loads(raw.decode("utf-8-sig"))
    configured_root = Path(str(value.get("authoritative_output_dir") or ""))
    if not configured_root.is_absolute():
        configured_root = config.crawler_root / configured_root
    configured_db = Path(str(value.get("db_path") or ""))
    if not configured_db.is_absolute():
        configured_db = config.crawler_root / configured_db
    configured_contract = Path(
        str(value.get("authoritative_native_contract") or "")
    )
    if not configured_contract.is_absolute():
        configured_contract = config.crawler_root / configured_contract
    binding = native_contract_binding(config.native_contract)
    settings = {
        "target": int(value.get("authoritative_target") or 0),
        "root": str(configured_root.resolve()),
        "db": str(configured_db.resolve()),
        "contract": str(configured_contract.resolve()),
        "contract_canonical_sha256": binding["canonical_sha256"],
        "contract_file_sha256": binding["file_sha256"],
        "game_version": str(value.get("authoritative_game_version") or ""),
    }
    expected = {
        "target": config.target,
        "root": str(config.authoritative_root.resolve()),
        "db": str(config.authoritative_db.resolve()),
        "contract": str(config.native_contract.resolve()),
    }
    mismatches = {
        key: (settings[key], wanted)
        for key, wanted in expected.items()
        if settings[key] != wanted
    }
    if mismatches:
        raise OneClickError(
            f"authoritative crawler config does not match one-click contract: {mismatches}"
        )
    if config.authoritative_root.name.casefold() != EXPECTED_AUTHORITATIVE_ROOT_NAME:
        raise OneClickError(
            "authoritative root must be the isolated contract-v3 corpus named "
            f"{EXPECTED_AUTHORITATIVE_ROOT_NAME!r}"
        )
    return settings


def _authoritative_count(config: OneClickConfig) -> int:
    binding = native_contract_binding(config.native_contract)
    database = config.authoritative_db.resolve(strict=True)
    connection = sqlite3.connect(database, timeout=30)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM authoritative_results "
            "WHERE status='accepted' AND tier='native_static_v2' "
            "AND contract_sha256=?",
            (binding["canonical_sha256"],),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _authoritative_db_invariants(config: OneClickConfig) -> dict[str, Any]:
    """Prove the whole progress DB is pinned to the v3 contract."""

    binding = native_contract_binding(config.native_contract)
    connection = sqlite3.connect(config.authoritative_db.resolve(strict=True), timeout=30)
    try:
        contract_counts = {
            str(contract): int(count)
            for contract, count in connection.execute(
                "SELECT contract_sha256,COUNT(*) FROM authoritative_results "
                "GROUP BY contract_sha256"
            )
        }
        invalid_status = int(connection.execute(
            "SELECT COUNT(*) FROM authoritative_results "
            "WHERE status NOT IN ('queued','rejected','accepted')"
        ).fetchone()[0])
        invalid_accepted = int(connection.execute(
            "SELECT COUNT(*) FROM authoritative_results "
            "WHERE status='accepted' AND (tier!='native_static_v2' "
            "OR contract_sha256!=?)",
            (binding["canonical_sha256"],),
        ).fetchone()[0])
        accepted = int(connection.execute(
            "SELECT COUNT(*) FROM authoritative_results "
            "WHERE status='accepted' AND tier='native_static_v2' "
            "AND contract_sha256=?",
            (binding["canonical_sha256"],),
        ).fetchone()[0])
        total = int(connection.execute(
            "SELECT COUNT(*) FROM authoritative_results"
        ).fetchone()[0])
    finally:
        connection.close()
    expected_contracts = (
        {} if total == 0 else {str(binding["canonical_sha256"]): total}
    )
    if (
        contract_counts != expected_contracts
        or invalid_status
        or invalid_accepted
        or accepted > config.target
    ):
        raise OneClickError(
            "authoritative DB violates all-v3 invariant: "
            f"contracts={contract_counts}, invalid_status={invalid_status}, "
            f"invalid_accepted={invalid_accepted}, accepted={accepted}/{config.target}"
        )
    return {
        "rows": total,
        "accepted": accepted,
        "contract_sha256": binding["canonical_sha256"],
        "contract_file_sha256": binding["file_sha256"],
    }


def _sqlite_quick_check_and_checkpoint(path: Path) -> str:
    connection = sqlite3.connect(path, timeout=60)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        result = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if result != "ok":
            raise OneClickError(f"authoritative SQLite quick_check: {result}")
        return result
    finally:
        connection.close()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":  # pragma: no cover
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _crawler_active(config: OneClickConfig) -> bool:
    lock = config.crawler_root / "logs" / "authoritative-production.lock"
    value = _read_json(lock)
    if not _pid_alive(int(value.get("pid") or 0)):
        return False
    binding = native_contract_binding(config.native_contract)
    try:
        locked_config = Path(str(value.get("config") or "")).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise OneClickError("active crawler lock has no valid resolved config") from error
    if (
        locked_config != config.crawler_config.resolve(strict=True)
        or str(value.get("contract_sha256") or "")
        != binding["canonical_sha256"]
    ):
        raise OneClickError(
            "active authoritative crawler belongs to another config/contract: "
            f"lock={value}"
        )
    return True


def validate_schema5_candidate_queue(
    path: Path,
    *,
    authoritative_root: Path,
    verify_source_bytes: bool = True,
    frozen_manifest: Path | None = None,
    native_contract: Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, int]:
    """Reject any legacy row before the native generator is allowed to run."""

    root = authoritative_root.resolve(strict=True)
    frozen: dict[str, dict[str, Any]] | None = None
    binding = (
        None if native_contract is None else native_contract_binding(native_contract)
    )
    if frozen_manifest is not None:
        frozen = {}
        with frozen_manifest.resolve(strict=True).open(
            "r", encoding="utf-8-sig"
        ) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                tag = str(row.get("battle_tag") or "")
                if not tag or tag in frozen:
                    raise OneClickError(
                        f"missing/duplicate frozen tag at line {line_number}"
                    )
                frozen[tag] = row
        if expected_rows is not None and len(frozen) != expected_rows:
            raise OneClickError(
                f"frozen manifest coverage changed: {len(frozen)}/{expected_rows}"
            )
    rows = ability_positive = 0
    seen: set[str] = set()
    with path.resolve(strict=True).open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            tag = str(value.get("battle_tag") or "")
            if not tag or tag in seen:
                raise OneClickError(
                    f"missing/duplicate candidate tag at line {line_number}"
                )
            seen.add(tag)
            source_path = Path(str(value.get("source_path") or "")).resolve()
            if (
                int(value.get("source_schema_version") or 0) != 5
                or value.get("schema5_authoritative_contract_verified") is not True
                or value.get("authoritative_native_full_candidate") is not True
                or not source_path.is_relative_to(root)
            ):
                raise OneClickError(
                    f"legacy/non-authoritative candidate is forbidden: {tag}"
                )
            declared = str(value.get("source_sha256") or "")
            if len(declared) != 64:
                raise OneClickError(f"candidate source SHA is malformed: {tag}")
            if verify_source_bytes and (
                not source_path.is_file() or sha256_file(source_path) != declared
            ):
                raise OneClickError(f"candidate source SHA changed: {tag}")
            if frozen is not None:
                frozen_row = frozen.get(tag)
                if frozen_row is None:
                    raise OneClickError(
                        f"candidate tag is absent from frozen manifest: {tag}"
                    )
                exact = {
                    "source_path": (
                        str(source_path),
                        str(Path(str(frozen_row.get("source_path") or "")).resolve()),
                    ),
                    "source_sha256": (
                        declared,
                        str(frozen_row.get("source_sha256") or ""),
                    ),
                    "source_schema_version": (
                        int(value.get("source_schema_version") or 0),
                        int(frozen_row.get("source_schema_version") or 0),
                    ),
                    "contract_sha256": (
                        str(value.get("contract_sha256") or ""),
                        str(frozen_row.get("contract_sha256") or ""),
                    ),
                    "contract_file_sha256": (
                        str(value.get("contract_file_sha256") or ""),
                        str(frozen_row.get("contract_file_sha256") or ""),
                    ),
                }
                mismatches = {
                    name: pair for name, pair in exact.items() if pair[0] != pair[1]
                }
                if mismatches:
                    raise OneClickError(
                        f"candidate/frozen row mismatch for {tag}: {mismatches}"
                    )
                if binding is not None and (
                    exact["contract_sha256"][0] != binding["canonical_sha256"]
                    or exact["contract_file_sha256"][0]
                    != binding["file_sha256"]
                ):
                    raise OneClickError(
                        f"candidate native contract mismatch: {tag}"
                    )
            rows += 1
            ability_positive += int(int(value.get("ability_events_observed") or 0) > 0)
    if rows == 0:
        raise OneClickError("Schema5 contract-v3 native candidate queue is empty")
    if expected_rows is not None and rows != expected_rows:
        raise OneClickError(
            "Schema5 contract-v3 candidate coverage is not complete: "
            f"{rows}/{expected_rows}"
        )
    if frozen is not None and seen != set(frozen):
        raise OneClickError("candidate queue is not an exact frozen-tag join")
    return {
        "rows": rows,
        "ability_positive": ability_positive,
        "ability_zero": rows - ability_positive,
    }


def validate_native_result_records(
    results_path: Path,
    candidate_queue: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Prove every frozen candidate has exactly one final native attempt."""

    expected: dict[str, bool] = {}
    with candidate_queue.resolve(strict=True).open(
        "r", encoding="utf-8-sig"
    ) as source:
        for line in source:
            if not line.strip():
                continue
            candidate = json.loads(line)
            tag = str(candidate.get("battle_tag") or "")
            if not tag or tag in expected:
                raise OneClickError("candidate queue tag set is malformed")
            expected[tag] = int(candidate.get("ability_events_observed") or 0) > 0
    if len(expected) != expected_rows:
        raise OneClickError("candidate queue tag set is malformed")
    seen: set[str] = set()
    successes = 0
    success_tags: set[str] = set()
    prefix_tags: set[str] = set()
    unframed_tags: set[str] = set()
    failure_classes: dict[str, int] = {}
    cohorts: dict[str, dict[str, Any]] = {
        "ability_positive": {
            "candidates": sum(expected.values()),
            "attempted": 0,
            "successes": 0,
            "failures": 0,
            "failure_class_counts": {},
        },
        "ability_zero": {
            "candidates": len(expected) - sum(expected.values()),
            "attempted": 0,
            "successes": 0,
            "failures": 0,
            "failure_class_counts": {},
        },
    }
    with results_path.resolve(strict=True).open(
        "r", encoding="utf-8-sig"
    ) as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            tag = str(row.get("battle_tag") or "")
            if (
                row.get("kind") != "expert_authoritative_native_tick_result_v1"
                or not tag
                or tag not in expected
                or tag in seen
                or row.get("final_attempt") is not True
                or not isinstance(row.get("teacher_forced_success"), bool)
            ):
                raise OneClickError(
                    f"native result is not a unique final attempt at line {line_number}"
                )
            seen.add(tag)
            cohort_name = (
                "ability_positive" if expected[tag] else "ability_zero"
            )
            cohort = cohorts[cohort_name]
            cohort["attempted"] += 1
            if row["teacher_forced_success"]:
                if row.get("audit_prefix_tick_store_entry") is not None:
                    raise OneClickError("successful native result references prefix frame")
                successes += 1
                success_tags.add(tag)
                cohort["successes"] += 1
            else:
                prefix_entry = row.get("audit_prefix_tick_store_entry")
                extent = row.get("audit_prefix_extent")
                if isinstance(prefix_entry, Mapping):
                    if (
                        not isinstance(extent, Mapping)
                        or extent.get("kind") != "cr_native_replay_extent_v1"
                        or extent.get("extent") != "valid_prefix"
                        or extent.get("training_admission") != "audit_only"
                        or extent.get("terminal_target") != "unknown_censored"
                        or extent.get("failure_tick_has_labels") is not False
                        or row.get("failure_domain") != "semantic"
                        or row.get("failure_prefix_semantic_match") is not True
                        or int(prefix_entry.get("ticks", 0)) <= 0
                        or int(
                            row.get("native_deployment_mask_probes_attempted")
                            or 0
                        ) != 0
                    ):
                        raise OneClickError("audit-prefix result contract changed")
                    prefix_tags.add(tag)
                else:
                    unframed_tags.add(tag)
                cohort["failures"] += 1
                failure_class = str(row.get("failure_class") or "unknown")
                failure_classes[failure_class] = (
                    failure_classes.get(failure_class, 0) + 1
                )
                cohort_failures = cohort["failure_class_counts"]
                cohort_failures[failure_class] = (
                    cohort_failures.get(failure_class, 0) + 1
                )
    if seen != set(expected):
        raise OneClickError(
            f"native result/candidate exact join failed: "
            f"results={len(seen)}, candidates={len(expected)}"
        )
    for cohort in cohorts.values():
        attempted = int(cohort["attempted"])
        cohort["success_rate"] = (
            float(cohort["successes"]) / attempted if attempted else None
        )
        cohort["failure_class_counts"] = dict(
            sorted(cohort["failure_class_counts"].items())
        )
    return {
        "rows": len(seen),
        "successes": successes,
        "failures": len(seen) - successes,
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "success_tags": sorted(success_tags),
        "audit_prefix_tags": sorted(prefix_tags),
        "unframed_tags": sorted(unframed_tags),
        "audit_tick_episodes": len(success_tags) + len(prefix_tags),
        **cohorts,
    }


def evaluate_ability_positive_coverage(
    queue_summary: Mapping[str, Any],
    result_audit: Mapping[str, Any],
    *,
    minimum_success_count: int,
    minimum_success_rate: float,
    waived: bool,
    waiver_reason: str | None,
) -> dict[str, Any]:
    """Build the immutable ability/non-ability admission classification."""

    positive = dict(result_audit.get("ability_positive") or {})
    zero = dict(result_audit.get("ability_zero") or {})
    candidate_positive = int(queue_summary.get("ability_positive", -1))
    candidate_zero = int(queue_summary.get("ability_zero", -1))
    if (
        candidate_positive < 0
        or candidate_zero < 0
        or int(positive.get("candidates", -2)) != candidate_positive
        or int(positive.get("attempted", -2)) != candidate_positive
        or int(zero.get("candidates", -2)) != candidate_zero
        or int(zero.get("attempted", -2)) != candidate_zero
    ):
        raise OneClickError(
            "ability-positive/zero attempt classification does not cover candidates"
        )
    positive_successes = int(positive.get("successes", -1))
    positive_failures = int(positive.get("failures", -1))
    if positive_successes < 0 or positive_successes + positive_failures != candidate_positive:
        raise OneClickError("ability-positive success/failure classification is open")
    applicable = candidate_positive > 0
    positive_rate = (
        positive_successes / candidate_positive if applicable else None
    )
    raw_passed = (
        not applicable
        or (
            positive_successes >= int(minimum_success_count)
            and positive_rate is not None
            and positive_rate >= float(minimum_success_rate)
        )
    )
    reason = str(waiver_reason or "").strip() or None
    if waived and reason is None:
        raise OneClickError("ability-positive coverage waiver requires a reason")
    return {
        "schema_version": 1,
        "kind": "cr_expert_ability_native_coverage_v1",
        "candidate_counts": {
            "ability_positive": candidate_positive,
            "ability_zero": candidate_zero,
        },
        "attempt_counts": {
            "ability_positive": int(positive["attempted"]),
            "ability_zero": int(zero["attempted"]),
        },
        "success_counts": {
            "ability_positive": positive_successes,
            "ability_zero": int(zero["successes"]),
        },
        "failure_counts": {
            "ability_positive": positive_failures,
            "ability_zero": int(zero["failures"]),
        },
        "success_rates": {
            "ability_positive": positive_rate,
            "ability_zero": zero.get("success_rate"),
        },
        "failure_class_counts": {
            "ability_positive": positive.get("failure_class_counts") or {},
            "ability_zero": zero.get("failure_class_counts") or {},
        },
        "gate": {
            "applicable": applicable,
            "minimum_success_count": int(minimum_success_count),
            "minimum_success_rate": float(minimum_success_rate),
            "raw_passed": raw_passed,
            "waiver_applied": bool(waived),
            "waiver_reason": reason,
            "admitted": bool(raw_passed or waived),
        },
    }


class OneClickOrchestrator:
    def __init__(
        self,
        config: OneClickConfig,
        *,
        runner: CommandRunner | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.runner = runner or CommandRunner(config.logs_root)
        self.sleep = sleep
        self.journal = StageJournal(config.state_path)

    def _run_stage(
        self,
        stage: str,
        inputs: Sequence[Mapping[str, Any]],
        function: Any,
    ) -> None:
        if not self.journal.begin(stage, inputs):
            print(f"[resume] {stage}: SHA verified, skipped", flush=True)
            return
        print(f"[stage] {stage}", flush=True)
        try:
            outputs, details = function()
            self.journal.complete(stage, outputs, details)
        except BaseException as error:
            self.journal.fail(stage, error)
            raise

    def _best_effort_stop_native(self) -> None:
        """Emergency cleanup only; preserve the original stage exception."""

        try:
            self.runner.run(
                native_worker_command(self.config, "stop"),
                cwd=self.config.project_root,
                log_name="native-workers-emergency-stop",
                check=False,
            )
        except BaseException as cleanup_error:
            print(
                f"[warning] best-effort native cleanup failed: {cleanup_error}",
                flush=True,
            )

    def _ensure_native_layout(self) -> None:
        """Select once, after crawler shutdown, then freeze in the journal."""

        config = self.config
        if config.avds == 0:
            available_ram = available_physical_memory_bytes()
            avds = 2 if available_ram >= TWO_AVD_MIN_AVAILABLE_RAM_BYTES else 1
            workers_per_avd = 4
            workers = avds * workers_per_avd
            if (
                config.requested_workers is not None
                and int(config.requested_workers) != workers
            ):
                raise OneClickError(
                    f"--workers={config.requested_workers} conflicts with "
                    f"post-collection RAM layout {avds}x4={workers}"
                )
            config = replace(
                config,
                avds=avds,
                workers_per_avd=workers_per_avd,
                workers=workers,
                ports=DEFAULT_PORTS[:workers],
                native_layout_reason=(
                    "post_collection_available_ram_at_least_16gib"
                    if avds == 2
                    else "post_collection_available_ram_below_16gib"
                ),
                available_ram_at_selection=available_ram,
            )
            self.config = config
        if (
            config.avds not in {1, 2}
            or config.workers_per_avd != 4
            or config.workers != config.avds * config.workers_per_avd
            or config.ports != DEFAULT_PORTS[: config.workers]
        ):
            raise OneClickError("persisted native layout/ports are invalid")
        layout = {
            "schema_version": 1,
            "kind": "cr_native_hardware_layout_v1",
            "avds": config.avds,
            "workers_per_avd": config.workers_per_avd,
            "workers": config.workers,
            "ports": list(config.ports),
            "reason": config.native_layout_reason,
            "available_ram_at_selection": config.available_ram_at_selection,
        }
        existing = self.journal.value.get("native_layout")
        if existing is not None and existing != layout:
            raise OneClickError("persisted native hardware layout changed")
        if existing is None:
            self.journal.value["native_layout"] = layout
            self.journal.save()

    def collect(self) -> None:
        config = self.config
        settings = _authoritative_settings(config)
        _authoritative_db_invariants(config)
        inputs = [
            file_fingerprint(config.crawler_config),
            file_fingerprint(config.native_contract),
            file_fingerprint(
                config.crawler_root / "crawler" / "authoritative_production.py"
            ),
            file_fingerprint(config.crawler_root / "crawler" / "authoritative.py"),
            file_fingerprint(config.crawler_root / "crawler" / "main.py"),
            value_fingerprint("authoritative-settings", settings),
        ]

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            while True:
                accepted = _authoritative_count(config)
                if accepted > config.target:
                    raise OneClickError(
                        f"authoritative accepted count exceeds target: "
                        f"{accepted}/{config.target}"
                    )
                progress = {
                    "accepted": accepted,
                    "target": config.target,
                    "remaining": config.target - accepted,
                    "crawler_active": _crawler_active(config),
                }
                self.journal.progress("collect_schema5_v3", progress)
                print(
                    f"[collection] {accepted:,}/{config.target:,} "
                    f"remaining={config.target - accepted:,}",
                    flush=True,
                )
                if accepted == config.target:
                    break
                if not _crawler_active(config):
                    self.runner.run(
                        crawler_command(config, "start"),
                        cwd=config.crawler_root,
                        log_name="collect-schema5-v3",
                    )
                    if not _crawler_active(config):
                        raise OneClickError(
                            "authoritative supervisor did not remain alive after start"
                        )
                self.sleep(config.poll_seconds)

            if _crawler_active(config):
                self.runner.run(
                    crawler_command(config, "stop"),
                    cwd=config.crawler_root,
                    log_name="collect-schema5-v3",
                )
            deadline = time.monotonic() + 60
            while _crawler_active(config) and time.monotonic() < deadline:
                self.sleep(1.0)
            if _crawler_active(config):
                raise OneClickError("authoritative supervisor did not stop at target")
            quick_check = _sqlite_quick_check_and_checkpoint(
                config.authoritative_db
            )
            invariants = _authoritative_db_invariants(config)
            if _authoritative_count(config) != config.target:
                raise OneClickError("authoritative count changed during freeze fence")
            index = config.authoritative_root / "index.jsonl"
            return fingerprint_files([config.authoritative_db, index]), {
                "accepted": config.target,
                "quick_check": quick_check,
                "all_v3_invariant": invariants,
            }

        self._run_stage("collect_schema5_v3", inputs, action)

    def freeze(self) -> None:
        config = self.config
        index = config.authoritative_root / "index.jsonl"
        inputs = fingerprint_files(
            [config.authoritative_db, index, config.native_contract]
        ) + component_fingerprints(
            config, "expert_v1/freeze_schema5_manifest.py"
        ) + [value_fingerprint("target", config.target)]

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            self.runner.run(
                (
                    str(config.training_python),
                    "-m",
                    "expert_v1.freeze_schema5_manifest",
                    "--db",
                    str(config.authoritative_db),
                    "--authoritative-root",
                    str(config.authoritative_root),
                    "--output",
                    str(config.frozen_manifest),
                    "--target",
                    str(config.target),
                    "--native-contract",
                    str(config.native_contract),
                ),
                cwd=config.project_root,
                log_name="freeze-schema5-v3",
            )
            metadata = _read_json(config.frozen_metadata)
            binding = native_contract_binding(config.native_contract)
            if (
                metadata.get("production_ready") is not True
                or int(metadata.get("accepted_battles", -1)) != config.target
                or metadata.get("manifest_sha256")
                != sha256_file(config.frozen_manifest)
                or metadata.get("native_contract_sha256")
                != binding["canonical_sha256"]
                or metadata.get("native_contract_file_sha256")
                != binding["file_sha256"]
                or Path(str(metadata.get("native_contract_path") or "")).resolve()
                != config.native_contract.resolve()
            ):
                raise OneClickError(
                    "frozen Schema5 contract-v3 manifest failed admission"
                )
            return fingerprint_files(
                [config.frozen_manifest, config.frozen_metadata]
            ), {
                "accepted_battles": config.target,
                "manifest_sha256": metadata["manifest_sha256"],
            }

        self._run_stage("freeze_schema5_v3", inputs, action)

    def audit(self) -> None:
        config = self.config
        inputs = fingerprint_files(
            [
                config.frozen_manifest,
                config.native_contract,
                config.project_root / "expert_v1" / "audit_native_eligibility.py",
            ]
        ) + component_fingerprints(
            config,
            "scripts/audit_expert_100k_native_eligibility.py",
            "expert_v1/native_replay_plan.py",
            "expert_v1/native_ingest_contract.py",
            "expert_v1/native_capabilities.py",
            "native_core/card_catalog.py",
            "native_core/decks.py",
            "native_core/data/live_card_catalog.json",
            "bindings/runtime-150535029-x86_64.json",
        )

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            self.runner.run(
                (
                    str(config.training_python),
                    str(
                        config.project_root
                        / "scripts"
                        / "audit_expert_100k_native_eligibility.py"
                    ),
                    "--manifest",
                    str(config.frozen_manifest),
                    "--output",
                    str(config.eligibility_root),
                    "--workers",
                    str(config.audit_workers),
                    "--native-contract",
                    str(config.native_contract),
                ),
                cwd=config.project_root,
                log_name="audit-schema5-v3",
            )
            queue_summary = validate_schema5_candidate_queue(
                config.candidate_queue,
                authoritative_root=config.authoritative_root,
                verify_source_bytes=False,
                frozen_manifest=config.frozen_manifest,
                native_contract=config.native_contract,
                expected_rows=config.target,
            )
            audit_manifest = config.eligibility_root / "manifest.json"
            audit_summary = config.eligibility_root / "summary.json"
            inventory = _read_json(audit_manifest)
            summary = _read_json(audit_summary)
            binding = native_contract_binding(config.native_contract)
            for name, value in (("manifest", inventory), ("summary", summary)):
                source = value.get("source_manifest") or {}
                contract = value.get("native_contract") or {}
                if (
                    Path(str(source.get("path") or "")).resolve()
                    != config.frozen_manifest.resolve()
                    or source.get("sha256")
                    != sha256_file(config.frozen_manifest)
                    or int(source.get("rows", -1)) != config.target
                    or Path(str(contract.get("path") or "")).resolve()
                    != config.native_contract.resolve()
                    or contract.get("contract_sha256")
                    != binding["canonical_sha256"]
                    or contract.get("file_sha256") != binding["file_sha256"]
                ):
                    raise OneClickError(
                        f"audit {name} is not bound to the frozen source/contract"
                    )
            queues = {
                str(item.get("path")): item
                for item in inventory.get("queues") or []
                if isinstance(item, Mapping)
            }
            queue_entry = queues.get("queues/authoritative-native-full.jsonl")
            if (
                not isinstance(queue_entry, Mapping)
                or int(queue_entry.get("rows", -1)) != config.target
                or queue_entry.get("sha256")
                != sha256_file(config.candidate_queue)
            ):
                raise OneClickError("audit candidate queue coverage/SHA is open")
            return fingerprint_files(
                [audit_manifest, audit_summary, config.candidate_queue]
            ), queue_summary

        self._run_stage("audit_schema5_v3", inputs, action)

    def generate_native(self) -> None:
        config = self.config
        queue_summary = validate_schema5_candidate_queue(
            config.candidate_queue,
            authoritative_root=config.authoritative_root,
            verify_source_bytes=False,
            frozen_manifest=config.frozen_manifest,
            native_contract=config.native_contract,
            expected_rows=config.target,
        )
        inputs = fingerprint_files(
            [config.candidate_queue, config.frozen_manifest, config.native_contract, config.template]
        ) + component_fingerprints(
            config,
            "scripts/generate_expert_native_ticks.py",
            "expert_v1/native_dataset_generator.py",
            "expert_v1/native_replay_plan.py",
            "expert_v1/native_replay_runner.py",
            "expert_v1/native_ingest_contract.py",
            "expert_v1/native_capabilities.py",
            "expert_v1/native_profile.py",
            "expert_v1/native_seed_search.py",
            "native_core/card_catalog.py",
            "native_core/decks.py",
            "native_core/data/live_card_catalog.json",
            "bindings/runtime-150535029-x86_64.json",
            "expert_v1/tick_store_v1/codec.py",
            "expert_v1/tick_store_v1/deployment_masks.py",
            "expert_v1/tick_store_v1/schema.py",
            "expert_v1/tick_store_v1/shard.py",
        ) + [value_fingerprint("native-worker-layout", {
            "avds": config.avds,
            "workers_per_avd": config.workers_per_avd,
            "workers": config.workers,
            "ports": config.ports[: config.workers],
        })]

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            self.runner.run(
                native_worker_command(config, "start"),
                cwd=config.project_root,
                log_name="native-workers",
            )
            self.runner.run(
                native_generation_command(config),
                cwd=config.project_root,
                log_name="generate-native-ticks",
            )
            summary_path = config.native_root / "summary.json"
            manifest_path = config.native_root / "manifest.json"
            store_manifest = config.tick_store_root / "manifest.json"
            prefix_store_manifest = (
                config.audit_prefix_store_root / "manifest.json"
            )
            mask_manifest = (
                config.tick_store_root / "deployment-masks-v1" / "manifest.json"
            )
            results_path = config.native_root / "results.jsonl"
            summary = _read_json(summary_path)
            native_manifest = _read_json(manifest_path)
            result_audit = validate_native_result_records(
                results_path,
                config.candidate_queue,
                expected_rows=config.target,
            )
            selected = int(summary.get("selected_battles", -1))
            processed = int(summary.get("processed_battles", -1))
            successes = int(summary.get("teacher_forced_successes", -1))
            failures = int(summary.get("teacher_forced_failures", -1))
            stored = int(summary.get("stored_episodes", -1))
            prefix_stored = int(summary.get("audit_prefix_episodes", -1))
            unframed = int(summary.get("unframed_episodes", -1))
            success_rate = successes / config.target
            ability_coverage = evaluate_ability_positive_coverage(
                queue_summary,
                result_audit,
                minimum_success_count=(
                    config.minimum_ability_positive_success_count
                ),
                minimum_success_rate=(
                    config.minimum_ability_positive_success_rate
                ),
                waived=config.waive_ability_positive_coverage,
                waiver_reason=config.ability_positive_waiver_reason,
            )
            coverage_receipt = {
                "schema_version": 2,
                "kind": "cr_expert_native_generation_coverage_v2",
                "created_utc": utc_now(),
                "frozen_manifest": file_fingerprint(config.frozen_manifest),
                "candidate_queue": file_fingerprint(config.candidate_queue),
                "results": file_fingerprint(results_path),
                "native_contract": native_contract_binding(
                    config.native_contract
                ),
                "target_battles": config.target,
                "selected_battles": selected,
                "processed_battles": processed,
                "teacher_forced_successes": successes,
                "teacher_forced_failures": failures,
                "stored_episodes": stored,
                "audit_prefix_episodes": prefix_stored,
                "audit_tick_episodes": int(
                    summary.get("audit_tick_episodes", -1)
                ),
                "unframed_episodes": unframed,
                "audit_tick_coverage_rate": summary.get(
                    "audit_tick_coverage_rate"
                ),
                "audit_prefix_store": file_fingerprint(
                    prefix_store_manifest
                ),
                "success_rate": success_rate,
                "minimum_success_rate": config.minimum_native_success_rate,
                "ability_coverage": ability_coverage,
                "failure_class_counts": summary.get("failure_class_counts") or {},
                "failure_domain_counts": summary.get("failure_domain_counts") or {},
                "terminal_diagnostic_counts": summary.get(
                    "terminal_diagnostic_counts"
                ) or {},
                "queue_counts": summary.get("queue_counts") or {},
                "native_actions_attempted": int(
                    summary.get("native_actions_attempted", 0)
                ),
                "native_actions_accepted": int(
                    summary.get("native_actions_accepted", 0)
                ),
            }
            # Publish the classification even when a gate fails so the failed
            # one-click stage preserves exact evidence for diagnosis/waiver.
            _atomic_json(config.native_generation_receipt, coverage_receipt)
            if (
                summary.get("publication_ready") is not True
                or summary.get("infrastructure_complete") is not True
                or selected != config.target
                or processed != config.target
                or successes + failures != config.target
                or stored != successes
                or prefix_stored != failures
                or unframed != 0
                or int(summary.get("audit_tick_episodes", -1)) != config.target
                or summary.get("audit_tick_coverage_complete") is not True
                or set(result_audit["success_tags"])
                & set(result_audit["audit_prefix_tags"])
                or len(result_audit["success_tags"])
                + len(result_audit["audit_prefix_tags"])
                != config.target
                or summary.get("missing_result_tags") != []
                or summary.get("unexpected_result_tags") != []
                or success_rate < config.minimum_native_success_rate
                or (ability_coverage.get("gate") or {}).get("admitted") is not True
                or int(result_audit["successes"]) != successes
                or int(result_audit["failures"]) != failures
                or result_audit["failure_class_counts"]
                != (summary.get("failure_class_counts") or {})
                or ((native_manifest.get("content") or {}).get("results_sha256"))
                != sha256_file(results_path)
            ):
                raise OneClickError(
                    "native generator failed complete-attempt/success coverage: "
                    f"selected={selected}, processed={processed}, "
                    f"successes={successes}, failures={failures}, stored={stored}, "
                    f"rate={success_rate:.6f}, "
                    f"minimum={config.minimum_native_success_rate:.6f}, "
                    f"ability={json.dumps(ability_coverage, ensure_ascii=False)}"
                )
            return fingerprint_files(
                [
                    summary_path,
                    manifest_path,
                    store_manifest,
                    mask_manifest,
                    prefix_store_manifest,
                    results_path,
                    config.native_generation_receipt,
                ]
            ), {
                **queue_summary,
                "processed_battles": processed,
                "teacher_forced_successes": successes,
                "teacher_forced_failures": failures,
                "success_rate": success_rate,
                "failure_class_counts": coverage_receipt[
                    "failure_class_counts"
                ],
                "ability_coverage": ability_coverage,
                "stored_episodes": stored,
                "audit_prefix_episodes": prefix_stored,
                "audit_tick_episodes": int(summary["audit_tick_episodes"]),
                "unframed_episodes": unframed,
                "stored_ticks": int(summary["stored_ticks"]),
            }

        self._run_stage("generate_native_ticks", inputs, action)

    def validate_tick_store(self) -> None:
        config = self.config
        native_manifest = config.native_root / "manifest.json"
        summary_path = config.native_root / "summary.json"
        store_manifest = config.tick_store_root / "manifest.json"
        prefix_store_manifest = (
            config.audit_prefix_store_root / "manifest.json"
        )
        mask_manifest = (
            config.tick_store_root / "deployment-masks-v1" / "manifest.json"
        )
        inputs = fingerprint_files(
            [
                native_manifest,
                summary_path,
                store_manifest,
                prefix_store_manifest,
                mask_manifest,
                config.native_root / "results.jsonl",
                config.native_generation_receipt,
                config.worker_stop_receipt,
            ]
        ) + component_fingerprints(
            config,
            "expert_v1/native_dataset_generator.py",
            "expert_v1/tick_store_v1/deployment_masks.py",
            "expert_v1/tick_store_v1/codec.py",
        )

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            # This is deliberately a full physical scan.  It checks every
            # immutable CRTS/index SHA and every episode's content-addressed
            # deployment-mask metadata before compilation can begin.
            from expert_v1.native_dataset_generator import (
                verify_published_audit_prefix_store,
                verify_published_tick_store,
            )

            physical = verify_published_tick_store(config.tick_store_root)
            prefix_physical = verify_published_audit_prefix_store(
                config.audit_prefix_store_root
            )
            summary = _read_json(summary_path)
            coverage = _read_json(config.native_generation_receipt)
            if (
                physical["episodes"] != int(summary.get("stored_episodes", -1))
                or physical["ticks"] != int(summary.get("stored_ticks", -1))
                or physical["deployment_mask_sidecars_referenced"] <= 0
                or int(coverage.get("processed_battles", -1)) != config.target
                or int(coverage.get("stored_episodes", -1))
                != physical["episodes"]
                or coverage.get("kind")
                != "cr_expert_native_generation_coverage_v2"
                or ((coverage.get("ability_coverage") or {}).get("gate") or {}).get(
                    "admitted"
                )
                is not True
                or prefix_physical["episodes"]
                != int(summary.get("audit_prefix_episodes", -1))
                or prefix_physical["ticks"]
                != int(summary.get("audit_prefix_ticks", -1))
                or set(physical["battle_tags"])
                & set(prefix_physical["battle_tags"])
                or len(physical["battle_tags"])
                + len(prefix_physical["battle_tags"])
                != config.target
            ):
                raise OneClickError(
                    "Tick Store/Mask physical validation disagrees with summary"
                )
            receipt = {
                "schema_version": 1,
                "kind": "cr_expert_tick_store_mask_validation_v1",
                "created_utc": utc_now(),
                "inputs": inputs,
                "physical": physical,
                "audit_prefix_physical": prefix_physical,
            }
            _atomic_json(config.tick_validation_receipt, receipt)
            return [file_fingerprint(config.tick_validation_receipt)], physical

        self._run_stage("validate_tick_store_and_masks", inputs, action)

    def compile(self) -> None:
        config = self.config
        inputs = fingerprint_files(
            [
                config.tick_validation_receipt,
                config.tick_store_root / "manifest.json",
                config.tick_store_root / "deployment-masks-v1" / "manifest.json",
                config.frozen_manifest,
                config.native_contract,
                config.native_generation_receipt,
            ]
        ) + component_fingerprints(
            config,
            "expert_v1/compile_native_bc_dataset.py",
            "expert_v1/tick_store_v1/deployment_masks.py",
            "expert_v1/tick_store_v1/codec.py",
            "expert_v1/tick_store_v1/schema.py",
            "expert_v1/training_v1/schema.py",
        )

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            self.runner.run(
                compile_command(config),
                cwd=config.project_root,
                log_name="compile-native-bc",
            )
            manifest_path = config.compiled_root / "manifest.json"
            manifest_sha_path = config.compiled_root / "manifest.sha256"
            result_path = config.compiled_root / "compile-result.json"
            manifest = _read_json(manifest_path)
            coverage_receipt = _read_json(config.native_generation_receipt)
            source = manifest.get("source_manifest") or {}
            gates = manifest.get("quality_gates") or {}
            if (
                manifest.get("production_ready") is not True
                or manifest.get("native_replay_validated") is not True
                or Path(str(source.get("path") or "")).resolve()
                != config.frozen_manifest.resolve()
                or source.get("sha256") != sha256_file(config.frozen_manifest)
                or int((manifest.get("coverage") or {}).get("battles", -1))
                != int(coverage_receipt.get("stored_episodes", -2))
                or (manifest.get("native_generation_coverage") or {}).get(
                    "receipt_sha256"
                )
                != sha256_file(config.native_generation_receipt)
                or (manifest.get("native_generation_coverage") or {}).get(
                    "ability_coverage"
                )
                != coverage_receipt.get("ability_coverage")
            ):
                raise OneClickError("compiled dataset failed production admission")
            required_zero = (
                "split_collisions",
                "forbidden_actor_features",
                "nonfinite_features",
                "expert_label_mask_violations",
                "native_action_rejections",
                "terminal_mismatches",
                "missing_mask_sidecars",
            )
            failures = {
                name: gates.get(name)
                for name in required_zero
                if gates.get(name) != 0
            }
            if failures:
                raise OneClickError(f"compiled quality gates failed: {failures}")
            return fingerprint_files(
                [manifest_path, manifest_sha_path, result_path]
            ), {
                "dataset_content_sha256": manifest.get("dataset_content_sha256"),
                "battles": int((manifest.get("coverage") or {}).get("battles", 0)),
                "rows": int((manifest.get("coverage") or {}).get("rows", 0)),
            }

        self._run_stage("compile_native_bc", inputs, action)

    def stop_workers(self) -> None:
        config = self.config
        generation_artifacts = [
            config.native_root / "summary.json",
            config.native_root / "manifest.json",
            config.native_root / "results.jsonl",
            config.tick_store_root / "manifest.json",
            config.audit_prefix_store_root / "manifest.json",
            config.tick_store_root / "deployment-masks-v1" / "manifest.json",
            config.native_generation_receipt,
        ]
        inputs = fingerprint_files(generation_artifacts) + [
            value_fingerprint("worker-layout", {
                "workers": config.workers,
                "avds": config.avds,
                "workers_per_avd": config.workers_per_avd,
                "ports": config.ports[: config.workers],
            }),
        ] + component_fingerprints(
            config,
            "native_core/worker.py",
            "scripts/stop_direct_service.ps1",
        )

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            result = self.runner.run(
                native_worker_command(config, "stop"),
                cwd=config.project_root,
                log_name="native-workers",
            )
            try:
                stopped = json.loads(result.stdout)
            except Exception as error:
                raise OneClickError(
                    "native worker stop did not return a JSON receipt"
                ) from error
            instances = (
                stopped.get("instances") if isinstance(stopped, dict) else None
            )
            valid = bool(
                isinstance(instances, list)
                and len(instances) == config.avds
                and sum(
                    len(item.get("services") or [])
                    for item in instances
                    if isinstance(item, Mapping)
                )
                == config.workers
                and all(
                    isinstance(item, Mapping)
                    and item.get("vm_stopped") is True
                    and isinstance(item.get("services"), list)
                    and len(item["services"]) == config.workers_per_avd
                    and all(
                        isinstance(service, Mapping)
                        and service.get("stopped") is True
                        for service in item["services"]
                    )
                    for item in instances
                )
            )
            if not valid:
                raise OneClickError(
                    f"native workers/AVD did not fully stop: {stopped}"
                )
            receipt = {
                "schema_version": 1,
                "kind": "cr_expert_native_workers_stopped_v1",
                "created_utc": utc_now(),
                "command": list(result.command),
                "returncode": result.returncode,
                "stdout_sha256": hashlib.sha256(
                    result.stdout.encode("utf-8")
                ).hexdigest(),
                "verified_stop_result": stopped,
                "generation_inputs": inputs,
            }
            _atomic_json(config.worker_stop_receipt, receipt)
            return [file_fingerprint(config.worker_stop_receipt)], {
                "vm_stop_requested": True,
                "workers": config.workers,
                "avds": config.avds,
            }

        self._run_stage("stop_native_workers", inputs, action)

    def smoke(self) -> None:
        config = self.config
        inputs = fingerprint_files(
            [
                config.compiled_root / "manifest.json",
                config.frozen_manifest,
                config.worker_stop_receipt,
            ]
        ) + component_fingerprints(
            config,
            "expert_v1/training_v1/train.py",
            "expert_v1/training_v1/dataset.py",
            "expert_v1/training_v1/model.py",
            "expert_v1/training_v1/losses.py",
            "expert_v1/training_v1/schema.py",
        )

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            command = training_smoke_command(config)
            run_id = command[command.index("--run-id") + 1]
            self.runner.run(
                command,
                cwd=config.project_root,
                log_name="real-data-training-smoke",
            )
            run_root = config.smoke_output_root / run_id
            result_path = run_root / "result.json"
            latest = run_root / "checkpoints" / "latest.pt"
            best = run_root / "checkpoints" / "best.pt"
            result = _read_json(result_path)
            if result.get("event") != "run_complete":
                raise OneClickError("real-data training smoke did not complete")
            return fingerprint_files([result_path, latest, best]), {
                "run_id": run_id,
                "dataset_manifest_sha256": result.get(
                    "dataset_manifest_sha256"
                ),
            }

        self._run_stage("real_data_training_smoke", inputs, action)

    def train(self) -> None:
        config = self.config
        smoke_stage = self.journal.value["stages"].get(
            "real_data_training_smoke", {}
        )
        smoke_outputs = smoke_stage.get("outputs") or []
        inputs = fingerprint_files(
            [config.compiled_root / "manifest.json", config.frozen_manifest]
        ) + component_fingerprints(
            config,
            "expert_v1/training_v1/train.py",
            "expert_v1/training_v1/dataset.py",
            "expert_v1/training_v1/model.py",
            "expert_v1/training_v1/losses.py",
            "expert_v1/training_v1/schema.py",
        ) + [dict(item) for item in smoke_outputs]

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            self.runner.run(
                formal_training_command(config),
                cwd=config.project_root,
                log_name="formal-training",
            )
            run_root = config.training_output_root / config.training_run_id
            result_path = run_root / "result.json"
            latest = run_root / "checkpoints" / "latest.pt"
            best = run_root / "checkpoints" / "best.pt"
            result = _read_json(result_path)
            if result.get("event") != "run_complete":
                raise OneClickError("formal training returned without run_complete")
            return fingerprint_files([result_path, latest, best]), {
                "run_id": config.training_run_id,
                "dataset_manifest_sha256": result.get(
                    "dataset_manifest_sha256"
                ),
            }

        self._run_stage("formal_training", inputs, action)

    def run(self) -> None:
        static_configuration = value_fingerprint(
            "one-click-static-config",
            self.config.declared(include_native_layout=False),
        )
        existing_static = self.journal.value.get("static_configuration")
        if existing_static is not None and existing_static != static_configuration:
            raise OneClickError(
                "one-click static configuration changed; use a new --data-root"
            )
        if existing_static is None:
            self.journal.value["static_configuration"] = static_configuration
            self.journal.save()
        self.collect()
        # The crawler supervisor has now stopped and released its Chromium
        # workers.  Only at this point is available RAM representative of the
        # native phase.  The result is immutable on every resume.
        self._ensure_native_layout()
        configuration = value_fingerprint(
            "one-click-config", self.config.declared()
        )
        existing = self.journal.value.get("configuration")
        if existing is not None and existing != configuration:
            raise OneClickError(
                "one-click configuration changed; use a new --data-root"
            )
        if existing is None:
            self.journal.value["configuration"] = configuration
            self.journal.save()
        self.freeze()
        self.audit()
        with OneClickLock(
            self.config.native_hardware_lock_path,
            conflict_message=(
                "native Android hardware is owned by another one-click run"
            ),
        ):
            try:
                self.generate_native()
                self.stop_workers()
            except BaseException:
                self._best_effort_stop_native()
                raise
        self.validate_tick_store()
        self.compile()
        self.smoke()
        self.train()

    def run_smoke_only(self) -> None:
        """Run the real compiled-data smoke only; never starts crawler/AVD."""

        if not (self.config.compiled_root / "manifest.json").is_file():
            raise OneClickError(
                "--smoke requires an already compiled production dataset; "
                "it will not fabricate or collect input"
            )
        if not self.config.worker_stop_receipt.is_file():
            raise OneClickError(
                "--smoke requires the persisted AVD-stop receipt"
            )
        self.smoke()


def status(config: OneClickConfig) -> dict[str, Any]:
    value = _read_json(config.state_path)
    _validate_state_generation(value, config.state_path)
    try:
        accepted = _authoritative_count(config)
    except (OSError, sqlite3.Error):
        accepted = None
    stage_rows: list[dict[str, Any]] = []
    stored = value.get("stages") if isinstance(value, Mapping) else {}
    for stage in STAGES:
        record = stored.get(stage, {}) if isinstance(stored, Mapping) else {}
        stage_rows.append(
            {
                "stage": stage,
                "status": record.get("status", "pending"),
                "details": record.get("details", {}),
            }
        )
    return {
        "schema_version": 1,
        "kind": "cr_expert_one_click_status_v1",
        "contract_generation": STATE_CONTRACT_GENERATION,
        "state_path": str(config.state_path),
        "active_stage": value.get("active_stage") if value else None,
        "last_error": value.get("last_error") if value else None,
        "authoritative_accepted": accepted,
        "authoritative_target": config.target,
        "crawler_active": _crawler_active(config),
        "native_layout": (
            value.get("native_layout")
            if value and value.get("native_layout") is not None
            else {"status": "pending_post_collection_ram_preflight"}
        ),
        "stages": stage_rows,
        "training_entry_ready": bool(
            value
            and (value.get("stages") or {}).get(
                "real_data_training_smoke", {}
            ).get("status")
            == "completed"
        ),
    }


def _default_config(args: argparse.Namespace) -> OneClickConfig:
    data_root = args.data_root.resolve()
    crawler_root = args.crawler_root.resolve()
    authoritative_root = args.authoritative_root.resolve()
    if authoritative_root.name.casefold() != EXPECTED_AUTHORITATIVE_ROOT_NAME:
        raise OneClickError(
            "--authoritative-root must end in "
            f"{EXPECTED_AUTHORITATIVE_ROOT_NAME!r}; old v2 output is read-only"
        )
    if data_root.name.casefold() in LEGACY_DATA_ROOT_NAMES:
        raise OneClickError(
            "--data-root points at a legacy v2 state namespace; use the v3 root"
        )
    persisted_state = _read_json(data_root / "control" / "state.json")
    _validate_state_generation(
        persisted_state, data_root / "control" / "state.json"
    )
    persisted_layout = persisted_state.get("native_layout")
    if isinstance(persisted_layout, Mapping):
        avds = int(persisted_layout.get("avds") or 0)
        workers_per_avd = int(
            persisted_layout.get("workers_per_avd") or 0
        )
        workers = int(persisted_layout.get("workers") or 0)
        selected_ports = tuple(int(item) for item in persisted_layout.get("ports") or [])
        layout_reason = str(persisted_layout.get("reason") or "persisted")
        available_ram = int(
            persisted_layout.get("available_ram_at_selection") or 0
        )
    else:
        # Deliberately unresolved here.  Collection owns Chromium/proxy memory;
        # layout is selected only after collect() has stopped that process.
        available_ram = 0
        avds = 0
        workers_per_avd = 4
        workers = 0
        selected_ports = ()
        layout_reason = "pending_post_collection_ram_preflight"
    if (
        args.workers is not None
        and workers > 0
        and int(args.workers) != workers
    ):
        raise OneClickError(
            f"--workers={args.workers} conflicts with frozen/automatic "
            f"{avds} AVD x {workers_per_avd} layout ({workers})"
        )
    return OneClickConfig(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        crawler_root=crawler_root,
        crawler_python=args.crawler_python.resolve(),
        training_python=args.training_python.resolve(),
        crawler_config=args.crawler_config.resolve(),
        authoritative_db=args.authoritative_db.resolve(),
        authoritative_root=authoritative_root,
        native_contract=args.native_contract.resolve(),
        template=args.template.resolve(),
        target=args.target,
        workers=workers,
        avds=avds,
        workers_per_avd=workers_per_avd,
        ports=selected_ports,
        minimum_native_success_rate=args.minimum_native_success_rate,
        minimum_ability_positive_success_count=(
            args.minimum_ability_positive_success_count
        ),
        minimum_ability_positive_success_rate=(
            args.minimum_ability_positive_success_rate
        ),
        waive_ability_positive_coverage=args.waive_ability_positive_coverage,
        ability_positive_waiver_reason=args.ability_positive_waiver_reason,
        native_layout_reason=layout_reason,
        available_ram_at_selection=available_ram,
        requested_workers=args.workers,
        poll_seconds=args.poll_seconds,
        audit_workers=args.audit_workers,
        compile_io_workers=args.compile_io_workers,
        compile_process_workers=args.compile_process_workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run only the real compiled-data training smoke; no collection/AVD",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--crawler-root", type=Path, default=DEFAULT_CRAWLER_ROOT)
    parser.add_argument(
        "--crawler-python", type=Path, default=DEFAULT_CRAWLER_PYTHON
    )
    parser.add_argument(
        "--training-python", type=Path, default=DEFAULT_TRAINING_PYTHON
    )
    parser.add_argument(
        "--crawler-config", type=Path, default=DEFAULT_CRAWLER_CONFIG
    )
    parser.add_argument(
        "--authoritative-db",
        type=Path,
        default=DEFAULT_CRAWLER_ROOT / "data" / "authoritative-progress.sqlite3",
    )
    parser.add_argument(
        "--authoritative-root",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
            r"\authoritative-schema5-v3"
        ),
    )
    parser.add_argument("--native-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--workers",
        type=int,
        help="attest the auto/frozen worker total; does not override layout",
    )
    parser.add_argument("--ports", nargs="+", type=int, default=list(DEFAULT_PORTS))
    parser.add_argument(
        "--minimum-native-success-rate", type=float, default=0.50
    )
    parser.add_argument(
        "--allow-lower-native-success-rate",
        action="store_true",
        help="explicit waiver required to lower the default 50%% coverage gate",
    )
    parser.add_argument(
        "--minimum-ability-positive-success-count",
        type=int,
        default=DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_COUNT,
        help="minimum successful native replays among ability-positive candidates",
    )
    parser.add_argument(
        "--minimum-ability-positive-success-rate",
        type=float,
        default=DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_RATE,
        help="minimum native success rate within the ability-positive cohort",
    )
    parser.add_argument(
        "--waive-ability-positive-coverage",
        action="store_true",
        help=(
            "explicitly waive the ability-positive native coverage gate; "
            "requires --ability-positive-waiver-reason"
        ),
    )
    parser.add_argument("--ability-positive-waiver-reason")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--audit-workers", type=int, default=max(8, os.cpu_count() or 8)
    )
    parser.add_argument(
        "--compile-io-workers",
        type=int,
        default=min(32, max(4, (os.cpu_count() or 4) * 2)),
    )
    parser.add_argument(
        "--compile-process-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.target <= 0
        or tuple(args.ports) != DEFAULT_PORTS
        or args.poll_seconds <= 0
        or args.audit_workers <= 0
        or args.compile_io_workers <= 0
        or args.compile_process_workers <= 0
        or not 0 < args.minimum_native_success_rate <= 1
        or args.minimum_ability_positive_success_count < 0
        or not 0 <= args.minimum_ability_positive_success_rate <= 1
    ):
        raise OneClickError("invalid one-click concurrency/target configuration")
    if (
        args.minimum_native_success_rate < 0.50
        and not args.allow_lower_native_success_rate
    ):
        raise OneClickError(
            "lowering native success coverage below 50% requires "
            "--allow-lower-native-success-rate"
        )
    lowered_ability_gate = (
        args.minimum_ability_positive_success_count
        < DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_COUNT
        or args.minimum_ability_positive_success_rate
        < DEFAULT_MINIMUM_ABILITY_POSITIVE_SUCCESS_RATE
    )
    waiver_reason = str(args.ability_positive_waiver_reason or "").strip()
    if lowered_ability_gate and not args.waive_ability_positive_coverage:
        raise OneClickError(
            "lowering ability-positive native coverage requires "
            "--waive-ability-positive-coverage"
        )
    if args.waive_ability_positive_coverage and not waiver_reason:
        raise OneClickError(
            "--waive-ability-positive-coverage requires "
            "--ability-positive-waiver-reason"
        )
    if waiver_reason and not args.waive_ability_positive_coverage:
        raise OneClickError(
            "--ability-positive-waiver-reason requires "
            "--waive-ability-positive-coverage"
        )
    config = _default_config(args)
    if config.avds != 0 and (
        config.avds not in {1, 2}
        or config.workers_per_avd != 4
        or config.workers != config.avds * config.workers_per_avd
        or config.ports != DEFAULT_PORTS[: config.workers]
    ):
        raise OneClickError("persisted native layout/ports are invalid")
    if args.status:
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return 0
    for required in (
        config.training_python,
        config.crawler_python,
        config.crawler_config,
        config.native_contract,
        config.template,
    ):
        if not required.is_file():
            raise OneClickError(f"required one-click dependency missing: {required}")
    with OneClickLock(config.lock_path):
        orchestrator = OneClickOrchestrator(config)
        if args.smoke:
            orchestrator.run_smoke_only()
        else:
            orchestrator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
