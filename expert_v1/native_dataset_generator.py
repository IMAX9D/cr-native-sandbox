"""Resume-safe authoritative native Tick-dataset generation.

This module is intentionally an orchestration layer.  It does not implement
or patch battle rules: every transition is produced by ``libg`` through
``native_replay_runner.execute_plan``.  Source battle JSON is read in place,
SHA-256 verified, and never copied into the generated dataset.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback is supported
    orjson = None

from native_core.env import NativeRoyaleEnv

from .native_ingest_contract import load_native_ingest_contract
from .native_freeze import NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK
from .native_profile import (
    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET,
    action_tick_provenance,
    native_teacher_forced_profile,
)
from .native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    compile_battle,
)
from .native_replay_runner import NativeReplayResult, execute_plan, load_template
from .native_seed_search import (
    DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    NativeSeedSearchError,
)
from .tick_store_v1.codec import EpisodeReader, encode_episode
from .tick_store_v1.deployment_masks import (
    EPISODE_METADATA_KEY,
    MASK_STORE_DIRECTORY,
    DeploymentMaskStore,
    NativeDeploymentMaskCapture,
)
from .tick_store_v1.schema import TickState, require_consecutive
from .tick_store_v1.shard import (
    FRAME_HEADER,
    FRAME_MAGIC,
    SHARD_KIND,
    WorkerShardSink,
    _scan_frames,
    build_store_manifest,
    sha256_file,
)
from .tick_store_v1.work_queue import ClaimedTask, TickStoreWorkQueue


GENERATOR_SCHEMA_VERSION = 1
GENERATOR_KIND = "expert_authoritative_native_tick_generator_v1"
TASK_KIND = "expert_authoritative_native_tick_task_v1"
RESULT_KIND = "expert_authoritative_native_tick_result_v1"
DIAGNOSTIC_KIND = "expert_authoritative_native_tick_failure_v1"
COORDINATE_PROVENANCE = "royaleapi_raw_data_i_to_native_v1"
EXACT_ABILITY_TIERS = {
    "source_reports_zero",
    "observed_ticks_identity_runtime_resolved",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _loads(raw: bytes | str) -> Any:
    return orjson.loads(raw) if orjson is not None else json.loads(raw)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for value in values:
            output.write(
                json.dumps(
                    dict(value), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _selection_digest(seed: str, battle_tag: str) -> str:
    return hashlib.sha256(f"{seed}\0{battle_tag}".encode("utf-8")).hexdigest()


def _stable_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeDatasetTask:
    selection_index: int
    selection_digest: str
    battle_tag: str
    source_path: str
    source_sha256: str
    source_schema_version: int
    duration_ticks: int
    deployment_actions: int
    ability_count_reported: int
    ability_events_observed: int
    ability_log_tier: str
    coordinate_tier: str
    eligibility_tier: str

    @property
    def ability_positive(self) -> bool:
        return self.ability_events_observed > 0

    def json(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATOR_SCHEMA_VERSION,
            "kind": TASK_KIND,
            **asdict(self),
        }

    def queue_row(self) -> dict[str, Any]:
        value = self.json()
        return {
            "battle_tag": self.battle_tag,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "payload": {
                key: item
                for key, item in value.items()
                if key not in {"battle_tag", "source_path", "source_sha256"}
            },
        }

    @classmethod
    def from_claim(cls, claim: ClaimedTask) -> "NativeDatasetTask":
        value = {
            "battle_tag": claim.battle_tag,
            "source_path": claim.source_path,
            "source_sha256": claim.source_sha256,
            **claim.payload,
        }
        if value.pop("kind", None) != TASK_KIND:
            raise ValueError(f"invalid queued task kind for {claim.battle_tag}")
        if int(value.pop("schema_version", -1)) != GENERATOR_SCHEMA_VERSION:
            raise ValueError(f"invalid queued task schema for {claim.battle_tag}")
        return cls(**{
            key: value[key] for key in cls.__dataclass_fields__
        })


def _validate_candidate_row(value: Mapping[str, Any], line_number: int) -> None:
    tag = str(value.get("battle_tag") or "")
    if not tag:
        raise ValueError(f"candidate line {line_number} has no battle_tag")
    if value.get("authoritative_native_full_candidate") is not True:
        raise ValueError(f"candidate {tag} is not authoritative-native-full")
    source_schema = int(value.get("source_schema_version") or 0)
    if source_schema not in {3, 5}:
        raise ValueError(f"candidate {tag} is not source schema 3 or 5")
    if str(value.get("coordinate_tier")) != "all_card_events_raw_data_i":
        raise ValueError(f"candidate {tag} lacks exact raw data_i coordinates")
    if str(value.get("ability_log_tier")) not in EXACT_ABILITY_TIERS:
        raise ValueError(f"candidate {tag} lacks exact ability semantics")
    source_sha = str(value.get("source_sha256") or "")
    if len(source_sha) != 64:
        raise ValueError(f"candidate {tag} has invalid source SHA-256")
    if source_schema == 5:
        required = (
            "schema5_authoritative_contract_verified",
            "tower_troop_levels_complete",
            "king_tower_levels_complete",
            "final_tower_hp_complete",
        )
        missing = [name for name in required if value.get(name) is not True]
        if missing:
            raise ValueError(
                f"candidate {tag} lacks authoritative schema5 fields: {missing}"
            )
        mode = value.get("numeric_game_mode_id")
        if not isinstance(mode, int) or isinstance(mode, bool) or mode <= 0:
            raise ValueError(f"candidate {tag} lacks schema5 numeric game mode")
        execution_mode = value.get("native_execution_game_mode_id")
        if (
            not isinstance(execution_mode, int)
            or isinstance(execution_mode, bool)
            or execution_mode <= 0
        ):
            raise ValueError(
                f"candidate {tag} lacks schema5 native execution game mode"
            )
        if (
            value.get("native_execution_game_mode_provenance")
            != "frozen_native_ingest_contract_mode_map_v1"
        ):
            raise ValueError(
                f"candidate {tag} lacks schema5 execution-mode provenance"
            )
        battle_index = value.get("battle_index")
        if (
            not isinstance(battle_index, int)
            or isinstance(battle_index, bool)
            or battle_index <= 0
        ):
            raise ValueError(f"candidate {tag} lacks schema5 battle index")


def select_tasks(
    candidate_queue: Path,
    *,
    limit: int | None = None,
    selection_seed: str = "authoritative-native-full-v1",
    ensure_mixed_when_limited: bool = True,
    deployment_zero_quota: int | None = None,
    ability_exact_quota: int | None = None,
) -> tuple[list[NativeDatasetTask], dict[str, Any]]:
    """Load the immutable pointer queue and choose a deterministic subset.

    A limited run is SHA-ranked.  When at least two rows are requested and
    both strata exist, one exact-ability-positive and one source-reports-zero
    row are reserved before filling the remaining slots from the same global
    ranking.  This makes a ten-battle smoke exercise both execution paths.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    quotas_supplied = (
        deployment_zero_quota is not None or ability_exact_quota is not None
    )
    if quotas_supplied:
        if deployment_zero_quota is None or ability_exact_quota is None:
            raise ValueError("both explicit stratum quotas must be supplied")
        if deployment_zero_quota < 0 or ability_exact_quota < 0:
            raise ValueError("explicit stratum quotas must be non-negative")
        if limit is None:
            raise ValueError("explicit stratum quotas require --limit")
        if deployment_zero_quota + ability_exact_quota != limit:
            raise ValueError("explicit stratum quotas must sum exactly to limit")
    candidate_queue = candidate_queue.resolve(strict=True)
    raw_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with candidate_queue.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            value = _loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"candidate line {line_number} is not an object")
            _validate_candidate_row(value, line_number)
            tag = str(value["battle_tag"])
            if tag in seen:
                raise ValueError(f"duplicate candidate battle_tag: {tag}")
            seen.add(tag)
            raw_rows.append(value)
    ranked = sorted(
        raw_rows,
        key=lambda row: (
            _selection_digest(selection_seed, str(row["battle_tag"])),
            str(row["battle_tag"]),
        ),
    )
    selected_rows: list[dict[str, Any]]
    mixed_reserved: list[str] = []
    if quotas_supplied:
        assert deployment_zero_quota is not None
        assert ability_exact_quota is not None
        zero_rows = [
            row for row in ranked if int(row["ability_events_observed"]) == 0
        ]
        ability_rows = [
            row for row in ranked if int(row["ability_events_observed"]) > 0
        ]
        if len(zero_rows) < deployment_zero_quota:
            raise RuntimeError(
                "deployment-zero stratum quota cannot be satisfied: "
                f"requested {deployment_zero_quota}, available {len(zero_rows)}"
            )
        if len(ability_rows) < ability_exact_quota:
            raise RuntimeError(
                "ability-exact stratum quota cannot be satisfied: "
                f"requested {ability_exact_quota}, available {len(ability_rows)}"
            )
        selected_rows = (
            zero_rows[:deployment_zero_quota]
            + ability_rows[:ability_exact_quota]
        )
        selected_rows.sort(
            key=lambda row: (
                _selection_digest(selection_seed, str(row["battle_tag"])),
                str(row["battle_tag"]),
            )
        )
    elif limit is None or limit >= len(ranked):
        selected_rows = ranked
    elif ensure_mixed_when_limited and limit >= 2:
        positive = next(
            (row for row in ranked if int(row["ability_events_observed"]) > 0),
            None,
        )
        zero = next(
            (row for row in ranked if int(row["ability_events_observed"]) == 0),
            None,
        )
        reserved = [row for row in (positive, zero) if row is not None]
        reserved_tags = {str(row["battle_tag"]) for row in reserved}
        mixed_reserved = sorted(reserved_tags)
        selected_rows = list(reserved)
        selected_rows.extend(
            row for row in ranked
            if str(row["battle_tag"]) not in reserved_tags
        )
        selected_rows = selected_rows[:limit]
        selected_rows.sort(
            key=lambda row: (
                _selection_digest(selection_seed, str(row["battle_tag"])),
                str(row["battle_tag"]),
            )
        )
    else:
        selected_rows = ranked[:limit]
    tasks = [
        NativeDatasetTask(
            selection_index=index,
            selection_digest=_selection_digest(
                selection_seed, str(row["battle_tag"])
            ),
            battle_tag=str(row["battle_tag"]),
            source_path=str(Path(str(row["source_path"])).resolve()),
            source_sha256=str(row["source_sha256"]),
            source_schema_version=int(row["source_schema_version"]),
            duration_ticks=int(row["duration_ticks"]),
            deployment_actions=int(row["deployment_actions"]),
            ability_count_reported=int(row["ability_count_reported"]),
            ability_events_observed=int(row["ability_events_observed"]),
            ability_log_tier=str(row["ability_log_tier"]),
            coordinate_tier=str(row["coordinate_tier"]),
            eligibility_tier=str(row["eligibility_tier"]),
        )
        for index, row in enumerate(selected_rows)
    ]
    summary = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": "expert_authoritative_native_tick_selection_v1",
        "created_utc": utc_now(),
        "candidate_queue": str(candidate_queue),
        "candidate_queue_sha256": sha256_file(candidate_queue),
        "candidate_rows": len(raw_rows),
        "selection_seed": selection_seed,
        "selection_algorithm": (
            "sha256(seed\\0battle_tag), ascending within each explicit stratum"
            if quotas_supplied else
            "sha256(seed\\0battle_tag), ascending; limited runs reserve one "
            "ability-positive and one source-reports-zero row when available"
        ),
        "requested_limit": limit,
        "explicit_stratum_quotas": (
            None if not quotas_supplied else {
                "authoritative_native_deployment_only": deployment_zero_quota,
                "authoritative_native_ability_exact": ability_exact_quota,
            }
        ),
        "selected_rows": len(tasks),
        "selected_rows_digest": _stable_rows_digest(
            [task.json() for task in tasks]
        ),
        "mixed_reserved_battle_tags": mixed_reserved,
        "ability_positive_battles": sum(task.ability_positive for task in tasks),
        "ability_zero_battles": sum(not task.ability_positive for task in tasks),
        "deployment_actions": sum(task.deployment_actions for task in tasks),
        "ability_events": sum(task.ability_events_observed for task in tasks),
        "duration_ticks": sum(task.duration_ticks for task in tasks),
    }
    return tasks, summary


def write_selection(
    output_root: Path,
    tasks: Sequence[NativeDatasetTask],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    selection = output_root / "selection.jsonl"
    selection_summary = output_root / "selection.summary.json"
    rows = [task.json() for task in tasks]
    if selection.exists():
        existing = []
        with selection.open("rb") as source:
            existing = [_loads(line) for line in source if line.strip()]
        if existing != rows:
            raise RuntimeError("resume selection differs from existing selection")
    else:
        atomic_jsonl(selection, rows)
    published = {
        **dict(summary),
        "selection_manifest": str(selection.resolve()),
        "selection_manifest_sha256": sha256_file(selection),
    }
    if selection_summary.exists():
        old = load_json(selection_summary)
        comparable_old = dict(old)
        comparable_new = dict(published)
        comparable_old.pop("created_utc", None)
        comparable_new.pop("created_utc", None)
        if comparable_old != comparable_new:
            raise RuntimeError("resume selection summary contract changed")
    else:
        atomic_json(selection_summary, published)
    return selection, selection_summary


def _component_hashes(project_root: Path) -> dict[str, str]:
    relative = (
        "expert_v1/native_dataset_generator.py",
        "expert_v1/native_replay_runner.py",
        "expert_v1/native_replay_plan.py",
        "expert_v1/native_profile.py",
        "expert_v1/tick_store_v1/codec.py",
        "expert_v1/tick_store_v1/deployment_masks.py",
        "expert_v1/tick_store_v1/schema.py",
        "expert_v1/tick_store_v1/shard.py",
    )
    return {
        name: sha256_file(project_root / name)
        for name in relative
    }


def prepare_run(
    *,
    candidate_queue: Path,
    output_root: Path,
    template_path: Path,
    limit: int | None,
    selection_seed: str,
    deployment_zero_quota: int | None = None,
    ability_exact_quota: int | None = None,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    episodes_per_shard: int,
    anchor_interval: int = 256,
    compression_level: int = 1,
    native_contract_path: Path | None = None,
) -> tuple[list[NativeDatasetTask], Path, Path, dict[str, Any]]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for child in ("shards", "results", "diagnostics", "workers"):
        (output_root / child).mkdir(exist_ok=True)
    tasks, selection_summary = select_tasks(
        candidate_queue,
        limit=limit,
        selection_seed=selection_seed,
        ensure_mixed_when_limited=True,
        deployment_zero_quota=deployment_zero_quota,
        ability_exact_quota=ability_exact_quota,
    )
    selection_path, _ = write_selection(output_root, tasks, selection_summary)
    template_path = template_path.resolve(strict=True)
    native_contract = (
        None
        if native_contract_path is None
        else load_native_ingest_contract(native_contract_path)
    )
    project_root = Path(__file__).resolve().parents[1]
    contract = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": GENERATOR_KIND,
        "candidate_queue": str(candidate_queue.resolve(strict=True)),
        "candidate_queue_sha256": sha256_file(candidate_queue.resolve(strict=True)),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_seed": selection_seed,
        "explicit_stratum_quotas": selection_summary[
            "explicit_stratum_quotas"
        ],
        "requested_limit": limit,
        "selected_battles": len(tasks),
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "native_ingest_contract": (
            None
            if native_contract is None
            else {
                "path": str(native_contract.source_path),
                "contract_sha256": str(
                    native_contract.value.get("contract_sha256") or ""
                ),
                "file_sha256": str(native_contract.file_sha256),
            }
        ),
        "native_teacher_forced_profile": native_teacher_forced_profile(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        "action_tick_provenance": action_tick_provenance(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        "preferred_seed": int(seed),
        "preferred_seed_role": (
            "legacy preferred only; bounded source-order search chooses each seed"
        ),
        "maximum_seeds_to_test": int(maximum_seeds_to_test),
        "trace_batch_steps": int(trace_batch_steps),
        "episodes_per_shard": int(episodes_per_shard),
        "anchor_interval": int(anchor_interval),
        "compression": f"zlib-level-{compression_level}",
        "coordinate_provenance_required": COORDINATE_PROVENANCE,
        "ability_branch_policy": "branch_required fails closed; entity is never guessed",
        "native_deployment_masks": {
            "schema_version": 1,
            "capture": "one native probe per side/deck slot when first in hand",
            "expected_base_probe_rpcs_per_success": 16,
            "dynamic_choice_probe_policy": (
                "one additional probe only at each expert play Tick whose "
                "native selection is resource-dependent"
            ),
            "per_tick_rpc": False,
            "content_store": f"shards/{MASK_STORE_DIRECTORY}",
            "dynamic_rule": "native_base_and_tower_state_projection_v1",
            "failure_policy": "missing slot or invalid sidecar fails closed",
        },
        "component_sha256": _component_hashes(project_root),
    }
    contract_path = output_root / "run-contract.json"
    if contract_path.exists():
        existing = load_json(contract_path)
        if existing != contract:
            raise RuntimeError(
                "resume contract changed; use a new output directory"
            )
    else:
        atomic_json(contract_path, contract)
    queue_path = output_root / "work-queue.sqlite3"
    with TickStoreWorkQueue(queue_path) as queue:
        queue.add_tasks(task.queue_row() for task in tasks)
    return tasks, selection_path, queue_path, contract


class RecordingCountingEnv:
    """Transparent environment proxy with exact native-action denominators."""

    def __init__(self, env: Any, *, history_size: int = 12) -> None:
        self.env = env
        self.latest_state: dict[str, Any] | None = None
        self.reset_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.action_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.trace_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.native_action_batches_attempted = 0
        self.native_actions_attempted = 0
        self.native_actions_responded = 0
        self.native_actions_accepted = 0
        self.native_actions_rejected = 0
        self.native_deploy_actions_attempted = 0
        self.native_deploy_actions_accepted = 0
        self.native_ability_actions_attempted = 0
        self.native_ability_actions_accepted = 0
        self.native_action_exceptions = 0
        self.native_deployment_mask_probes_attempted = 0
        self.native_deployment_mask_probes_responded = 0
        self.native_deployment_mask_probe_exceptions = 0
        self.first_rejection: dict[str, Any] | None = None

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

    def trace_train(
        self, steps: int, *, allow_nonterminal_freeze: bool = False
    ) -> dict[str, Any]:
        response = self.env.trace_train(
            steps, allow_nonterminal_freeze=allow_nonterminal_freeze
        )
        frames = response.get("frames") or []
        complete = [
            frame for frame in [response.get("initial_frame"), *frames]
            if isinstance(frame, Mapping)
            and frame.get("observation_complete") is True
            and isinstance(frame.get("state"), Mapping)
        ]
        if complete:
            self.latest_state = deepcopy(dict(complete[-1]["state"]))
        self.trace_history.append({
            "requested_steps": int(steps),
            "stepped": int(response.get("stepped", -1)),
            "initial_tick": int(
                response.get("initial_frame", {}).get("state", {}).get("tick", -1)
            ),
            "final_tick": int(response.get("final_tick", -1)),
            "terminal": bool(response.get("terminal", False)),
            "nonterminal_freeze": bool(response.get("nonterminal_freeze", False)),
        })
        return response

    def joint_act(self, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        request = deepcopy(list(actions))
        self.native_action_batches_attempted += 1
        self.native_actions_attempted += len(request)
        self.native_deploy_actions_attempted += sum(
            str(action.get("type")) == "play" for action in request
        )
        self.native_ability_actions_attempted += sum(
            str(action.get("type")) == "ability" for action in request
        )
        audit: dict[str, Any] = {
            "pre_action_state": deepcopy(self.latest_state),
            "request": request,
            "response": None,
            "exception": None,
        }
        try:
            response = self.env.joint_act(request)
            audit["response"] = deepcopy(response)
            results = response.get("actions") or []
            self.native_actions_responded += len(results)
            for index, (action, item) in enumerate(zip(request, results)):
                accepted = bool(item.get("result", {}).get("accepted", False))
                if accepted:
                    self.native_actions_accepted += 1
                    if action.get("type") == "ability":
                        self.native_ability_actions_accepted += 1
                    else:
                        self.native_deploy_actions_accepted += 1
                else:
                    self.native_actions_rejected += 1
                    if self.first_rejection is None:
                        self.first_rejection = {
                            "batch_index": self.native_action_batches_attempted - 1,
                            "action_index_in_batch": index,
                            "request": deepcopy(action),
                            "response": deepcopy(item),
                            "pre_action_tick": (
                                None if self.latest_state is None
                                else int(self.latest_state.get("tick", -1))
                            ),
                        }
            # A short/malformed response is a real attempted action without a
            # response, not an accepted action.  execute_plan fails closed.
            return response
        except Exception as error:
            self.native_action_exceptions += 1
            audit["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.action_history.append(audit)

    def probe_grid(self, *, side: int, deck_index: int) -> dict[str, Any]:
        self.native_deployment_mask_probes_attempted += 1
        try:
            value = self.env.probe_grid(side=side, deck_index=deck_index)
            self.native_deployment_mask_probes_responded += 1
            return value
        except Exception:
            self.native_deployment_mask_probe_exceptions += 1
            raise

    def metrics(self) -> dict[str, Any]:
        no_response = max(
            0, self.native_actions_attempted - self.native_actions_responded
        )
        response_excess = max(
            0, self.native_actions_responded - self.native_actions_attempted
        )
        return {
            "native_action_batches_attempted": self.native_action_batches_attempted,
            "native_actions_attempted": self.native_actions_attempted,
            "native_actions_responded": self.native_actions_responded,
            "native_actions_accepted": self.native_actions_accepted,
            "native_actions_rejected": self.native_actions_rejected,
            "native_actions_no_response": no_response,
            "native_action_response_excess": response_excess,
            "native_deploy_actions_attempted": self.native_deploy_actions_attempted,
            "native_deploy_actions_accepted": self.native_deploy_actions_accepted,
            "native_ability_actions_attempted": self.native_ability_actions_attempted,
            "native_ability_actions_accepted": self.native_ability_actions_accepted,
            "native_action_exceptions": self.native_action_exceptions,
            "native_deployment_mask_probes_attempted": (
                self.native_deployment_mask_probes_attempted
            ),
            "native_deployment_mask_probes_responded": (
                self.native_deployment_mask_probes_responded
            ),
            "native_deployment_mask_probe_exceptions": (
                self.native_deployment_mask_probe_exceptions
            ),
            "true_attempted_acceptance_rate": (
                self.native_actions_accepted / self.native_actions_attempted
                if self.native_actions_attempted else None
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "latest_state": deepcopy(self.latest_state),
            "reset_history": list(deepcopy(self.reset_history)),
            "recent_action_history": list(deepcopy(self.action_history)),
            "recent_trace_history": list(deepcopy(self.trace_history)),
            "native_action_metrics": self.metrics(),
            "first_rejection": deepcopy(self.first_rejection),
        }


@dataclass(slots=True)
class StagedEpisode:
    battle_tag: str
    states: tuple[TickState, ...]
    metadata: dict[str, Any]
    deployment_mask_payloads: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


class StagedTickSink:
    """Hold one successful episode until all postconditions are checked."""

    def __init__(self) -> None:
        self.episode: StagedEpisode | None = None
        self._deployment_mask_metadata: dict[str, Any] | None = None
        self._deployment_mask_payloads: dict[str, dict[str, Any]] = {}

    def stage_deployment_masks(
        self, capture: NativeDeploymentMaskCapture
    ) -> None:
        if self.episode is not None:
            raise RuntimeError("deployment masks must be staged before episode")
        if self._deployment_mask_metadata is not None:
            raise RuntimeError("deployment masks may be staged only once")
        self._deployment_mask_metadata = capture.metadata(require_complete=True)
        self._deployment_mask_payloads = capture.payloads

    def append(
        self,
        battle_tag: str,
        states: Iterable[TickState],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.episode is not None:
            raise RuntimeError("staged sink accepts exactly one episode")
        materialized = tuple(states)
        require_consecutive(materialized)
        if not materialized:
            raise ValueError("successful native replay produced no Tick states")
        normalized_metadata = {
            **dict(metadata),
            "tick_start": materialized[0].tick,
            "tick_stop": materialized[-1].tick + 1,
        }
        if self._deployment_mask_metadata is not None:
            if (
                normalized_metadata.get(EPISODE_METADATA_KEY)
                != self._deployment_mask_metadata
            ):
                raise RuntimeError(
                    "episode deployment-mask metadata differs from staged capture"
                )
        self.episode = StagedEpisode(
            battle_tag=battle_tag,
            states=materialized,
            metadata=normalized_metadata,
            deployment_mask_payloads=dict(self._deployment_mask_payloads),
        )
        return {
            "battle_tag": battle_tag,
            "ticks": len(materialized),
            "tick_start": materialized[0].tick,
            "tick_stop": materialized[-1].tick + 1,
            "staged_not_committed": True,
        }


class StoredFrameRegistry:
    """Detect/reuse a checksummed orphan frame after a hard-process crash."""

    def __init__(
        self,
        root: Path,
        deployment_mask_store: DeploymentMaskStore | None = None,
    ) -> None:
        self.root = root.resolve()
        self.deployment_mask_store = (
            deployment_mask_store
            if deployment_mask_store is not None
            else DeploymentMaskStore(self.root)
        )
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def _register(self, entry: Mapping[str, Any]) -> None:
        tag = str(entry["battle_tag"])
        old = self._entries.get(tag)
        if old is not None and (
            str(old["payload_sha256"]) != str(entry["payload_sha256"])
            or str(old["shard"]) != str(entry["shard"])
            or int(old["offset"]) != int(entry["offset"])
        ):
            raise RuntimeError(f"duplicate immutable Tick frames for {tag}")
        self._entries[tag] = dict(entry)

    def _load_existing(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for index_path in sorted(self.root.glob("*.index.jsonl")):
            shard = index_path.name.removesuffix(".index.jsonl")
            with index_path.open("r", encoding="utf-8-sig") as source:
                for line in source:
                    if line.strip():
                        self._register({**json.loads(line), "shard": shard})
        for partial in sorted(self.root.glob("*.crts.partial")):
            shard = partial.name.removesuffix(".crts.partial")
            for entry in _scan_frames(partial, truncate_invalid_tail=True):
                self._register({**entry, "shard": shard})

    def commit_or_reuse(
        self, sink: WorkerShardSink, episode: StagedEpisode
    ) -> dict[str, Any]:
        with self._lock:
            mask_metadata = episode.metadata.get(EPISODE_METADATA_KEY)
            if episode.deployment_mask_payloads:
                self.deployment_mask_store.publish_many(
                    episode.deployment_mask_payloads
                )
                self.deployment_mask_store.verify_episode_metadata(
                    episode.metadata, allow_cached=True
                )
            elif mask_metadata is not None:
                raise RuntimeError(
                    "episode references deployment masks without staged payloads"
                )
            existing = self._entries.get(episode.battle_tag)
            if existing is not None:
                blob, _ = encode_episode(
                    episode.states,
                    {**episode.metadata, "battle_tag": episode.battle_tag},
                    anchor_interval=sink.anchor_interval,
                    compression_level=sink.compression_level,
                )
                actual = hashlib.sha256(blob).hexdigest()
                if actual != str(existing["payload_sha256"]):
                    raise RuntimeError(
                        "resume Tick payload diverged for " + episode.battle_tag
                    )
                return {**existing, "resume_reused_existing_frame": True}
            # The existence check, append, and registry mutation form one
            # critical section.  SQLite normally prevents two executions of
            # the same task; this also fences the rare lease-expiry race.
            entry = sink.append(
                episode.battle_tag, episode.states, episode.metadata
            )
            entry["resume_reused_existing_frame"] = False
            self._register(entry)
            return entry


def _verify_plan(
    task: NativeDatasetTask,
    source: Mapping[str, Any],
    *,
    native_ingest_contract: Any | None = None,
) -> BattlePlan:
    if str(source.get("battle_tag") or "") != task.battle_tag:
        raise RuntimeError("source battle_tag changed")
    if int(source.get("schema_version") or 0) != task.source_schema_version:
        raise RuntimeError("source schema version changed")
    crowns = (int(source["team_crowns"]), int(source["opponent_crowns"]))
    if any(value < 0 or value > 3 for value in crowns):
        raise ValueError(f"invalid source terminal crowns: {crowns}")
    plan = compile_battle(
        source,
        terminal_crowns=crowns,
        native_ingest_contract=native_ingest_contract,
    )
    if not plan.native_replay_ready:
        raise RuntimeError(f"compiled plan is not native ready: {plan.replay_tier}")
    if plan.source_schema_version not in {3, 5}:
        raise RuntimeError("compiled plan is not schema 3 or 5")
    if plan.source_schema_version == 5:
        if (
            plan.authoritative_contract_provenance
            != "schema5_authoritative_native_contract_verified"
            or plan.numeric_game_mode_id is None
            or plan.native_execution_game_mode_id is None
            or plan.battle_index is None
            or any(
                side.tower_troop_level is None
                or side.king_tower_level is None
                or side.final_tower_hp is None
                for side in plan.sides
            )
        ):
            raise RuntimeError("compiled schema5 authoritative metadata is incomplete")
    if plan.coordinate_provenance != COORDINATE_PROVENANCE:
        raise RuntimeError(
            f"coordinate provenance changed: {plan.coordinate_provenance}"
        )
    coordinate = asdict(plan.coordinate_audit)
    if (
        int(coordinate["raw_data_i_events"]) != task.deployment_actions
        or int(coordinate["legacy_xy_fallback_events"]) != 0
    ):
        raise RuntimeError("compiled coordinate audit differs from candidate")
    if len(plan.actions) != task.deployment_actions:
        raise RuntimeError("compiled deployment count differs from candidate")
    if len(plan.ability_events) != task.ability_events_observed:
        raise RuntimeError("compiled ability event count differs from candidate")
    if plan.ability_log_tier != task.ability_log_tier:
        raise RuntimeError("compiled ability tier differs from candidate")
    missing = sum(side.missing_ability_event_count for side in plan.sides)
    if missing:
        raise RuntimeError(f"compiled source is missing {missing} ability Ticks")
    return plan


def _failure_class(
    result: NativeReplayResult | None,
    error: Exception | None,
    stage: str,
) -> str:
    if stage == "source_sha_verification":
        return "source_sha_mismatch"
    if isinstance(error, NativeSeedSearchError):
        return "native_seed_search_exhausted"
    if error is not None:
        if stage == "immutable_tick_store_commit":
            return "infrastructure_tick_store_commit_failed"
        if stage == "tick_store_postcondition":
            return "infrastructure_tick_store_postcondition_failed"
        if stage == "native_teacher_forced_replay":
            return "infrastructure_native_replay_exception"
        if stage == "compile_and_provenance_validation":
            return "source_contract_or_compile_error"
        return "infrastructure_exception"
    assert result is not None
    failure = str(result.failure or "unknown")
    if "ability_branch_required" in failure:
        return "ability_branch_required"
    if "ability_no_legal" in failure:
        return "ability_entity_missing"
    if "native_rejected" in failure:
        return "native_action_rejected"
    if failure.startswith("native_terminal_before_"):
        return "native_terminal_before_source_event"
    if failure.startswith("hand_mismatch_event_"):
        return "source_hand_sequence_mismatch"
    if failure.startswith((
        "source_ability_ticks_missing_count_",
        "source_tick_",
        "execution_tick_",
        "native_deployment_mask_capture_incomplete_slots_",
        "derived_deployment_mask_rejected_source_event_",
    )):
        return "source_plan_contract_mismatch"
    if failure.startswith("native_seed_search_layout_revalidation_failed"):
        return "infrastructure_seed_layout_revalidation_failed"
    if failure.startswith((
        "native_action_count_mismatch_",
        "native_tick_mismatch_",
        "observation_tick_",
    )):
        return "infrastructure_native_protocol_mismatch"
    if "tick_store_write" in failure:
        return "tick_store_staging_failed"
    if failure.startswith(NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK):
        return NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK
    return "unclassified_native_failure"


def _failure_domain(failure_class: str | None) -> str | None:
    if failure_class is None:
        return None
    if failure_class in {
        "native_seed_search_exhausted",
        "ability_branch_required",
        "ability_entity_missing",
        "native_action_rejected",
        "native_terminal_before_source_event",
        NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK,
    }:
        return "semantic"
    if failure_class == "source_sha_mismatch":
        return "infrastructure"
    if failure_class in {
        "source_contract_or_compile_error",
        "source_hand_sequence_mismatch",
        "source_plan_contract_mismatch",
    }:
        return "source_integrity"
    return "infrastructure"


@dataclass(slots=True)
class TaskExecution:
    record: dict[str, Any]
    diagnostic: dict[str, Any] | None


def execute_task(
    env: Any,
    task: NativeDatasetTask,
    template: Mapping[str, Any],
    sink: WorkerShardSink,
    registry: StoredFrameRegistry,
    *,
    worker_id: str,
    port: int,
    attempt: int,
    seed: int = DEFAULT_NATIVE_SEED,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    trace_batch_steps: int = 64,
    commit_guard: Callable[[], bool] | None = None,
    native_ingest_contract: Any | None = None,
) -> TaskExecution:
    """Execute one source and commit its Tick frame only after full success."""
    recorder = RecordingCountingEnv(env)
    plan: BattlePlan | None = None
    result: NativeReplayResult | None = None
    staged = StagedTickSink()
    error: Exception | None = None
    error_traceback: str | None = None
    stage = "source_sha_verification"
    source_sha_verified = False
    store_entry: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        source_path = Path(task.source_path).resolve(strict=True)
        actual_sha = sha256_file(source_path)
        if actual_sha != task.source_sha256:
            raise RuntimeError(
                f"source SHA changed: {actual_sha} != {task.source_sha256}"
            )
        source_sha_verified = True
        stage = "compile_and_provenance_validation"
        source = load_json(source_path)
        plan = _verify_plan(
            task,
            source,
            native_ingest_contract=native_ingest_contract,
        )
        stage = "native_teacher_forced_replay"
        result = execute_plan(
            recorder,
            plan,
            template,
            None,
            seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            capture_decisions=False,
            ability_branch_choices=None,
            tick_sink=staged,
            tick_store_metadata={
                "source_path": str(source_path),
                "source_sha256": actual_sha,
                "source_schema_version": plan.source_schema_version,
                "selection_index": task.selection_index,
                "selection_digest": task.selection_digest,
                "eligibility_tier": task.eligibility_tier,
                "exact_tick_ability_events": task.ability_events_observed,
                "every_native_tick_present": True,
                "action_execution_tick_offset": (
                    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
                ),
                "native_teacher_forced_profile": native_teacher_forced_profile(
                    ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
                ),
            },
            trace_batch_steps=trace_batch_steps,
            capture_deployment_masks=True,
            action_execution_tick_offset=(
                ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
            ),
        )
        if not result.teacher_forced_success:
            stage = "first_native_difference"
        else:
            stage = "tick_store_postcondition"
            if staged.episode is None:
                raise RuntimeError("successful replay did not stage a Tick episode")
            if len(staged.episode.states) != result.tick_trace_complete_frames:
                raise RuntimeError(
                    "successful replay Tick count differs from complete trace frames"
                )
            if (
                recorder.native_actions_attempted != result.source_actions
                or recorder.native_actions_accepted != result.source_actions
            ):
                raise RuntimeError(
                    "successful replay action counters differ from source actions"
                )
            if (
                recorder.native_deployment_mask_probes_attempted
                != result.deployment_mask_probe_rpc_count
                or recorder.native_deployment_mask_probes_responded
                != result.deployment_mask_probe_rpc_count
                or recorder.native_deployment_mask_probe_exceptions != 0
                or result.deployment_mask_base_probe_rpc_count != 16
                or result.deployment_mask_slots_captured != 16
                or not result.deployment_mask_capture_complete
                or result.deployment_mask_label_checks
                != result.source_deploy_actions
                or result.deployment_mask_label_rejections != 0
            ):
                raise RuntimeError(
                    "successful replay deployment-mask base/dynamic accounting is open"
                )
            stage = "immutable_tick_store_commit"
            if commit_guard is not None and not commit_guard():
                raise RuntimeError(
                    "native task lease ownership was lost before Tick commit"
                )
            store_entry = registry.commit_or_reuse(sink, staged.episode)
    except Exception as caught:
        error = caught
        error_traceback = traceback.format_exc()

    success = bool(
        error is None
        and result is not None
        and result.teacher_forced_success
        and store_entry is not None
    )
    metrics = recorder.metrics()
    if success:
        failure_class = failure = None
        first_difference = None
    else:
        failure_class = _failure_class(result, error, stage)
        failure = (
            f"{type(error).__name__}: {error}"
            if error is not None else str(result.failure if result else "unknown")
        )
        first_difference = {
            "stage": stage,
            "failure_class": failure_class,
            "failure": failure,
            "first_native_rejection": deepcopy(recorder.first_rejection),
            "logic_freeze": (
                None if result is None else result.logic_freeze_diagnostic
            ),
        }
    failure_domain = _failure_domain(failure_class)
    coordinate_provenance = (
        None if plan is None else plan.coordinate_provenance
    )
    coordinate_audit = (
        None if plan is None else asdict(plan.coordinate_audit)
    )
    record: dict[str, Any] = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "completed_utc": utc_now(),
        "battle_tag": task.battle_tag,
        "selection_index": task.selection_index,
        "selection_digest": task.selection_digest,
        "worker_id": worker_id,
        "port": int(port),
        "attempt": int(attempt),
        "source_path": task.source_path,
        "source_sha256": task.source_sha256,
        "source_sha_verified": source_sha_verified,
        "source_schema_version": task.source_schema_version,
        "source_json_copied": False,
        "teacher_forced_success": success,
        "failure_class": failure_class,
        "failure_domain": failure_domain,
        "failure": failure,
        "first_difference": first_difference,
        "planned_deploy_actions": task.deployment_actions,
        "planned_ability_actions": task.ability_events_observed,
        "planned_actions": task.deployment_actions + task.ability_events_observed,
        **metrics,
        "native_teacher_forced_profile": native_teacher_forced_profile(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        "action_tick_provenance": action_tick_provenance(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        "coordinate_provenance": coordinate_provenance,
        "coordinate_audit": coordinate_audit,
        "numeric_game_mode_id": (
            None if plan is None else plan.numeric_game_mode_id
        ),
        "numeric_game_mode_provenance": (
            None if plan is None else plan.numeric_game_mode_provenance
        ),
        "native_execution_game_mode_id": (
            None if plan is None else plan.native_execution_game_mode_id
        ),
        "native_execution_game_mode_provenance": (
            None
            if plan is None
            else plan.native_execution_game_mode_provenance
        ),
        "king_tower_levels": (
            None
            if plan is None
            else [side.king_tower_level for side in plan.sides]
        ),
        "king_tower_level_provenance": (
            None
            if plan is None
            else [side.king_tower_level_provenance for side in plan.sides]
        ),
        "battle_index": None if plan is None else plan.battle_index,
        "battle_index_provenance": (
            None if plan is None else plan.battle_index_provenance
        ),
        "authoritative_contract_sha256": (
            None if plan is None else plan.authoritative_contract_sha256
        ),
        "ability_log_tier": task.ability_log_tier,
        "ability_branch_policy": "branch_required fails closed; no guessed entity",
        "ability_resolution_counts": (
            {} if result is None else result.ability_resolution_counts
        ),
        "chosen_seed": None if result is None else result.chosen_seed,
        "seeds_tested": 0 if result is None else result.seeds_tested,
        "seed_search_cache_hit": (
            False if result is None else result.seed_search_cache_hit
        ),
        "source_seed_recovered": (
            False if result is None else result.source_seed_recovered
        ),
        "native_ticks_advanced": (
            0 if result is None else result.native_ticks_advanced
        ),
        "collected_tick_state_count": (
            0 if result is None else len(result.collected_tick_states)
        ),
        "tick_trace_complete_frames": (
            0 if result is None else result.tick_trace_complete_frames
        ),
        "tick_trace_incomplete_terminal_frames": (
            0 if result is None else result.tick_trace_incomplete_terminal_frames
        ),
        "tick_trace_incomplete_nonterminal_freeze_frames": (
            0 if result is None
            else result.tick_trace_incomplete_nonterminal_freeze_frames
        ),
        "deployment_mask_probe_seconds": (
            0.0 if result is None else result.deployment_mask_probe_seconds
        ),
        "deployment_mask_probe_rpc_count": (
            0 if result is None else result.deployment_mask_probe_rpc_count
        ),
        "deployment_mask_slots_captured": (
            0 if result is None else result.deployment_mask_slots_captured
        ),
        "deployment_mask_base_probe_rpc_count": (
            0
            if result is None
            else result.deployment_mask_base_probe_rpc_count
        ),
        "deployment_mask_dynamic_label_probe_rpc_count": (
            0
            if result is None
            else result.deployment_mask_dynamic_label_probe_rpc_count
        ),
        "deployment_mask_capture_complete": (
            False if result is None else result.deployment_mask_capture_complete
        ),
        "deployment_mask_metadata": (
            None if result is None else result.deployment_mask_metadata
        ),
        "deployment_mask_label_checks": (
            0 if result is None else result.deployment_mask_label_checks
        ),
        "deployment_mask_label_rejections": (
            0 if result is None else result.deployment_mask_label_rejections
        ),
        "deployment_mask_first_label_rejection": (
            None
            if result is None
            else result.deployment_mask_first_label_rejection
        ),
        "logic_freeze_diagnostic": (
            None if result is None else result.logic_freeze_diagnostic
        ),
        "terminal_diagnostic_status": (
            "not_reached" if result is None else result.terminal_diagnostic_status
        ),
        "terminal_tower_hp_diagnostic_status": (
            "not_reached"
            if result is None
            else result.terminal_tower_hp_diagnostic_status
        ),
        "terminal_tower_hp_validated": (
            False if result is None else result.terminal_tower_hp_validated
        ),
        "terminal_tower_hp_match": (
            None if result is None else result.terminal_tower_hp_match
        ),
        "source_final_tower_hp": (
            None if result is None else result.source_final_tower_hp
        ),
        "observed_final_tower_hp": (
            None if result is None else result.observed_final_tower_hp
        ),
        "tick_store_entry": store_entry,
        "wall_seconds": time.perf_counter() - started,
    }
    if success:
        return TaskExecution(record=record, diagnostic=None)
    diagnostic = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "battle_tag": task.battle_tag,
        "source_identity": {
            "path": task.source_path,
            "sha256": task.source_sha256,
            "sha_verified": source_sha_verified,
            "source_json_copied": False,
        },
        "task": task.json(),
        "failure_class": failure_class,
        "failure_domain": failure_domain,
        "failure": failure,
        "first_difference": first_difference,
        "native_action_metrics": metrics,
        "native_teacher_forced_profile": record[
            "native_teacher_forced_profile"
        ],
        "coordinate_provenance": coordinate_provenance,
        "coordinate_audit": coordinate_audit,
        "plan": None if plan is None else plan.json(),
        "native_result": None if result is None else result.json(),
        "native_boundary_snapshot": recorder.snapshot(),
        "exception_traceback": error_traceback,
    }
    return TaskExecution(record=record, diagnostic=diagnostic)


class LeaseHeartbeat(AbstractContextManager["LeaseHeartbeat"]):
    def __init__(
        self,
        queue_path: Path,
        worker_id: str,
        battle_tag: str,
        *,
        lease_seconds: float,
    ) -> None:
        self.queue_path = queue_path
        self.worker_id = worker_id
        self.battle_tag = battle_tag
        self.lease_seconds = lease_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        interval = max(5.0, min(60.0, self.lease_seconds / 3.0))
        while not self.stop_event.wait(interval):
            try:
                with TickStoreWorkQueue(self.queue_path) as queue:
                    queue.heartbeat(
                        self.worker_id,
                        [self.battle_tag],
                        lease_seconds=self.lease_seconds,
                    )
            except Exception:
                # The owning worker's final queue mutation remains the
                # authority.  A heartbeat failure must not mask native output.
                pass

    def __enter__(self) -> "LeaseHeartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def _safe_tag(tag: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in tag)


def _result_path(output_root: Path, battle_tag: str) -> Path:
    return output_root / "results" / f"{_safe_tag(battle_tag)}.json"


def _diagnostic_path(
    output_root: Path, battle_tag: str, attempt: int
) -> Path:
    return (
        output_root / "diagnostics"
        / f"{_safe_tag(battle_tag)}.attempt-{attempt:03d}.json"
    )


def should_retry_failure(
    record: Mapping[str, Any], attempt: int, *, maximum_attempts: int = 3
) -> bool:
    return bool(
        record.get("teacher_forced_success") is not True
        and record.get("failure_domain") == "infrastructure"
        and int(attempt) < int(maximum_attempts)
    )


def _lease_is_owned(
    queue: TickStoreWorkQueue, worker_id: str, battle_tag: str
) -> bool:
    row = queue.connection.execute(
        """
        SELECT status, lease_owner, lease_until FROM tasks
        WHERE battle_tag=?
        """,
        (battle_tag,),
    ).fetchone()
    return bool(
        row is not None
        and row["status"] == "leased"
        and row["lease_owner"] == worker_id
        and float(row["lease_until"] or 0.0) > time.time()
    )


def recover_unmanifested_final_shards(
    root: Path,
    worker_id: str,
    *,
    anchor_interval: int = 256,
    compression_level: int = 1,
) -> int:
    """Repair the crash window after .crts rename but before manifest write."""
    recovered = 0
    root.mkdir(parents=True, exist_ok=True)
    for data_path in sorted(root.glob(f"{worker_id}-*.crts")):
        stem = data_path.name.removesuffix(".crts")
        manifest_path = root / f"{stem}.manifest.json"
        if manifest_path.exists():
            continue
        index_path = root / f"{stem}.index.jsonl"
        if not index_path.exists():
            raise RuntimeError(
                f"final shard has no recoverable index: {data_path}"
            )
        entries = _scan_frames(data_path, truncate_invalid_tail=False)
        index_entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if entries != index_entries:
            raise RuntimeError(
                f"final shard/index differ during recovery: {stem}"
            )
        manifest = {
            "schema_version": 1,
            "kind": SHARD_KIND,
            "created_utc": utc_now(),
            "name": stem,
            "data_file": data_path.name,
            "index_file": index_path.name,
            "episode_count": len(entries),
            "tick_count": sum(int(entry["ticks"]) for entry in entries),
            "anchor_interval": anchor_interval,
            "compression": f"zlib-level-{compression_level}",
            "data_sha256": sha256_file(data_path),
            "index_sha256": sha256_file(index_path),
            "bytes": data_path.stat().st_size,
            "recovered_after_finalize_crash": True,
        }
        atomic_json(manifest_path, manifest)
        recovered += 1
    return recovered


def worker_loop(
    *,
    worker_index: int,
    port: int,
    queue_path: Path,
    output_root: Path,
    template_path: Path,
    registry: StoredFrameRegistry,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    episodes_per_shard: int,
    lease_seconds: float,
    native_contract_path: Path | None = None,
) -> dict[str, Any]:
    worker_id = f"native-worker-port-{port}"
    completed = successes = failures = stored_ticks = 0
    worker_error: str | None = None
    started = time.perf_counter()
    sink: WorkerShardSink | None = None
    manifests: list[dict[str, Any]] = []
    recovered_final_shards = 0
    try:
        template = load_template(template_path)
        native_ingest_contract = (
            None
            if native_contract_path is None
            else load_native_ingest_contract(native_contract_path)
        )
        recovered_final_shards = recover_unmanifested_final_shards(
            output_root / "shards", worker_id
        )
        sink = WorkerShardSink(
            output_root / "shards",
            worker_id,
            episodes_per_shard=episodes_per_shard,
            anchor_interval=256,
            compression_level=1,
        )
        with NativeRoyaleEnv(port=port, timeout=60.0) as env:
            with TickStoreWorkQueue(queue_path) as queue:
                while True:
                    claimed = queue.claim(
                        worker_id,
                        limit=1,
                        lease_seconds=lease_seconds,
                        maximum_attempts=100,
                    )
                    if not claimed:
                        break
                    raw_task = claimed[0]
                    task = NativeDatasetTask.from_claim(raw_task)
                    with LeaseHeartbeat(
                        queue_path,
                        worker_id,
                        task.battle_tag,
                        lease_seconds=lease_seconds,
                    ):
                        execution = execute_task(
                            env,
                            task,
                            template,
                            sink,
                            registry,
                            worker_id=worker_id,
                            port=port,
                            attempt=raw_task.attempts,
                            seed=seed,
                            maximum_seeds_to_test=maximum_seeds_to_test,
                            trace_batch_steps=trace_batch_steps,
                            native_ingest_contract=native_ingest_contract,
                            commit_guard=lambda: _lease_is_owned(
                                queue, worker_id, task.battle_tag
                            ),
                        )
                        record = execution.record
                        retry = should_retry_failure(
                            record, raw_task.attempts, maximum_attempts=3
                        )
                        record["retry_scheduled"] = retry
                        record["final_attempt"] = not retry
                        diagnostic_path = None
                        if execution.diagnostic is not None:
                            diagnostic_path = _diagnostic_path(
                                output_root, task.battle_tag, raw_task.attempts
                            )
                            atomic_json(diagnostic_path, execution.diagnostic)
                        record["diagnostic_path"] = (
                            None if diagnostic_path is None
                            else str(diagnostic_path.resolve())
                        )
                        atomic_json(
                            _result_path(output_root, task.battle_tag), record
                        )
                        if record["teacher_forced_success"]:
                            entry = record["tick_store_entry"]
                            queue.complete(
                                worker_id,
                                task.battle_tag,
                                output_shard=str(entry["shard"]),
                                frame_offset=int(entry["offset"]),
                                frame_size=int(entry["frame_size"]),
                                episode_sha256=str(entry["payload_sha256"]),
                            )
                            successes += 1
                            stored_ticks += int(entry["ticks"])
                        else:
                            queue.fail(
                                worker_id,
                                task.battle_tag,
                                str(record["failure"]),
                                retry=retry,
                            )
                            if not retry:
                                failures += 1
                    completed += 1
    except Exception as error:
        worker_error = f"{type(error).__name__}: {error}"
    finally:
        if sink is not None:
            try:
                manifests = sink.finalize()
            except Exception as error:
                if worker_error is None:
                    worker_error = f"shard_finalize_{type(error).__name__}: {error}"
    report = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": "expert_authoritative_native_tick_worker_report_v1",
        "worker_index": worker_index,
        "worker_id": worker_id,
        "port": int(port),
        "completed": completed,
        "successes": successes,
        "failures": failures,
        "stored_ticks": stored_ticks,
        "wall_seconds": time.perf_counter() - started,
        "worker_error": worker_error,
        "recovered_final_shards": recovered_final_shards,
        "newly_finalized_shards": manifests,
    }
    atomic_json(output_root / "workers" / f"{worker_id}.json", report)
    return report


def load_results(
    output_root: Path,
    tasks: Sequence[NativeDatasetTask],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    expected = {task.battle_tag for task in tasks}
    found: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "results").glob("*.json")):
        value = load_json(path)
        if value.get("kind") != RESULT_KIND:
            continue
        tag = str(value["battle_tag"])
        if tag in found:
            raise RuntimeError(f"duplicate result record for {tag}")
        found[tag] = value
    ordered = [found[task.battle_tag] for task in tasks if task.battle_tag in found]
    return ordered, sorted(expected - found.keys()), sorted(found.keys() - expected)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def summarize_results(
    tasks: Sequence[NativeDatasetTask],
    results: Sequence[Mapping[str, Any]],
    *,
    queue_counts: Mapping[str, int],
    worker_reports: Sequence[Mapping[str, Any]],
    wall_seconds: float,
    missing_tags: Sequence[str],
    unexpected_tags: Sequence[str],
) -> dict[str, Any]:
    successes = [row for row in results if row.get("teacher_forced_success")]
    failures = [row for row in results if not row.get("teacher_forced_success")]
    attempted = sum(int(row.get("native_actions_attempted") or 0) for row in results)
    accepted = sum(int(row.get("native_actions_accepted") or 0) for row in results)
    responded = sum(int(row.get("native_actions_responded") or 0) for row in results)
    rejected = sum(int(row.get("native_actions_rejected") or 0) for row in results)
    no_response = sum(
        int(row.get("native_actions_no_response") or 0) for row in results
    )
    response_excess = sum(
        int(row.get("native_action_response_excess") or 0) for row in results
    )
    action_exceptions = sum(
        int(row.get("native_action_exceptions") or 0) for row in results
    )
    deploy_attempted = sum(
        int(row.get("native_deploy_actions_attempted") or 0) for row in results
    )
    deploy_accepted = sum(
        int(row.get("native_deploy_actions_accepted") or 0) for row in results
    )
    ability_attempted = sum(
        int(row.get("native_ability_actions_attempted") or 0) for row in results
    )
    ability_accepted = sum(
        int(row.get("native_ability_actions_accepted") or 0) for row in results
    )
    stored_ticks = sum(
        int(row.get("tick_store_entry", {}).get("ticks", 0))
        for row in successes
    )
    mask_probe_rpcs = sum(
        int(row.get("native_deployment_mask_probes_attempted") or 0)
        for row in results
    )
    mask_probe_responses = sum(
        int(row.get("native_deployment_mask_probes_responded") or 0)
        for row in results
    )
    mask_probe_exceptions = sum(
        int(row.get("native_deployment_mask_probe_exceptions") or 0)
        for row in results
    )
    successful_mask_probe_rpcs = sum(
        int(row.get("deployment_mask_probe_rpc_count") or 0)
        for row in successes
    )
    dynamic_label_probe_rpcs = sum(
        int(row.get("deployment_mask_dynamic_label_probe_rpc_count") or 0)
        for row in results
    )
    successful_mask_integrity = all(
        row.get("deployment_mask_capture_complete") is True
        and int(row.get("deployment_mask_slots_captured") or 0) == 16
        and int(row.get("deployment_mask_base_probe_rpc_count") or 0) == 16
        and int(row.get("deployment_mask_probe_rpc_count") or 0)
        == 16 + int(
            row.get("deployment_mask_dynamic_label_probe_rpc_count") or 0
        )
        and isinstance(row.get("deployment_mask_metadata"), Mapping)
        and int(row.get("deployment_mask_label_checks") or 0)
        == int(row.get("planned_deploy_actions") or 0)
        and int(row.get("deployment_mask_label_rejections") or 0) == 0
        and int(row.get("native_deployment_mask_probes_attempted") or 0)
        == int(row.get("deployment_mask_probe_rpc_count") or 0)
        and int(row.get("native_deployment_mask_probes_responded") or 0)
        == int(row.get("deployment_mask_probe_rpc_count") or 0)
        and int(row.get("native_deployment_mask_probe_exceptions") or 0) == 0
        for row in successes
    )
    failure_classes = Counter(
        str(row.get("failure_class") or "unknown") for row in failures
    )
    failure_domains = Counter(
        str(row.get("failure_domain") or "unknown") for row in failures
    )
    terminal = Counter(
        str(row.get("terminal_diagnostic_status") or "unknown")
        for row in results
    )
    coordinates = Counter(
        str(row.get("coordinate_provenance") or "unknown") for row in results
    )
    profiles = Counter(
        json.dumps(row.get("native_teacher_forced_profile"), sort_keys=True)
        for row in results
    )
    worker_errors = [
        str(row["worker_error"]) for row in worker_reports
        if row.get("worker_error")
    ]
    infrastructure_complete = bool(
        len(results) == len(tasks)
        and not missing_tags
        and not unexpected_tags
        and not worker_errors
        and int(queue_counts.get("pending", 0)) == 0
        and int(queue_counts.get("leased", 0)) == 0
        and int(failure_domains.get("infrastructure", 0)) == 0
        and int(failure_domains.get("source_integrity", 0)) == 0
    )
    source_integrity = all(bool(row.get("source_sha_verified")) for row in results)
    profile_integrity = all(
        row.get("native_teacher_forced_profile")
        == native_teacher_forced_profile(
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        )
        for row in results
    )
    coordinate_integrity = all(
        row.get("coordinate_provenance") == COORDINATE_PROVENANCE
        for row in results
        if row.get("source_sha_verified")
    )
    action_accounting_closed = bool(
        attempted == accepted + rejected + no_response
        and response_excess == 0
    )
    return {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "kind": "expert_authoritative_native_tick_summary_v1",
        "created_utc": utc_now(),
        "selected_battles": len(tasks),
        "processed_battles": len(results),
        "teacher_forced_successes": len(successes),
        "teacher_forced_failures": len(failures),
        "ability_positive_selected": sum(task.ability_positive for task in tasks),
        "ability_zero_selected": sum(not task.ability_positive for task in tasks),
        "planned_actions": sum(
            task.deployment_actions + task.ability_events_observed for task in tasks
        ),
        "native_actions_attempted": attempted,
        "native_actions_responded": responded,
        "native_actions_accepted": accepted,
        "native_actions_rejected": rejected,
        "native_actions_no_response": no_response,
        "native_action_response_excess": response_excess,
        "native_action_exceptions": action_exceptions,
        "native_action_accounting_closed": action_accounting_closed,
        "true_attempted_acceptance_rate": _ratio(accepted, attempted),
        "native_deploy_actions_attempted": deploy_attempted,
        "native_deploy_actions_accepted": deploy_accepted,
        "true_attempted_deploy_acceptance_rate": _ratio(
            deploy_accepted, deploy_attempted
        ),
        "native_ability_actions_attempted": ability_attempted,
        "native_ability_actions_accepted": ability_accepted,
        "true_attempted_ability_acceptance_rate": _ratio(
            ability_accepted, ability_attempted
        ),
        "stored_episodes": len(successes),
        "stored_ticks": stored_ticks,
        "native_deployment_mask_probe_rpcs": mask_probe_rpcs,
        "native_deployment_mask_probe_responses": mask_probe_responses,
        "native_deployment_mask_probe_exceptions": mask_probe_exceptions,
        "native_deployment_mask_dynamic_label_probe_rpcs": (
            dynamic_label_probe_rpcs
        ),
        "native_deployment_mask_probe_rpcs_per_success": _ratio(
            successful_mask_probe_rpcs, len(successes)
        ),
        "native_deployment_mask_integrity": successful_mask_integrity,
        "wall_seconds": wall_seconds,
        "stored_ticks_per_wall_second": _ratio(stored_ticks, wall_seconds),
        "processed_episodes_per_hour": _ratio(
            len(results) * 3600.0, wall_seconds
        ),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "failure_domain_counts": dict(sorted(failure_domains.items())),
        "terminal_diagnostic_counts": dict(sorted(terminal.items())),
        "coordinate_provenance_counts": dict(sorted(coordinates.items())),
        "native_profile_counts": dict(sorted(profiles.items())),
        "logic_freeze_battles": int(
            failure_classes[NATIVE_LOGIC_FROZEN_BEFORE_EXECUTION_TICK]
        ),
        "branch_required_battles": int(
            failure_classes["ability_branch_required"]
        ),
        "queue_counts": dict(queue_counts),
        "missing_result_tags": list(missing_tags),
        "unexpected_result_tags": list(unexpected_tags),
        "worker_errors": worker_errors,
        "worker_reports": list(worker_reports),
        "source_sha_integrity": source_integrity,
        "profile_integrity": profile_integrity,
        "coordinate_integrity": coordinate_integrity,
        "infrastructure_complete": infrastructure_complete,
        "publication_ready": bool(
            infrastructure_complete
            and source_integrity
            and profile_integrity
            and coordinate_integrity
            and action_accounting_closed
            and successful_mask_integrity
        ),
        "source_json_copied": False,
        "semantic_rejections_are_expected_subset_evidence": True,
    }


def _physical_frame_valid(output_root: Path, record: Mapping[str, Any]) -> bool:
    entry = record.get("tick_store_entry")
    if not isinstance(entry, Mapping):
        return False
    stem = str(entry.get("shard") or "")
    paths = [
        output_root / "shards" / f"{stem}.crts",
        output_root / "shards" / f"{stem}.crts.partial",
    ]
    path = next((item for item in paths if item.exists()), None)
    if path is None:
        return False
    offset = int(entry["offset"])
    with path.open("rb") as source:
        source.seek(offset)
        header = source.read(FRAME_HEADER.size)
        if len(header) != FRAME_HEADER.size:
            return False
        magic, payload_size, payload_crc, _tag_hash, ticks, _reserved = (
            FRAME_HEADER.unpack(header)
        )
        payload = source.read(payload_size)
    import zlib
    return bool(
        magic == FRAME_MAGIC
        and len(payload) == payload_size
        and zlib.crc32(payload) == payload_crc
        and hashlib.sha256(payload).hexdigest() == entry["payload_sha256"]
        and int(ticks) == int(entry["ticks"])
    )


def reconcile_result_files(output_root: Path, queue_path: Path) -> int:
    """Finish the DB mutation if a crash occurred after atomic result write."""
    reconciled = 0
    with TickStoreWorkQueue(queue_path) as queue:
        for path in sorted((output_root / "results").glob("*.json")):
            record = load_json(path)
            if record.get("kind") != RESULT_KIND:
                continue
            tag = str(record["battle_tag"])
            row = queue.connection.execute(
                "SELECT status FROM tasks WHERE battle_tag=?", (tag,)
            ).fetchone()
            if row is None or row["status"] in {"done", "failed"}:
                continue
            if record.get("teacher_forced_success"):
                if not _physical_frame_valid(output_root, record):
                    continue
                entry = record["tick_store_entry"]
                with queue.connection:
                    queue.connection.execute(
                        """
                        UPDATE tasks SET status='done', lease_owner=NULL,
                            lease_until=NULL, output_shard=?, frame_offset=?,
                            frame_size=?, episode_sha256=?, updated_at=?
                        WHERE battle_tag=? AND status IN ('pending','leased')
                        """,
                        (
                            str(entry["shard"]), int(entry["offset"]),
                            int(entry["frame_size"]), str(entry["payload_sha256"]),
                            time.time(), tag,
                        ),
                    )
                reconciled += 1
            else:
                if record.get("retry_scheduled") is True:
                    # The prior process died before returning this retryable
                    # attempt to pending.  The exclusive run lock lets the
                    # caller release the lease and execute the next attempt.
                    continue
                with queue.connection:
                    queue.connection.execute(
                        """
                        UPDATE tasks SET status='failed', lease_owner=NULL,
                            lease_until=NULL, last_error=?, updated_at=?
                        WHERE battle_tag=? AND status IN ('pending','leased')
                        """,
                        (str(record.get("failure") or "failed")[:4000], time.time(), tag),
                    )
                reconciled += 1
    return reconciled


def release_interrupted_leases(queue_path: Path) -> int:
    """Immediately reclaim leases from a dead prior process.

    ``run_generation`` holds the output-root OS lock while calling this.  A
    second live generator therefore cannot own these leases; waiting for the
    normal 900-second expiry would only make crash resume unnecessarily slow.
    """
    with TickStoreWorkQueue(queue_path) as queue:
        with queue.connection:
            cursor = queue.connection.execute(
                """
                UPDATE tasks SET status='pending', lease_owner=NULL,
                    lease_until=NULL, updated_at=?
                WHERE status='leased'
                """,
                (time.time(),),
            )
        return int(cursor.rowcount)


def requeue_failed_infrastructure(
    output_root: Path, queue_path: Path
) -> int:
    """Reopen only prior infrastructure failures on an explicit new run.

    Semantic and source-contract failures remain terminal.  Attempts reset so
    a repaired host, transport, or disk receives a fresh bounded retry budget.
    """
    requeued = 0
    with TickStoreWorkQueue(queue_path) as queue:
        for path in sorted((output_root / "results").glob("*.json")):
            record = load_json(path)
            if (
                record.get("kind") != RESULT_KIND
                or record.get("teacher_forced_success") is True
                or record.get("failure_domain") != "infrastructure"
            ):
                continue
            tag = str(record["battle_tag"])
            with queue.connection:
                cursor = queue.connection.execute(
                    """
                    UPDATE tasks SET status='pending', lease_owner=NULL,
                        lease_until=NULL, attempts=0, output_shard=NULL,
                        frame_offset=NULL, frame_size=NULL,
                        episode_sha256=NULL, last_error=NULL, updated_at=?
                    WHERE battle_tag=? AND status='failed'
                    """,
                    (time.time(), tag),
                )
            requeued += int(cursor.rowcount)
    return requeued


def verify_published_tick_store(root: Path) -> dict[str, int]:
    """Read-only validation of every data/index hash behind the store manifest."""
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "cr_native_tick_store_v1":
        raise RuntimeError("published Tick Store manifest kind changed")
    mask_contract = (manifest.get("metadata") or {}).get(
        "native_deployment_masks"
    )
    mask_store: DeploymentMaskStore | None = None
    referenced_masks: set[str] = set()
    if mask_contract is not None:
        if (
            not isinstance(mask_contract, Mapping)
            or mask_contract.get("required") is not True
        ):
            raise RuntimeError("published Tick Store mask contract is malformed")
        expected_path = f"{MASK_STORE_DIRECTORY}/manifest.json"
        if str(mask_contract.get("manifest") or "") != expected_path:
            raise RuntimeError("published Tick Store mask manifest path changed")
        mask_manifest_path = root / expected_path
        if not mask_manifest_path.is_file():
            raise RuntimeError("published Tick Store mask manifest is missing")
        if sha256_file(mask_manifest_path) != str(
            mask_contract.get("manifest_sha256") or ""
        ):
            raise RuntimeError("published Tick Store mask manifest hash changed")
        mask_store = DeploymentMaskStore(root, create=False)
        mask_store.verify_manifest()
    episodes = ticks = total_bytes = 0
    shard_names: set[str] = set()
    for shard in manifest.get("shards") or []:
        if not isinstance(shard, Mapping):
            raise RuntimeError("published Tick Store has malformed shard entry")
        name = str(shard.get("name") or "")
        if not name or name in shard_names:
            raise RuntimeError("published Tick Store has duplicate/empty shard name")
        shard_names.add(name)
        data = root / str(shard["data_file"])
        index = root / str(shard["index_file"])
        if not data.is_file() or not index.is_file():
            raise RuntimeError(f"published Tick Store shard is missing: {name}")
        if sha256_file(data) != str(shard["data_sha256"]):
            raise RuntimeError(f"published Tick Store data hash changed: {name}")
        if sha256_file(index) != str(shard["index_sha256"]):
            raise RuntimeError(f"published Tick Store index hash changed: {name}")
        entries = _scan_frames(data, truncate_invalid_tail=False)
        indexed = [
            json.loads(line)
            for line in index.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if entries != indexed:
            raise RuntimeError(f"published Tick Store index content changed: {name}")
        if len(entries) != int(shard["episode_count"]):
            raise RuntimeError(f"published Tick Store episode count changed: {name}")
        shard_ticks = sum(int(entry["ticks"]) for entry in entries)
        if shard_ticks != int(shard["tick_count"]):
            raise RuntimeError(f"published Tick Store Tick count changed: {name}")
        episodes += len(entries)
        ticks += shard_ticks
        total_bytes += data.stat().st_size
        if mask_store is not None:
            with data.open("rb") as handle:
                for entry in entries:
                    handle.seek(int(entry["offset"]))
                    raw_header = handle.read(FRAME_HEADER.size)
                    if len(raw_header) != FRAME_HEADER.size:
                        raise RuntimeError("Tick Store frame header disappeared")
                    _, payload_size, _, _, _, _ = FRAME_HEADER.unpack(raw_header)
                    payload = handle.read(payload_size)
                    reader = EpisodeReader(payload)
                    metadata = mask_store.verify_episode_metadata(
                        reader.metadata, allow_cached=True
                    )
                    referenced_masks.update(
                        str(item["content_sha256"])
                        for item in metadata["entries"]
                    )
                    referenced_masks.update(
                        str(variant["content_sha256"])
                        for item in metadata["entries"]
                        for variant in item["dynamic_label_variants"]
                    )
    if (
        episodes != int(manifest.get("episode_count", -1))
        or ticks != int(manifest.get("tick_count", -1))
        or total_bytes != int(manifest.get("total_bytes", -1))
    ):
        raise RuntimeError("published Tick Store global counts changed")
    return {
        "episodes": episodes,
        "ticks": ticks,
        "bytes": total_bytes,
        "deployment_mask_sidecars_referenced": len(referenced_masks),
    }


def completed_run_summary(
    output_root: Path,
    tasks: Sequence[NativeDatasetTask],
    queue_path: Path,
) -> dict[str, Any] | None:
    """Return an already-published run without reopening any shard writer."""
    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary.json"
    results_path = output_root / "results.jsonl"
    store_manifest_path = output_root / "shards" / "manifest.json"
    if not all(path.exists() for path in (
        manifest_path, summary_path, results_path, store_manifest_path,
    )):
        return None
    with TickStoreWorkQueue(queue_path) as queue:
        counts = queue.counts()
    if int(counts.get("pending", 0)) or int(counts.get("leased", 0)):
        return None
    results, missing, unexpected = load_results(output_root, tasks)
    if missing or unexpected or len(results) != len(tasks):
        return None
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    if manifest.get("kind") != "expert_authoritative_native_tick_dataset_manifest_v1":
        raise RuntimeError("published native dataset manifest kind changed")
    content = manifest.get("content")
    if not isinstance(content, Mapping):
        raise RuntimeError("published native dataset manifest has no content audit")
    expected_hashes = {
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "tick_store_manifest_sha256": sha256_file(store_manifest_path),
    }
    mask_manifest_path = (
        output_root / "shards" / MASK_STORE_DIRECTORY / "manifest.json"
    )
    if "deployment_mask_store_manifest_sha256" in content:
        if not mask_manifest_path.is_file():
            raise RuntimeError(
                "published native dataset deployment-mask manifest is missing"
            )
        expected_hashes["deployment_mask_store_manifest_sha256"] = (
            sha256_file(mask_manifest_path)
        )
    for key, expected in expected_hashes.items():
        if str(content.get(key) or "") != expected:
            raise RuntimeError(f"published native dataset {key} changed")
    physical = verify_published_tick_store(output_root / "shards")
    if int(summary.get("processed_battles", -1)) != len(tasks):
        raise RuntimeError("published summary task count changed")
    if (
        physical["episodes"] != int(summary.get("stored_episodes", -1))
        or physical["ticks"] != int(summary.get("stored_ticks", -1))
    ):
        raise RuntimeError("published summary and physical Tick Store differ")
    return {
        **summary,
        "resume_noop": True,
        "resume_reason": "all selected tasks already terminal and manifest verified",
    }


class RunLock(AbstractContextManager["RunLock"]):
    """OS-held non-blocking lock preventing two shard writers in one run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "RunLock":
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
            else:  # pragma: no cover - Windows is the production host
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError("another native dataset generator owns this run") from error
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


def run_generation(
    *,
    candidate_queue: Path,
    output_root: Path,
    template_path: Path,
    ports: Sequence[int],
    workers: int,
    limit: int | None = None,
    selection_seed: str = "authoritative-native-full-v1",
    deployment_zero_quota: int | None = None,
    ability_exact_quota: int | None = None,
    seed: int = DEFAULT_NATIVE_SEED,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    trace_batch_steps: int = 64,
    episodes_per_shard: int = 256,
    lease_seconds: float = 900.0,
    retry_infrastructure_failures: bool = True,
    native_contract_path: Path | None = None,
) -> dict[str, Any]:
    if workers <= 0 or workers > len(ports):
        raise ValueError("workers must be positive and no greater than port count")
    if len(set(ports[:workers])) != workers:
        raise ValueError("active native Worker ports must be unique")
    if not 1 <= trace_batch_steps <= 64:
        raise ValueError("trace_batch_steps must be in 1..64")
    if maximum_seeds_to_test <= 0 or episodes_per_shard <= 0:
        raise ValueError("seed limit and episodes_per_shard must be positive")
    if lease_seconds < 30.0:
        raise ValueError("lease_seconds must be at least 30")
    output_root = output_root.resolve()
    with RunLock(output_root / "run.lock"):
        tasks, selection_path, queue_path, contract = prepare_run(
            candidate_queue=candidate_queue,
            output_root=output_root,
            template_path=template_path,
            limit=limit,
            selection_seed=selection_seed,
            deployment_zero_quota=deployment_zero_quota,
            ability_exact_quota=ability_exact_quota,
            seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            trace_batch_steps=trace_batch_steps,
            episodes_per_shard=episodes_per_shard,
            native_contract_path=native_contract_path,
        )
        reconciled = reconcile_result_files(output_root, queue_path)
        released_leases = release_interrupted_leases(queue_path)
        infrastructure_requeued = (
            requeue_failed_infrastructure(output_root, queue_path)
            if retry_infrastructure_failures else 0
        )
        completed = completed_run_summary(output_root, tasks, queue_path)
        if completed is not None:
            return completed
        registry = StoredFrameRegistry(output_root / "shards")
        started = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    worker_loop,
                    worker_index=index,
                    port=int(ports[index]),
                    queue_path=queue_path,
                    output_root=output_root,
                    template_path=template_path.resolve(strict=True),
                    registry=registry,
                    seed=seed,
                    maximum_seeds_to_test=maximum_seeds_to_test,
                    trace_batch_steps=trace_batch_steps,
                    episodes_per_shard=episodes_per_shard,
                    lease_seconds=lease_seconds,
                    native_contract_path=native_contract_path,
                )
                for index in range(workers)
            ]
            worker_reports = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        with TickStoreWorkQueue(queue_path) as queue:
            queue_counts = queue.counts()
        results, missing, unexpected = load_results(output_root, tasks)
        atomic_jsonl(output_root / "results.jsonl", results)
        summary = summarize_results(
            tasks,
            results,
            queue_counts=queue_counts,
            worker_reports=worker_reports,
            wall_seconds=wall_seconds,
            missing_tags=missing,
            unexpected_tags=unexpected,
        )
        summary.update({
            "candidate_queue": contract["candidate_queue"],
            "candidate_queue_sha256": contract["candidate_queue_sha256"],
            "selection_manifest": str(selection_path.resolve()),
            "selection_manifest_sha256": sha256_file(selection_path),
            "run_contract": str((output_root / "run-contract.json").resolve()),
            "run_contract_sha256": sha256_file(output_root / "run-contract.json"),
            "resume_result_records_reconciled": reconciled,
            "resume_interrupted_leases_released": released_leases,
            "resume_infrastructure_failures_requeued": infrastructure_requeued,
        })
        if summary["infrastructure_complete"]:
            mask_manifest = registry.deployment_mask_store.build_manifest()
            mask_manifest_path = (
                output_root / "shards" / MASK_STORE_DIRECTORY / "manifest.json"
            )
            store = build_store_manifest(
                output_root / "shards",
                source_manifest=selection_path,
                expected_episodes=int(summary["stored_episodes"]),
                expected_ticks=int(summary["stored_ticks"]),
                store_metadata={
                    "generator_kind": GENERATOR_KIND,
                    "native_teacher_forced_profile": contract[
                        "native_teacher_forced_profile"
                    ],
                    "coordinate_provenance": COORDINATE_PROVENANCE,
                    "ability_branch_policy": contract["ability_branch_policy"],
                    "source_json_copied": False,
                    "native_deployment_masks": {
                        "required": True,
                        "schema_version": 1,
                        "dynamic_rule": (
                            "native_base_and_tower_state_projection_v1"
                        ),
                        "manifest": (
                            f"{MASK_STORE_DIRECTORY}/manifest.json"
                        ),
                        "manifest_sha256": sha256_file(mask_manifest_path),
                        "sidecars": int(mask_manifest["sidecars"]),
                    },
                },
            )
            summary["stored_bytes"] = int(store["total_bytes"])
            summary["bytes_per_tick"] = _ratio(
                int(store["total_bytes"]), int(store["tick_count"])
            )
            summary["tick_store_manifest"] = str(
                (output_root / "shards" / "manifest.json").resolve()
            )
            summary["tick_store_manifest_sha256"] = sha256_file(
                output_root / "shards" / "manifest.json"
            )
            summary["deployment_mask_store_manifest"] = str(
                mask_manifest_path.resolve()
            )
            summary["deployment_mask_store_manifest_sha256"] = sha256_file(
                mask_manifest_path
            )
            summary["deployment_mask_unique_sidecars"] = int(
                mask_manifest["sidecars"]
            )
            summary["deployment_mask_sidecar_bytes"] = int(
                mask_manifest["bytes"]
            )
        atomic_json(output_root / "summary.json", summary)
        if summary["publication_ready"]:
            manifest = {
                "schema_version": GENERATOR_SCHEMA_VERSION,
                "kind": "expert_authoritative_native_tick_dataset_manifest_v1",
                "created_utc": utc_now(),
                "status": (
                    "complete" if not summary["teacher_forced_failures"]
                    else "complete_with_fail_closed_semantic_rejections"
                ),
                "source": {
                    "candidate_queue": contract["candidate_queue"],
                    "candidate_queue_sha256": contract["candidate_queue_sha256"],
                    "selection_manifest": str(selection_path.resolve()),
                    "selection_manifest_sha256": sha256_file(selection_path),
                    "source_json_copied": False,
                },
                "semantics": {
                    "battle_core": "original libg.so",
                    "tick_hz": 20,
                    "native_teacher_forced_profile": contract[
                        "native_teacher_forced_profile"
                    ],
                    "coordinate_provenance": COORDINATE_PROVENANCE,
                    "ability_branch_policy": contract["ability_branch_policy"],
                    "first_difference_policy": "fail_closed",
                    "native_deployment_masks": contract[
                        "native_deployment_masks"
                    ],
                },
                "counts": {
                    key: summary[key]
                    for key in (
                        "selected_battles", "processed_battles",
                        "teacher_forced_successes", "teacher_forced_failures",
                        "stored_episodes", "stored_ticks",
                        "native_actions_attempted", "native_actions_accepted",
                        "native_deployment_mask_probe_rpcs",
                        "native_deployment_mask_dynamic_label_probe_rpcs",
                        "deployment_mask_unique_sidecars",
                    )
                },
                "content": {
                    "results": "results.jsonl",
                    "results_sha256": sha256_file(output_root / "results.jsonl"),
                    "summary": "summary.json",
                    "summary_sha256": sha256_file(output_root / "summary.json"),
                    "tick_store_manifest": "shards/manifest.json",
                    "tick_store_manifest_sha256": sha256_file(
                        output_root / "shards" / "manifest.json"
                    ),
                    "deployment_mask_store_manifest": (
                        f"shards/{MASK_STORE_DIRECTORY}/manifest.json"
                    ),
                    "deployment_mask_store_manifest_sha256": sha256_file(
                        output_root / "shards" / MASK_STORE_DIRECTORY
                        / "manifest.json"
                    ),
                },
            }
            atomic_json(output_root / "manifest.json", manifest)
            _atomic_text(
                output_root / "manifest.sha256",
                f"{sha256_file(output_root / 'manifest.json')}  manifest.json\n",
            )
        return summary
