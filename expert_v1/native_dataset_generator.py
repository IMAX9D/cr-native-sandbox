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
import re
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback is supported
    orjson = None

from native_core.env import NativeRoyaleEnv

from .native_capabilities import ability_cards
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
    DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS,
    DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    NativeSeedSearchError,
    compatible_native_seed_search,
)
from .tick_store_v1.codec import EpisodeReader, encode_episode
from .tick_store_v1.deployment_masks import (
    DYNAMIC_RULE,
    EPISODE_METADATA_KEY,
    MASK_STORE_DIRECTORY,
    DeploymentMaskStore,
    NativeDeploymentMaskCapture,
    locked_enemy_princess_pocket_proof_valid,
    resolve_deployment_reference,
    validate_episode_mask_metadata,
)
from .tick_store_v1.schema import TickState, require_consecutive
from .tick_store_v1.shard import (
    AUDIT_PREFIX_STORE_KIND,
    FRAME_HEADER,
    FRAME_MAGIC,
    SHARD_KIND,
    STORE_KIND,
    WorkerShardSink,
    _tag_hash,
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
RUN_CONTRACT_VERSION = 4
NATIVE_PREFLIGHT_CONTRACT_VERSION = 4
NATIVE_PREFLIGHT_MODE = (
    "single_semantic_seed_preflight_then_fixed_seed_trace_v4"
)
SEMANTIC_SEED_AUDIT_SCHEMA_VERSION = 2
SEMANTIC_SEED_AUDIT_KIND = "single_semantic_seed_preflight_v2"
SEMANTIC_SEED_SELECTION_RULE = "first_layout_compatible_seed_only"
AUDIT_PREFIX_DIRECTORY = "audit-prefix-shards"
REPLAY_EXTENT_METADATA_KEY = "native_replay_extent_v1"
REPLAY_EXTENT_KIND = "cr_native_replay_extent_v1"
MASK_INVALID_CENSOR_KIND = "native_mask_invalid_safe_censor_v3"
MASK_INVALID_FAILURE_CLASS = "native_deployment_mask_invalid_censored"
MASK_INVALID_FAILURE_DOMAIN = "semantic_mask_invalid"
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
        "expert_v1/native_seed_search.py",
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
        "run_contract_version": RUN_CONTRACT_VERSION,
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
        "native_execution_pipeline": {
            "contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
            "mode": NATIVE_PREFLIGHT_MODE,
            "semantic_seed_preflight": {
                "layout_compatible_candidate_limit": (
                    DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
                ),
                "selection_rule": SEMANTIC_SEED_SELECTION_RULE,
                "raw_seed_scan": (
                    "canonical ascending libg resets until first compatible layout"
                ),
            },
            "preflight": {
                "tick_sink": False,
                "deployment_mask_capture": False,
                "decision_capture": False,
                "seed_resolution": "bounded_source_order_search",
            },
            "full_trace": {
                "tick_sink": True,
                "deployment_mask_capture": True,
                "decision_capture": False,
                "seed_resolution": "single_reset_with_preflight_chosen_seed",
                "layout_revalidation": True,
            },
            "semantic_diff": (
                "action_acceptance_sequence+failure+terminal+tower_hp; "
                "any difference fails closed"
            ),
        },
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
        self,
        capture: NativeDeploymentMaskCapture | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        payloads: Mapping[str, Mapping[str, Any]] | None = None,
        require_complete: bool = True,
    ) -> None:
        if self.episode is not None:
            raise RuntimeError("deployment masks must be staged before episode")
        if self._deployment_mask_metadata is not None:
            raise RuntimeError("deployment masks may be staged only once")
        if capture is not None:
            if metadata is not None or payloads is not None:
                raise RuntimeError("mask capture and explicit payloads are exclusive")
            metadata = capture.metadata(require_complete=require_complete)
            payloads = capture.payloads
        if not isinstance(metadata, Mapping) or not isinstance(payloads, Mapping):
            raise RuntimeError("deployment mask metadata/payloads are required")
        validate_episode_mask_metadata(
            metadata, require_complete=require_complete
        )
        self._deployment_mask_metadata = dict(metadata)
        self._deployment_mask_payloads = {
            str(digest): dict(value) for digest, value in payloads.items()
        }

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


def build_full_success_token_evidence(
    plan: BattlePlan,
    result: NativeReplayResult,
    episode: StagedEpisode,
    *,
    prefix_extent: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bind source markers to accepted libg actions for later BC coverage.

    This evidence is emitted only for a complete teacher-forced replay.  It is
    still *native* evidence rather than compiled supervision: the compiler
    independently joins it to the frozen source, Tick Store and final shard
    labels before setting ``compiled=true``.
    """

    censored_prefix = prefix_extent is not None
    if censored_prefix:
        extent = dict(prefix_extent or {})
        cutoff = int(extent.get("action_label_tick_stop_exclusive", -1))
        if (
            result.teacher_forced_success
            or cutoff <= 0
            or extent.get("training_admission") not in {
                "actor_bc_censored_prefix_v1",
                "actor_bc_mask_invalid_censored_prefix_v1",
            }
            or (
                extent.get("training_admission")
                == "actor_bc_mask_invalid_censored_prefix_v1"
                and not mask_invalid_censor_provenance_valid(
                    extent.get("censor_provenance")
                )
            )
            or extent.get("failure_tick_has_labels") is not False
            or episode.battle_tag != plan.battle_tag
        ):
            raise RuntimeError("prefix token evidence extent is invalid")
    else:
        extent = {}
        cutoff = 2**63 - 1
    if (
        (not censored_prefix and (
            not result.teacher_forced_success
            or result.accepted_actions != result.source_actions
            or result.tick_store_entry is None
        ))
        or episode.battle_tag != plan.battle_tag
    ):
        raise RuntimeError("token evidence requires one complete native replay")
    raw_masks = episode.metadata.get(EPISODE_METADATA_KEY)
    if not isinstance(raw_masks, Mapping):
        raise RuntimeError("token evidence requires deployment-mask metadata")
    masks = validate_episode_mask_metadata(
        raw_masks, require_complete=not censored_prefix
    )
    entries = list(masks["entries"])
    payloads = episode.deployment_mask_payloads
    state_by_tick = {state.tick: state for state in episode.states}
    if len(state_by_tick) != len(episode.states):
        raise RuntimeError("token evidence Tick Store contains duplicate Ticks")

    acceptance: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for raw in result.action_acceptance_sequence:
        row = _mapping_for_token_evidence(raw, "native action acceptance")
        key = (
            str(row.get("type") or ""),
            int(row.get("side", -1)),
            int(row.get("source_event_index", -1)),
            int(row.get("source_tick", -1)),
        )
        execution_tick = int(row.get("execution_tick", -1))
        if execution_tick >= cutoff:
            continue
        if key in acceptance or row.get("accepted") is not True:
            raise RuntimeError("token evidence action acceptance is open/duplicate")
        acceptance[key] = row
    expected_source_actions = sum(
        int(action.tick) + int(result.action_execution_tick_offset) < cutoff
        for action in plan.actions
    ) + sum(
        int(event.tick) + int(result.action_execution_tick_offset) < cutoff
        for event in plan.ability_events
    )
    if len(acceptance) != expected_source_actions:
        raise RuntimeError("token evidence does not cover every native action")

    actor_rows = [
        {
            "schema_version": 1,
            "kind": (
                "cr_native_censored_prefix_actor_token_evidence_v1"
                if censored_prefix
                else "cr_native_full_success_actor_token_evidence_v1"
            ),
            "battle_tag": plan.battle_tag,
            "actor_side": side,
            "full_success": not censored_prefix,
            "censored_prefix": censored_prefix,
            "prefix_admission": censored_prefix,
            "action_label_tick_stop_exclusive": (
                cutoff if censored_prefix else None
            ),
            "timing_target": (
                "right_censored_at_failure_tick_v1"
                if censored_prefix else "complete_episode_v1"
            ),
            "replay_extent_sha256": (
                hashlib.sha256(json.dumps(
                    extent,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                if censored_prefix else None
            ),
            "deck_tokens": [card.source_token for card in plan.sides[side].deck],
            "deploy_labels": [],
            "ability_labels": [],
        }
        for side in (0, 1)
    ]

    for action in plan.actions:
        side = int(action.side)
        spec = plan.sides[side].deck[int(action.logical_card_index)]
        execution_tick = int(action.tick) + int(result.action_execution_tick_offset)
        if execution_tick >= cutoff:
            continue
        action_key = ("play", side, int(action.source_event_index), int(action.tick))
        accepted = acceptance.get(action_key)
        if (
            accepted is None
            or int(accepted.get("execution_tick", -1)) != execution_tick
        ):
            raise RuntimeError("deployment evidence/source acceptance join failed")
        matching = [
            entry
            for entry in entries
            if int(entry["side"]) == side
            and int(entry["card_id"]) == int(spec.card_id)
            and int(entry["form_flags"]) == int(spec.form_flags)
        ]
        if len(matching) != 1:
            raise RuntimeError("deployment evidence deck-slot identity is ambiguous")
        reference = resolve_deployment_reference(
            matching[0], tick=execution_tick, require_dynamic_exact=True
        )
        if reference is None:
            raise RuntimeError("deployment evidence has no exact Tick mask")
        digest = str(reference.get("content_sha256") or "")
        payload = payloads.get(digest)
        if not isinstance(payload, Mapping):
            raise RuntimeError("deployment evidence mask payload is missing")
        label = {
            "source_event_index": int(action.source_event_index),
            "source_marker_index": int(action.source_marker_index),
            "source_tick": int(action.tick),
            "execution_tick": execution_tick,
            "source_token": str(spec.source_token),
            "resolved_native_form_id": int(payload["resolved_data_id"]),
            "mask_content_sha256": digest,
            "identity_provenance": (
                "libg_dynamic_choice_exact_v1"
                if str(spec.source_token) == "mirror"
                else "libg_deployment_mask_exact_v1"
            ),
            "accepted": True,
            "mask_legal": True,
            "compiled": False,
        }
        label["native_evidence_sha256"] = hashlib.sha256(
            json.dumps(
                label, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        actor_rows[side]["deploy_labels"].append(label)

    resolution_by_marker: dict[int, Mapping[str, Any]] = {}
    for raw in result.ability_resolutions:
        row = _mapping_for_token_evidence(raw, "native ability resolution")
        marker = int(row.get("source_marker_index", -1))
        if marker < 0 or marker in resolution_by_marker:
            raise RuntimeError("ability evidence marker is invalid/duplicate")
        resolution_by_marker[marker] = row
    for event in plan.ability_events:
        side = int(event.side)
        execution_tick = int(event.tick) + int(result.action_execution_tick_offset)
        if execution_tick >= cutoff:
            continue
        resolution = resolution_by_marker.get(int(event.source_marker_index))
        accepted = acceptance.get(
            ("ability", side, int(event.source_event_index), int(event.tick))
        )
        if (
            resolution is None
            or accepted is None
            or int(resolution.get("source_tick", -1)) != int(event.tick)
            or int(resolution.get("execution_tick", -1)) != execution_tick
            or int(accepted.get("execution_tick", -1)) != execution_tick
        ):
            raise RuntimeError("ability evidence source/native join failed")
        execution = str(resolution.get("execution") or "")
        if execution not in {"unique_executed", "explicit_branch_executed"}:
            raise RuntimeError("ability evidence is not an executed native branch")
        entity_id = int(resolution.get("selected_entity_id", -1))
        candidate_entities = tuple(
            int(value) for value in resolution.get("candidate_entity_ids") or ()
        )
        candidate_cards = tuple(
            int(value) for value in resolution.get("candidate_card_ids") or ()
        )
        if (
            entity_id < 0
            or len(candidate_entities) != len(candidate_cards)
            or entity_id not in candidate_entities
        ):
            raise RuntimeError("ability evidence selected entity is not a candidate")
        state = state_by_tick.get(execution_tick)
        if state is None:
            raise RuntimeError("ability evidence Tick is absent from full Tick Store")
        entities = [entity for entity in state.entities if entity.key == entity_id]
        if len(entities) != 1:
            raise RuntimeError("ability evidence selected entity is absent/duplicate")
        entity = entities[0]
        if (
            entity.side != side
            or entity.ability_slot <= 0
            or entity.ability_available != 1
        ):
            raise RuntimeError("ability evidence selected entity was not legally usable")
        allowed = [
            card
            for card in ability_cards(plan.sides[side].deck)
            if int(entity.card_id) == int(card.native_form_id)
        ]
        if len(allowed) != 1:
            raise RuntimeError("ability evidence native form does not identify one token")
        card = allowed[0]
        selected_index = candidate_entities.index(entity_id)
        if int(candidate_cards[selected_index]) != int(card.base_card_id):
            raise RuntimeError("ability evidence entity/base-card transcript changed")
        label = {
            "source_event_index": int(event.source_event_index),
            "source_marker_index": int(event.source_marker_index),
            "source_tick": int(event.tick),
            "execution_tick": execution_tick,
            "resolved_token": str(card.source_token),
            "resolved_native_form_id": int(entity.card_id),
            "selected_entity_id": entity_id,
            "identity_provenance": (
                "libg_live_entity_unique_v1"
                if execution == "unique_executed"
                else "libg_live_entity_explicit_branch_v1"
            ),
            "branch_verified": execution == "explicit_branch_executed",
            "accepted": True,
            "legal": True,
            "compiled": False,
            "libg_resolution": {
                "status": str(resolution.get("status") or ""),
                "execution": execution,
                "side": side,
                "source_tick": int(event.tick),
                "execution_tick": execution_tick,
                "source_event_index": int(event.source_event_index),
                "source_marker_index": int(event.source_marker_index),
                "candidate_entity_ids": list(candidate_entities),
                "candidate_card_ids": list(candidate_cards),
                "selected_entity_id": entity_id,
                "selected_native_form_id": int(entity.card_id),
            },
        }
        label["native_evidence_sha256"] = hashlib.sha256(
            json.dumps(
                label, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        actor_rows[side]["ability_labels"].append(label)
    safe_ability_markers = {
        int(event.source_marker_index)
        for event in plan.ability_events
        if int(event.tick) + int(result.action_execution_tick_offset) < cutoff
    }
    if not safe_ability_markers <= set(resolution_by_marker):
        raise RuntimeError("ability evidence has unexpected native resolutions")

    for row in actor_rows:
        body = {
            key: value for key, value in row.items()
            if key != "native_evidence_sha256"
        }
        row["native_evidence_sha256"] = hashlib.sha256(
            json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    return actor_rows


def _mapping_for_token_evidence(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    return value


class StoredFrameRegistry:
    """Detect/reuse a checksummed orphan frame after a hard-process crash."""

    def __init__(
        self,
        root: Path,
        deployment_mask_store: DeploymentMaskStore | None = None,
        *,
        require_complete_masks: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.deployment_mask_store = (
            deployment_mask_store
            if deployment_mask_store is not None
            else DeploymentMaskStore(self.root)
        )
        self.require_complete_masks = bool(require_complete_masks)
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
                    episode.metadata,
                    allow_cached=True,
                    require_complete=self.require_complete_masks,
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
    if isinstance(error, PreflightFullTraceDivergence):
        return "infrastructure_preflight_full_trace_semantic_divergence"
    if error is not None:
        if stage in {
            "immutable_tick_store_commit",
            "immutable_prefix_tick_store_commit",
        }:
            return "infrastructure_tick_store_commit_failed"
        if stage == "tick_store_postcondition":
            return "infrastructure_tick_store_postcondition_failed"
        if stage in {
            "native_teacher_forced_replay",
            "native_preflight",
            "native_full_trace_replay",
        }:
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
    if failure.startswith("derived_deployment_mask_rejected_source_event_"):
        return MASK_INVALID_FAILURE_CLASS
    if failure.startswith("native_terminal_before_"):
        return "native_terminal_before_source_event"
    if failure.startswith("hand_mismatch_event_"):
        return "source_hand_sequence_mismatch"
    if failure.startswith((
        "source_ability_ticks_missing_count_",
        "source_tick_",
        "execution_tick_",
        "native_deployment_mask_capture_incomplete_slots_",
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
    if failure_class == MASK_INVALID_FAILURE_CLASS:
        return MASK_INVALID_FAILURE_DOMAIN
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


class PreflightFullTraceDivergence(RuntimeError):
    """The instrumented replay changed fixed-seed native semantics."""

    def __init__(self, differences: Mapping[str, Any]) -> None:
        self.differences = dict(differences)
        super().__init__(
            "native preflight/full-trace semantic divergence: "
            + ",".join(sorted(self.differences))
        )


def native_replay_semantics(result: NativeReplayResult) -> dict[str, Any]:
    """Project only semantics that instrumentation is forbidden to change."""
    return {
        "teacher_forced_success": bool(result.teacher_forced_success),
        "failure": result.failure,
        "accepted_actions": int(result.accepted_actions),
        "accepted_deploy_actions": int(result.accepted_deploy_actions),
        "accepted_ability_actions": int(result.accepted_ability_actions),
        "action_acceptance_sequence": [
            dict(row) for row in result.action_acceptance_sequence
        ],
        "final_tick": int(result.final_tick),
        "terminal": {
            "validated": bool(result.terminal_validated),
            "match": result.terminal_match,
            "diagnostic_status": result.terminal_diagnostic_status,
            "source_crowns": result.source_crowns,
            "observed_crowns": result.observed_crowns,
        },
        "tower_hp": {
            "validated": bool(result.terminal_tower_hp_validated),
            "match": result.terminal_tower_hp_match,
            "diagnostic_status": result.terminal_tower_hp_diagnostic_status,
            "source": result.source_final_tower_hp,
            "observed": result.observed_final_tower_hp,
        },
    }


def diff_native_replay_semantics(
    preflight: NativeReplayResult,
    full_trace: NativeReplayResult,
) -> dict[str, Any]:
    """Return an auditable top-level diff; an empty mapping is exact parity."""
    left = native_replay_semantics(preflight)
    right = native_replay_semantics(full_trace)
    differences = {
        key: {"preflight": left[key], "full_trace": right[key]}
        for key in left
        if left[key] != right[key]
    }
    if preflight.chosen_seed != full_trace.chosen_seed:
        differences["chosen_seed"] = {
            "preflight": preflight.chosen_seed,
            "full_trace": full_trace.chosen_seed,
        }
    if full_trace.layout_resolution_mode != "fixed_preflight_seed_replay":
        differences["full_trace_layout_resolution_mode"] = {
            "preflight": "fixed_preflight_seed_replay",
            "full_trace": full_trace.layout_resolution_mode,
        }
    if full_trace.seed_search_native_resets != 1 or full_trace.seeds_tested != 0:
        differences["full_trace_seed_reuse"] = {
            "expected": {"native_resets": 1, "seeds_tested": 0},
            "actual": {
                "native_resets": full_trace.seed_search_native_resets,
                "seeds_tested": full_trace.seeds_tested,
            },
        }
    return differences


def _semantic_digest(result: NativeReplayResult) -> str:
    payload = json.dumps(
        native_replay_semantics(result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def accepted_source_event_prefix(result: NativeReplayResult) -> int:
    """Count the contiguous accepted source-event prefix of one preflight."""

    accepted = 0
    for row in result.action_acceptance_sequence:
        if row.get("accepted") is not True:
            break
        accepted += 1
    return accepted


@dataclass(frozen=True)
class SemanticSeedPreflightSelection:
    selected: NativeReplayResult
    audit: dict[str, Any]


def execute_bounded_semantic_preflights(
    env: Any,
    recorder: RecordingCountingEnv,
    plan: BattlePlan,
    template: Mapping[str, Any],
    *,
    seed: int,
    maximum_seeds_to_test: int,
    maximum_compatible_seeds: int = (
        DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
    ),
    trace_batch_steps: int,
) -> SemanticSeedPreflightSelection:
    """Run exactly one semantic preflight on the first compatible seed.

    Every candidate replay is no-trace/no-mask and uses an exact fixed seed.
    Raw seed scanning remains canonical and bounded, but only its first
    layout-compatible result enters semantic replay.  Ability branches remain
    unselected, so this search can never invent an ability identity.
    """

    if maximum_compatible_seeds != DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS:
        raise ValueError(
            "production semantic preflight requires exactly one "
            "layout-compatible seed"
        )

    search = compatible_native_seed_search(
        env,
        plan,
        template,
        preferred_seed=seed,
        maximum_seeds_to_test=maximum_seeds_to_test,
        maximum_compatible_seeds=maximum_compatible_seeds,
        warmup_tick=10,
    )
    selected: NativeReplayResult | None = None
    selected_prefix = -1
    candidate_rows: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(search):
        result = execute_plan(
            recorder,
            plan,
            template,
            None,
            seed=seed,
            fixed_seed=candidate.chosen_seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            capture_decisions=False,
            ability_branch_choices=None,
            tick_sink=None,
            trace_batch_steps=trace_batch_steps,
            capture_deployment_masks=False,
            action_execution_tick_offset=(
                ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
            ),
        )
        prefix = accepted_source_event_prefix(result)
        candidate_rows.append({
            "ordinal": ordinal,
            "seed": int(candidate.chosen_seed),
            "raw_seeds_scanned_when_found": int(candidate.seeds_tested),
            "teacher_forced_success": bool(result.teacher_forced_success),
            "accepted_source_event_prefix": prefix,
            "failure": result.failure,
            "semantics_sha256": _semantic_digest(result),
        })
        if result.teacher_forced_success:
            selected = result
            selected_prefix = prefix
            break
        if selected is None or prefix > selected_prefix:
            selected = result
            selected_prefix = prefix
    if selected is None:
        # The search iterator raises NativeSeedSearchError when it scans no
        # compatible layout.  This guard protects mock/custom iterators too.
        raise NativeSeedSearchError(
            battle_tag=plan.battle_tag,
            seeds_tested=int(search.seeds_scanned),
            maximum_seeds_to_test=int(maximum_seeds_to_test),
            preferred_seed=int(seed),
        )
    audit = {
        "schema_version": SEMANTIC_SEED_AUDIT_SCHEMA_VERSION,
        "kind": SEMANTIC_SEED_AUDIT_KIND,
        "maximum_compatible_seeds": int(maximum_compatible_seeds),
        "raw_seed_scan_limit": int(maximum_seeds_to_test),
        "raw_seeds_scanned": int(search.seeds_scanned),
        "layout_compatible_candidates_tested": len(candidate_rows),
        "layout_compatible_candidates_found": int(
            search.compatible_seeds_yielded
        ),
        "layout_scan_native_resets": int(search.native_resets),
        "semantic_preflight_native_resets": len(candidate_rows),
        "selected_seed": int(selected.chosen_seed),
        "selected_accepted_source_event_prefix": int(selected_prefix),
        "selected_teacher_forced_success": bool(
            selected.teacher_forced_success
        ),
        "selection_rule": SEMANTIC_SEED_SELECTION_RULE,
        "ability_identity_policy": "branch_required_fails_closed_no_guess",
        "candidates": candidate_rows,
    }
    return SemanticSeedPreflightSelection(selected=selected, audit=audit)


def semantic_seed_audit_valid(
    audit: Any,
    *,
    chosen_seed: Any,
    teacher_forced_success: Any,
) -> bool:
    """Recompute the bounded selection rule from a persisted audit row."""

    if not isinstance(audit, Mapping):
        return False
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    try:
        maximum = int(audit["maximum_compatible_seeds"])
        tested = int(audit["layout_compatible_candidates_tested"])
        found = int(audit["layout_compatible_candidates_found"])
        raw_scanned = int(audit["raw_seeds_scanned"])
        raw_limit = int(audit["raw_seed_scan_limit"])
        layout_resets = int(audit["layout_scan_native_resets"])
        semantic_resets = int(audit["semantic_preflight_native_resets"])
        selected_seed = int(audit["selected_seed"])
        selected_prefix = int(audit["selected_accepted_source_event_prefix"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        audit.get("schema_version") != SEMANTIC_SEED_AUDIT_SCHEMA_VERSION
        or audit.get("kind") != SEMANTIC_SEED_AUDIT_KIND
        or maximum != DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
        or tested != len(candidates)
        or not 1 <= tested <= maximum
        or found != tested
        or not tested <= raw_scanned <= raw_limit
        or layout_resets != raw_scanned
        or semantic_resets != tested
        or audit.get("ability_identity_policy")
        != "branch_required_fails_closed_no_guess"
        or audit.get("selection_rule") != SEMANTIC_SEED_SELECTION_RULE
        or selected_seed != chosen_seed
        or audit.get("selected_teacher_forced_success")
        is not teacher_forced_success
    ):
        return False
    normalized: list[tuple[int, int, bool, int]] = []
    previous_seed = 0
    for expected_ordinal, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            return False
        try:
            ordinal = int(row["ordinal"])
            seed = int(row["seed"])
            prefix = int(row["accepted_source_event_prefix"])
            scan_at_found = int(row["raw_seeds_scanned_when_found"])
        except (KeyError, TypeError, ValueError):
            return False
        success = row.get("teacher_forced_success")
        digest = str(row.get("semantics_sha256") or "")
        if (
            ordinal != expected_ordinal
            or seed <= previous_seed
            or prefix < 0
            or not ordinal + 1 <= scan_at_found <= raw_scanned
            or not isinstance(success, bool)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            return False
        previous_seed = seed
        normalized.append((ordinal, seed, success, prefix))
    expected = normalized[0]
    return bool(
        selected_seed == expected[1]
        and selected_prefix == expected[3]
        and teacher_forced_success is expected[2]
        and int(candidates[0]["raw_seeds_scanned_when_found"]) == raw_scanned
    )


def native_result_pipeline_contract_valid(row: Any) -> bool:
    """Validate one persisted result against the current cap=1 pipeline."""

    if not isinstance(row, Mapping):
        return False
    success = row.get("preflight_teacher_forced_success")
    chosen = row.get("preflight_chosen_seed")
    if (
        row.get("native_preflight_contract_version")
        == NATIVE_PREFLIGHT_CONTRACT_VERSION
        and row.get("native_execution_pipeline_mode") == NATIVE_PREFLIGHT_MODE
        and success is None
    ):
        return bool(
            chosen is None
            and row.get("chosen_seed") is None
            and row.get("semantic_seed_preflight") is None
            and row.get("tick_store_entry") is None
            and row.get("audit_prefix_tick_store_entry") is None
            and row.get("token_coverage_actor_evidence") in (None, [])
            and row.get("prefix_token_coverage_actor_evidence") in (None, [])
        )
    if (
        row.get("native_preflight_contract_version")
        != NATIVE_PREFLIGHT_CONTRACT_VERSION
        or row.get("native_execution_pipeline_mode") != NATIVE_PREFLIGHT_MODE
        or not isinstance(success, bool)
        or isinstance(chosen, bool)
        or not isinstance(chosen, int)
        or chosen <= 0
        or row.get("chosen_seed") != chosen
    ):
        return False
    return semantic_seed_audit_valid(
        row.get("semantic_seed_preflight"),
        chosen_seed=chosen,
        teacher_forced_success=success,
    )


def native_episode_pipeline_contract_valid(
    metadata: Any,
    result_row: Any,
) -> bool:
    """Bind one Full/Prefix episode to its authenticated current result."""

    if (
        not native_result_pipeline_contract_valid(result_row)
        or not isinstance(
            result_row.get("preflight_teacher_forced_success"), bool
        )
    ):
        return False
    if not isinstance(metadata, Mapping):
        return False
    pipeline = metadata.get("native_execution_pipeline")
    if not isinstance(pipeline, Mapping):
        return False
    chosen = int(result_row["preflight_chosen_seed"])
    audit = result_row["semantic_seed_preflight"]
    candidates = audit.get("candidates") if isinstance(audit, Mapping) else None
    selected = candidates[0] if isinstance(candidates, list) and candidates else None
    if (
        pipeline.get("contract_version") != NATIVE_PREFLIGHT_CONTRACT_VERSION
        or pipeline.get("mode") != NATIVE_PREFLIGHT_MODE
        or pipeline.get("preflight_chosen_seed") != chosen
        or pipeline.get("semantic_seed_selection") != audit
        or not isinstance(selected, Mapping)
        or pipeline.get("preflight_semantics_sha256")
        != selected.get("semantics_sha256")
    ):
        return False
    preflight_sha = selected.get("semantics_sha256")
    extent = metadata.get(REPLAY_EXTENT_METADATA_KEY)
    if isinstance(extent, Mapping):
        provenance = extent.get("censor_provenance")
        if extent.get("training_admission") == (
            "actor_bc_mask_invalid_censored_prefix_v1"
        ):
            return bool(
                extent.get("failure_class") == MASK_INVALID_FAILURE_CLASS
                and extent.get("failure_domain") == MASK_INVALID_FAILURE_DOMAIN
                and extent.get("semantic_match") is False
                and extent.get("maskless_reference_semantic_match") is True
                and extent.get("pre_censor_tick_state_parity") is True
                and extent.get("fixed_seed_replay") is True
                and extent.get("chosen_seed") == chosen
                and extent.get("preflight_semantics_sha256") == preflight_sha
                and mask_invalid_censor_provenance_valid(provenance)
                and provenance.get("preflight_semantics_sha256") == preflight_sha
            )
        return bool(
            extent.get("fixed_seed_replay") is True
            and extent.get("chosen_seed") == chosen
            and extent.get("preflight_semantics_sha256") == preflight_sha
            and extent.get("prefix_replay_semantics_sha256") == preflight_sha
            and extent.get("semantic_match") is True
        )
    return bool(
        pipeline.get("full_trace_semantics_sha256") == preflight_sha
        and pipeline.get("semantic_diff_count") == 0
        and pipeline.get("semantic_match") is True
    )


@dataclass(slots=True)
class TwoPhaseNativeReplay:
    preflight: NativeReplayResult
    full_trace: NativeReplayResult | None
    preflight_recorder: RecordingCountingEnv
    full_trace_recorder: RecordingCountingEnv | None
    preflight_seconds: float
    full_trace_seconds: float
    semantic_diff: dict[str, Any] | None
    failure_prefix: NativeReplayResult | None = None
    failure_prefix_recorder: RecordingCountingEnv | None = None
    failure_prefix_seconds: float = 0.0
    failure_prefix_staged: bool = False
    semantic_seed_audit: dict[str, Any] = field(default_factory=dict)
    mask_invalid_prefix: bool = False
    mask_invalid_semantic_diff: dict[str, Any] | None = None
    maskless_reference: NativeReplayResult | None = None
    maskless_reference_recorder: RecordingCountingEnv | None = None
    maskless_reference_seconds: float = 0.0

    @property
    def result(self) -> NativeReplayResult:
        if self.mask_invalid_prefix and self.failure_prefix is not None:
            return self.failure_prefix
        return self.preflight if self.full_trace is None else self.full_trace


@dataclass(slots=True)
class MasklessReferenceAttempt:
    result: NativeReplayResult
    recorder: RecordingCountingEnv
    seconds: float
    evidence: dict[str, Any] | None


_FAILURE_TICK_PATTERNS = (
    re.compile(r"native_rejected_tick_(\d+)(?:_source_tick_(\d+))?"),
    re.compile(r"native_terminal_before_execution_tick_(\d+)_source_tick_(\d+)"),
    re.compile(r"native_terminal_before_source_tick_(\d+)"),
    re.compile(r"native_logic_frozen_before_execution_tick_(\d+)_source_tick_(\d+)"),
)
_MASK_REJECTION_FAILURE_RE = re.compile(
    r"^derived_deployment_mask_rejected_source_event_(\d+)$"
)
_MASK_REJECTION_FIELDS = {
    "tick", "side", "deck_index", "card_id", "x", "y",
    "content_sha256", "legal", "reasons", "source_event_index",
    "source_marker_index", "locked_pocket",
}
_ACCEPTED_ACTION_ROW_FIELDS = {
    "accepted", "execution_tick", "result_code", "side",
    "source_event_index", "source_tick", "type",
}


def _canonical_rows_sha256(rows: Any) -> str:
    return hashlib.sha256(json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def mask_invalid_censor_provenance_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    required = {
        "schema_version", "kind", "rejected_source_event_index", "source_marker_index",
        "source_tick", "execution_tick", "side", "deck_index", "card_id",
        "x", "y", "mask_content_sha256", "boundary_deploy_labels_checked",
        "mask_rejection_count", "failure_event_executed",
        "failure_label_compiled", "label_or_mask_repair_applied",
        "censored_tick_event_indices", "safe_action_count",
        "safe_action_transcript_sha256", "maskless_reference_reset_count",
        "mask_lane_action_metrics",
        "maskless_reference_layout_mode", "pre_censor_tick_start",
        "pre_censor_tick_stop_exclusive", "pre_censor_tick_count",
        "mask_lane_tick_sha256", "maskless_tick_sha256", "tick_state_parity",
        "preflight_semantics_sha256", "maskless_reference_semantics_sha256",
        "preflight_boundary_accepted_action",
        "preflight_boundary_accepted_action_sha256",
        "locked_pocket",
    }
    sha_fields = {
        "mask_content_sha256", "safe_action_transcript_sha256",
        "mask_lane_tick_sha256", "maskless_tick_sha256",
        "preflight_semantics_sha256", "maskless_reference_semantics_sha256",
        "preflight_boundary_accepted_action_sha256",
    }
    try:
        return bool(
            set(value) == required
            and int(value["schema_version"]) == 3
            and value.get("kind") == MASK_INVALID_CENSOR_KIND
            and int(value["side"]) in (0, 1)
            and 0 <= int(value["deck_index"]) < 8
            and 0 <= int(value["x"]) < 18_000
            and 0 <= int(value["y"]) < 32_000
            and locked_enemy_princess_pocket_proof_valid(
                value.get("locked_pocket")
            )
            and int(value["locked_pocket"]["tower_side"])
            == 1 - int(value["side"])
            and int(value["locked_pocket"]["row"]) == int(value["y"]) // 1000
            and int(value["locked_pocket"]["column"])
            == int(value["x"]) // 1000
            and int(value["locked_pocket"]["lane"])
            == int(int(value["x"]) // 1000 >= 9)
            and int(value["locked_pocket"]["lane"])
            == int(int(value["locked_pocket"]["tower_x"]) >= 9000)
            and int(value["locked_pocket"]["row"])
            in (range(17, 21) if int(value["side"]) == 0 else range(11, 15))
            and int(value["mask_rejection_count"]) == 1
            and int(value["boundary_deploy_labels_checked"]) >= 1
            and value.get("failure_event_executed") is False
            and value.get("failure_label_compiled") is False
            and value.get("label_or_mask_repair_applied") is False
            and value.get("tick_state_parity") is True
            and int(value["maskless_reference_reset_count"]) == 1
            and value.get("maskless_reference_layout_mode")
            == "fixed_preflight_seed_replay"
            and int(value["pre_censor_tick_stop_exclusive"])
            == int(value["execution_tick"])
            and int(value["pre_censor_tick_count"]) > 0
            and int(value["safe_action_count"]) >= 0
            and value["mask_lane_tick_sha256"]
            == value["maskless_tick_sha256"]
            and value["preflight_semantics_sha256"]
            == value["maskless_reference_semantics_sha256"]
            and isinstance(
                value.get("preflight_boundary_accepted_action"), Mapping
            )
            and set(value["preflight_boundary_accepted_action"])
            == _ACCEPTED_ACTION_ROW_FIELDS
            and value["preflight_boundary_accepted_action"].get("accepted")
            is True
            and int(value["preflight_boundary_accepted_action"]["result_code"])
            == 0
            and _canonical_rows_sha256(
                value["preflight_boundary_accepted_action"]
            ) == value["preflight_boundary_accepted_action_sha256"]
            and isinstance(value.get("censored_tick_event_indices"), list)
            and isinstance(value.get("mask_lane_action_metrics"), Mapping)
            and value["mask_lane_action_metrics"] == {
                "attempted": int(value["safe_action_count"]),
                "responded": int(value["safe_action_count"]),
                "accepted": int(value["safe_action_count"]),
                "rejected": 0,
                "no_response": 0,
                "exceptions": 0,
            }
            and int(value["rejected_source_event_index"])
            in value["censored_tick_event_indices"]
            and all(re.fullmatch(r"[0-9a-f]{64}", str(value[field]))
                    for field in sha_fields)
        )
    except (KeyError, TypeError, ValueError):
        return False


def prefix_extent_common_contract_valid(
    value: Any, *, tick_count: int
) -> bool:
    """Shared extent/range/accounting receiver for every trainable Prefix."""

    if not isinstance(value, Mapping):
        return False
    coverage = value.get("mask_coverage")
    if not isinstance(coverage, Mapping):
        return False
    try:
        start = int(value["observation_tick_start"])
        stop = int(value["observation_tick_stop_exclusive"])
        action_stop = int(value["action_label_tick_stop_exclusive"])
        censor = int(value["timing_censor_tick_exclusive"])
        compiled_ticks = censor - start
        return bool(
            value.get("kind") == REPLAY_EXTENT_KIND
            and value.get("extent") == "valid_prefix"
            and value.get("source_episode_complete") is False
            and value.get("every_native_tick_present_within_extent") is True
            and value.get("failure_tick_has_labels") is False
            and value.get("terminal_target") == "unknown_censored"
            and value.get("terminal_validated") is False
            and value.get("deployment_masks")
            == "partial_native_visible_hand_complete_v1"
            and value.get("timing_target")
            == "right_censored_at_failure_tick_v1"
            and stop - start == int(tick_count)
            and start < action_stop <= censor <= stop
            and compiled_ticks > 0
            and coverage.get("all_retained_visible_hand_slots_covered") is True
            and int(coverage["retained_ticks"]) == compiled_ticks
            and int(coverage["actor_ticks"]) == compiled_ticks * 2
            and int(coverage["rejected_deploy_labels"]) == 0
            and int(coverage["checked_deploy_labels"])
            >= int(coverage["safe_deploy_labels"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _unique_mask_rejection_boundary(
    masked: NativeReplayResult,
    plan: BattlePlan,
) -> dict[str, Any] | None:
    """Parse the one complete mask-rejection identity before another reset."""

    match = _MASK_REJECTION_FAILURE_RE.fullmatch(str(masked.failure or ""))
    rejection = masked.deployment_mask_first_label_rejection
    raw_sequence = getattr(
        masked, "deployment_mask_label_rejection_sequence", ()
    )
    if (
        match is None
        or int(masked.deployment_mask_label_rejections) != 1
        or not isinstance(raw_sequence, (list, tuple))
        or len(raw_sequence) != 1
        or not isinstance(raw_sequence[0], Mapping)
        or not isinstance(rejection, Mapping)
        or set(rejection) != _MASK_REJECTION_FIELDS
        or dict(raw_sequence[0]) != dict(rejection)
        or rejection.get("legal") is not False
        or rejection.get("reasons")
        != ["position_not_in_derived_native_mask"]
        or not locked_enemy_princess_pocket_proof_valid(
            rejection.get("locked_pocket")
        )
        or re.fullmatch(
            r"[0-9a-f]{64}", str(rejection.get("content_sha256") or "")
        ) is None
    ):
        return None
    try:
        event_index = int(match.group(1))
        actions = [
            action for action in plan.actions
            if int(action.source_event_index) == event_index
        ]
        if len(actions) != 1:
            return None
        action = actions[0]
        execution_tick = (
            int(action.tick) + int(masked.action_execution_tick_offset)
        )
        card = plan.sides[int(action.side)].deck[
            int(action.logical_card_index)
        ]
        if (
            int(rejection["source_event_index"]) != event_index
            or int(rejection["source_marker_index"])
            != int(action.source_marker_index)
            or int(rejection["tick"]) != execution_tick
            or int(rejection["side"]) != int(action.side)
            or int(rejection["deck_index"])
            != int(action.logical_card_index)
            or int(rejection["card_id"]) != int(card.card_id)
            or int(rejection["x"]) != int(action.x)
            or int(rejection["y"]) != int(action.y)
        ):
            return None
        actual_actions = tuple(
            dict(row) for row in masked.action_acceptance_sequence
        )
        if (
            int(masked.accepted_actions) != len(actual_actions)
            or any(row.get("accepted") is not True for row in actual_actions)
            or any(
                int(row.get("source_event_index", -1)) == event_index
                or int(row.get("execution_tick", -1)) >= execution_tick
                for row in actual_actions
            )
        ):
            return None
        safe_deploys = sum(
            row.get("type") == "play" for row in actual_actions
        )
        boundary_deploys = sum(
            int(candidate.tick) + int(masked.action_execution_tick_offset)
            == execution_tick
            for candidate in plan.actions
        )
        if (
            boundary_deploys <= 0
            or int(masked.deployment_mask_label_checks)
            != safe_deploys + boundary_deploys
        ):
            return None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return {
        "event_index": event_index,
        "action": action,
        "card": card,
        "execution_tick": execution_tick,
        "rejection": dict(rejection),
        "actual_actions": actual_actions,
        "boundary_deploys": boundary_deploys,
    }


def _mask_invalid_censor_evidence(
    preflight: NativeReplayResult,
    masked: NativeReplayResult,
    maskless_reference: NativeReplayResult,
    plan: BattlePlan,
) -> dict[str, Any] | None:
    """Authenticate one mask-invalid boundary without legalizing its label."""

    boundary = _unique_mask_rejection_boundary(masked, plan)
    if boundary is None or diff_native_replay_semantics(
        preflight, maskless_reference
    ):
        return None
    event_index = int(boundary["event_index"])
    action = boundary["action"]
    execution_tick = int(boundary["execution_tick"])
    card = boundary["card"]
    rejection = boundary["rejection"]
    preflight_boundary_rows = [
        dict(row) for row in preflight.action_acceptance_sequence
        if int(row.get("source_event_index", -1)) == event_index
    ]
    if (
        len(preflight_boundary_rows) != 1
        or set(preflight_boundary_rows[0]) != _ACCEPTED_ACTION_ROW_FIELDS
        or preflight_boundary_rows[0].get("accepted") is not True
        or int(preflight_boundary_rows[0].get("result_code", -1)) != 0
        or preflight_boundary_rows[0].get("type") != "play"
        or int(preflight_boundary_rows[0].get("source_tick", -1))
        != int(action.tick)
        or int(preflight_boundary_rows[0].get("execution_tick", -1))
        != execution_tick
        or int(preflight_boundary_rows[0].get("side", -1))
        != int(action.side)
    ):
        return None
    preflight_boundary_row = preflight_boundary_rows[0]
    expected_actions = tuple(
        dict(row) for row in preflight.action_acceptance_sequence
        if int(row["execution_tick"]) < execution_tick
    )
    actual_actions = boundary["actual_actions"]
    if actual_actions != expected_actions:
        return None
    boundary_deploys = int(boundary["boundary_deploys"])
    masked_through_boundary = tuple(
        state for state in masked.collected_tick_states
        if int(state.tick) <= execution_tick
    )
    reference_through_boundary = tuple(
        state for state in maskless_reference.collected_tick_states
        if int(state.tick) <= execution_tick
    )
    if (
        not masked_through_boundary
        or masked_through_boundary[-1].tick != execution_tick
        or masked_through_boundary != reference_through_boundary
    ):
        return None
    masked_states = tuple(
        state for state in masked.collected_tick_states
        if int(state.tick) < execution_tick
    )
    reference_states = tuple(
        state for state in maskless_reference.collected_tick_states
        if int(state.tick) < execution_tick
    )
    if not masked_states or masked_states != reference_states:
        return None
    state_rows = [asdict(state) for state in masked_states]
    action_rows = list(actual_actions)
    return {
        "schema_version": 3,
        "kind": MASK_INVALID_CENSOR_KIND,
        "rejected_source_event_index": event_index,
        "source_marker_index": int(action.source_marker_index),
        "source_tick": int(action.tick),
        "execution_tick": execution_tick,
        "side": int(action.side),
        "deck_index": int(action.logical_card_index),
        "card_id": int(card.card_id),
        "x": int(action.x),
        "y": int(action.y),
        "mask_content_sha256": str(rejection["content_sha256"]),
        "boundary_deploy_labels_checked": boundary_deploys,
        "mask_rejection_count": 1,
        "failure_event_executed": False,
        "failure_label_compiled": False,
        "label_or_mask_repair_applied": False,
        "censored_tick_event_indices": sorted(
            int(candidate.source_event_index)
            for candidate in (*plan.actions, *plan.ability_events)
            if int(candidate.tick) + int(masked.action_execution_tick_offset)
            == execution_tick
        ),
        "safe_action_count": len(action_rows),
        "safe_action_transcript_sha256": _canonical_rows_sha256(action_rows),
        "mask_lane_action_metrics": {
            "attempted": len(action_rows),
            "responded": len(action_rows),
            "accepted": int(masked.accepted_actions),
            "rejected": 0,
            "no_response": 0,
            "exceptions": 0,
        },
        "maskless_reference_reset_count": int(
            maskless_reference.seed_search_native_resets
        ),
        "maskless_reference_layout_mode": str(
            maskless_reference.layout_resolution_mode
        ),
        "pre_censor_tick_start": int(masked_states[0].tick),
        "pre_censor_tick_stop_exclusive": execution_tick,
        "pre_censor_tick_count": len(state_rows),
        "mask_lane_tick_sha256": _canonical_rows_sha256(state_rows),
        "maskless_tick_sha256": _canonical_rows_sha256(
            [asdict(state) for state in reference_states]
        ),
        "tick_state_parity": True,
        "preflight_semantics_sha256": _semantic_digest(preflight),
        "maskless_reference_semantics_sha256": _semantic_digest(
            maskless_reference
        ),
        "preflight_boundary_accepted_action": preflight_boundary_row,
        "preflight_boundary_accepted_action_sha256": _canonical_rows_sha256(
            preflight_boundary_row
        ),
        "locked_pocket": dict(rejection["locked_pocket"]),
    }


def _failure_boundary(
    result: NativeReplayResult,
) -> tuple[int, int | None, int | None] | None:
    """Return execution/source/event boundary for a traceable semantic failure."""
    rejected = [
        row for row in result.action_acceptance_sequence
        if row.get("accepted") is not True
    ]
    if rejected:
        row = rejected[0]
        return (
            int(row["execution_tick"]),
            int(row["source_tick"]),
            int(row["source_event_index"]),
        )
    if result.ability_resolutions:
        row = result.ability_resolutions[-1]
        if row.get("execution") not in {"unique_executed", "explicit_branch_executed"}:
            return (
                int(row["execution_tick"]),
                int(row["source_tick"]),
                int(row["source_event_index"]),
            )
    diagnostic = result.logic_freeze_diagnostic
    if isinstance(diagnostic, Mapping):
        return (
            int(diagnostic["execution_tick"]),
            int(diagnostic["source_tick"]),
            None,
        )
    failure = str(result.failure or "")
    for pattern in _FAILURE_TICK_PATTERNS:
        match = pattern.search(failure)
        if match:
            execution_tick = int(match.group(1))
            source_tick = (
                int(match.group(2)) if match.lastindex and match.lastindex >= 2
                and match.group(2) is not None else execution_tick
            )
            return execution_tick, source_tick, None
    return None


def _traceable_semantic_prefix(result: NativeReplayResult) -> bool:
    failure_class = _failure_class(result, None, "preflight_first_native_difference")
    return (
        _failure_domain(failure_class) == "semantic"
        and failure_class != "native_seed_search_exhausted"
        and _failure_boundary(result) is not None
    )


def _prefix_extent_metadata(
    preflight: NativeReplayResult,
    prefix: NativeReplayResult,
    *,
    base: Mapping[str, Any],
    mask_invalid_censor: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    states = prefix.collected_tick_states
    boundary = (
        (
            int(mask_invalid_censor["execution_tick"]),
            int(mask_invalid_censor["source_tick"]),
            int(mask_invalid_censor["rejected_source_event_index"]),
        )
        if isinstance(mask_invalid_censor, Mapping)
        else _failure_boundary(prefix)
    )
    if not states or boundary is None:
        return None
    require_consecutive(states)
    if int(prefix.tick_trace_complete_frames) != len(states):
        return None
    failure_execution_tick, failure_source_tick, failure_event_index = boundary
    last_tick = states[-1].tick
    if (
        failure_execution_tick < states[0].tick
        or last_tick > failure_execution_tick
        or (mask_invalid_censor is not None and last_tick != failure_execution_tick)
    ):
        return None
    failure_class = (
        MASK_INVALID_FAILURE_CLASS
        if mask_invalid_censor is not None
        else _failure_class(prefix, None, "preflight_first_native_difference")
    )
    failure_domain = (
        MASK_INVALID_FAILURE_DOMAIN
        if mask_invalid_censor is not None else "semantic"
    )
    accepted_before_boundary = [
        dict(row)
        for row in prefix.action_acceptance_sequence
        if row.get("accepted") is True
        and int(row["execution_tick"]) < failure_execution_tick
    ]
    mask_metadata = prefix.deployment_mask_metadata
    if not isinstance(mask_metadata, Mapping):
        return None
    try:
        normalized_masks = validate_episode_mask_metadata(
            mask_metadata, require_complete=False
        )
    except Exception:
        return None
    mask_keys = {
        (int(row["side"]), int(row["deck_index"]))
        for row in normalized_masks["entries"]
    }
    label_stop = min(failure_execution_tick, last_tick + 1)
    training_states = tuple(
        state for state in states if int(state.tick) < label_stop
    )
    if not training_states:
        return None
    retained_actor_ticks = 0
    visible_slot_references = 0
    empty_slot_actor_ticks = 0
    for state in training_states:
        for player in state.players:
            hand = tuple(int(value) for value in player.hand)
            if (
                len(hand) != 4
                or any(value < -1 or value > 7 for value in hand)
                or len({value for value in hand if value >= 0})
                != sum(value >= 0 for value in hand)
            ):
                return None
            visible = [value for value in hand if value >= 0]
            if any((int(player.side), value) not in mask_keys for value in visible):
                return None
            if len(visible) < 4:
                empty_slot_actor_ticks += 1
                if player.next_deck_index not in range(8):
                    return None
            retained_actor_ticks += 1
            visible_slot_references += len(visible)
    safe_deploy_labels = sum(
        row.get("accepted") is True
        and row.get("type") == "play"
        and int(row["execution_tick"]) < failure_execution_tick
        for row in prefix.action_acceptance_sequence
    )
    if (
        prefix.deployment_mask_label_rejections
        != (1 if mask_invalid_censor is not None else 0)
        or prefix.deployment_mask_label_checks < safe_deploy_labels
        or not prefix.deployment_mask_payloads
    ):
        return None
    extent = {
        "schema_version": 1,
        "kind": REPLAY_EXTENT_KIND,
        "extent": "valid_prefix",
        "training_admission": (
            "actor_bc_mask_invalid_censored_prefix_v1"
            if mask_invalid_censor is not None
            else "actor_bc_censored_prefix_v1"
        ),
        "source_episode_complete": False,
        "every_native_tick_present_within_extent": True,
        "fixed_seed_replay": True,
        "chosen_seed": int(prefix.chosen_seed),
        "preflight_semantics_sha256": _semantic_digest(preflight),
        "prefix_replay_semantics_sha256": _semantic_digest(prefix),
        "semantic_match": mask_invalid_censor is None,
        "maskless_reference_semantic_match": (
            None if mask_invalid_censor is None else True
        ),
        "pre_censor_tick_state_parity": (
            None if mask_invalid_censor is None else True
        ),
        "failure_class": failure_class,
        "failure_domain": failure_domain,
        "failure": prefix.failure,
        "failure_source_tick": failure_source_tick,
        "failure_execution_tick": failure_execution_tick,
        "first_invalid_source_event_index": failure_event_index,
        "safe_accepted_event_count": len(accepted_before_boundary),
        "safe_accepted_action_transcript_sha256": hashlib.sha256(
            json.dumps(
                accepted_before_boundary,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "observation_tick_start": int(states[0].tick),
        "observation_tick_stop_exclusive": int(last_tick + 1),
        "action_label_tick_stop_exclusive": int(label_stop),
        "timing_censor_tick_exclusive": int(label_stop),
        "timing_target": "right_censored_at_failure_tick_v1",
        "failure_tick_has_labels": False,
        "terminal_target": "unknown_censored",
        "terminal_validated": False,
        "deployment_masks": "partial_native_visible_hand_complete_v1",
        "censor_provenance": (
            {"kind": "native_semantic_failure_censor_v1"}
            if mask_invalid_censor is None
            else dict(mask_invalid_censor)
        ),
        "mask_coverage": {
            "all_retained_visible_hand_slots_covered": True,
            "retained_ticks": len(training_states),
            "actor_ticks": retained_actor_ticks,
            "visible_slot_references": visible_slot_references,
            "empty_slot_actor_ticks": empty_slot_actor_ticks,
            "captured_slots": int(normalized_masks["captured_slots"]),
            "safe_deploy_labels": safe_deploy_labels,
            "checked_deploy_labels": int(
                prefix.deployment_mask_label_checks
            ),
            "rejected_deploy_labels": 0,
        },
        "trace_batches": int(prefix.tick_trace_batches),
        "trace_complete_frames": int(prefix.tick_trace_complete_frames),
        "trace_incomplete_terminal_frames": int(
            prefix.tick_trace_incomplete_terminal_frames
        ),
        "trace_incomplete_nonterminal_freeze_frames": int(
            prefix.tick_trace_incomplete_nonterminal_freeze_frames
        ),
    }
    return {
        **dict(base),
        REPLAY_EXTENT_METADATA_KEY: extent,
        EPISODE_METADATA_KEY: dict(mask_metadata),
        "every_native_tick_present": True,
    }


def _authenticate_mask_invalid_prefix(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    preflight: NativeReplayResult,
    masked: NativeReplayResult,
    *,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    phase_state: dict[str, Any] | None = None,
) -> MasklessReferenceAttempt | None:
    """Replay maskless once and authenticate the mask-invalid safe prefix."""

    if _unique_mask_rejection_boundary(masked, plan) is None:
        return None
    recorder = RecordingCountingEnv(env)
    if phase_state is not None:
        phase_state["maskless_reference_recorder"] = recorder
    started = time.perf_counter()
    try:
        reference = execute_plan(
            recorder,
            plan,
            template,
            None,
            seed=seed,
            fixed_seed=preflight.chosen_seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            capture_decisions=False,
            ability_branch_choices=None,
            tick_sink=None,
            trace_batch_steps=trace_batch_steps,
            capture_deployment_masks=False,
            collect_tick_states_on_failure=True,
            action_execution_tick_offset=(
                ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
            ),
        )
    finally:
        seconds = time.perf_counter() - started
        if phase_state is not None:
            phase_state["maskless_reference_seconds"] = seconds
    evidence = _mask_invalid_censor_evidence(
        preflight, masked, reference, plan
    )
    return MasklessReferenceAttempt(reference, recorder, seconds, evidence)


def execute_two_phase_plan(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    staged: StagedTickSink,
    prefix_staged: StagedTickSink | None = None,
    *,
    seed: int,
    maximum_seeds_to_test: int,
    trace_batch_steps: int,
    tick_store_metadata: Mapping[str, Any],
    phase_state: dict[str, Any] | None = None,
    maximum_compatible_seeds: int = (
        DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
    ),
) -> TwoPhaseNativeReplay:
    """Cheap semantic preflight followed by a fixed-seed traced replay.

    Rejected candidate preflights never invoke ``trace_train`` or
    ``probe_grid``.  Only the first layout-compatible seed is semantically
    replayed.  It is reset exactly once for the traced full/prefix pass and
    parity is mandatory.
    """
    phase_state = {} if phase_state is None else phase_state
    preflight_recorder = RecordingCountingEnv(env)
    phase_state["preflight_recorder"] = preflight_recorder
    phase_started = time.perf_counter()
    try:
        semantic_selection = execute_bounded_semantic_preflights(
            env,
            preflight_recorder,
            plan,
            template,
            seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            trace_batch_steps=trace_batch_steps,
            maximum_compatible_seeds=maximum_compatible_seeds,
        )
        preflight = semantic_selection.selected
        semantic_seed_audit = semantic_selection.audit
        phase_state["semantic_seed_audit"] = semantic_seed_audit
    finally:
        preflight_seconds = time.perf_counter() - phase_started
        phase_state["preflight_seconds"] = preflight_seconds
    if (
        preflight.tick_trace_batches != 0
        or preflight.tick_trace_complete_frames != 0
        or preflight.deployment_mask_probe_rpc_count != 0
        or preflight_recorder.trace_history
        or preflight_recorder.native_deployment_mask_probes_attempted != 0
    ):
        raise RuntimeError(
            "native preflight unexpectedly produced Tick trace or mask probes"
        )
    if not preflight.teacher_forced_success:
        prefix_result: NativeReplayResult | None = None
        prefix_recorder: RecordingCountingEnv | None = None
        prefix_seconds = 0.0
        prefix_semantic_diff: dict[str, Any] | None = None
        prefix_was_staged = False
        if _traceable_semantic_prefix(preflight):
            prefix_recorder = RecordingCountingEnv(env)
            phase_state["failure_prefix_recorder"] = prefix_recorder
            phase_state["failure_prefix_executed"] = True
            phase_started = time.perf_counter()
            try:
                prefix_result = execute_plan(
                    prefix_recorder,
                    plan,
                    template,
                    None,
                    seed=seed,
                    fixed_seed=preflight.chosen_seed,
                    maximum_seeds_to_test=maximum_seeds_to_test,
                    capture_decisions=False,
                    ability_branch_choices=None,
                    tick_sink=None,
                    trace_batch_steps=trace_batch_steps,
                    capture_deployment_masks=True,
                    collect_tick_states_on_failure=True,
                    action_execution_tick_offset=(
                        ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
                    ),
                )
            finally:
                prefix_seconds = time.perf_counter() - phase_started
                phase_state["failure_prefix_seconds"] = prefix_seconds
            mask_invalid = _authenticate_mask_invalid_prefix(
                env,
                plan,
                template,
                preflight,
                prefix_result,
                seed=seed,
                maximum_seeds_to_test=maximum_seeds_to_test,
                trace_batch_steps=trace_batch_steps,
                phase_state=phase_state,
            )
            mask_invalid_evidence = None
            if mask_invalid is None:
                prefix_semantic_diff = diff_native_replay_semantics(
                    preflight, prefix_result
                )
            else:
                reference = mask_invalid.result
                reference_recorder = mask_invalid.recorder
                mask_invalid_evidence = mask_invalid.evidence
                prefix_semantic_diff = (
                    None
                    if mask_invalid_evidence is not None
                    else diff_native_replay_semantics(preflight, prefix_result)
                )
                phase_state["mask_invalid_reference_semantics_sha256"] = (
                    _semantic_digest(reference)
                )
                phase_state["mask_invalid_reference_metrics"] = (
                    reference_recorder.metrics()
                )
            extent_metadata = (
                None
                if prefix_semantic_diff or prefix_result.teacher_forced_success
                else _prefix_extent_metadata(
                    preflight,
                    prefix_result,
                    mask_invalid_censor=mask_invalid_evidence,
                    base={
                        **dict(tick_store_metadata),
                        "native_execution_pipeline": {
                            "contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
                            "mode": NATIVE_PREFLIGHT_MODE,
                            "preflight_chosen_seed": preflight.chosen_seed,
                            "preflight_semantics_sha256": _semantic_digest(
                                preflight
                            ),
                            "semantic_seed_selection": semantic_seed_audit,
                        },
                    },
                )
            )
            if extent_metadata is not None and prefix_staged is not None:
                prefix_staged.stage_deployment_masks(
                    metadata=prefix_result.deployment_mask_metadata,
                    payloads=prefix_result.deployment_mask_payloads,
                    require_complete=False,
                )
                prefix_staged.append(
                    plan.battle_tag,
                    prefix_result.collected_tick_states,
                    extent_metadata,
                )
                prefix_was_staged = True
            return TwoPhaseNativeReplay(
                preflight=preflight,
                full_trace=(
                    prefix_result if mask_invalid_evidence is not None else None
                ),
                preflight_recorder=preflight_recorder,
                full_trace_recorder=(
                    prefix_recorder if mask_invalid_evidence is not None else None
                ),
                preflight_seconds=preflight_seconds,
                full_trace_seconds=(
                    prefix_seconds if mask_invalid_evidence is not None else 0.0
                ),
                semantic_diff=prefix_semantic_diff,
                failure_prefix=prefix_result,
                failure_prefix_recorder=(
                    None if mask_invalid_evidence is not None else prefix_recorder
                ),
                failure_prefix_seconds=(
                    0.0 if mask_invalid_evidence is not None else prefix_seconds
                ),
                failure_prefix_staged=prefix_was_staged,
                semantic_seed_audit=semantic_seed_audit,
                mask_invalid_prefix=mask_invalid_evidence is not None,
                mask_invalid_semantic_diff=(
                    diff_native_replay_semantics(preflight, prefix_result)
                    if mask_invalid_evidence is not None else None
                ),
                maskless_reference=(
                    None if mask_invalid is None else mask_invalid.result
                ),
                maskless_reference_recorder=(
                    None if mask_invalid is None else mask_invalid.recorder
                ),
                maskless_reference_seconds=(
                    0.0 if mask_invalid is None else mask_invalid.seconds
                ),
            )

    full_trace_recorder = RecordingCountingEnv(env)
    phase_state["full_trace_recorder"] = full_trace_recorder
    phase_state["full_trace_executed"] = True
    full_metadata = {
        **dict(tick_store_metadata),
        "native_execution_pipeline": {
            "contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
            "mode": NATIVE_PREFLIGHT_MODE,
            "preflight_chosen_seed": preflight.chosen_seed,
            "preflight_seeds_tested": semantic_seed_audit["raw_seeds_scanned"],
            "preflight_semantics_sha256": _semantic_digest(preflight),
            "semantic_seed_selection": semantic_seed_audit,
        },
    }
    phase_started = time.perf_counter()
    try:
        full_trace = execute_plan(
            full_trace_recorder,
            plan,
            template,
            None,
            seed=seed,
            fixed_seed=preflight.chosen_seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            capture_decisions=False,
            ability_branch_choices=None,
            tick_sink=staged,
            tick_store_metadata=full_metadata,
            trace_batch_steps=trace_batch_steps,
            capture_deployment_masks=True,
            collect_tick_states_on_failure=True,
            action_execution_tick_offset=(
                ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
            ),
        )
    finally:
        full_trace_seconds = time.perf_counter() - phase_started
        phase_state["full_trace_seconds"] = full_trace_seconds
    mask_invalid = _authenticate_mask_invalid_prefix(
        env,
        plan,
        template,
        preflight,
        full_trace,
        seed=seed,
        maximum_seeds_to_test=maximum_seeds_to_test,
        trace_batch_steps=trace_batch_steps,
        phase_state=phase_state,
    )
    if mask_invalid is not None and mask_invalid.evidence is not None:
        reference = mask_invalid.result
        reference_recorder = mask_invalid.recorder
        mask_invalid_evidence = mask_invalid.evidence
        phase_state["mask_invalid_reference_semantics_sha256"] = (
            _semantic_digest(reference)
        )
        phase_state["mask_invalid_reference_metrics"] = (
            reference_recorder.metrics()
        )
        extent_metadata = _prefix_extent_metadata(
            preflight,
            full_trace,
            mask_invalid_censor=mask_invalid_evidence,
            base={
                **dict(tick_store_metadata),
                "native_execution_pipeline": {
                    "contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
                    "mode": NATIVE_PREFLIGHT_MODE,
                    "preflight_chosen_seed": preflight.chosen_seed,
                    "preflight_semantics_sha256": _semantic_digest(preflight),
                    "semantic_seed_selection": semantic_seed_audit,
                },
            },
        )
        if extent_metadata is not None and prefix_staged is not None:
            prefix_staged.stage_deployment_masks(
                metadata=full_trace.deployment_mask_metadata,
                payloads=full_trace.deployment_mask_payloads,
                require_complete=False,
            )
            prefix_staged.append(
                plan.battle_tag,
                full_trace.collected_tick_states,
                extent_metadata,
            )
            return TwoPhaseNativeReplay(
                preflight=preflight,
                full_trace=full_trace,
                preflight_recorder=preflight_recorder,
                full_trace_recorder=full_trace_recorder,
                preflight_seconds=preflight_seconds,
                full_trace_seconds=full_trace_seconds,
                semantic_diff=None,
                failure_prefix=full_trace,
                failure_prefix_recorder=None,
                failure_prefix_seconds=0.0,
                failure_prefix_staged=True,
                semantic_seed_audit=semantic_seed_audit,
                mask_invalid_prefix=True,
                mask_invalid_semantic_diff=diff_native_replay_semantics(
                    preflight, full_trace
                ),
                maskless_reference=mask_invalid.result,
                maskless_reference_recorder=mask_invalid.recorder,
                maskless_reference_seconds=mask_invalid.seconds,
            )
    semantic_diff = diff_native_replay_semantics(preflight, full_trace)
    if not semantic_diff and staged.episode is not None:
        staged.episode.metadata["native_execution_pipeline"].update({
            "full_trace_semantics_sha256": _semantic_digest(full_trace),
            "semantic_diff_count": 0,
            "semantic_match": True,
        })
    return TwoPhaseNativeReplay(
        preflight=preflight,
        full_trace=full_trace,
        preflight_recorder=preflight_recorder,
        full_trace_recorder=full_trace_recorder,
        preflight_seconds=preflight_seconds,
        full_trace_seconds=full_trace_seconds,
        semantic_diff=semantic_diff,
        semantic_seed_audit=semantic_seed_audit,
        maskless_reference=(
            None if mask_invalid is None else mask_invalid.result
        ),
        maskless_reference_recorder=(
            None if mask_invalid is None else mask_invalid.recorder
        ),
        maskless_reference_seconds=(
            0.0 if mask_invalid is None else mask_invalid.seconds
        ),
    )


_COUNTING_METRIC_KEYS = (
    "native_action_batches_attempted",
    "native_actions_attempted",
    "native_actions_responded",
    "native_actions_accepted",
    "native_actions_rejected",
    "native_deploy_actions_attempted",
    "native_deploy_actions_accepted",
    "native_ability_actions_attempted",
    "native_ability_actions_accepted",
    "native_action_exceptions",
    "native_deployment_mask_probes_attempted",
    "native_deployment_mask_probes_responded",
    "native_deployment_mask_probe_exceptions",
)


def _combined_phase_metrics(
    preflight: RecordingCountingEnv,
    full_trace: RecordingCountingEnv | None,
    failure_prefix: RecordingCountingEnv | None = None,
    maskless_reference: RecordingCountingEnv | None = None,
) -> dict[str, Any]:
    rows = [preflight.metrics()]
    if full_trace is not None:
        rows.append(full_trace.metrics())
    if failure_prefix is not None:
        rows.append(failure_prefix.metrics())
    if maskless_reference is not None:
        rows.append(maskless_reference.metrics())
    combined = {
        key: sum(int(row[key]) for row in rows)
        for key in _COUNTING_METRIC_KEYS
    }
    combined["native_actions_no_response"] = max(
        0,
        combined["native_actions_attempted"]
        - combined["native_actions_responded"],
    )
    combined["native_action_response_excess"] = max(
        0,
        combined["native_actions_responded"]
        - combined["native_actions_attempted"],
    )
    combined["true_attempted_acceptance_rate"] = (
        combined["native_actions_accepted"]
        / combined["native_actions_attempted"]
        if combined["native_actions_attempted"] else None
    )
    return combined


def execute_task(
    env: Any,
    task: NativeDatasetTask,
    template: Mapping[str, Any],
    sink: WorkerShardSink,
    registry: StoredFrameRegistry,
    prefix_sink: WorkerShardSink | None = None,
    prefix_registry: StoredFrameRegistry | None = None,
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
    preflight_recorder = RecordingCountingEnv(env)
    full_trace_recorder: RecordingCountingEnv | None = None
    failure_prefix_recorder: RecordingCountingEnv | None = None
    maskless_reference_recorder: RecordingCountingEnv | None = None
    plan: BattlePlan | None = None
    preflight_result: NativeReplayResult | None = None
    result: NativeReplayResult | None = None
    two_phase: TwoPhaseNativeReplay | None = None
    staged = StagedTickSink()
    prefix_staged = StagedTickSink()
    error: Exception | None = None
    error_traceback: str | None = None
    stage = "source_sha_verification"
    source_sha_verified = False
    store_entry: dict[str, Any] | None = None
    audit_prefix_store_entry: dict[str, Any] | None = None
    preflight_seconds = 0.0
    full_trace_seconds = 0.0
    failure_prefix_seconds = 0.0
    maskless_reference_seconds = 0.0
    semantic_diff: dict[str, Any] | None = None
    semantic_seed_audit: dict[str, Any] | None = None
    token_coverage_actor_evidence: list[dict[str, Any]] = []
    prefix_token_coverage_actor_evidence: list[dict[str, Any]] = []
    phase_state: dict[str, Any] = {}
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
        stage = "native_preflight"
        two_phase = execute_two_phase_plan(
            env,
            plan,
            template,
            staged,
            prefix_staged=prefix_staged,
            seed=seed,
            maximum_seeds_to_test=maximum_seeds_to_test,
            trace_batch_steps=trace_batch_steps,
            phase_state=phase_state,
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
                "state_provenance": plan.state_provenance,
                "action_provenance": plan.action_provenance,
                "coordinate_provenance": plan.coordinate_provenance,
                "coordinate_audit": asdict(plan.coordinate_audit),
                "ability_provenance": plan.ability_provenance,
                "terminal_provenance": plan.terminal_provenance,
                "numeric_game_mode_id": plan.numeric_game_mode_id,
                "numeric_game_mode_provenance": (
                    plan.numeric_game_mode_provenance
                ),
                "native_execution_game_mode_id": (
                    plan.native_execution_game_mode_id
                ),
                "native_execution_game_mode_provenance": (
                    plan.native_execution_game_mode_provenance
                ),
                "king_tower_levels": tuple(
                    side.king_tower_level for side in plan.sides
                ),
                "king_tower_level_provenance": tuple(
                    side.king_tower_level_provenance for side in plan.sides
                ),
                "battle_index": plan.battle_index,
                "battle_index_provenance": plan.battle_index_provenance,
                "authoritative_contract_game_version": (
                    plan.authoritative_contract_game_version
                ),
                "authoritative_contract_sha256": (
                    plan.authoritative_contract_sha256
                ),
                "authoritative_contract_file_sha256": (
                    plan.authoritative_contract_file_sha256
                ),
                "source_final_tower_hp": tuple(
                    asdict(side.final_tower_hp)  # type: ignore[arg-type]
                    for side in plan.sides
                ) if all(
                    side.final_tower_hp is not None for side in plan.sides
                ) else None,
                "source_actions": len(plan.actions) + len(plan.ability_events),
            },
        )
        preflight_result = two_phase.preflight
        result = two_phase.result
        preflight_recorder = two_phase.preflight_recorder
        full_trace_recorder = two_phase.full_trace_recorder
        failure_prefix_recorder = two_phase.failure_prefix_recorder
        maskless_reference_recorder = two_phase.maskless_reference_recorder
        preflight_seconds = two_phase.preflight_seconds
        full_trace_seconds = two_phase.full_trace_seconds
        failure_prefix_seconds = two_phase.failure_prefix_seconds
        maskless_reference_seconds = two_phase.maskless_reference_seconds
        semantic_diff = two_phase.semantic_diff
        semantic_seed_audit = two_phase.semantic_seed_audit
        if two_phase.failure_prefix_staged:
            stage = "preflight_first_native_difference"
            if semantic_diff:
                stage = "preflight_failure_prefix_semantic_diff"
                raise PreflightFullTraceDivergence(semantic_diff)
            if prefix_staged.episode is None:
                raise RuntimeError("failure prefix was reported staged but is absent")
            prefix_result = two_phase.failure_prefix
            if prefix_result is None:
                raise RuntimeError("failure prefix result is absent")
            prefix_extent = prefix_staged.episode.metadata.get(
                REPLAY_EXTENT_METADATA_KEY
            )
            if not isinstance(prefix_extent, Mapping):
                raise RuntimeError("failure prefix extent is absent")
            prefix_token_coverage_actor_evidence = (
                build_full_success_token_evidence(
                    plan,
                    prefix_result,
                    prefix_staged.episode,
                    prefix_extent=prefix_extent,
                )
            )
            if prefix_sink is None or prefix_registry is None:
                # Unit callers may intentionally omit persistence.  The
                # production worker always supplies the isolated store.
                pass
            else:
                stage = "immutable_prefix_tick_store_commit"
                if commit_guard is not None and not commit_guard():
                    raise RuntimeError(
                        "native task lease ownership was lost before prefix commit"
                    )
                audit_prefix_store_entry = prefix_registry.commit_or_reuse(
                    prefix_sink, prefix_staged.episode
                )
                stage = "preflight_first_native_difference"
        elif not preflight_result.teacher_forced_success:
            stage = "preflight_first_native_difference"
            if semantic_diff:
                stage = "preflight_failure_prefix_semantic_diff"
                raise PreflightFullTraceDivergence(semantic_diff)
        elif semantic_diff:
            stage = "preflight_full_trace_semantic_diff"
            raise PreflightFullTraceDivergence(semantic_diff)
        elif not result.teacher_forced_success:
            # A successful preflight followed by a failed full trace must have
            # appeared in the semantic diff above.  Keep this independent
            # guard fail-closed if that contract is ever weakened by mistake.
            stage = "preflight_full_trace_semantic_diff"
            raise PreflightFullTraceDivergence({
                "unclassified_full_trace_failure": {
                    "preflight": preflight_result.failure,
                    "full_trace": result.failure,
                }
            })
        else:
            stage = "tick_store_postcondition"
            assert full_trace_recorder is not None
            if staged.episode is None:
                raise RuntimeError("successful replay did not stage a Tick episode")
            if len(staged.episode.states) != result.tick_trace_complete_frames:
                raise RuntimeError(
                    "successful replay Tick count differs from complete trace frames"
                )
            if (
                full_trace_recorder.native_actions_attempted != result.source_actions
                or full_trace_recorder.native_actions_accepted != result.source_actions
            ):
                raise RuntimeError(
                    "successful replay action counters differ from source actions"
                )
            if (
                full_trace_recorder.native_deployment_mask_probes_attempted
                != result.deployment_mask_probe_rpc_count
                or full_trace_recorder.native_deployment_mask_probes_responded
                != result.deployment_mask_probe_rpc_count
                or full_trace_recorder.native_deployment_mask_probe_exceptions != 0
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
            token_coverage_actor_evidence = build_full_success_token_evidence(
                plan, result, staged.episode
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
        preflight_recorder = phase_state.get(
            "preflight_recorder", preflight_recorder
        )
        full_trace_recorder = phase_state.get("full_trace_recorder")
        failure_prefix_recorder = phase_state.get("failure_prefix_recorder")
        maskless_reference_recorder = phase_state.get(
            "maskless_reference_recorder"
        )
        preflight_seconds = float(
            phase_state.get("preflight_seconds", preflight_seconds)
        )
        full_trace_seconds = float(
            phase_state.get("full_trace_seconds", full_trace_seconds)
        )
        failure_prefix_seconds = float(
            phase_state.get("failure_prefix_seconds", failure_prefix_seconds)
        )
        maskless_reference_seconds = float(
            phase_state.get(
                "maskless_reference_seconds", maskless_reference_seconds
            )
        )
        semantic_seed_audit = phase_state.get(
            "semantic_seed_audit", semantic_seed_audit
        )

    success = bool(
        error is None
        and result is not None
        and result.teacher_forced_success
        and store_entry is not None
    )
    metrics = _combined_phase_metrics(
        preflight_recorder,
        full_trace_recorder,
        failure_prefix_recorder,
        maskless_reference_recorder,
    )
    active_recorder = (
        full_trace_recorder or failure_prefix_recorder or preflight_recorder
    )
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
            "first_native_rejection": deepcopy(active_recorder.first_rejection),
            "preflight_full_trace_semantic_diff": semantic_diff,
            "logic_freeze": (
                None if result is None else result.logic_freeze_diagnostic
            ),
        }
    failure_domain = _failure_domain(failure_class)
    capture_result = (
        two_phase.failure_prefix
        if (
            two_phase is not None
            and two_phase.failure_prefix is not None
            and audit_prefix_store_entry is not None
        )
        else result
    )
    coordinate_provenance = (
        None if plan is None else plan.coordinate_provenance
    )
    coordinate_audit = (
        None if plan is None else asdict(plan.coordinate_audit)
    )
    preflight_metrics = preflight_recorder.metrics()
    full_trace_metrics = (
        None if full_trace_recorder is None else full_trace_recorder.metrics()
    )
    full_trace_executed = full_trace_recorder is not None
    avoided_trace_ticks = (
        int(preflight_result.native_ticks_advanced)
        if preflight_result is not None and not full_trace_executed
        else 0
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
        "token_coverage_actor_evidence": (
            token_coverage_actor_evidence if success else []
        ),
        "prefix_token_coverage_actor_evidence": (
            prefix_token_coverage_actor_evidence
            if audit_prefix_store_entry is not None else []
        ),
        "failure_class": failure_class,
        "failure_domain": failure_domain,
        "failure": failure,
        "first_difference": first_difference,
        "planned_deploy_actions": task.deployment_actions,
        "planned_ability_actions": task.ability_events_observed,
        "planned_actions": task.deployment_actions + task.ability_events_observed,
        "native_preflight_contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
        "native_execution_pipeline_mode": NATIVE_PREFLIGHT_MODE,
        "preflight_teacher_forced_success": (
            None
            if preflight_result is None
            else preflight_result.teacher_forced_success
        ),
        "preflight_seconds": preflight_seconds,
        "full_trace_executed": full_trace_executed,
        "full_trace_seconds": full_trace_seconds,
        "failure_prefix_executed": failure_prefix_recorder is not None,
        "failure_prefix_seconds": failure_prefix_seconds,
        "maskless_reference_executed": maskless_reference_recorder is not None,
        "maskless_reference_seconds": maskless_reference_seconds,
        "maskless_reference_native_ticks_advanced": (
            0
            if two_phase is None or two_phase.maskless_reference is None
            else two_phase.maskless_reference.native_ticks_advanced
        ),
        "maskless_reference_semantics_sha256": (
            None
            if two_phase is None or two_phase.maskless_reference is None
            else _semantic_digest(two_phase.maskless_reference)
        ),
        "failure_prefix_semantic_match": (
            None
            if two_phase is None or two_phase.failure_prefix is None
            else (
                False
                if two_phase.mask_invalid_prefix
                else not bool(two_phase.semantic_diff)
            )
        ),
        "mask_invalid_censor_validated": bool(
            two_phase is not None
            and two_phase.mask_invalid_prefix
            and two_phase.failure_prefix_staged
            and not two_phase.semantic_diff
        ),
        "mask_invalid_full_semantic_diff": (
            None if two_phase is None else two_phase.mask_invalid_semantic_diff
        ),
        "failure_prefix_tick_count": (
            0
            if prefix_staged.episode is None
            else len(prefix_staged.episode.states)
        ),
        "avoided_trace_ticks": avoided_trace_ticks,
        "preflight_native_ticks_advanced": (
            0
            if preflight_result is None
            else preflight_result.native_ticks_advanced
        ),
        "full_trace_native_ticks_advanced": (
            0
            if two_phase is None or two_phase.full_trace is None
            else two_phase.full_trace.native_ticks_advanced
        ),
        "preflight_chosen_seed": (
            None if preflight_result is None else preflight_result.chosen_seed
        ),
        "preflight_seeds_tested": (
            0
            if semantic_seed_audit is None
            else int(semantic_seed_audit.get("raw_seeds_scanned") or 0)
        ),
        "semantic_seed_preflight": semantic_seed_audit,
        "full_trace_layout_resolution_mode": (
            None if result is None or not full_trace_executed
            else result.layout_resolution_mode
        ),
        "preflight_native_action_metrics": preflight_metrics,
        "full_trace_native_action_metrics": full_trace_metrics,
        "failure_prefix_native_action_metrics": (
            None
            if failure_prefix_recorder is None
            else failure_prefix_recorder.metrics()
        ),
        "maskless_reference_native_action_metrics": (
            None
            if maskless_reference_recorder is None
            else maskless_reference_recorder.metrics()
        ),
        "maskless_reference_action_acceptance_sequence": (
            []
            if two_phase is None or two_phase.maskless_reference is None
            else [
                dict(row)
                for row in two_phase.maskless_reference.action_acceptance_sequence
            ]
        ),
        "preflight_action_acceptance_sequence": (
            []
            if preflight_result is None
            else [dict(row) for row in preflight_result.action_acceptance_sequence]
        ),
        "full_trace_action_acceptance_sequence": (
            []
            if two_phase is None or two_phase.full_trace is None
            else [
                dict(row)
                for row in two_phase.full_trace.action_acceptance_sequence
            ]
        ),
        "preflight_full_trace_semantic_match": (
            None
            if (
                not full_trace_executed
                or (two_phase is not None and two_phase.mask_invalid_prefix)
            )
            else (
                not bool(semantic_diff)
            )
        ),
        "preflight_full_trace_semantic_diff": (
            None
            if two_phase is not None and two_phase.mask_invalid_prefix
            else semantic_diff
        ),
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
        "chosen_seed": (
            None if preflight_result is None else preflight_result.chosen_seed
        ),
        "seeds_tested": (
            0
            if semantic_seed_audit is None
            else int(semantic_seed_audit.get("raw_seeds_scanned") or 0)
        ),
        "seed_search_cache_hit": (
            False
            if preflight_result is None
            else preflight_result.seed_search_cache_hit
        ),
        "source_seed_recovered": (
            False
            if preflight_result is None
            else preflight_result.source_seed_recovered
        ),
        "native_ticks_advanced": (
            0 if result is None else result.native_ticks_advanced
        ),
        "collected_tick_state_count": (
            0 if result is None else len(result.collected_tick_states)
        ),
        "tick_trace_complete_frames": (
            0 if capture_result is None else capture_result.tick_trace_complete_frames
        ),
        "tick_trace_incomplete_terminal_frames": (
            0 if capture_result is None
            else capture_result.tick_trace_incomplete_terminal_frames
        ),
        "tick_trace_incomplete_nonterminal_freeze_frames": (
            0 if capture_result is None
            else capture_result.tick_trace_incomplete_nonterminal_freeze_frames
        ),
        "deployment_mask_probe_seconds": (
            0.0 if capture_result is None
            else capture_result.deployment_mask_probe_seconds
        ),
        "deployment_mask_probe_rpc_count": (
            0 if capture_result is None
            else capture_result.deployment_mask_probe_rpc_count
        ),
        "deployment_mask_slots_captured": (
            0 if capture_result is None
            else capture_result.deployment_mask_slots_captured
        ),
        "deployment_mask_base_probe_rpc_count": (
            0
            if capture_result is None
            else capture_result.deployment_mask_base_probe_rpc_count
        ),
        "deployment_mask_dynamic_label_probe_rpc_count": (
            0
            if capture_result is None
            else capture_result.deployment_mask_dynamic_label_probe_rpc_count
        ),
        "deployment_mask_capture_complete": (
            False if capture_result is None
            else capture_result.deployment_mask_capture_complete
        ),
        "deployment_mask_metadata": (
            None if capture_result is None else capture_result.deployment_mask_metadata
        ),
        "deployment_mask_label_checks": (
            0 if capture_result is None else capture_result.deployment_mask_label_checks
        ),
        "deployment_mask_label_rejections": (
            0 if capture_result is None
            else capture_result.deployment_mask_label_rejections
        ),
        "deployment_mask_first_label_rejection": (
            None
            if capture_result is None
            else capture_result.deployment_mask_first_label_rejection
        ),
        "deployment_mask_label_rejection_sequence": (
            []
            if capture_result is None
            else [
                dict(row)
                for row in getattr(
                    capture_result,
                    "deployment_mask_label_rejection_sequence",
                    (),
                )
            ]
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
        "audit_prefix_tick_store_entry": audit_prefix_store_entry,
        "audit_prefix_extent": (
            None
            if prefix_staged.episode is None
            else prefix_staged.episode.metadata.get(
                REPLAY_EXTENT_METADATA_KEY
            )
        ),
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
        "preflight_native_result": (
            None if preflight_result is None else preflight_result.json()
        ),
        "full_trace_native_result": (
            None
            if two_phase is None or two_phase.full_trace is None
            else two_phase.full_trace.json()
        ),
        "failure_prefix_native_result": (
            None
            if two_phase is None or two_phase.failure_prefix is None
            else two_phase.failure_prefix.json()
        ),
        "maskless_reference_native_result": (
            None
            if two_phase is None or two_phase.maskless_reference is None
            else two_phase.maskless_reference.json()
        ),
        "preflight_full_trace_semantic_diff": semantic_diff,
        "native_boundary_snapshot": active_recorder.snapshot(),
        "preflight_native_boundary_snapshot": preflight_recorder.snapshot(),
        "full_trace_native_boundary_snapshot": (
            None
            if full_trace_recorder is None
            else full_trace_recorder.snapshot()
        ),
        "failure_prefix_native_boundary_snapshot": (
            None
            if failure_prefix_recorder is None
            else failure_prefix_recorder.snapshot()
        ),
        "maskless_reference_native_boundary_snapshot": (
            None
            if maskless_reference_recorder is None
            else maskless_reference_recorder.snapshot()
        ),
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
    prefix_registry: StoredFrameRegistry,
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
    prefix_sink: WorkerShardSink | None = None
    manifests: list[dict[str, Any]] = []
    recovered_final_shards = 0
    recovered_prefix_shards = 0
    prefix_manifests: list[dict[str, Any]] = []
    prefix_episodes = prefix_ticks = 0
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
        recovered_prefix_shards = recover_unmanifested_final_shards(
            output_root / AUDIT_PREFIX_DIRECTORY, worker_id + "-prefix"
        )
        prefix_sink = WorkerShardSink(
            output_root / AUDIT_PREFIX_DIRECTORY,
            worker_id + "-prefix",
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
                            prefix_sink,
                            prefix_registry,
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
                            prefix_entry = record.get(
                                "audit_prefix_tick_store_entry"
                            )
                            if isinstance(prefix_entry, Mapping):
                                prefix_episodes += 1
                                prefix_ticks += int(prefix_entry["ticks"])
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
        if prefix_sink is not None:
            try:
                prefix_manifests = prefix_sink.finalize()
            except Exception as error:
                if worker_error is None:
                    worker_error = (
                        f"prefix_shard_finalize_{type(error).__name__}: {error}"
                    )
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
        "audit_prefix_episodes": prefix_episodes,
        "audit_prefix_ticks": prefix_ticks,
        "wall_seconds": time.perf_counter() - started,
        "worker_error": worker_error,
        "recovered_final_shards": recovered_final_shards,
        "recovered_prefix_shards": recovered_prefix_shards,
        "newly_finalized_shards": manifests,
        "newly_finalized_prefix_shards": prefix_manifests,
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


def _result_prefix_integrity(row: Mapping[str, Any]) -> bool:
    extent = row.get("audit_prefix_extent")
    if not isinstance(extent, Mapping):
        return False
    prefix_entry = row.get("audit_prefix_tick_store_entry")
    common = bool(
        isinstance(prefix_entry, Mapping)
        and prefix_extent_common_contract_valid(
            extent, tick_count=int(prefix_entry.get("ticks", -1))
        )
        and int(row.get("failure_prefix_tick_count") or 0) > 0
        and int(row.get("deployment_mask_probe_rpc_count") or 0) > 0
        and int(row.get("native_deployment_mask_probes_attempted") or 0)
        == int(row.get("deployment_mask_probe_rpc_count") or 0)
        and int(row.get("native_deployment_mask_probe_exceptions") or 0) == 0
    )
    if not common:
        return False
    if extent.get("training_admission") == "actor_bc_censored_prefix_v1":
        return bool(
            extent.get("failure_domain") == "semantic"
            and extent.get("semantic_match") is True
            and row.get("failure_domain") == "semantic"
            and row.get("failure_prefix_semantic_match") is True
            and int(row.get("deployment_mask_label_rejections") or 0) == 0
        )
    if extent.get("training_admission") != (
        "actor_bc_mask_invalid_censored_prefix_v1"
    ):
        return False
    provenance = extent.get("censor_provenance")
    actions = row.get("full_trace_action_acceptance_sequence")
    preflight_actions = row.get("preflight_action_acceptance_sequence")
    reference_actions = row.get(
        "maskless_reference_action_acceptance_sequence"
    )
    metrics = row.get("full_trace_native_action_metrics")
    reference_metrics = row.get("maskless_reference_native_action_metrics")
    rejection = row.get("deployment_mask_first_label_rejection")
    rejection_sequence = row.get("deployment_mask_label_rejection_sequence")
    if not mask_invalid_censor_provenance_valid(provenance):
        return False
    boundary_row = (
        provenance.get("preflight_boundary_accepted_action")
        if isinstance(provenance, Mapping) else None
    )
    seed_audit = row.get("semantic_seed_preflight")
    candidates = (
        seed_audit.get("candidates")
        if isinstance(seed_audit, Mapping) else None
    )
    selected = (
        candidates[0]
        if isinstance(candidates, list)
        and len(candidates) == 1
        and isinstance(candidates[0], Mapping)
        else None
    )
    if not isinstance(reference_actions, list):
        return False
    reference_accepted = sum(
        action.get("accepted") is True
        for action in reference_actions
        if isinstance(action, Mapping)
    )
    reference_rejected = sum(
        action.get("accepted") is False
        for action in reference_actions
        if isinstance(action, Mapping)
    )
    if reference_accepted + reference_rejected != len(reference_actions):
        return False
    try:
        return bool(
            extent.get("failure_class") == MASK_INVALID_FAILURE_CLASS
        and extent.get("failure_domain") == MASK_INVALID_FAILURE_DOMAIN
        and extent.get("semantic_match") is False
        and extent.get("maskless_reference_semantic_match") is True
        and extent.get("pre_censor_tick_state_parity") is True
        and row.get("failure_class") == MASK_INVALID_FAILURE_CLASS
        and row.get("failure_domain") == MASK_INVALID_FAILURE_DOMAIN
        and row.get("failure_prefix_semantic_match") is False
        and row.get("preflight_full_trace_semantic_match") is None
        and row.get("preflight_full_trace_semantic_diff") is None
        and row.get("mask_invalid_censor_validated") is True
        and row.get("maskless_reference_executed") is True
        and float(row.get("maskless_reference_seconds", -1.0)) >= 0.0
        and int(row.get("deployment_mask_label_rejections") or 0) == 1
        and isinstance(rejection, Mapping)
        and rejection_sequence == [rejection]
        and rejection.get("source_event_index")
        == provenance["rejected_source_event_index"]
        and rejection.get("content_sha256") == provenance["mask_content_sha256"]
        and rejection.get("reasons")
        == ["position_not_in_derived_native_mask"]
        and rejection.get("locked_pocket") == provenance["locked_pocket"]
        and isinstance(selected, Mapping)
        and selected.get("semantics_sha256")
        == provenance["preflight_semantics_sha256"]
        and isinstance(preflight_actions, list)
        and isinstance(boundary_row, Mapping)
        and [
            action for action in preflight_actions
            if action.get("source_event_index")
            == provenance["rejected_source_event_index"]
        ] == [boundary_row]
        and _canonical_rows_sha256(boundary_row)
        == provenance["preflight_boundary_accepted_action_sha256"]
        and isinstance(actions, list)
        and len(actions) == int(provenance["safe_action_count"])
        and _canonical_rows_sha256(actions)
        == provenance["safe_action_transcript_sha256"]
        and isinstance(metrics, Mapping)
        and int(metrics.get("native_actions_attempted", -1))
        == int(provenance["safe_action_count"])
        and int(metrics.get("native_actions_responded", -1))
        == int(provenance["safe_action_count"])
        and int(metrics.get("native_actions_accepted", -1))
        == int(provenance["safe_action_count"])
        and int(metrics.get("native_actions_rejected", -1)) == 0
        and int(metrics.get("native_actions_no_response", -1)) == 0
        and int(metrics.get("native_action_exceptions", -1)) == 0
        and isinstance(reference_actions, list)
        and reference_actions == preflight_actions
        and row.get("maskless_reference_semantics_sha256")
        == provenance["maskless_reference_semantics_sha256"]
        and isinstance(reference_metrics, Mapping)
        and int(reference_metrics.get("native_actions_attempted", -1))
        == len(reference_actions)
        and int(reference_metrics.get("native_actions_responded", -1))
        == len(reference_actions)
        and int(reference_metrics.get("native_actions_accepted", -1))
        == reference_accepted
        and int(reference_metrics.get("native_actions_rejected", -1))
        == reference_rejected
        and int(reference_metrics.get("native_actions_no_response", -1)) == 0
        and int(reference_metrics.get("native_action_exceptions", -1)) == 0
        and str(row.get("failure") or "")
        == "derived_deployment_mask_rejected_source_event_"
        + str(provenance["rejected_source_event_index"])
        )
    except (KeyError, TypeError, ValueError):
        return False


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
    prefixes = [
        row for row in failures
        if isinstance(row.get("audit_prefix_tick_store_entry"), Mapping)
    ]
    unframed = [row for row in failures if row not in prefixes]
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
    preflight_seconds = sum(
        float(row.get("preflight_seconds") or 0.0) for row in results
    )
    full_trace_seconds = sum(
        float(row.get("full_trace_seconds") or 0.0) for row in results
    )
    maskless_reference_seconds = sum(
        float(row.get("maskless_reference_seconds") or 0.0)
        for row in results
    )
    avoided_trace_ticks = sum(
        int(row.get("avoided_trace_ticks") or 0) for row in results
    )
    preflight_rejections = sum(
        row.get("preflight_teacher_forced_success") is False for row in results
    )
    full_trace_executions = sum(
        row.get("full_trace_executed") is True for row in results
    )
    maskless_reference_executions = sum(
        row.get("maskless_reference_executed") is True for row in results
    )
    maskless_reference_ticks = sum(
        int(row.get("maskless_reference_native_ticks_advanced") or 0)
        for row in results
    )
    mask_invalid_censored_prefixes = sum(
        row.get("mask_invalid_censor_validated") is True for row in results
    )
    semantic_divergences = sum(
        bool(row.get("preflight_full_trace_semantic_diff")) for row in results
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
    audit_prefix_ticks = sum(
        int(row["audit_prefix_tick_store_entry"].get("ticks", 0))
        for row in prefixes
    )
    prefix_integrity = all(_result_prefix_integrity(row) for row in prefixes)
    audit_tick_coverage_complete = bool(
        len(successes) + len(prefixes) == len(tasks)
        and not unframed
        and prefix_integrity
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
    two_phase_integrity = all(
        int(row.get("native_preflight_contract_version", -1))
        == NATIVE_PREFLIGHT_CONTRACT_VERSION
        and row.get("native_execution_pipeline_mode") == NATIVE_PREFLIGHT_MODE
        and (
            row.get("preflight_teacher_forced_success") is None
            or semantic_seed_audit_valid(
                row.get("semantic_seed_preflight"),
                chosen_seed=row.get("preflight_chosen_seed"),
                teacher_forced_success=row.get(
                    "preflight_teacher_forced_success"
                ),
            )
        )
        and (
            (
                row.get("preflight_teacher_forced_success") is False
                and row.get("full_trace_executed") is False
                and row.get("failure_prefix_executed") is not True
                and int(row.get("tick_trace_complete_frames") or 0) == 0
                and int(row.get("deployment_mask_probe_rpc_count") or 0) == 0
            )
            or (
                row.get("preflight_teacher_forced_success") is False
                and row.get("full_trace_executed") is False
                and row.get("failure_prefix_executed") is True
                and row.get("failure_prefix_semantic_match") is True
                and isinstance(
                    row.get("audit_prefix_tick_store_entry"), Mapping
                )
                and _result_prefix_integrity(row)
            )
            or (
                row.get("preflight_teacher_forced_success") is True
                and row.get("full_trace_executed") is True
                and row.get("preflight_full_trace_semantic_match") is True
                and not row.get("preflight_full_trace_semantic_diff")
                and row.get("full_trace_layout_resolution_mode")
                == "fixed_preflight_seed_replay"
            )
            or (
                row.get("preflight_teacher_forced_success") is True
                and row.get("full_trace_executed") is True
                and row.get("preflight_full_trace_semantic_match") is None
                and row.get("preflight_full_trace_semantic_diff") is None
                and row.get("mask_invalid_censor_validated") is True
                and isinstance(
                    row.get("mask_invalid_full_semantic_diff"), Mapping
                )
                and row.get("full_trace_layout_resolution_mode")
                == "fixed_preflight_seed_replay"
                and _result_prefix_integrity(row)
            )
            or (
                row.get("preflight_teacher_forced_success") is False
                and row.get("full_trace_executed") is True
                and row.get("preflight_full_trace_semantic_match") is None
                and row.get("preflight_full_trace_semantic_diff") is None
                and row.get("mask_invalid_censor_validated") is True
                and isinstance(
                    row.get("mask_invalid_full_semantic_diff"), Mapping
                )
                and row.get("full_trace_layout_resolution_mode")
                == "fixed_preflight_seed_replay"
                and _result_prefix_integrity(row)
            )
            or row.get("preflight_teacher_forced_success") is None
        )
        for row in results
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
        "native_preflight_contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
        "native_execution_pipeline_mode": NATIVE_PREFLIGHT_MODE,
        "preflight_seconds": preflight_seconds,
        "full_trace_seconds": full_trace_seconds,
        "maskless_reference_seconds": maskless_reference_seconds,
        "preflight_rejections": preflight_rejections,
        "full_trace_executions": full_trace_executions,
        "maskless_reference_executions": maskless_reference_executions,
        "maskless_reference_native_ticks_advanced": maskless_reference_ticks,
        "mask_invalid_censored_prefixes": mask_invalid_censored_prefixes,
        "avoided_trace_ticks": avoided_trace_ticks,
        "preflight_full_trace_semantic_divergences": semantic_divergences,
        "two_phase_preflight_integrity": two_phase_integrity,
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
        "audit_prefix_episodes": len(prefixes),
        "audit_prefix_ticks": audit_prefix_ticks,
        "audit_tick_episodes": len(successes) + len(prefixes),
        "unframed_episodes": len(unframed),
        "audit_tick_coverage_rate": _ratio(
            len(successes) + len(prefixes), len(tasks)
        ),
        "audit_prefix_integrity": prefix_integrity,
        "audit_tick_coverage_complete": audit_tick_coverage_complete,
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
            and two_phase_integrity
            and semantic_divergences == 0
            and audit_tick_coverage_complete
        ),
        "source_json_copied": False,
        "semantic_rejections_are_expected_subset_evidence": True,
    }


def _physical_frame_reader(
    output_root: Path,
    record: Mapping[str, Any],
    *,
    prefix: bool = False,
) -> EpisodeReader | None:
    entry = record.get(
        "audit_prefix_tick_store_entry" if prefix else "tick_store_entry"
    )
    if not isinstance(entry, Mapping):
        return None
    stem = str(entry.get("shard") or "")
    paths = [
        output_root
        / (AUDIT_PREFIX_DIRECTORY if prefix else "shards")
        / f"{stem}.crts",
        output_root
        / (AUDIT_PREFIX_DIRECTORY if prefix else "shards")
        / f"{stem}.crts.partial",
    ]
    path = next((item for item in paths if item.exists()), None)
    if path is None:
        return None
    offset = int(entry["offset"])
    with path.open("rb") as source:
        source.seek(offset)
        header = source.read(FRAME_HEADER.size)
        if len(header) != FRAME_HEADER.size:
            return None
        magic, payload_size, payload_crc, tag_hash, ticks, _reserved = (
            FRAME_HEADER.unpack(header)
        )
        payload = source.read(payload_size)
    import zlib
    tag = str(record.get("battle_tag") or "")
    try:
        reader = EpisodeReader(payload)
    except Exception:
        return None
    if not (
        magic == FRAME_MAGIC
        and payload_size > 0
        and len(payload) == payload_size
        and zlib.crc32(payload) == payload_crc
        and tag_hash == _tag_hash(tag)
        and hashlib.sha256(payload).hexdigest() == entry["payload_sha256"]
        and int(payload_size) == int(entry.get("payload_size", -1))
        and int(entry.get("frame_size", -1))
        == FRAME_HEADER.size + payload_size
        and int(ticks) == int(entry["ticks"]) == reader.tick_count
        and str(reader.metadata.get("battle_tag") or "") == tag
    ):
        return None
    return reader


def _physical_frame_valid(
    output_root: Path,
    record: Mapping[str, Any],
    *,
    prefix: bool = False,
) -> bool:
    return _physical_frame_reader(output_root, record, prefix=prefix) is not None


def reconcile_result_files(output_root: Path, queue_path: Path) -> int:
    """Finish the DB mutation if a crash occurred after atomic result write."""
    reconciled = 0
    with TickStoreWorkQueue(queue_path) as queue:
        for path in sorted((output_root / "results").glob("*.json")):
            record = load_json(path)
            if record.get("kind") != RESULT_KIND:
                continue
            tag = str(record["battle_tag"])
            if not native_result_pipeline_contract_valid(record):
                raise RuntimeError(
                    f"native result pipeline contract is stale/invalid: {tag}"
                )
            if isinstance(
                record.get("audit_prefix_tick_store_entry"), Mapping
            ) and not _result_prefix_integrity(record):
                raise RuntimeError(
                    f"native prefix result proof is stale/invalid: {tag}"
                )
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
                if isinstance(
                    record.get("audit_prefix_tick_store_entry"), Mapping
                ):
                    prefix_reader = _physical_frame_reader(
                        output_root, record, prefix=True
                    )
                    if prefix_reader is None:
                        continue
                    metadata = prefix_reader.metadata
                    if (
                        metadata.get(REPLAY_EXTENT_METADATA_KEY)
                        != record.get("audit_prefix_extent")
                        or not native_episode_pipeline_contract_valid(
                            metadata, record
                        )
                        or str(metadata.get("source_sha256") or "")
                        != str(record.get("source_sha256") or "")
                        or int(metadata.get("selection_index", -1))
                        != int(record.get("selection_index", -2))
                        or str(metadata.get("selection_digest") or "")
                        != str(record.get("selection_digest") or "")
                    ):
                        raise RuntimeError(
                            f"native prefix frame/result proof changed: {tag}"
                        )
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


def verify_published_tick_store(
    root: Path,
    *,
    expected_kind: str = STORE_KIND,
) -> dict[str, Any]:
    """Read-only validation of every data/index hash behind the store manifest."""
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("kind") != expected_kind:
        raise RuntimeError("published Tick Store manifest kind changed")
    mask_contract = (manifest.get("metadata") or {}).get(
        "native_deployment_masks"
    )
    mask_store: DeploymentMaskStore | None = None
    referenced_masks: set[str] = set()
    if mask_contract is not None:
        if not isinstance(mask_contract, Mapping):
            raise RuntimeError("published Tick Store mask contract is malformed")
        if (
            mask_contract.get("required") is not True
            or (
                expected_kind == AUDIT_PREFIX_STORE_KIND
                and mask_contract.get("partial") is not True
            )
            or (
                expected_kind != AUDIT_PREFIX_STORE_KIND
                and mask_contract.get("partial") is True
            )
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
    battle_tags: set[str] = set()
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
        for entry in entries:
            tag = str(entry.get("battle_tag") or "")
            if not tag or tag in battle_tags:
                raise RuntimeError("published Tick Store battle tags are not unique")
            battle_tags.add(tag)
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
                        reader.metadata,
                        allow_cached=True,
                        require_complete=(
                            expected_kind != AUDIT_PREFIX_STORE_KIND
                        ),
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
                    if expected_kind == AUDIT_PREFIX_STORE_KIND:
                        states = tuple(reader.iter_ticks())
                        extent = reader.metadata.get(REPLAY_EXTENT_METADATA_KEY)
                        coverage = (
                            extent.get("mask_coverage")
                            if isinstance(extent, Mapping) else None
                        )
                        keys = {
                            (int(item["side"]), int(item["deck_index"]))
                            for item in metadata["entries"]
                        }
                        censor_tick = (
                            int(extent.get("timing_censor_tick_exclusive", -1))
                            if isinstance(extent, Mapping) else -1
                        )
                        training_states = tuple(
                            state for state in states
                            if int(state.tick) < censor_tick
                        )
                        actor_ticks = visible_references = empty_actor_ticks = 0
                        visible_complete = True
                        for state in training_states:
                            for player in state.players:
                                visible = [
                                    int(value) for value in player.hand
                                    if int(value) >= 0
                                ]
                                visible_complete = visible_complete and all(
                                    (int(player.side), value) in keys
                                    for value in visible
                                )
                                actor_ticks += 1
                                visible_references += len(visible)
                                empty_actor_ticks += len(visible) < 4
                        normal_extent = bool(
                            isinstance(extent, Mapping)
                            and extent.get("training_admission")
                            == "actor_bc_censored_prefix_v1"
                            and extent.get("semantic_match") is True
                            and extent.get("failure_domain") == "semantic"
                        )
                        special_extent = bool(
                            isinstance(extent, Mapping)
                            and extent.get("training_admission")
                            == "actor_bc_mask_invalid_censored_prefix_v1"
                            and extent.get("failure_class")
                            == MASK_INVALID_FAILURE_CLASS
                            and extent.get("failure_domain")
                            == MASK_INVALID_FAILURE_DOMAIN
                            and extent.get("semantic_match") is False
                            and extent.get("maskless_reference_semantic_match")
                            is True
                            and extent.get("pre_censor_tick_state_parity") is True
                            and mask_invalid_censor_provenance_valid(
                                extent.get("censor_provenance")
                            )
                        )
                        if (
                            not isinstance(extent, Mapping)
                            or extent.get("kind") != REPLAY_EXTENT_KIND
                            or extent.get("extent") != "valid_prefix"
                            or not (normal_extent or special_extent)
                            or extent.get("source_episode_complete") is not False
                            or extent.get("failure_tick_has_labels") is not False
                            or extent.get("terminal_target") != "unknown_censored"
                            or extent.get("terminal_validated") is not False
                            or extent.get("timing_target")
                            != "right_censored_at_failure_tick_v1"
                            or extent.get("deployment_masks")
                            != "partial_native_visible_hand_complete_v1"
                            or not isinstance(coverage, Mapping)
                            or coverage.get(
                                "all_retained_visible_hand_slots_covered"
                            ) is not True
                            or not visible_complete
                            or int(coverage.get("retained_ticks", -1))
                            != len(training_states)
                            or int(coverage.get("actor_ticks", -1)) != actor_ticks
                            or int(coverage.get("visible_slot_references", -1))
                            != visible_references
                            or int(coverage.get("empty_slot_actor_ticks", -1))
                            != empty_actor_ticks
                            or int(coverage.get("captured_slots", -1))
                            != int(metadata["captured_slots"])
                            or int(coverage.get("rejected_deploy_labels", -1)) != 0
                            or int(extent.get("trace_complete_frames", -1))
                            != len(states)
                            or not states
                            or len(states) != int(entry["ticks"])
                            or int(extent.get("observation_tick_start", -1))
                            != states[0].tick
                            or int(extent.get(
                                "observation_tick_stop_exclusive", -1
                            )) != states[-1].tick + 1
                            or int(extent.get(
                                "action_label_tick_stop_exclusive", -1
                            )) > states[-1].tick + 1
                        ):
                            raise RuntimeError(
                                "published training prefix metadata changed: "
                                f"{entry.get('battle_tag')}"
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
        "battle_tags": sorted(battle_tags),
    }


def verify_published_audit_prefix_store(root: Path) -> dict[str, Any]:
    return verify_published_tick_store(
        root, expected_kind=AUDIT_PREFIX_STORE_KIND
    )


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
    prefix_manifest_path = (
        output_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
    )
    if not all(path.exists() for path in (
        manifest_path, summary_path, results_path, store_manifest_path,
        prefix_manifest_path,
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
        "audit_prefix_tick_store_manifest_sha256": sha256_file(
            prefix_manifest_path
        ),
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
    prefix_physical = verify_published_audit_prefix_store(
        output_root / AUDIT_PREFIX_DIRECTORY
    )
    if int(summary.get("processed_battles", -1)) != len(tasks):
        raise RuntimeError("published summary task count changed")
    if (
        physical["episodes"] != int(summary.get("stored_episodes", -1))
        or physical["ticks"] != int(summary.get("stored_ticks", -1))
        or prefix_physical["episodes"]
        != int(summary.get("audit_prefix_episodes", -1))
        or prefix_physical["ticks"]
        != int(summary.get("audit_prefix_ticks", -1))
        or set(physical["battle_tags"]) & set(prefix_physical["battle_tags"])
        or len(physical["battle_tags"]) + len(prefix_physical["battle_tags"])
        != len(tasks)
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
        prefix_registry = StoredFrameRegistry(
            output_root / AUDIT_PREFIX_DIRECTORY,
            require_complete_masks=False,
        )
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
                    prefix_registry=prefix_registry,
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
                        "dynamic_rule": DYNAMIC_RULE,
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
            prefix_mask_manifest = (
                prefix_registry.deployment_mask_store.build_manifest()
            )
            prefix_mask_manifest_path = (
                output_root / AUDIT_PREFIX_DIRECTORY
                / MASK_STORE_DIRECTORY / "manifest.json"
            )
            prefix_store = build_store_manifest(
                output_root / AUDIT_PREFIX_DIRECTORY,
                source_manifest=selection_path,
                expected_episodes=int(summary["audit_prefix_episodes"]),
                expected_ticks=int(summary["audit_prefix_ticks"]),
                store_kind=AUDIT_PREFIX_STORE_KIND,
                store_metadata={
                    "generator_kind": GENERATOR_KIND,
                    "episode_extent": "valid_prefix",
                    "training_admission": "actor_bc_censored_prefix_v1",
                    "terminal_target": "unknown_censored",
                    "timing_target": "right_censored_at_failure_tick_v1",
                    "native_deployment_masks": {
                        "required": True,
                        "partial": True,
                        "schema_version": 1,
                        "dynamic_rule": DYNAMIC_RULE,
                        "manifest": f"{MASK_STORE_DIRECTORY}/manifest.json",
                        "manifest_sha256": sha256_file(
                            prefix_mask_manifest_path
                        ),
                        "sidecars": int(prefix_mask_manifest["sidecars"]),
                    },
                },
            )
            prefix_manifest_path = (
                output_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
            )
            summary["audit_prefix_stored_bytes"] = int(
                prefix_store["total_bytes"]
            )
            summary["audit_prefix_tick_store_manifest"] = str(
                prefix_manifest_path.resolve()
            )
            summary["audit_prefix_tick_store_manifest_sha256"] = (
                sha256_file(prefix_manifest_path)
            )
            summary["audit_prefix_deployment_mask_store_manifest"] = str(
                prefix_mask_manifest_path.resolve()
            )
            summary["audit_prefix_deployment_mask_store_manifest_sha256"] = (
                sha256_file(prefix_mask_manifest_path)
            )
            summary["audit_prefix_deployment_mask_unique_sidecars"] = int(
                prefix_mask_manifest["sidecars"]
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
                        "audit_prefix_episodes", "audit_prefix_ticks",
                        "audit_tick_episodes", "unframed_episodes",
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
                    "audit_prefix_tick_store_manifest": (
                        f"{AUDIT_PREFIX_DIRECTORY}/manifest.json"
                    ),
                    "audit_prefix_tick_store_manifest_sha256": sha256_file(
                        output_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
                    ),
                },
            }
            atomic_json(output_root / "manifest.json", manifest)
            _atomic_text(
                output_root / "manifest.sha256",
                f"{sha256_file(output_root / 'manifest.json')}  manifest.json\n",
            )
        return summary
