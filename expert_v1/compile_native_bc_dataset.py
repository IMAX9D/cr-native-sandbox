"""Compile immutable native Tick Store episodes into actor-safe BC shards.

This compiler is deliberately downstream of the native generator.  It never
simulates a battle and never repairs source data.  Its inputs are:

* a checksummed ``cr_native_tick_store_v1``;
* an immutable JSONL index whose rows point at authoritative schema-v5 JSON;
* the per-episode, content-addressed native deployment-mask sidecars.

The output uses :mod:`expert_v1.training_v1`'s mmap shard contract.  Work is
partitioned into deterministic, independently atomic shards, so separate
processes may compile different ``worker_index`` values and a stopped run can
resume without rewriting completed shards.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .native_capabilities import ability_cards
from .native_ingest_contract import load_native_ingest_contract
from .native_replay_plan import BattlePlan, compile_battle
from .tick_store_v1.deployment_masks import (
    DeploymentMaskStore,
    resolve_deployment_reference,
    verify_deployment_labels,
)
from .tick_store_v1.shard import SHARD_KIND as TICK_SHARD_KIND
from .tick_store_v1.shard import AUDIT_PREFIX_STORE_KIND
from .tick_store_v1.shard import STORE_KIND as TICK_STORE_KIND
from .tick_store_v1.shard import ShardReader, sha256_file
from .tick_store_v1.schema import ActorTick, TickState, actor_projection
from .training_v1.schema import (
    ARENA_COLUMNS,
    ARENA_ROWS,
    DATASET_KIND,
    GRID_STORAGE,
    POSITION_COUNT,
    POSITION_MASK_BYTES,
    POSITION_MASK_STORAGE,
    SCHEMA_VERSION,
    SHARD_KIND,
    validate_manifest,
    validate_shard,
)
from .token_coverage_v1 import (
    RECEIPT_KIND as TOKEN_COVERAGE_RECEIPT_KIND,
    RECEIPT_SCHEMA_VERSION as TOKEN_COVERAGE_RECEIPT_SCHEMA_VERSION,
    authenticate_generator_ability_evidence,
    build_adaptive_token_quotas,
    build_token_coverage_receipt,
    canonical_json_bytes,
    coverage_receipt_sha256,
    evaluate_token_coverage,
    freeze_source_token_coverage,
    summarize_success_token_coverage,
)


COMPILER_KIND = "cr_native_tick_store_bc_compiler_v4"
PLAN_KIND = "cr_native_tick_store_bc_compile_plan_v4"
ASSIGNMENT_KIND = "cr_native_tick_store_bc_split_assignment_v1"
OBSERVATION_MODE = "native_state_v1"
ACTION_EXECUTION_OFFSET_METADATA = "action_execution_tick_offset"
REPLAY_EXTENT_METADATA_KEY = "native_replay_extent_v1"
REPLAY_EXTENT_KIND = "cr_native_replay_extent_v1"
FULL_SUCCESS_EXTENT = "full_success"
VALID_PREFIX_EXTENT = "valid_prefix"
PREFIX_TRAINING_ADMISSION = "actor_bc_censored_prefix_v1"
PREFIX_MASK_PROVENANCE = "partial_native_visible_hand_complete_v1"
PREFIX_TIMING_TARGET = "right_censored_at_failure_tick_v1"
ENTITY_NUMERIC_FIELDS = ("level_ratio", "hp_ratio", "log_max_hp")
MAX_ABILITY_SLOTS = 16
CAPACITY_PREFLIGHT_KIND = "cr_native_bc_capacity_preflight_v1"
CAPACITY_PREFLIGHT_FILENAME = "capacity-preflight.json"
CAPACITY_SAMPLE_BATTLES = 100
CAPACITY_SAFETY_FACTOR = 1.35
CAPACITY_MINIMUM_RESERVE_BYTES = 10 * 1024**3
CAPACITY_FILESYSTEM_RESERVE_FRACTION = 0.05
CAPACITY_SHARD_OVERHEAD_BYTES = 64 * 1024
CAPACITY_MEMORY_SCALE = 1.25
CAPACITY_MEMORY_BUDGET_FRACTION = 0.75
CAPACITY_MEMORY_GATE_FRACTION = 0.80
CAPACITY_SELECTION_STRATEGY = (
    "sha256_tick_count_payload_density_stratified_v1"
)
CAPACITY_STRATA_PER_DIMENSION = 4
CAPACITY_RESERVATION_DIRECTORY = ".capacity-reservations-v1"
GRID_CHANNELS = (
    "own_tower_occupancy",
    "own_tower_hp_ratio",
    "enemy_tower_occupancy",
    "enemy_tower_hp_ratio",
    "own_entity_count",
    "own_entity_hp_ratio",
    "enemy_entity_count",
    "enemy_entity_hp_ratio",
)
PUBLIC_SCALARS = (
    "tick_fraction",
    "own_elixir_ratio",
    "commands_allowed",
    "terminated",
    "own_crowns_ratio",
    "enemy_crowns_ratio",
    "own_king_hp_ratio",
    "own_left_hp_ratio",
    "own_right_hp_ratio",
    "enemy_king_hp_ratio",
    "enemy_left_hp_ratio",
    "enemy_right_hp_ratio",
    "own_visible_entity_count_log",
    "enemy_visible_entity_count_log",
    "own_visible_entity_hp_log",
    "enemy_visible_entity_hp_log",
)


class NativeBcCompileError(RuntimeError):
    """An input or output cannot satisfy the native BC contract."""


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    battle_tag: str
    tick_data_path: str
    tick_index_path: str
    tick_count: int
    tick_payload_sha256: str
    tick_payload_size: int
    source_path: str
    source_sha256: str
    source_group: str
    player_tags: tuple[str, ...]
    split: str
    component_sha256: str
    tick_store_root: str = ""
    replay_extent: str = FULL_SUCCESS_EXTENT
    compiled_tick_count: int = 0
    observation_tick_start: int = -1
    observation_tick_stop_exclusive: int = -1
    action_label_tick_stop_exclusive: int = -1
    timing_censor_tick_exclusive: int = -1
    timing_target: str = "native_tick_hazard_v1"
    terminal_target: str = "unknown_unanchored_v1"
    extent_sha256: str = "0" * 64
    mask_metadata_sha256: str = "0" * 64
    prefix_ability_evidence: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class OutputShard:
    relative_path: str
    split: str
    index: int
    episodes: tuple[EpisodeInput, ...]
    estimated_rows: int
    content_sha256: str


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as error:
                raise NativeBcCompileError(
                    f"invalid JSONL row {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise NativeBcCompileError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _source_path(row: Mapping[str, Any], manifest: Path) -> Path:
    raw = row.get("source_path") or row.get("saved_path")
    if not raw:
        raise NativeBcCompileError(
            f"schema-v5 index row lacks source_path/saved_path: {row.get('battle_tag')}"
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(strict=True)


def _normalize_source_group(source: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    explicit = (
        source.get("source_group_id")
        or row.get("source_group_id")
        or row.get("source_group")
    )
    if explicit:
        return "explicit:" + str(explicit).strip()
    metadata = source.get("deck_metadata")
    url = metadata.get("source_list_url") if isinstance(metadata, Mapping) else None
    if not url:
        url = row.get("source_list_url") or row.get("url")
    if not url:
        raise NativeBcCompileError(
            f"schema-v5 source lacks a source group: {source.get('battle_tag')}"
        )
    parts = urlsplit(str(url))
    # Query cursors identify pages, not independent sources.  Keeping only the
    # endpoint prevents adjacent pages from the same crawl/player crossing a
    # split.
    return "url:" + urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def _player_tags(source: Mapping[str, Any]) -> tuple[str, ...]:
    result: set[str] = set()
    for field in ("team_tags", "opponent_tags"):
        raw = source.get(field)
        if isinstance(raw, list):
            result.update(str(value).strip().lstrip("#") for value in raw if str(value).strip())
    rounds = source.get("rounds")
    if isinstance(rounds, list):
        for round_value in rounds:
            if not isinstance(round_value, Mapping):
                continue
            for side in ("team", "opponent"):
                raw_side = round_value.get(side)
                if not isinstance(raw_side, list):
                    continue
                for player in raw_side:
                    if isinstance(player, Mapping) and player.get("player_tag"):
                        result.add(str(player["player_tag"]).strip().lstrip("#"))
    result.discard("")
    if len(result) != 2:
        raise NativeBcCompileError(
            f"normal 1v1 schema-v5 source must expose exactly two player tags: "
            f"{source.get('battle_tag')} ({sorted(result)})"
        )
    return tuple(sorted(result))


def _validate_tick_store(
    root: Path,
    *,
    workers: int,
    expected_kind: str = TICK_STORE_KIND,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NativeBcCompileError(f"Tick Store manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        expected_kind == TICK_STORE_KIND
        and manifest.get("kind") == AUDIT_PREFIX_STORE_KIND
    ):
        raise NativeBcCompileError(
            "audit-prefix Tick Store is training_admission=audit_only and "
            "cannot be compiled as BC data"
        )
    if (
        manifest.get("kind") != expected_kind
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("every_native_tick_present") is not True
        or int(manifest.get("tick_hz", -1)) != 20
    ):
        raise NativeBcCompileError("Tick Store manifest contract changed")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or (not raw_shards and not allow_empty):
        raise NativeBcCompileError("Tick Store contains no immutable shards")

    def verify_one(raw: Mapping[str, Any]) -> dict[str, Any]:
        if raw.get("kind") != TICK_SHARD_KIND:
            raise NativeBcCompileError("Tick Store shard kind changed")
        data = (root / str(raw["data_file"])).resolve(strict=True)
        index = (root / str(raw["index_file"])).resolve(strict=True)
        if sha256_file(data) != str(raw["data_sha256"]):
            raise NativeBcCompileError(f"Tick Store data SHA mismatch: {data}")
        if sha256_file(index) != str(raw["index_sha256"]):
            raise NativeBcCompileError(f"Tick Store index SHA mismatch: {index}")
        return {**dict(raw), "data_path": str(data), "index_path": str(index)}

    maximum = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        shards = list(executor.map(verify_one, raw_shards))
    calculated = hashlib.sha256(
        json.dumps(
            {
            str(item["name"]): {
                "data": item["data_sha256"], "index": item["index_sha256"]
            }
            for item in shards
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if calculated != str(manifest.get("content_sha256")):
        raise NativeBcCompileError("Tick Store global content SHA mismatch")
    return manifest, shards


def _episode_index(shards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for shard in shards:
        with Path(str(shard["index_path"])).open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                tag = str(entry["battle_tag"])
                if tag in result:
                    raise NativeBcCompileError(f"duplicate Tick Store battle: {tag}")
                result[tag] = {
                    **entry,
                    "tick_data_path": str(shard["data_path"]),
                    "tick_index_path": str(shard["index_path"]),
                }
    return result


def _episode_extent_contract(
    metadata: Mapping[str, Any],
    *,
    tick_count: int,
    replay_extent: str,
) -> dict[str, Any]:
    """Normalize the immutable row/censor provenance for one Tick episode."""

    if replay_extent == FULL_SUCCESS_EXTENT:
        body = {
            "replay_extent": FULL_SUCCESS_EXTENT,
            "compiled_tick_count": int(tick_count),
            "observation_tick_start": -1,
            "observation_tick_stop_exclusive": -1,
            "action_label_tick_stop_exclusive": -1,
            "timing_censor_tick_exclusive": -1,
            "timing_target": "native_tick_hazard_v1",
            "terminal_target": "unknown_unanchored_v1",
        }
        return {
            **body,
            "extent_sha256": hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest(),
        }
    if replay_extent != VALID_PREFIX_EXTENT:
        raise NativeBcCompileError(f"unsupported replay extent: {replay_extent}")
    raw = metadata.get(REPLAY_EXTENT_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise NativeBcCompileError("audit-prefix episode lacks replay extent metadata")
    required = {
        "kind", "extent", "training_admission", "source_episode_complete",
        "every_native_tick_present_within_extent", "semantic_match",
        "failure_domain", "failure_tick_has_labels", "terminal_target",
        "terminal_validated", "deployment_masks", "mask_coverage",
        "observation_tick_start", "observation_tick_stop_exclusive",
        "action_label_tick_stop_exclusive", "timing_censor_tick_exclusive",
        "timing_target",
    }
    if not required <= set(raw):
        raise NativeBcCompileError("audit-prefix replay extent fields are incomplete")
    start = _require_integer(raw.get("observation_tick_start"), "prefix observation start")
    stop = _require_integer(raw.get("observation_tick_stop_exclusive"), "prefix observation stop")
    action_stop = _require_integer(
        raw.get("action_label_tick_stop_exclusive"), "prefix action-label stop"
    )
    censor = _require_integer(
        raw.get("timing_censor_tick_exclusive"), "prefix timing censor"
    )
    if (
        raw.get("kind") != REPLAY_EXTENT_KIND
        or raw.get("extent") != VALID_PREFIX_EXTENT
        or raw.get("training_admission") != PREFIX_TRAINING_ADMISSION
        or raw.get("source_episode_complete") is not False
        or raw.get("every_native_tick_present_within_extent") is not True
        or raw.get("semantic_match") is not True
        or raw.get("failure_domain") != "semantic"
        or raw.get("failure_tick_has_labels") is not False
        or raw.get("terminal_target") != "unknown_censored"
        or raw.get("terminal_validated") is not False
        or raw.get("deployment_masks") != PREFIX_MASK_PROVENANCE
        or raw.get("timing_target") != PREFIX_TIMING_TARGET
        or stop - start != int(tick_count)
        or not start < action_stop <= censor <= stop
    ):
        raise NativeBcCompileError("audit-prefix replay extent contract changed")
    compiled_ticks = censor - start
    if compiled_ticks <= 0:
        raise NativeBcCompileError("audit-prefix has no pre-censor Tick rows")
    coverage = raw.get("mask_coverage")
    if not isinstance(coverage, Mapping):
        raise NativeBcCompileError("audit-prefix mask coverage is missing")
    coverage_fields = {
        "all_retained_visible_hand_slots_covered", "retained_ticks",
        "actor_ticks", "visible_slot_references", "empty_slot_actor_ticks",
        "safe_deploy_labels", "checked_deploy_labels", "rejected_deploy_labels",
    }
    if not coverage_fields <= set(coverage):
        raise NativeBcCompileError("audit-prefix mask coverage fields are incomplete")
    if (
        coverage.get("all_retained_visible_hand_slots_covered") is not True
        or _require_integer(coverage.get("retained_ticks"), "prefix retained_ticks")
        != compiled_ticks
        or _require_integer(coverage.get("actor_ticks"), "prefix actor_ticks")
        != compiled_ticks * 2
        or _require_integer(
            coverage.get("rejected_deploy_labels"), "prefix rejected deploy labels"
        ) != 0
        or _require_integer(
            coverage.get("checked_deploy_labels"), "prefix checked deploy labels"
        )
        < _require_integer(
            coverage.get("safe_deploy_labels"), "prefix safe deploy labels"
        )
    ):
        raise NativeBcCompileError("audit-prefix mask coverage accounting changed")
    body = dict(raw)
    return {
        "replay_extent": VALID_PREFIX_EXTENT,
        "compiled_tick_count": compiled_ticks,
        "observation_tick_start": start,
        "observation_tick_stop_exclusive": stop,
        "action_label_tick_stop_exclusive": action_stop,
        "timing_censor_tick_exclusive": censor,
        "timing_target": PREFIX_TIMING_TARGET,
        "terminal_target": "unknown_censored",
        "extent_sha256": hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest(),
    }


def _read_source_input(
    row: Mapping[str, Any], manifest: Path, wanted: set[str]
) -> tuple[str, dict[str, Any], Path, str, str, tuple[str, ...]] | None:
    tag = str(row.get("battle_tag") or "")
    if tag not in wanted:
        return None
    row_schema = row.get("source_schema_version", row.get("schema_version", -1))
    if int(row_schema) != 5:
        raise NativeBcCompileError(f"index row is not schema-v5: {tag}")
    path = _source_path(row, manifest)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    declared = row.get("source_sha256")
    if declared is not None and str(declared) != digest:
        raise NativeBcCompileError(f"schema-v5 source SHA mismatch: {tag}")
    try:
        source = json.loads(raw)
    except Exception as error:
        raise NativeBcCompileError(f"invalid schema-v5 source JSON: {path}") from error
    if (
        not isinstance(source, dict)
        or int(source.get("schema_version", -1)) != 5
        or str(source.get("battle_tag") or "") != tag
    ):
        raise NativeBcCompileError(f"schema-v5 source identity mismatch: {tag}")
    if source.get("normal_1v1") is not True:
        raise NativeBcCompileError(f"source is not authoritative normal 1v1: {tag}")
    return (
        tag,
        source,
        path,
        digest,
        _normalize_source_group(source, row),
        _player_tags(source),
    )


def _assign_components(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation/test fractions must be in (0,1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation+test fractions must be below one")
    tags = [str(row["battle_tag"]) for row in rows]
    union = _UnionFind(tags)
    owners: dict[str, str] = {}
    for row in rows:
        tag = str(row["battle_tag"])
        keys = [
            *("player:" + value for value in row["player_tags"]),
            "source-group:" + str(row["source_group"]),
            "source-file:" + str(row["source_sha256"]),
        ]
        for key in keys:
            previous = owners.setdefault(key, tag)
            union.union(previous, tag)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        components[union.find(str(row["battle_tag"]))].append(row)
    if len(components) < 3:
        raise NativeBcCompileError(
            "at least three disconnected player/source components are required "
            "for train/validation/test"
        )
    ordered: list[tuple[str, list[dict[str, Any]], int]] = []
    for values in components.values():
        tags_in_component = sorted(str(value["battle_tag"]) for value in values)
        component_sha = _digest({"seed": seed, "battle_tags": tags_in_component})
        weight = sum(
            int(value.get("compiled_tick_count", value["tick_count"])) * 2
            for value in values
        )
        ordered.append((component_sha, values, weight))
    ordered.sort(key=lambda value: value[0])
    total = sum(value[2] for value in ordered)
    targets = {
        "test": max(1, round(total * test_fraction)),
        "validation": max(1, round(total * validation_fraction)),
    }
    split_by_component: dict[str, str] = {}
    consumed = {"test": 0, "validation": 0, "train": 0}
    # A deterministic shuffled component order plus a fill-to-target policy
    # preserves whole graph components and guarantees non-empty holdouts.
    for index, (digest, _values, weight) in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [name for name in ("test", "validation", "train") if consumed[name] == 0]
        if remaining == len(empty):
            split = empty[0]
        elif consumed["test"] < targets["test"]:
            split = "test"
        elif consumed["validation"] < targets["validation"]:
            split = "validation"
        else:
            split = "train"
        split_by_component[digest] = split
        consumed[split] += weight
    assignments: dict[str, tuple[str, str]] = {}
    for digest, values, _weight in ordered:
        split = split_by_component[digest]
        for row in values:
            assignments[str(row["battle_tag"])] = (split, digest)
    return assignments, {
        "component_count": len(components),
        "component_rows_by_split": consumed,
        "player_holdout_leaks": 0,
        "source_group_leaks": 0,
        "source_file_leaks": 0,
        "battle_tag_leaks": 0,
    }


def _card_vocab(contract: Mapping[str, Any]) -> tuple[list[str], dict[int, int], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    source_to_native: dict[str, int] = {}
    for card in contract.get("cards", []):
        if not isinstance(card, Mapping):
            continue
        base_id = int(card["card_id"])
        tokens = [str(value) for value in card.get("allowed_tokens", [])]
        base_tokens = [value for value in tokens if not value.endswith(("-ev1", "-hero"))]
        if len(base_tokens) != 1:
            raise NativeBcCompileError(f"native contract has ambiguous base token: {base_id}")
        id_to_name.setdefault(base_id, base_tokens[0])
        source_to_native[base_tokens[0]] = base_id
        evolution = card.get("evolution")
        if isinstance(evolution, Mapping) and evolution.get("native_form_id") is not None:
            candidates = [value for value in tokens if value.endswith("-ev1")]
            if len(candidates) != 1:
                raise NativeBcCompileError(f"native contract has ambiguous evolution: {base_id}")
            form_id = int(evolution["native_form_id"])
            id_to_name.setdefault(form_id, candidates[0])
            source_to_native[candidates[0]] = form_id
        hero = card.get("hero")
        if isinstance(hero, Mapping) and hero.get("native_form_id") is not None:
            candidates = [value for value in tokens if value.endswith("-hero")]
            if len(candidates) != 1:
                raise NativeBcCompileError(f"native contract has ambiguous hero: {base_id}")
            form_id = int(hero["native_form_id"])
            id_to_name.setdefault(form_id, candidates[0])
            source_to_native[candidates[0]] = form_id
    native_ids = sorted(id_to_name)
    vocabulary = ["<PAD>", *(f"{id_to_name[value]}@{value}" for value in native_ids)]
    id_to_token = {value: index + 1 for index, value in enumerate(native_ids)}
    source_to_token = {
        source: id_to_token[native_id] for source, native_id in source_to_native.items()
    }
    return vocabulary, id_to_token, source_to_token


def _ability_vocab(contract: Mapping[str, Any]) -> tuple[list[str], dict[int, int]]:
    names: dict[int, str] = {}
    for row in contract.get("ability_sources", []):
        if isinstance(row, Mapping):
            names[int(row["native_form_id"])] = str(row["token"])
            # Native state may expose the base ID for non-form abilities.
            names.setdefault(int(row["base_card_id"]), str(row["token"]))
    ids = sorted(names)
    return ["<PAD>", *(f"{names[value]}@{value}" for value in ids)], {
        value: index + 1 for index, value in enumerate(ids)
    }


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        loaded = load_native_ingest_contract(path)
    except Exception as error:
        raise NativeBcCompileError("native ingest contract authentication failed") from error
    return dict(loaded.value), loaded.file_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise NativeBcCompileError(
            f"{label} fields changed: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "")
    if not _SHA256_RE.fullmatch(result):
        raise NativeBcCompileError(f"{label} is not lowercase SHA-256")
    return result


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeBcCompileError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _verify_coverage_fingerprint(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(value, Mapping):
        raise NativeBcCompileError(f"native coverage {label} fingerprint is missing")
    path = Path(str(value.get("path") or "")).resolve(strict=True)
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise NativeBcCompileError(f"native coverage {label} path changed")
    if (
        value.get("kind") != "file_sha256_v1"
        or int(value.get("bytes", -1)) != path.stat().st_size
        or _require_sha(value.get("sha256"), f"native coverage {label} SHA")
        != sha256_file(path)
    ):
        raise NativeBcCompileError(f"native coverage {label} bytes changed")
    return path


def _authenticate_native_generation_receipt(
    receipt_path: Path,
    *,
    schema5_manifest: Path,
    native_contract: Path,
    contract_sha256: str,
    contract_file_sha256: str,
    expected_episodes: int,
) -> dict[str, Any]:
    """Recompute ability cohort coverage before any BC shard is admitted."""

    receipt_path = receipt_path.resolve(strict=True)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    except Exception as error:
        raise NativeBcCompileError("native generation coverage receipt is invalid") from error
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("kind") != "cr_expert_native_generation_coverage_v2"
        or int(receipt.get("schema_version", -1)) != 2
    ):
        raise NativeBcCompileError("native generation coverage receipt kind/schema changed")
    _verify_coverage_fingerprint(
        receipt.get("frozen_manifest"),
        "frozen manifest",
        expected_path=schema5_manifest,
    )
    candidate_queue = _verify_coverage_fingerprint(
        receipt.get("candidate_queue"), "candidate queue"
    )
    results_path = _verify_coverage_fingerprint(receipt.get("results"), "results")
    contract_binding = receipt.get("native_contract")
    if not isinstance(contract_binding, Mapping) or (
        Path(str(contract_binding.get("path") or "")).resolve()
        != native_contract.resolve()
        or contract_binding.get("canonical_sha256") != contract_sha256
        or contract_binding.get("file_sha256") != contract_file_sha256
    ):
        raise NativeBcCompileError("native coverage contract identity changed")
    target = _require_integer(receipt.get("target_battles"), "coverage target", minimum=1)
    selected = _require_integer(receipt.get("selected_battles"), "coverage selected")
    processed = _require_integer(receipt.get("processed_battles"), "coverage processed")
    successes = _require_integer(
        receipt.get("teacher_forced_successes"), "coverage successes"
    )
    failures = _require_integer(
        receipt.get("teacher_forced_failures"), "coverage failures"
    )
    stored = _require_integer(receipt.get("stored_episodes"), "coverage stored")
    prefix_stored = _require_integer(
        receipt.get("audit_prefix_episodes"), "coverage audit prefixes"
    )
    audit_episodes = _require_integer(
        receipt.get("audit_tick_episodes"), "coverage audited episodes"
    )
    unframed = _require_integer(
        receipt.get("unframed_episodes"), "coverage unframed episodes"
    )
    prefix_manifest_path = _verify_coverage_fingerprint(
        receipt.get("audit_prefix_store"), "audit-prefix store"
    )
    prefix_manifest = json.loads(
        prefix_manifest_path.read_text(encoding="utf-8-sig")
    )
    if (
        selected != target
        or processed != target
        or successes + failures != target
        or stored != successes
        or stored != expected_episodes
        or prefix_stored != failures
        or audit_episodes != target
        or unframed != 0
        or float(receipt.get("audit_tick_coverage_rate", -1)) != 1.0
        or prefix_manifest.get("kind") != AUDIT_PREFIX_STORE_KIND
        or int(prefix_manifest.get("episode_count", -1)) != prefix_stored
        or float(receipt.get("success_rate", -1)) != successes / target
    ):
        raise NativeBcCompileError(
            "native generation coverage does not match compiled episode admission"
        )
    ability = receipt.get("ability_coverage")
    if not isinstance(ability, Mapping):
        raise NativeBcCompileError("native ability coverage is missing")
    if (
        ability.get("kind") != "cr_expert_ability_native_coverage_v2"
        or int(ability.get("schema_version", -1)) != 2
    ):
        raise NativeBcCompileError("native ability coverage kind/schema changed")
    gate = ability.get("gate")
    if not isinstance(gate, Mapping):
        raise NativeBcCompileError("native ability coverage gate is missing")
    if (
        gate.get("authority")
        != "final_compiled_array_token_coverage_v1"
        or gate.get("final_array_gate_deferred") is not True
    ):
        raise NativeBcCompileError("native ability coverage authority changed")
    minimum_count = _require_integer(
        gate.get("minimum_success_count"), "ability minimum success count"
    )
    minimum_rate = float(gate.get("minimum_success_rate", -1))
    waived = gate.get("waiver_applied") is True
    waiver_reason = str(gate.get("waiver_reason") or "").strip() or None
    if not 0 <= minimum_rate <= 1:
        raise NativeBcCompileError("native ability minimum success rate is invalid")
    if (minimum_count < 1 or minimum_rate < 0.10) and not waived:
        raise NativeBcCompileError(
            "native ability coverage is below the default gate without a waiver"
        )
    try:
        from .one_click_v1 import (
            evaluate_ability_positive_coverage,
            validate_native_result_records,
        )

        audit = validate_native_result_records(
            results_path,
            candidate_queue,
            expected_rows=target,
            require_token_evidence=isinstance(
                receipt.get("source_token_coverage"), Mapping
            ),
        )
        recomputed = evaluate_ability_positive_coverage(
            {
                "ability_positive": int(audit["ability_positive"]["candidates"]),
                "ability_zero": int(audit["ability_zero"]["candidates"]),
            },
            audit,
            minimum_success_count=minimum_count,
            minimum_success_rate=minimum_rate,
            waived=waived,
            waiver_reason=waiver_reason,
        )
    except Exception as error:
        raise NativeBcCompileError(
            "native ability coverage candidate/result join failed"
        ) from error
    if dict(ability) != recomputed or (recomputed.get("gate") or {}).get(
        "admitted"
    ) is not True:
        raise NativeBcCompileError("native ability coverage receipt is not admitted")
    if (
        len(audit.get("success_tags") or []) != successes
        or len(audit.get("audit_prefix_tags") or []) != prefix_stored
        or audit.get("unframed_tags") != []
        or set(audit.get("success_tags") or [])
        & set(audit.get("audit_prefix_tags") or [])
    ):
        raise NativeBcCompileError("native audit-prefix coverage is not exact")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "source_token_coverage": receipt.get("source_token_coverage"),
        "ability_coverage": recomputed,
        "stored_episodes": stored,
        "audit_prefix_episodes": prefix_stored,
        "success_tags": sorted(str(value) for value in audit.get("success_tags") or []),
        "audit_prefix_tags": sorted(
            str(value) for value in audit.get("audit_prefix_tags") or []
        ),
        "audit_prefix_manifest_path": str(prefix_manifest_path),
        "audit_prefix_manifest_sha256": sha256_file(prefix_manifest_path),
    }


def _authenticate_source_token_coverage_receipt(
    receipt_path: Path,
    *,
    schema5_manifest: Path,
    native_contract: Path,
    contract_sha256: str,
    contract_file_sha256: str,
) -> dict[str, Any]:
    """Independently recompute source statistics from every frozen JSON."""

    receipt_path = receipt_path.resolve(strict=True)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    except Exception as error:
        raise NativeBcCompileError(
            "source token coverage receipt is invalid"
        ) from error
    if not isinstance(receipt, Mapping):
        raise NativeBcCompileError("source token coverage receipt is not an object")
    body = dict(receipt)
    claimed = str(body.pop("canonical_sha256", ""))
    if (
        body.get("kind") != "cr_expert_frozen_source_token_coverage_v1"
        or int(body.get("schema_version", -1)) != 1
        or claimed != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise NativeBcCompileError(
            "source token coverage receipt canonical identity changed"
        )
    _verify_coverage_fingerprint(
        body.get("frozen_manifest"),
        "source token frozen manifest",
        expected_path=schema5_manifest,
    )
    contract_binding = body.get("native_contract")
    if not isinstance(contract_binding, Mapping) or (
        Path(str(contract_binding.get("path") or "")).resolve()
        != native_contract.resolve()
        or contract_binding.get("canonical_sha256") != contract_sha256
        or contract_binding.get("file_sha256") != contract_file_sha256
    ):
        raise NativeBcCompileError("source token coverage contract binding changed")
    contract = json.loads(native_contract.read_bytes())

    def battles() -> Iterable[Mapping[str, Any]]:
        seen: set[str] = set()
        for line_number, row in enumerate(_json_lines(schema5_manifest), start=1):
            tag = str(row.get("battle_tag") or "")
            path = Path(str(row.get("source_path") or "")).resolve(strict=True)
            payload = path.read_bytes()
            if (
                not tag
                or tag in seen
                or hashlib.sha256(payload).hexdigest()
                != str(row.get("source_sha256") or "")
            ):
                raise NativeBcCompileError(
                    f"source token manifest identity failed at row {line_number}"
                )
            seen.add(tag)
            value = json.loads(payload)
            if not isinstance(value, Mapping) or value.get("battle_tag") != tag:
                raise NativeBcCompileError(
                    f"source token battle identity changed: {tag}"
                )
            yield value

    recomputed_source = freeze_source_token_coverage(battles(), contract)
    recomputed_quotas = build_adaptive_token_quotas(recomputed_source)
    if (
        body.get("source_coverage") != recomputed_source
        or body.get("adaptive_quotas") != recomputed_quotas
    ):
        raise NativeBcCompileError(
            "source token coverage receipt differs from frozen source bytes"
        )
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "canonical_sha256": claimed,
        "source_coverage": recomputed_source,
        "adaptive_quotas": recomputed_quotas,
    }


def _validate_plan_vocabulary(
    plan: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    cards, native_cards, source_cards = _card_vocab(contract)
    abilities, native_abilities = _ability_vocab(contract)
    if plan.get("card_vocabulary") != cards:
        raise NativeBcCompileError("compile-plan card vocabulary changed")
    if plan.get("ability_vocabulary") != abilities:
        raise NativeBcCompileError("compile-plan ability vocabulary changed")
    if plan.get("native_card_id_to_token") != {
        str(key): value for key, value in native_cards.items()
    }:
        raise NativeBcCompileError("compile-plan native card-token map changed")
    if plan.get("source_card_to_token") != source_cards:
        raise NativeBcCompileError("compile-plan source card-token map changed")
    if plan.get("native_ability_id_to_token") != {
        str(key): value for key, value in native_abilities.items()
    }:
        raise NativeBcCompileError("compile-plan native ability-token map changed")


def validate_compile_plan(
    plan: Mapping[str, Any],
    *,
    plan_path: Path | None = None,
    verify_live_inputs: bool = True,
) -> dict[str, Any]:
    """Authenticate a plan and recompute every deterministic contract field."""
    if not isinstance(plan, Mapping):
        raise NativeBcCompileError("compile-plan root is not an object")
    _require_keys(
        plan,
        {
            "kind", "schema_version", "created_utc", "input_content_sha256",
            "plan_content_sha256",
            "inputs", "compiler", "tick_store_root", "output_root", "episodes",
            "estimated_rows", "split_audit", "card_vocabulary",
            "native_card_id_to_token", "source_card_to_token", "ability_vocabulary",
            "native_ability_id_to_token", "capacity_preflight", "shards",
        },
        "compile-plan",
    )
    if (
        plan.get("kind") != PLAN_KIND
        or _require_integer(plan.get("schema_version"), "plan.schema_version", minimum=1)
        != 4
    ):
        raise NativeBcCompileError("compile-plan kind/schema changed")
    if not isinstance(plan.get("created_utc"), str) or not plan["created_utc"]:
        raise NativeBcCompileError("compile-plan created_utc is missing")
    inputs = plan.get("inputs")
    compiler = plan.get("compiler")
    if not isinstance(inputs, Mapping) or not isinstance(compiler, Mapping):
        raise NativeBcCompileError("compile-plan input/compiler contract is malformed")
    _require_keys(
        inputs,
        {
            "tick_store_manifest_path", "tick_store_manifest_sha256",
            "tick_store_content_sha256", "schema5_manifest_path",
            "schema5_manifest_sha256", "native_contract_path",
            "native_contract_file_sha256", "native_contract_sha256",
            "referenced_deployment_mask_sha256", "deployment_mask_manifest_sha256",
            "deployment_mask_content_sha256", "native_generation_receipt_path",
            "native_generation_receipt_sha256", "native_generation_results_path",
            "native_generation_results_sha256",
            "source_token_coverage_receipt_path",
            "source_token_coverage_receipt_sha256",
            "source_token_coverage_canonical_sha256",
            "audit_prefix_store_root", "audit_prefix_store_manifest_path",
            "audit_prefix_store_manifest_sha256",
            "audit_prefix_store_content_sha256",
            "referenced_audit_prefix_deployment_mask_sha256",
            "audit_prefix_deployment_mask_manifest_sha256",
            "audit_prefix_deployment_mask_content_sha256",
        },
        "compile-plan inputs",
    )
    _require_keys(
        compiler,
        {
            "kind", "schema_version", "seed", "validation_fraction",
            "test_fraction", "maximum_rows_per_shard", "actor_information",
            "action_alignment", "entity_identity", "mask_policy", "storage_schema",
            "components",
        },
        "compile-plan compiler",
    )
    if (
        compiler.get("kind") != COMPILER_KIND
        or _require_integer(
            compiler.get("schema_version"), "compiler.schema_version", minimum=1
        )
        != 4
        or compiler.get("actor_information") != "public_only_v1"
        or compiler.get("action_alignment") != "source_tick_plus_episode_metadata_offset"
        or compiler.get("entity_identity") != "discrete_native_card_token_v1"
        or compiler.get("mask_policy")
        != "full_complete_prefix_visible_hand_partial_fail_closed_v1"
        or compiler.get("storage_schema") != {
            "grid": GRID_STORAGE,
            "selected_position_mask": POSITION_MASK_STORAGE,
            "ability_position_mask": POSITION_MASK_STORAGE,
        }
    ):
        raise NativeBcCompileError("compile-plan compiler semantics changed")
    seed = _require_integer(compiler.get("seed"), "compiler.seed")
    maximum_rows = _require_integer(
        compiler.get("maximum_rows_per_shard"),
        "compiler.maximum_rows_per_shard",
        minimum=1,
    )
    validation_fraction = float(compiler.get("validation_fraction", -1))
    test_fraction = float(compiler.get("test_fraction", -1))
    if (
        not 0 < validation_fraction < 1
        or not 0 < test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise NativeBcCompileError("compile-plan split fractions are invalid")
    components = compiler.get("components")
    if not isinstance(components, Mapping):
        raise NativeBcCompileError("compile-plan component hashes are malformed")
    _require_keys(
        components,
        {
            "compiler_sha256", "training_schema_sha256",
            "deployment_masks_sha256", "native_coverage_validator_sha256",
            "token_coverage_validator_sha256",
        },
        "compile-plan components",
    )
    for name, value in components.items():
        _require_sha(value, f"compiler.components.{name}")

    input_sha = _require_sha(plan.get("input_content_sha256"), "input_content_sha256")
    if input_sha != _digest({"inputs": dict(inputs), "compiler": dict(compiler)}):
        raise NativeBcCompileError("compile-plan canonical input content SHA changed")
    plan_content_sha = _require_sha(
        plan.get("plan_content_sha256"), "plan_content_sha256"
    )
    if plan_content_sha != _digest(
        {key: value for key, value in plan.items() if key != "plan_content_sha256"}
    ):
        raise NativeBcCompileError("compile-plan canonical full-content SHA changed")
    for name in (
        "tick_store_manifest_sha256", "tick_store_content_sha256",
        "schema5_manifest_sha256", "native_contract_file_sha256",
        "native_contract_sha256", "deployment_mask_manifest_sha256",
        "deployment_mask_content_sha256", "native_generation_receipt_sha256",
        "native_generation_results_sha256",
        "audit_prefix_store_manifest_sha256",
        "audit_prefix_store_content_sha256",
        "audit_prefix_deployment_mask_manifest_sha256",
        "audit_prefix_deployment_mask_content_sha256",
    ):
        _require_sha(inputs.get(name), f"inputs.{name}")
    if inputs.get("source_token_coverage_receipt_path") is None:
        if (
            inputs.get("source_token_coverage_receipt_sha256") is not None
            or inputs.get("source_token_coverage_canonical_sha256") is not None
        ):
            raise NativeBcCompileError(
                "compile-plan has partial source token coverage identity"
            )
    else:
        _require_sha(
            inputs.get("source_token_coverage_receipt_sha256"),
            "inputs.source_token_coverage_receipt_sha256",
        )
        _require_sha(
            inputs.get("source_token_coverage_canonical_sha256"),
            "inputs.source_token_coverage_canonical_sha256",
        )
    referenced_masks = inputs.get("referenced_deployment_mask_sha256")
    if (
        not isinstance(referenced_masks, list)
        or referenced_masks != sorted(set(str(value) for value in referenced_masks))
    ):
        raise NativeBcCompileError("referenced deployment masks are not sorted unique")
    for value in referenced_masks:
        _require_sha(value, "referenced deployment mask")
    referenced_prefix_masks = inputs.get(
        "referenced_audit_prefix_deployment_mask_sha256"
    )
    if (
        not isinstance(referenced_prefix_masks, list)
        or referenced_prefix_masks
        != sorted(set(str(value) for value in referenced_prefix_masks))
    ):
        raise NativeBcCompileError(
            "referenced audit-prefix deployment masks are not sorted unique"
        )
    for value in referenced_prefix_masks:
        _require_sha(value, "referenced audit-prefix deployment mask")

    tick_root = Path(str(plan.get("tick_store_root") or "")).resolve()
    prefix_root = Path(str(inputs.get("audit_prefix_store_root") or "")).resolve()
    output_root = Path(str(plan.get("output_root") or "")).resolve()
    if str(tick_root) != str(plan["tick_store_root"]):
        raise NativeBcCompileError("compile-plan Tick Store path is not canonical absolute")
    if str(output_root) != str(plan["output_root"]):
        raise NativeBcCompileError("compile-plan output path is not canonical absolute")
    if Path(str(inputs["tick_store_manifest_path"])).resolve() != tick_root / "manifest.json":
        raise NativeBcCompileError("compile-plan Tick Store manifest path disagrees with root")
    if (
        str(prefix_root) != str(inputs["audit_prefix_store_root"])
        or Path(str(inputs["audit_prefix_store_manifest_path"])).resolve()
        != prefix_root / "manifest.json"
    ):
        raise NativeBcCompileError(
            "compile-plan audit-prefix Tick Store path is not canonical"
        )
    if plan_path is not None and output_root != plan_path.resolve().parent:
        raise NativeBcCompileError("compile-plan output root disagrees with its location")
    capacity_reference = plan.get("capacity_preflight")
    if not isinstance(capacity_reference, Mapping):
        raise NativeBcCompileError("compile-plan capacity preflight reference is missing")
    _require_keys(
        capacity_reference,
        {
            "path", "file_sha256", "content_sha256", "sample_actor_rows",
            "sample_bytes_per_actor_row",
            "sample_max_episode_bytes_per_actor_row", "projected_output_bytes",
            "required_free_bytes",
        },
        "compile-plan capacity preflight",
    )
    capacity_path = Path(str(capacity_reference.get("path") or "")).resolve()
    if (
        capacity_path != output_root / CAPACITY_PREFLIGHT_FILENAME
        or str(capacity_path) != str(capacity_reference["path"])
        or not capacity_path.is_file()
        or sha256_file(capacity_path)
        != _require_sha(
            capacity_reference.get("file_sha256"),
            "capacity_preflight.file_sha256",
        )
    ):
        raise NativeBcCompileError("compile-plan capacity preflight file changed")
    capacity_value = _read_capacity_preflight(capacity_path)
    if (
        capacity_value["content_sha256"]
        != _require_sha(
            capacity_reference.get("content_sha256"),
            "capacity_preflight.content_sha256",
        )
        or int(capacity_value["sample_actor_rows"])
        != _require_integer(
            capacity_reference.get("sample_actor_rows"),
            "capacity_preflight.sample_actor_rows",
            minimum=1,
        )
        or float(capacity_value["sample_bytes_per_actor_row"])
        != float(capacity_reference.get("sample_bytes_per_actor_row", -1))
        or float(capacity_value["sample_max_episode_bytes_per_actor_row"])
        != float(
            capacity_reference.get(
                "sample_max_episode_bytes_per_actor_row", -1
            )
        )
        or int(capacity_value["projected_output_bytes"])
        != _require_integer(
            capacity_reference.get("projected_output_bytes"),
            "capacity_preflight.projected_output_bytes",
            minimum=1,
        )
        or int(capacity_value["required_free_bytes"])
        != _require_integer(
            capacity_reference.get("required_free_bytes"),
            "capacity_preflight.required_free_bytes",
            minimum=1,
        )
    ):
        raise NativeBcCompileError("compile-plan capacity preflight summary changed")

    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise NativeBcCompileError("compile-plan has no output shards")
    episode_fields = {
        "battle_tag", "tick_store_root", "tick_data_path", "tick_index_path", "tick_count",
        "tick_payload_sha256", "tick_payload_size", "source_path", "source_sha256", "source_group",
        "player_tags", "split", "component_sha256", "replay_extent",
        "compiled_tick_count", "observation_tick_start",
        "observation_tick_stop_exclusive", "action_label_tick_stop_exclusive",
        "timing_censor_tick_exclusive", "timing_target", "terminal_target",
        "extent_sha256", "mask_metadata_sha256",
        "prefix_ability_evidence",
    }
    shard_fields = {
        "relative_path", "split", "index", "episodes", "estimated_rows",
        "content_sha256",
    }
    all_episodes: list[dict[str, Any]] = []
    paths: set[str] = set()
    by_split_shards: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, Mapping):
            raise NativeBcCompileError("compile-plan shard is not an object")
        _require_keys(raw_shard, shard_fields, "compile-plan shard")
        split = str(raw_shard.get("split") or "")
        if split not in {"train", "validation", "test"}:
            raise NativeBcCompileError("compile-plan shard split is invalid")
        index = _require_integer(raw_shard.get("index"), "shard.index")
        relative = str(raw_shard.get("relative_path") or "").replace("\\", "/")
        if relative != f"shards/{split}-{index:05d}" or relative in paths:
            raise NativeBcCompileError("compile-plan shard path/index is invalid")
        paths.add(relative)
        episodes = raw_shard.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise NativeBcCompileError("compile-plan output shard has no episodes")
        normalized_episodes: list[dict[str, Any]] = []
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise NativeBcCompileError("compile-plan episode is not an object")
            _require_keys(episode, episode_fields, "compile-plan episode")
            item = dict(episode)
            tag = str(item.get("battle_tag") or "")
            players = item.get("player_tags")
            if (
                not tag
                or item.get("split") != split
                or not isinstance(players, list)
                or len(players) != 2
                or players != sorted(set(str(value) for value in players))
                or not str(item.get("source_group") or "")
            ):
                raise NativeBcCompileError(f"compile-plan episode identity is invalid: {tag}")
            _require_integer(item.get("tick_count"), "episode.tick_count", minimum=1)
            compiled_ticks = _require_integer(
                item.get("compiled_tick_count"),
                "episode.compiled_tick_count",
                minimum=1,
            )
            if compiled_ticks > int(item["tick_count"]):
                raise NativeBcCompileError("compiled prefix rows exceed stored Tick extent")
            _require_integer(
                item.get("tick_payload_size"),
                "episode.tick_payload_size",
                minimum=1,
            )
            for name in (
                "tick_payload_sha256", "source_sha256", "component_sha256",
                "extent_sha256", "mask_metadata_sha256",
            ):
                _require_sha(item.get(name), f"episode.{name}")
            for name in (
                "tick_store_root", "tick_data_path", "tick_index_path", "source_path"
            ):
                raw_path = str(item.get(name) or "")
                if not Path(raw_path).is_absolute() or str(Path(raw_path).resolve()) != raw_path:
                    raise NativeBcCompileError(f"episode.{name} is not canonical absolute")
            extent = str(item.get("replay_extent") or "")
            expected_root = tick_root if extent == FULL_SUCCESS_EXTENT else prefix_root
            if extent not in {FULL_SUCCESS_EXTENT, VALID_PREFIX_EXTENT}:
                raise NativeBcCompileError("compile-plan replay extent is invalid")
            if Path(str(item["tick_store_root"])).resolve() != expected_root:
                raise NativeBcCompileError("episode Tick root disagrees with replay extent")
            if expected_root not in Path(str(item["tick_data_path"])).resolve().parents:
                raise NativeBcCompileError("episode Tick data escapes Tick Store root")
            if expected_root not in Path(str(item["tick_index_path"])).resolve().parents:
                raise NativeBcCompileError("episode Tick index escapes Tick Store root")
            if extent == VALID_PREFIX_EXTENT:
                if (
                    item.get("timing_target") != PREFIX_TIMING_TARGET
                    or item.get("terminal_target") != "unknown_censored"
                    or _require_integer(
                        item.get("timing_censor_tick_exclusive"),
                        "episode timing censor",
                    )
                    - _require_integer(
                        item.get("observation_tick_start"),
                        "episode observation start",
                    )
                    != compiled_ticks
                    or _require_integer(
                        item.get("action_label_tick_stop_exclusive"),
                        "episode action label stop",
                    )
                    > int(item["timing_censor_tick_exclusive"])
                ):
                    raise NativeBcCompileError(
                        "compile-plan audit-prefix censor contract changed"
                    )
            prefix_ability = item.get("prefix_ability_evidence")
            if not isinstance(prefix_ability, list):
                raise NativeBcCompileError(
                    "compile-plan prefix ability evidence must be an array"
                )
            normalized_prefix_ability: list[tuple[int, int, int, int, str, str]] = []
            for raw_evidence in prefix_ability:
                if not isinstance(raw_evidence, Mapping):
                    raise NativeBcCompileError(
                        "compile-plan prefix ability evidence is malformed"
                    )
                if set(raw_evidence) != {
                    "actor_side", "source_event_index", "selected_entity_id",
                    "selected_native_form_id", "resolved_token",
                    "transcript_sha256",
                }:
                    raise NativeBcCompileError(
                        "compile-plan prefix ability evidence fields changed"
                    )
                actor_side = _require_integer(
                    raw_evidence.get("actor_side"), "prefix ability actor_side"
                )
                event_index = _require_integer(
                    raw_evidence.get("source_event_index"),
                    "prefix ability source_event_index",
                )
                entity_id = _require_integer(
                    raw_evidence.get("selected_entity_id"),
                    "prefix ability selected_entity_id",
                    minimum=1,
                )
                native_form_id = _require_integer(
                    raw_evidence.get("selected_native_form_id"),
                    "prefix ability selected_native_form_id",
                    minimum=1,
                )
                resolved_token = str(raw_evidence.get("resolved_token") or "")
                if not resolved_token:
                    raise NativeBcCompileError(
                        "prefix ability resolved token is missing"
                    )
                transcript_sha = _require_sha(
                    raw_evidence.get("transcript_sha256"),
                    "prefix ability transcript_sha256",
                )
                if actor_side not in (0, 1):
                    raise NativeBcCompileError("prefix ability actor side is invalid")
                normalized_prefix_ability.append(
                    (
                        actor_side, event_index, entity_id, native_form_id,
                        resolved_token, transcript_sha,
                    )
                )
            if normalized_prefix_ability != sorted(set(normalized_prefix_ability)):
                raise NativeBcCompileError(
                    "compile-plan prefix ability evidence is not sorted unique"
                )
            if extent == FULL_SUCCESS_EXTENT and prefix_ability:
                raise NativeBcCompileError(
                    "full-success episode carries prefix ability evidence"
                )
            normalized_episodes.append(item)
            all_episodes.append(item)
        if [value["battle_tag"] for value in normalized_episodes] != sorted(
            value["battle_tag"] for value in normalized_episodes
        ):
            raise NativeBcCompileError("compile-plan shard episodes are not deterministic")
        expected_rows = sum(
            int(value["compiled_tick_count"]) * 2
            for value in normalized_episodes
        )
        if int(raw_shard.get("estimated_rows", -1)) != expected_rows:
            raise NativeBcCompileError("compile-plan shard estimated_rows changed")
        expected_content = _digest({"split": split, "episodes": normalized_episodes})
        if _require_sha(raw_shard.get("content_sha256"), "shard.content_sha256") != expected_content:
            raise NativeBcCompileError("compile-plan shard canonical content SHA changed")
        by_split_shards[split].append(raw_shard)

    if len({str(value["battle_tag"]) for value in all_episodes}) != len(all_episodes):
        raise NativeBcCompileError("compile-plan battle appears more than once")
    if set(by_split_shards) != {"train", "validation", "test"}:
        raise NativeBcCompileError("compile-plan must contain all three splits")
    # Reconstruct the deterministic greedy row packing and exact shard order.
    expected_groups: list[tuple[str, int, list[str]]] = []
    for split in ("train", "validation", "test"):
        values = sorted(
            (value for value in all_episodes if value["split"] == split),
            key=lambda value: value["battle_tag"],
        )
        pending: list[str] = []
        rows = 0
        index = 0
        for value in values:
            cost = int(value["compiled_tick_count"]) * 2
            if pending and rows + cost > maximum_rows:
                expected_groups.append((split, index, pending))
                pending, rows, index = [], 0, index + 1
            pending.append(str(value["battle_tag"]))
            rows += cost
        if pending:
            expected_groups.append((split, index, pending))
    actual_groups = [
        (
            str(value["split"]),
            int(value["index"]),
            [str(episode["battle_tag"]) for episode in value["episodes"]],
        )
        for value in raw_shards
    ]
    if actual_groups != expected_groups:
        raise NativeBcCompileError("compile-plan shard partition/order changed")

    assignment_rows = [
        {
            "battle_tag": value["battle_tag"],
            "player_tags": value["player_tags"],
            "source_group": value["source_group"],
            "source_sha256": value["source_sha256"],
            "tick_count": value["tick_count"],
            "compiled_tick_count": value["compiled_tick_count"],
        }
        for value in all_episodes
    ]
    expected_assignments, expected_audit = _assign_components(
        assignment_rows,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if plan.get("split_audit") != expected_audit:
        raise NativeBcCompileError("compile-plan split audit changed")
    for value in all_episodes:
        if expected_assignments[str(value["battle_tag"])] != (
            value["split"], value["component_sha256"]
        ):
            raise NativeBcCompileError("compile-plan split/component assignment changed")
    if _require_integer(plan.get("episodes"), "plan.episodes") != len(all_episodes):
        raise NativeBcCompileError("compile-plan episode count changed")
    expected_rows = sum(
        int(value["compiled_tick_count"]) * 2 for value in all_episodes
    )
    if _require_integer(plan.get("estimated_rows"), "plan.estimated_rows") != expected_rows:
        raise NativeBcCompileError("compile-plan estimated row count changed")
    if (
        int(capacity_value["estimated_total_actor_rows"]) != expected_rows
        or int(capacity_value["maximum_rows_per_shard"]) != maximum_rows
        or int(capacity_value["planned_shards"]) != len(raw_shards)
    ):
        raise NativeBcCompileError("capacity preflight does not cover this exact plan")
    expected_capacity_sample = _stratified_capacity_sample(
        [EpisodeInput(**value) for value in all_episodes]
    )
    actual_capacity_sample = capacity_value["sample_episodes"]
    if [
        (
            value.battle_tag,
            value.tick_count,
            value.tick_payload_size,
        )
        for value in expected_capacity_sample
    ] != [
        (
            str(value["battle_tag"]),
            int(value["tick_count"]),
            int(value["tick_payload_size"]),
        )
        for value in actual_capacity_sample
    ]:
        raise NativeBcCompileError(
            "capacity preflight sample is not the deterministic stratified plan sample"
        )

    contract_path = Path(str(inputs["native_contract_path"])).resolve()
    if verify_live_inputs:
        files = (
            (Path(str(inputs["tick_store_manifest_path"])), inputs["tick_store_manifest_sha256"], "Tick Store manifest"),
            (Path(str(inputs["schema5_manifest_path"])), inputs["schema5_manifest_sha256"], "Schema5 manifest"),
            (contract_path, inputs["native_contract_file_sha256"], "native contract"),
            (
                Path(str(inputs["native_generation_receipt_path"])),
                inputs["native_generation_receipt_sha256"],
                "native generation receipt",
            ),
            (
                Path(str(inputs["native_generation_results_path"])),
                inputs["native_generation_results_sha256"],
                "native generation results",
            ),
            (tick_root / "deployment-masks-v1" / "manifest.json", inputs["deployment_mask_manifest_sha256"], "mask manifest"),
            (
                Path(str(inputs["audit_prefix_store_manifest_path"])),
                inputs["audit_prefix_store_manifest_sha256"],
                "audit-prefix Tick Store manifest",
            ),
            (
                prefix_root / "deployment-masks-v1" / "manifest.json",
                inputs["audit_prefix_deployment_mask_manifest_sha256"],
                "audit-prefix mask manifest",
            ),
        )
        if inputs.get("source_token_coverage_receipt_path") is not None:
            files += ((
                Path(str(inputs["source_token_coverage_receipt_path"])),
                inputs["source_token_coverage_receipt_sha256"],
                "source token coverage receipt",
            ),)
        for path, expected, label in files:
            if not path.is_file() or sha256_file(path) != str(expected):
                raise NativeBcCompileError(f"compile-plan {label} changed")
        tick_manifest = json.loads(
            Path(str(inputs["tick_store_manifest_path"])).read_text(encoding="utf-8-sig")
        )
        if tick_manifest.get("content_sha256") != inputs["tick_store_content_sha256"]:
            raise NativeBcCompileError("compile-plan Tick Store content SHA changed")
        prefix_manifest = json.loads(
            Path(str(inputs["audit_prefix_store_manifest_path"])).read_text(
                encoding="utf-8-sig"
            )
        )
        if (
            prefix_manifest.get("content_sha256")
            != inputs["audit_prefix_store_content_sha256"]
        ):
            raise NativeBcCompileError(
                "compile-plan audit-prefix Tick Store content SHA changed"
            )
        contract, contract_file_sha = _load_contract(contract_path)
        if (
            contract_file_sha != inputs["native_contract_file_sha256"]
            or contract.get("contract_sha256") != inputs["native_contract_sha256"]
        ):
            raise NativeBcCompileError("compile-plan native contract identity changed")
        native_coverage = _authenticate_native_generation_receipt(
            Path(str(inputs["native_generation_receipt_path"])),
            schema5_manifest=Path(str(inputs["schema5_manifest_path"])),
            native_contract=contract_path,
            contract_sha256=str(inputs["native_contract_sha256"]),
            contract_file_sha256=str(inputs["native_contract_file_sha256"]),
            expected_episodes=sum(
                value["replay_extent"] == FULL_SUCCESS_EXTENT
                for value in all_episodes
            ),
        )
        if native_coverage["receipt_sha256"] != inputs[
            "native_generation_receipt_sha256"
        ]:
            raise NativeBcCompileError("compile-plan native coverage receipt changed")
        if (
            native_coverage["results_sha256"]
            != inputs["native_generation_results_sha256"]
            or Path(native_coverage["results_path"]).resolve()
            != Path(str(inputs["native_generation_results_path"])).resolve()
        ):
            raise NativeBcCompileError("compile-plan native results identity changed")
        planned_full_tags = {
            str(value["battle_tag"])
            for value in all_episodes
            if value["replay_extent"] == FULL_SUCCESS_EXTENT
        }
        planned_prefix_tags = {
            str(value["battle_tag"])
            for value in all_episodes
            if value["replay_extent"] == VALID_PREFIX_EXTENT
        }
        if (
            set(native_coverage["success_tags"]) != planned_full_tags
            or set(native_coverage["audit_prefix_tags"]) != planned_prefix_tags
            or Path(native_coverage["audit_prefix_manifest_path"]).resolve()
            != prefix_root / "manifest.json"
        ):
            raise NativeBcCompileError(
                "compile-plan full/prefix result union changed"
            )
        if inputs.get("source_token_coverage_receipt_path") is not None:
            source_coverage = _authenticate_source_token_coverage_receipt(
                Path(str(inputs["source_token_coverage_receipt_path"])),
                schema5_manifest=Path(str(inputs["schema5_manifest_path"])),
                native_contract=contract_path,
                contract_sha256=str(inputs["native_contract_sha256"]),
                contract_file_sha256=str(inputs["native_contract_file_sha256"]),
            )
            if (
                source_coverage["receipt_sha256"]
                != inputs["source_token_coverage_receipt_sha256"]
                or source_coverage["canonical_sha256"]
                != inputs["source_token_coverage_canonical_sha256"]
            ):
                raise NativeBcCompileError(
                    "compile-plan source token coverage identity changed"
                )
            if planned_prefix_tags:
                result_rows = {
                    str(row.get("battle_tag") or ""): row
                    for row in _json_lines(
                        Path(str(inputs["native_generation_results_path"]))
                    )
                }
                raw_prefix_actors: list[Mapping[str, Any]] = []
                for tag in sorted(planned_prefix_tags):
                    actors = result_rows.get(tag, {}).get(
                        "prefix_token_coverage_actor_evidence"
                    )
                    if not isinstance(actors, list) or len(actors) != 2:
                        raise NativeBcCompileError(
                            f"compile-plan prefix ability evidence is missing: {tag}"
                        )
                    raw_prefix_actors.extend(
                        actor for actor in actors if isinstance(actor, Mapping)
                    )
                try:
                    rebuilt_prefix = authenticate_generator_ability_evidence(
                        raw_prefix_actors,
                        contract,
                        source_coverage["source_coverage"],
                        expected_source_events_sha256=str(
                            source_coverage["source_coverage"]
                            ["ability_event_registry"]["source_events_sha256"]
                        ),
                    )
                except Exception as error:
                    raise NativeBcCompileError(
                        f"compile-plan prefix ability authentication failed: {error}"
                    ) from error
                rebuilt_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for transcript in rebuilt_prefix["transcripts"]:
                    rebuilt_by_tag[str(transcript["battle_tag"])].append({
                        "actor_side": int(transcript["actor_side"]),
                        "source_event_index": int(
                            transcript["source_event_index"]
                        ),
                        "selected_entity_id": int(
                            transcript["selected_entity_id"]
                        ),
                        "selected_native_form_id": int(
                            transcript["selected_native_form_id"]
                        ),
                        "resolved_token": str(transcript["resolved_token"]),
                        "transcript_sha256": str(
                            transcript["transcript_sha256"]
                        ),
                    })
                for values in rebuilt_by_tag.values():
                    values.sort(
                        key=lambda value: (
                            int(value["actor_side"]),
                            int(value["source_event_index"]),
                            int(value["selected_entity_id"]),
                            str(value["transcript_sha256"]),
                        )
                    )
                for episode in all_episodes:
                    if episode["replay_extent"] == VALID_PREFIX_EXTENT and (
                        episode["prefix_ability_evidence"]
                        != rebuilt_by_tag.get(str(episode["battle_tag"]), [])
                    ):
                        raise NativeBcCompileError(
                            "compile-plan prefix ability evidence differs from results"
                        )
        _validate_plan_vocabulary(plan, contract)
        mask_store = DeploymentMaskStore(tick_root, create=False)
        mask_manifest = mask_store.verify_manifest()
        if mask_manifest.get("content_sha256") != inputs["deployment_mask_content_sha256"]:
            raise NativeBcCompileError("compile-plan mask-store content SHA changed")
        manifest_masks = {
            str(value["content_sha256"]) for value in mask_manifest.get("entries", [])
        }
        if not set(referenced_masks).issubset(manifest_masks):
            raise NativeBcCompileError("compile-plan references masks outside mask manifest")
        for digest in referenced_masks:
            mask_store.load(digest)
        prefix_mask_store = DeploymentMaskStore(prefix_root, create=False)
        prefix_mask_manifest = prefix_mask_store.verify_manifest()
        if (
            prefix_mask_manifest.get("content_sha256")
            != inputs["audit_prefix_deployment_mask_content_sha256"]
        ):
            raise NativeBcCompileError(
                "compile-plan audit-prefix mask-store content SHA changed"
            )
        prefix_manifest_masks = {
            str(value["content_sha256"])
            for value in prefix_mask_manifest.get("entries", [])
        }
        if not set(referenced_prefix_masks).issubset(prefix_manifest_masks):
            raise NativeBcCompileError(
                "compile-plan references masks outside audit-prefix mask manifest"
            )
        for digest in referenced_prefix_masks:
            prefix_mask_store.load(digest)
        # Rebuild the episode join from the immutable inputs.  This prevents a
        # re-signed plan from omitting a battle, swapping a source path, or
        # pointing a tag at a different Tick frame while preserving plausible
        # aggregate counts.
        _tick_manifest, tick_shards = _validate_tick_store(
            tick_root, workers=min(16, max(1, os.cpu_count() or 1))
        )
        tick_episodes = _episode_index(tick_shards)
        _prefix_tick_manifest, prefix_tick_shards = _validate_tick_store(
            prefix_root,
            workers=min(16, max(1, os.cpu_count() or 1)),
            expected_kind=AUDIT_PREFIX_STORE_KIND,
            allow_empty=True,
        )
        prefix_tick_episodes = _episode_index(prefix_tick_shards)
        planned_by_tag = {
            str(value["battle_tag"]): value for value in all_episodes
        }
        if (
            planned_full_tags != set(tick_episodes)
            or planned_prefix_tags != set(prefix_tick_episodes)
            or planned_full_tags & planned_prefix_tags
            or set(planned_by_tag) != planned_full_tags | planned_prefix_tags
        ):
            raise NativeBcCompileError(
                "compile-plan battle set differs from immutable Tick Store"
            )
        schema_manifest = Path(str(inputs["schema5_manifest_path"])).resolve(strict=True)
        source_index: dict[str, Mapping[str, Any]] = {}
        for row in _json_lines(schema_manifest):
            tag = str(row.get("battle_tag") or "")
            if not tag or tag in source_index:
                raise NativeBcCompileError(
                    "compile-plan Schema5 manifest has duplicate/missing battle tag"
                )
            source_index[tag] = row
        if not set(planned_by_tag).issubset(source_index):
            raise NativeBcCompileError(
                "compile-plan battle set is absent from Schema5 manifest"
            )
        readers: dict[tuple[str, str], ShardReader] = {}
        actual_referenced_masks: set[str] = set()
        actual_referenced_prefix_masks: set[str] = set()
        try:
            for tag in sorted(planned_by_tag):
                planned = planned_by_tag[tag]
                loaded_source = _read_source_input(
                    source_index[tag], schema_manifest, {tag}
                )
                assert loaded_source is not None
                (
                    _loaded_tag,
                    source_value,
                    source_path,
                    source_sha,
                    source_group,
                    players,
                ) = loaded_source
                if (
                    str(source_path) != planned["source_path"]
                    or source_sha != planned["source_sha256"]
                    or source_group != planned["source_group"]
                    or list(players) != planned["player_tags"]
                ):
                    raise NativeBcCompileError(
                        f"compile-plan Schema5 episode join changed: {tag}"
                    )
                is_prefix = planned["replay_extent"] == VALID_PREFIX_EXTENT
                tick_entry = (
                    prefix_tick_episodes[tag] if is_prefix else tick_episodes[tag]
                )
                expected_tick_fields = {
                    "tick_data_path": str(tick_entry["tick_data_path"]),
                    "tick_index_path": str(tick_entry["tick_index_path"]),
                    "tick_count": int(tick_entry["ticks"]),
                    "tick_payload_sha256": str(tick_entry["payload_sha256"]),
                }
                if any(planned[name] != value for name, value in expected_tick_fields.items()):
                    raise NativeBcCompileError(
                        f"compile-plan Tick episode join changed: {tag}"
                    )
                key = (
                    expected_tick_fields["tick_data_path"],
                    expected_tick_fields["tick_index_path"],
                )
                reader = readers.get(key)
                if reader is None:
                    reader = ShardReader(Path(key[0]), Path(key[1]))
                    readers[key] = reader
                native_episode = reader.episode(tag)
                metadata = native_episode.metadata
                source_contract = source_value.get("authoritative_native_contract") or {}
                identities = (
                    str(source_contract.get("contract_sha256") or ""),
                    str(source_contract.get("contract_file_sha256") or ""),
                    str(metadata.get("authoritative_contract_sha256") or ""),
                    str(metadata.get("authoritative_contract_file_sha256") or ""),
                )
                expected_identity = (
                    str(inputs["native_contract_sha256"]),
                    str(inputs["native_contract_file_sha256"]),
                    str(inputs["native_contract_sha256"]),
                    str(inputs["native_contract_file_sha256"]),
                )
                if identities != expected_identity:
                    raise NativeBcCompileError(
                        f"compile-plan source/episode contract join changed: {tag}"
                    )
                if str(metadata.get("source_sha256") or "") != source_sha:
                    raise NativeBcCompileError(
                        f"compile-plan source/Tick SHA join changed: {tag}"
                    )
                active_mask_store = prefix_mask_store if is_prefix else mask_store
                episode_masks = active_mask_store.verify_episode_metadata(
                    metadata, require_complete=not is_prefix
                )
                extent_contract = _episode_extent_contract(
                    metadata,
                    tick_count=int(tick_entry["ticks"]),
                    replay_extent=str(planned["replay_extent"]),
                )
                if any(
                    planned[name] != value
                    for name, value in extent_contract.items()
                ) or planned["mask_metadata_sha256"] != _digest(dict(episode_masks)):
                    raise NativeBcCompileError(
                        f"compile-plan replay extent/mask provenance changed: {tag}"
                    )
                actual_target = (
                    actual_referenced_prefix_masks
                    if is_prefix
                    else actual_referenced_masks
                )
                for entry in episode_masks["entries"]:
                    actual_target.add(str(entry["content_sha256"]))
                    actual_target.update(
                        str(variant["content_sha256"])
                        for variant in entry.get("dynamic_label_variants", [])
                    )
        finally:
            for reader in readers.values():
                reader.close()
        if sorted(actual_referenced_masks) != referenced_masks:
            raise NativeBcCompileError(
                "compile-plan referenced deployment-mask set changed"
            )
        if sorted(actual_referenced_prefix_masks) != referenced_prefix_masks:
            raise NativeBcCompileError(
                "compile-plan referenced audit-prefix deployment-mask set changed"
            )
        component_paths = {
            "compiler_sha256": Path(__file__).resolve(),
            "training_schema_sha256": (
                Path(__file__).parent / "training_v1" / "schema.py"
            ).resolve(),
            "deployment_masks_sha256": (
                Path(__file__).parent / "tick_store_v1" / "deployment_masks.py"
            ).resolve(),
            "native_coverage_validator_sha256": (
                Path(__file__).parent / "one_click_v1.py"
            ).resolve(),
            "token_coverage_validator_sha256": (
                Path(__file__).parent / "token_coverage_v1.py"
            ).resolve(),
        }
        for name, path in component_paths.items():
            if sha256_file(path) != components[name]:
                raise NativeBcCompileError(f"compile-plan component changed: {name}")
    else:
        # Even without filesystem reads, reject malformed/aliased vocab maps.
        for vocabulary_name in ("card_vocabulary", "ability_vocabulary"):
            vocabulary = plan.get(vocabulary_name)
            if (
                not isinstance(vocabulary, list)
                or not vocabulary
                or vocabulary[0] != "<PAD>"
                or len(vocabulary) != len(set(str(value) for value in vocabulary))
            ):
                raise NativeBcCompileError(f"compile-plan {vocabulary_name} is invalid")
    return dict(plan)


def load_compile_plan(path: Path, *, verify_live_inputs: bool = True) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sidecar = path.with_name("compile-plan.sha256")
    try:
        fields = sidecar.read_text(encoding="ascii").split()
    except OSError as error:
        raise NativeBcCompileError("compile-plan SHA sidecar is missing") from error
    if (
        len(fields) != 2
        or fields[1] != "compile-plan.json"
        or not _SHA256_RE.fullmatch(fields[0])
    ):
        raise NativeBcCompileError("compile-plan SHA sidecar format is invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != fields[0]:
        raise NativeBcCompileError("compile-plan SHA sidecar mismatch")
    try:
        value = json.loads(raw)
    except Exception as error:
        raise NativeBcCompileError("compile-plan JSON is invalid") from error
    if not isinstance(value, Mapping) or _canonical_bytes(value) != raw:
        raise NativeBcCompileError("compile-plan is not canonical JSON")
    return validate_compile_plan(
        value, plan_path=path, verify_live_inputs=verify_live_inputs
    )


def _authenticate_plan_argument(
    plan: Mapping[str, Any], *, verify_live_inputs: bool
) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or not plan.get("output_root"):
        raise NativeBcCompileError("compile-plan argument lacks output_root")
    authenticated = load_compile_plan(
        Path(str(plan["output_root"])) / "compile-plan.json",
        verify_live_inputs=verify_live_inputs,
    )
    if dict(plan) != authenticated:
        raise NativeBcCompileError(
            "in-memory compile-plan differs from authenticated on-disk plan"
        )
    return authenticated


def _stratified_capacity_sample(
    episodes: Sequence[EpisodeInput], *, limit: int = CAPACITY_SAMPLE_BATTLES
) -> list[EpisodeInput]:
    """Deterministically cover duration and encoded state-density strata."""
    if limit <= 0:
        raise ValueError("capacity sample limit must be positive")
    values = list(episodes)
    if len(values) <= limit:
        return sorted(values, key=lambda value: value.battle_tag)

    def rank_bins(key: Any) -> dict[str, int]:
        ordered = sorted(
            values,
            key=lambda value: (key(value), value.battle_tag),
        )
        return {
            value.battle_tag: min(
                CAPACITY_STRATA_PER_DIMENSION - 1,
                index * CAPACITY_STRATA_PER_DIMENSION // len(ordered),
            )
            for index, value in enumerate(ordered)
        }

    tick_bins = rank_bins(lambda value: value.tick_count)
    density_bins = rank_bins(
        lambda value: value.tick_payload_size / value.tick_count
    )
    strata: dict[tuple[int, int], list[EpisodeInput]] = defaultdict(list)
    for episode in values:
        strata[(tick_bins[episode.battle_tag], density_bins[episode.battle_tag])].append(
            episode
        )
    for group in strata.values():
        group.sort(
            key=lambda value: (
                _digest({"capacity_sample_battle_tag": value.battle_tag}),
                value.battle_tag,
            )
        )
    selected: list[EpisodeInput] = []
    depth = 0
    keys = sorted(strata)
    while len(selected) < limit:
        changed = False
        for key in keys:
            group = strata[key]
            if depth < len(group):
                selected.append(group[depth])
                changed = True
                if len(selected) == limit:
                    break
        if not changed:
            break
        depth += 1
    if len(selected) != limit:
        raise NativeBcCompileError("capacity stratified sampler under-filled")
    return selected


def _float_equal(left: Any, right: float) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, right, rel_tol=1e-12, abs_tol=1e-12
    )


def _read_capacity_preflight(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as error:
        raise NativeBcCompileError("capacity preflight is missing/invalid") from error
    if not isinstance(value, Mapping) or _canonical_bytes(value) != raw:
        raise NativeBcCompileError("capacity preflight is not canonical JSON")
    expected = {
        "kind", "schema_version", "content_sha256", "sample_selection_strategy",
        "sample_episodes", "sample_battles", "sample_battle_tags_sha256",
        "sample_actor_rows", "sample_array_payload_bytes", "sample_output_bytes",
        "sample_bytes_per_actor_row", "sample_compile_seconds",
        "sample_actor_rows_per_second", "sample_peak_rss_bytes",
        "sample_baseline_rss_bytes", "sample_peak_rss_delta_bytes",
        "sample_max_episode_bytes_per_actor_row", "maximum_rows_per_shard",
        "planned_shards", "estimated_total_actor_rows", "projected_output_bytes",
        "per_shard_overhead_bytes", "safety_factor",
        "projected_output_with_safety_bytes",
        "filesystem_total_bytes", "filesystem_free_bytes",
        "filesystem_reserve_fraction", "minimum_reserve_bytes",
        "required_free_bytes", "memory_scale", "memory_budget_fraction",
        "memory_gate_fraction",
        "estimated_peak_worker_rss_bytes", "system_available_memory_bytes",
        "recommended_max_parallel_compile_workers", "disk_gate_passed",
        "memory_gate_passed", "passed",
    }
    _require_keys(value, expected, "capacity preflight")
    if (
        value.get("kind") != CAPACITY_PREFLIGHT_KIND
        or _require_integer(value.get("schema_version"), "capacity.schema_version", minimum=1)
        != 1
    ):
        raise NativeBcCompileError("capacity preflight kind/schema changed")
    content_sha = _require_sha(value.get("content_sha256"), "capacity.content_sha256")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if content_sha != _digest(body):
        raise NativeBcCompileError("capacity preflight canonical content SHA changed")
    episodes = value.get("sample_episodes")
    if (
        value.get("sample_selection_strategy") != CAPACITY_SELECTION_STRATEGY
        or not isinstance(episodes, list)
        or not episodes
        or len(episodes) != int(value.get("sample_battles", -1))
    ):
        raise NativeBcCompileError("capacity sample selection contract changed")
    episode_fields = {
        "battle_tag", "tick_count", "tick_payload_size",
        "tick_payload_bytes_per_tick", "actor_rows", "array_payload_bytes",
        "array_payload_bytes_per_actor_row", "mean_entities_per_actor_row",
        "max_entities_per_actor_row",
    }
    tags: list[str] = []
    total_rows = 0
    total_payload = 0
    episode_rates: list[float] = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise NativeBcCompileError("capacity sample episode is malformed")
        _require_keys(episode, episode_fields, "capacity sample episode")
        tag = str(episode.get("battle_tag") or "")
        ticks = _require_integer(
            episode.get("tick_count"), "capacity.episode.tick_count", minimum=1
        )
        payload_size = _require_integer(
            episode.get("tick_payload_size"),
            "capacity.episode.tick_payload_size",
            minimum=1,
        )
        rows = _require_integer(
            episode.get("actor_rows"), "capacity.episode.actor_rows", minimum=1
        )
        payload = _require_integer(
            episode.get("array_payload_bytes"),
            "capacity.episode.array_payload_bytes",
            minimum=1,
        )
        maximum_entities = _require_integer(
            episode.get("max_entities_per_actor_row"),
            "capacity.episode.max_entities_per_actor_row",
        )
        if (
            not tag
            or tag in tags
            or rows != ticks * 2
            or not _float_equal(
                episode.get("tick_payload_bytes_per_tick"), payload_size / ticks
            )
            or not _float_equal(
                episode.get("array_payload_bytes_per_actor_row"), payload / rows
            )
            or not math.isfinite(float(episode.get("mean_entities_per_actor_row", -1)))
            or not 0 <= float(episode["mean_entities_per_actor_row"]) <= maximum_entities
        ):
            raise NativeBcCompileError("capacity sample episode arithmetic changed")
        tags.append(tag)
        total_rows += rows
        total_payload += payload
        episode_rates.append(payload / rows)
    if str(value.get("sample_battle_tags_sha256")) != _digest(
        {"battle_tags": tags}
    ):
        raise NativeBcCompileError("capacity sample battle selection SHA changed")
    sample_output = _require_integer(
        value.get("sample_output_bytes"), "capacity.sample_output_bytes", minimum=1
    )
    sample_seconds = float(value.get("sample_compile_seconds", -1))
    if (
        int(value.get("sample_actor_rows", -1)) != total_rows
        or int(value.get("sample_array_payload_bytes", -1)) != total_payload
        or sample_output < total_payload
        or not _float_equal(
            value.get("sample_bytes_per_actor_row"), sample_output / total_rows
        )
        or not math.isfinite(sample_seconds)
        or sample_seconds <= 0
        or not _float_equal(
            value.get("sample_actor_rows_per_second"), total_rows / sample_seconds
        )
        or not _float_equal(
            value.get("sample_max_episode_bytes_per_actor_row"), max(episode_rates)
        )
    ):
        raise NativeBcCompileError("capacity sample aggregate arithmetic changed")
    estimated_rows = _require_integer(
        value.get("estimated_total_actor_rows"),
        "capacity.estimated_total_actor_rows",
        minimum=1,
    )
    planned_shards = _require_integer(
        value.get("planned_shards"), "capacity.planned_shards", minimum=1
    )
    if int(value.get("per_shard_overhead_bytes", -1)) != CAPACITY_SHARD_OVERHEAD_BYTES:
        raise NativeBcCompileError("capacity per-shard overhead changed")
    projected = math.ceil(max(episode_rates) * estimated_rows) + (
        planned_shards * CAPACITY_SHARD_OVERHEAD_BYTES
    )
    if (
        int(value.get("projected_output_bytes", -1)) != projected
        or not _float_equal(value.get("safety_factor"), CAPACITY_SAFETY_FACTOR)
    ):
        raise NativeBcCompileError("capacity projection arithmetic changed")
    projected_safe = math.ceil(projected * CAPACITY_SAFETY_FACTOR)
    filesystem_total = _require_integer(
        value.get("filesystem_total_bytes"), "capacity.filesystem_total_bytes", minimum=1
    )
    filesystem_free = _require_integer(
        value.get("filesystem_free_bytes"), "capacity.filesystem_free_bytes"
    )
    reserve = max(
        CAPACITY_MINIMUM_RESERVE_BYTES,
        math.ceil(filesystem_total * CAPACITY_FILESYSTEM_RESERVE_FRACTION),
    )
    required_free = projected_safe + reserve
    if (
        int(value.get("projected_output_with_safety_bytes", -1)) != projected_safe
        or not _float_equal(
            value.get("filesystem_reserve_fraction"),
            CAPACITY_FILESYSTEM_RESERVE_FRACTION,
        )
        or int(value.get("minimum_reserve_bytes", -1)) != reserve
        or int(value.get("required_free_bytes", -1)) != required_free
    ):
        raise NativeBcCompileError("capacity disk arithmetic changed")
    baseline = _require_integer(
        value.get("sample_baseline_rss_bytes"),
        "capacity.sample_baseline_rss_bytes",
        minimum=1,
    )
    peak = _require_integer(
        value.get("sample_peak_rss_bytes"),
        "capacity.sample_peak_rss_bytes",
        minimum=baseline,
    )
    delta = peak - baseline
    maximum_rows = _require_integer(
        value.get("maximum_rows_per_shard"),
        "capacity.maximum_rows_per_shard",
        minimum=1,
    )
    available_memory = _require_integer(
        value.get("system_available_memory_bytes"),
        "capacity.system_available_memory_bytes",
        minimum=1,
    )
    estimated_peak = baseline + math.ceil(
        delta * (maximum_rows / total_rows) * CAPACITY_MEMORY_SCALE
    )
    recommended_workers = max(
        1,
        int(
            (available_memory * CAPACITY_MEMORY_BUDGET_FRACTION)
            // max(estimated_peak, 1)
        ),
    )
    disk_passed = filesystem_free >= required_free
    memory_passed = estimated_peak <= int(
        available_memory * CAPACITY_MEMORY_GATE_FRACTION
    )
    if (
        int(value.get("sample_peak_rss_delta_bytes", -1)) != delta
        or not _float_equal(value.get("memory_scale"), CAPACITY_MEMORY_SCALE)
        or not _float_equal(
            value.get("memory_budget_fraction"),
            CAPACITY_MEMORY_BUDGET_FRACTION,
        )
        or not _float_equal(
            value.get("memory_gate_fraction"), CAPACITY_MEMORY_GATE_FRACTION
        )
        or int(value.get("estimated_peak_worker_rss_bytes", -1)) != estimated_peak
        or int(value.get("recommended_max_parallel_compile_workers", -1))
        != recommended_workers
        or value.get("disk_gate_passed") is not disk_passed
        or value.get("memory_gate_passed") is not memory_passed
        or value.get("passed") is not bool(disk_passed and memory_passed)
    ):
        raise NativeBcCompileError("capacity memory/gate arithmetic changed")
    if value.get("passed") is not True:
        raise NativeBcCompileError("capacity preflight did not pass")
    return dict(value)


def _capacity_reference(path: Path) -> dict[str, Any]:
    value = _read_capacity_preflight(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "content_sha256": value["content_sha256"],
        "sample_actor_rows": int(value["sample_actor_rows"]),
        "sample_bytes_per_actor_row": float(value["sample_bytes_per_actor_row"]),
        "sample_max_episode_bytes_per_actor_row": float(
            value["sample_max_episode_bytes_per_actor_row"]
        ),
        "projected_output_bytes": int(value["projected_output_bytes"]),
        "required_free_bytes": int(value["required_free_bytes"]),
    }


@contextmanager
def _capacity_reservation_lock(output_root: Path) -> Iterable[None]:
    """OS-released cross-process lock for the reservation ledger."""
    path = output_root / f"{CAPACITY_RESERVATION_DIRECTORY}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _active_capacity_reservations(output_root: Path) -> tuple[int, list[Path]]:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception as error:
        raise NativeBcCompileError("psutil is required for disk reservations") from error
    root = output_root / CAPACITY_RESERVATION_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    active: list[Path] = []
    for path in sorted(root.glob("*.json")):
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except Exception as error:
            raise NativeBcCompileError(
                f"capacity reservation ledger is corrupt: {path}"
            ) from error
        if (
            not isinstance(value, Mapping)
            or _canonical_bytes(value) != raw
            or set(value) != {
                "kind", "schema_version", "pid", "process_create_time",
                "token", "relative_path", "reserved_bytes",
            }
            or value.get("kind") != "cr_native_bc_disk_reservation_v1"
            or int(value.get("schema_version", -1)) != 1
        ):
            raise NativeBcCompileError(
                f"capacity reservation contract changed: {path}"
            )
        pid = _require_integer(value.get("pid"), "reservation.pid", minimum=1)
        process_create_time = float(value.get("process_create_time", -1))
        alive = psutil.pid_exists(pid)
        if alive:
            try:
                alive = math.isclose(
                    float(psutil.Process(pid).create_time()),
                    process_create_time,
                    rel_tol=0.0,
                    abs_tol=1e-3,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                alive = False
        if not alive:
            path.unlink()
            continue
        total += _require_integer(
            value.get("reserved_bytes"), "reservation.reserved_bytes", minimum=1
        )
        active.append(path)
    return total, active


def _acquire_capacity_reservation(
    output_root: Path,
    *,
    relative_path: str,
    requested_bytes: int,
) -> Path:
    if requested_bytes <= 0:
        raise ValueError("capacity reservation must be positive")
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception as error:
        raise NativeBcCompileError("psutil is required for disk reservations") from error
    token = uuid.uuid4().hex
    reservation_root = output_root / CAPACITY_RESERVATION_DIRECTORY
    path = reservation_root / f"{os.getpid()}-{token}.json"
    with _capacity_reservation_lock(output_root):
        active_bytes, _active = _active_capacity_reservations(output_root)
        usage = shutil.disk_usage(output_root)
        reserve = max(
            CAPACITY_MINIMUM_RESERVE_BYTES,
            math.ceil(
                int(usage.total) * CAPACITY_FILESYSTEM_RESERVE_FRACTION
            ),
        )
        if int(usage.free) - active_bytes - requested_bytes < reserve:
            raise NativeBcCompileError(
                "cross-process shard disk reservation failed: "
                f"free={usage.free}, active_reserved={active_bytes}, "
                f"requested={requested_bytes}, mandatory_reserve={reserve}"
            )
        value = {
            "kind": "cr_native_bc_disk_reservation_v1",
            "schema_version": 1,
            "pid": os.getpid(),
            "process_create_time": float(psutil.Process(os.getpid()).create_time()),
            "token": token,
            "relative_path": relative_path,
            "reserved_bytes": int(requested_bytes),
        }
        _atomic_json(path, value)
    return path


def _release_capacity_reservation(output_root: Path, path: Path | None) -> None:
    if path is None:
        return
    with _capacity_reservation_lock(output_root):
        if not path.is_file():
            raise NativeBcCompileError(
                f"capacity reservation disappeared before release: {path}"
            )
        path.unlink()


def _build_capacity_preflight(
    *,
    output_root: Path,
    tick_store_root: Path,
    episodes: Sequence[EpisodeInput],
    raw_plan: Mapping[str, Any],
    estimated_total_rows: int,
    maximum_rows_per_shard: int,
    planned_shards: int,
) -> dict[str, Any]:
    """Compile a deterministic <=100-battle sample before full shard writes."""
    if not episodes or estimated_total_rows <= 0:
        raise NativeBcCompileError("capacity sample requires non-empty native episodes")
    selected = _stratified_capacity_sample(episodes)
    sample_root = output_root / f".capacity-sample-{os.getpid()}"
    if sample_root.exists():
        shutil.rmtree(sample_root)
    sample_root.mkdir(parents=True)
    raw_spec = {
        "relative_path": "sample",
        "split": "capacity",
        "index": 0,
        "episodes": [asdict(value) for value in selected],
        "estimated_rows": sum(value.compiled_tick_count * 2 for value in selected),
        "content_sha256": _digest(
            {"capacity_sample": [asdict(value) for value in selected]}
        ),
    }
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception as error:
        raise NativeBcCompileError(
            "psutil is required for the capacity memory preflight"
        ) from error
    process = psutil.Process(os.getpid())
    baseline_rss = int(process.memory_info().rss)
    peak_rss = [baseline_rss]
    finished = threading.Event()

    def watch_rss() -> None:
        while not finished.wait(0.01):
            try:
                peak_rss[0] = max(peak_rss[0], int(process.memory_info().rss))
            except Exception:
                return

    watcher = threading.Thread(target=watch_rss, daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        metadata = _compile_output_shard(
            str(sample_root), str(tick_store_root), raw_spec, raw_plan
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        peak_rss[0] = max(peak_rss[0], int(process.memory_info().rss))
        sample_path = sample_root / "sample"
        output_bytes = sum(
            path.stat().st_size for path in sample_path.rglob("*") if path.is_file()
        )
        sample_rows = int(metadata["rows"])
        if sample_rows <= 0 or output_bytes <= 0:
            raise NativeBcCompileError("capacity sample produced no actor rows/bytes")
        episode_estimates = metadata.get("capacity_episode_estimates")
        if (
            not isinstance(episode_estimates, list)
            or len(episode_estimates) != len(selected)
        ):
            raise NativeBcCompileError(
                "capacity sample lacks per-episode storage estimates"
            )
        selected_by_tag = {value.battle_tag: value for value in selected}
        if [value.get("battle_tag") for value in episode_estimates] != [
            value.battle_tag for value in selected
        ]:
            raise NativeBcCompileError(
                "capacity per-episode estimate order changed"
            )
        for estimate in episode_estimates:
            episode = selected_by_tag[str(estimate["battle_tag"])]
            if (
                int(estimate["tick_count"]) != episode.tick_count
                or int(estimate["tick_payload_size"]) != episode.tick_payload_size
            ):
                raise NativeBcCompileError(
                    "capacity estimate/source episode identity changed"
                )
        bytes_per_row = output_bytes / sample_rows
        sample_array_payload_bytes = sum(
            int(value["array_payload_bytes"]) for value in episode_estimates
        )
        maximum_episode_rate = max(
            float(value["array_payload_bytes_per_actor_row"])
            for value in episode_estimates
        )
        projected = math.ceil(maximum_episode_rate * estimated_total_rows)
        projected += int(planned_shards) * CAPACITY_SHARD_OVERHEAD_BYTES
        projected_safe = math.ceil(projected * CAPACITY_SAFETY_FACTOR)
        usage = shutil.disk_usage(output_root)
        reserve = max(
            CAPACITY_MINIMUM_RESERVE_BYTES,
            math.ceil(
                int(usage.total) * CAPACITY_FILESYSTEM_RESERVE_FRACTION
            ),
        )
        required_free = projected_safe + reserve
        available_memory = int(psutil.virtual_memory().available)
        delta_rss = max(0, peak_rss[0] - baseline_rss)
        estimated_peak_worker = baseline_rss + math.ceil(
            delta_rss
            * (maximum_rows_per_shard / max(1, sample_rows))
            * CAPACITY_MEMORY_SCALE
        )
        recommended_workers = max(
            1,
            int(
                (available_memory * CAPACITY_MEMORY_BUDGET_FRACTION)
                // max(estimated_peak_worker, 1)
            ),
        )
        disk_passed = int(usage.free) >= required_free
        memory_passed = estimated_peak_worker <= int(
            available_memory * CAPACITY_MEMORY_GATE_FRACTION
        )
        body: dict[str, Any] = {
            "kind": CAPACITY_PREFLIGHT_KIND,
            "schema_version": 1,
            "sample_selection_strategy": CAPACITY_SELECTION_STRATEGY,
            "sample_episodes": episode_estimates,
            "sample_battles": len(selected),
            "sample_battle_tags_sha256": _digest(
                {"battle_tags": [value.battle_tag for value in selected]}
            ),
            "sample_actor_rows": sample_rows,
            "sample_array_payload_bytes": sample_array_payload_bytes,
            "sample_output_bytes": output_bytes,
            "sample_bytes_per_actor_row": bytes_per_row,
            "sample_compile_seconds": elapsed,
            "sample_actor_rows_per_second": sample_rows / elapsed,
            "sample_baseline_rss_bytes": baseline_rss,
            "sample_peak_rss_bytes": peak_rss[0],
            "sample_peak_rss_delta_bytes": delta_rss,
            "sample_max_episode_bytes_per_actor_row": maximum_episode_rate,
            "maximum_rows_per_shard": int(maximum_rows_per_shard),
            "planned_shards": int(planned_shards),
            "estimated_total_actor_rows": int(estimated_total_rows),
            "projected_output_bytes": int(projected),
            "per_shard_overhead_bytes": CAPACITY_SHARD_OVERHEAD_BYTES,
            "safety_factor": CAPACITY_SAFETY_FACTOR,
            "projected_output_with_safety_bytes": int(projected_safe),
            "filesystem_total_bytes": int(usage.total),
            "filesystem_free_bytes": int(usage.free),
            "filesystem_reserve_fraction": CAPACITY_FILESYSTEM_RESERVE_FRACTION,
            "minimum_reserve_bytes": int(reserve),
            "required_free_bytes": int(required_free),
            "memory_scale": CAPACITY_MEMORY_SCALE,
            "memory_budget_fraction": CAPACITY_MEMORY_BUDGET_FRACTION,
            "memory_gate_fraction": CAPACITY_MEMORY_GATE_FRACTION,
            "estimated_peak_worker_rss_bytes": int(estimated_peak_worker),
            "system_available_memory_bytes": int(available_memory),
            "recommended_max_parallel_compile_workers": int(recommended_workers),
            "disk_gate_passed": disk_passed,
            "memory_gate_passed": memory_passed,
            "passed": bool(disk_passed and memory_passed),
        }
        value = {**body, "content_sha256": _digest(body)}
    finally:
        finished.set()
        watcher.join(timeout=1.0)
        try:
            shutil.rmtree(sample_root)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise NativeBcCompileError(
                f"capacity sample cleanup failed: {sample_root}"
            ) from error
        if sample_root.exists():
            raise NativeBcCompileError(
                f"capacity sample cleanup left output pollution: {sample_root}"
            )
    path = output_root / CAPACITY_PREFLIGHT_FILENAME
    _atomic_json(path, value)
    if value["passed"] is not True:
        raise NativeBcCompileError(
            "capacity preflight failed: "
            f"projected={value['projected_output_with_safety_bytes']} "
            f"free={value['filesystem_free_bytes']} reserve={value['minimum_reserve_bytes']} "
            f"peak_worker={value['estimated_peak_worker_rss_bytes']} "
            f"available_memory={value['system_available_memory_bytes']}"
        )
    return _capacity_reference(path)


def create_compile_plan(
    tick_store_root: Path,
    schema5_manifest: Path,
    output_root: Path,
    native_contract: Path,
    native_generation_receipt: Path,
    source_token_coverage_receipt: Path | None = None,
    *,
    audit_prefix_store_root: Path,
    seed: int = 20260827,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    maximum_rows_per_shard: int = 524_288,
    io_workers: int = 16,
) -> dict[str, Any]:
    """Verify all immutable inputs and atomically publish a deterministic plan."""
    tick_store_root = tick_store_root.resolve(strict=True)
    schema5_manifest = schema5_manifest.resolve(strict=True)
    native_contract = native_contract.resolve(strict=True)
    native_generation_receipt = native_generation_receipt.resolve(strict=True)
    audit_prefix_store_root = audit_prefix_store_root.resolve(strict=True)
    source_token_coverage_receipt = (
        None
        if source_token_coverage_receipt is None
        else source_token_coverage_receipt.resolve(strict=True)
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_plan_path = output_root / "compile-plan.json"
    existing_authenticated = (
        load_compile_plan(existing_plan_path)
        if existing_plan_path.is_file()
        else None
    )
    if maximum_rows_per_shard <= 0:
        raise ValueError("maximum_rows_per_shard must be positive")
    contract, contract_file_sha256 = _load_contract(native_contract)
    contract_sha256 = str(contract["contract_sha256"])
    store, shards = _validate_tick_store(tick_store_root, workers=io_workers)
    episode_entries = _episode_index(shards)
    prefix_store, prefix_shards = _validate_tick_store(
        audit_prefix_store_root,
        workers=io_workers,
        expected_kind=AUDIT_PREFIX_STORE_KIND,
        allow_empty=True,
    )
    prefix_episode_entries = _episode_index(prefix_shards)
    native_generation_coverage = _authenticate_native_generation_receipt(
        native_generation_receipt,
        schema5_manifest=schema5_manifest,
        native_contract=native_contract,
        contract_sha256=contract_sha256,
        contract_file_sha256=contract_file_sha256,
        expected_episodes=len(episode_entries),
    )
    if (
        Path(native_generation_coverage["audit_prefix_manifest_path"]).resolve()
        != audit_prefix_store_root / "manifest.json"
        or int(native_generation_coverage["audit_prefix_episodes"])
        != len(prefix_episode_entries)
        or set(native_generation_coverage["success_tags"]) != set(episode_entries)
        or set(native_generation_coverage["audit_prefix_tags"])
        != set(prefix_episode_entries)
        or set(episode_entries) & set(prefix_episode_entries)
    ):
        raise NativeBcCompileError(
            "full/prefix Tick Store episode sets differ from native results"
        )
    source_token_coverage = (
        None
        if source_token_coverage_receipt is None
        else _authenticate_source_token_coverage_receipt(
            source_token_coverage_receipt,
            schema5_manifest=schema5_manifest,
            native_contract=native_contract,
            contract_sha256=contract_sha256,
            contract_file_sha256=contract_file_sha256,
        )
    )
    if source_token_coverage is not None:
        declared_source_coverage = native_generation_coverage.get(
            "source_token_coverage"
        )
        if not isinstance(declared_source_coverage, Mapping) or (
            Path(str(declared_source_coverage.get("path") or "")).resolve()
            != source_token_coverage_receipt
            or declared_source_coverage.get("sha256")
            != source_token_coverage["receipt_sha256"]
        ):
            raise NativeBcCompileError(
                "native generation receipt is not bound to source token coverage"
            )
    index_rows = _json_lines(schema5_manifest)
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in index_rows:
        tag = str(row.get("battle_tag") or "")
        if not tag:
            raise NativeBcCompileError("schema-v5 index row lacks battle_tag")
        if tag in indexed:
            raise NativeBcCompileError(f"duplicate schema-v5 index battle: {tag}")
        indexed[tag] = row
    wanted = set(episode_entries) | set(prefix_episode_entries)
    missing = sorted(wanted - set(indexed))
    if missing:
        raise NativeBcCompileError(
            f"Tick Store episodes missing from schema-v5 index: {missing[:5]}"
        )
    with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
        loaded = list(
            executor.map(
                lambda row: _read_source_input(row, schema5_manifest, wanted),
                (indexed[tag] for tag in sorted(wanted)),
            )
        )
    prefix_ability_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if prefix_episode_entries and source_token_coverage is not None:
        result_rows = {
            str(row.get("battle_tag") or ""): row
            for row in _json_lines(
                Path(str(native_generation_coverage["results_path"]))
            )
        }
        prefix_actor_evidence: list[Mapping[str, Any]] = []
        for tag in sorted(prefix_episode_entries):
            actors = result_rows.get(tag, {}).get(
                "prefix_token_coverage_actor_evidence"
            )
            if not isinstance(actors, list) or len(actors) != 2:
                raise NativeBcCompileError(
                    f"audit-prefix result lacks actor token evidence: {tag}"
                )
            prefix_actor_evidence.extend(
                actor for actor in actors if isinstance(actor, Mapping)
            )
        if len(prefix_actor_evidence) != len(prefix_episode_entries) * 2:
            raise NativeBcCompileError(
                "audit-prefix actor token evidence sides are malformed"
            )
        try:
            authenticated_prefix = authenticate_generator_ability_evidence(
                prefix_actor_evidence,
                contract,
                source_token_coverage["source_coverage"],
                expected_source_events_sha256=str(
                    source_token_coverage["source_coverage"]
                    ["ability_event_registry"]["source_events_sha256"]
                ),
            )
        except Exception as error:
            raise NativeBcCompileError(
                f"audit-prefix ability evidence authentication failed: {error}"
            ) from error
        for transcript in authenticated_prefix["transcripts"]:
            prefix_ability_by_tag[str(transcript["battle_tag"])].append({
                "actor_side": int(transcript["actor_side"]),
                "source_event_index": int(transcript["source_event_index"]),
                "selected_entity_id": int(transcript["selected_entity_id"]),
                "selected_native_form_id": int(
                    transcript["selected_native_form_id"]
                ),
                "resolved_token": str(transcript["resolved_token"]),
                "transcript_sha256": str(transcript["transcript_sha256"]),
            })
        for values in prefix_ability_by_tag.values():
            values.sort(
                key=lambda value: (
                    int(value["actor_side"]),
                    int(value["source_event_index"]),
                    int(value["selected_entity_id"]),
                    str(value["transcript_sha256"]),
                )
            )
    source_rows: list[dict[str, Any]] = []
    mask_store = DeploymentMaskStore(tick_store_root, create=False)
    mask_manifest = mask_store.verify_manifest()
    prefix_mask_store = DeploymentMaskStore(
        audit_prefix_store_root, create=False
    )
    prefix_mask_manifest = prefix_mask_store.verify_manifest()
    replay_contract = load_native_ingest_contract(native_contract)
    referenced_masks: set[str] = set()
    referenced_prefix_masks: set[str] = set()
    for item in loaded:
        assert item is not None
        tag, source, path, source_sha, source_group, players = item
        replay_extent = (
            FULL_SUCCESS_EXTENT if tag in episode_entries else VALID_PREFIX_EXTENT
        )
        entry = (
            episode_entries[tag]
            if replay_extent == FULL_SUCCESS_EXTENT
            else prefix_episode_entries[tag]
        )
        active_mask_store = (
            mask_store
            if replay_extent == FULL_SUCCESS_EXTENT
            else prefix_mask_store
        )
        with ShardReader(Path(entry["tick_data_path"]), Path(entry["tick_index_path"])) as reader:
            episode = reader.episode(tag)
            metadata = dict(episode.metadata)
            if str(metadata.get("source_sha256") or "") != source_sha:
                raise NativeBcCompileError(f"Tick/source SHA identity mismatch: {tag}")
            if int(metadata.get("source_schema_version", -1)) != 5:
                raise NativeBcCompileError(f"Tick episode is not schema-v5: {tag}")
            masks = active_mask_store.verify_episode_metadata(
                metadata,
                require_complete=replay_extent == FULL_SUCCESS_EXTENT,
            )
            extent_contract = _episode_extent_contract(
                metadata,
                tick_count=int(entry["ticks"]),
                replay_extent=replay_extent,
            )
            if replay_extent == VALID_PREFIX_EXTENT:
                prefix_plan = compile_battle(
                    source,
                    native_ingest_contract=replay_contract,
                )
                prefix_events = {
                    (int(event.side), int(event.source_event_index)): event
                    for event in prefix_plan.ability_events
                }
                for evidence in prefix_ability_by_tag.get(tag, []):
                    event = prefix_events.get((
                        int(evidence["actor_side"]),
                        int(evidence["source_event_index"]),
                    ))
                    if (
                        event is None
                        or int(event.tick)
                        + int(metadata[ACTION_EXECUTION_OFFSET_METADATA])
                        >= int(extent_contract[
                            "action_label_tick_stop_exclusive"
                        ])
                    ):
                        raise NativeBcCompileError(
                            f"audit-prefix ability evidence crosses censor: {tag}"
                        )
            target_masks = (
                referenced_masks
                if replay_extent == FULL_SUCCESS_EXTENT
                else referenced_prefix_masks
            )
            for value in masks["entries"]:
                target_masks.add(str(value["content_sha256"]))
                target_masks.update(
                    str(variant["content_sha256"])
                    for variant in value.get("dynamic_label_variants", [])
                )
            if episode.tick_count != int(entry["ticks"]):
                raise NativeBcCompileError(f"Tick episode count mismatch: {tag}")
        source_contract = source.get("authoritative_native_contract") or {}
        identities = {
            "source_contract_sha256": source_contract.get("contract_sha256"),
            "source_contract_file_sha256": source_contract.get("contract_file_sha256"),
            "episode_contract_sha256": metadata.get("authoritative_contract_sha256"),
            "episode_contract_file_sha256": metadata.get(
                "authoritative_contract_file_sha256"
            ),
        }
        expected_identities = {
            "source_contract_sha256": contract_sha256,
            "source_contract_file_sha256": contract_file_sha256,
            "episode_contract_sha256": contract_sha256,
            "episode_contract_file_sha256": contract_file_sha256,
        }
        if {key: str(value or "") for key, value in identities.items()} != expected_identities:
            raise NativeBcCompileError(
                f"source/episode contract differs from CLI contract: {tag}"
            )
        indexed_contract_sha = indexed[tag].get("contract_sha256")
        indexed_contract_file_sha = indexed[tag].get("contract_file_sha256")
        if indexed_contract_sha is not None and str(indexed_contract_sha) != contract_sha256:
            raise NativeBcCompileError(f"manifest contract SHA differs from CLI contract: {tag}")
        if (
            indexed_contract_file_sha is not None
            and str(indexed_contract_file_sha) != contract_file_sha256
        ):
            raise NativeBcCompileError(
                f"manifest contract file SHA differs from CLI contract: {tag}"
            )
        source_rows.append(
            {
                "battle_tag": tag,
                "tick_store_root": str(
                    tick_store_root
                    if replay_extent == FULL_SUCCESS_EXTENT
                    else audit_prefix_store_root
                ),
                "tick_data_path": entry["tick_data_path"],
                "tick_index_path": entry["tick_index_path"],
                "tick_count": int(entry["ticks"]),
                "tick_payload_sha256": str(entry["payload_sha256"]),
                "tick_payload_size": int(entry["payload_size"]),
                "source_path": str(path),
                "source_sha256": source_sha,
                "source_group": source_group,
                "player_tags": players,
                **extent_contract,
                "mask_metadata_sha256": _digest(dict(masks)),
                "prefix_ability_evidence": prefix_ability_by_tag.get(tag, []),
            }
        )
    assignments, split_audit = _assign_components(
        source_rows,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    episodes: list[EpisodeInput] = []
    for row in source_rows:
        split, component = assignments[row["battle_tag"]]
        episodes.append(
            EpisodeInput(
                **row, split=split, component_sha256=component
            )
        )
    by_split: dict[str, list[EpisodeInput]] = defaultdict(list)
    for episode in sorted(episodes, key=lambda value: (value.split, value.battle_tag)):
        by_split[episode.split].append(episode)
    shard_specs: list[OutputShard] = []
    for split in ("train", "validation", "test"):
        pending: list[EpisodeInput] = []
        rows = 0
        shard_index = 0
        for episode in by_split[split]:
            cost = episode.compiled_tick_count * 2
            if pending and rows + cost > maximum_rows_per_shard:
                relative = f"shards/{split}-{shard_index:05d}"
                content = _digest(
                    {
                        "split": split,
                        "episodes": [asdict(value) for value in pending],
                    }
                )
                shard_specs.append(OutputShard(relative, split, shard_index, tuple(pending), rows, content))
                shard_index += 1
                pending, rows = [], 0
            pending.append(episode)
            rows += cost
        if pending:
            relative = f"shards/{split}-{shard_index:05d}"
            content = _digest(
                {"split": split, "episodes": [asdict(value) for value in pending]}
            )
            shard_specs.append(OutputShard(relative, split, shard_index, tuple(pending), rows, content))
    if {value.split for value in shard_specs} != {"train", "validation", "test"}:
        raise NativeBcCompileError("compile plan produced an empty split")
    card_vocabulary, id_to_card_token, source_to_card_token = _card_vocab(contract)
    ability_vocabulary, id_to_ability_token = _ability_vocab(contract)
    worker_contract = {
        "native_contract_path": str(native_contract),
        "native_card_id_to_token": {
            str(key): value for key, value in id_to_card_token.items()
        },
        "source_card_to_token": source_to_card_token,
        "native_ability_id_to_token": {
            str(key): value for key, value in id_to_ability_token.items()
        },
    }
    capacity_reference = (
        dict(existing_authenticated["capacity_preflight"])
        if existing_authenticated is not None
        else _build_capacity_preflight(
            output_root=output_root,
            tick_store_root=tick_store_root,
            episodes=episodes,
            raw_plan=worker_contract,
            estimated_total_rows=sum(
                value.compiled_tick_count * 2 for value in episodes
            ),
            maximum_rows_per_shard=maximum_rows_per_shard,
            planned_shards=len(shard_specs),
        )
    )
    input_contract = {
        "tick_store_manifest_path": str(tick_store_root / "manifest.json"),
        "tick_store_manifest_sha256": sha256_file(tick_store_root / "manifest.json"),
        "tick_store_content_sha256": store["content_sha256"],
        "audit_prefix_store_root": str(audit_prefix_store_root),
        "audit_prefix_store_manifest_path": str(
            audit_prefix_store_root / "manifest.json"
        ),
        "audit_prefix_store_manifest_sha256": sha256_file(
            audit_prefix_store_root / "manifest.json"
        ),
        "audit_prefix_store_content_sha256": prefix_store["content_sha256"],
        "schema5_manifest_path": str(schema5_manifest),
        "schema5_manifest_sha256": sha256_file(schema5_manifest),
        "native_contract_path": str(native_contract),
        "native_contract_file_sha256": contract_file_sha256,
        "native_contract_sha256": contract_sha256,
        "referenced_deployment_mask_sha256": sorted(referenced_masks),
        "deployment_mask_manifest_sha256": sha256_file(
            mask_store.root / "manifest.json"
        ),
        "deployment_mask_content_sha256": mask_manifest["content_sha256"],
        "referenced_audit_prefix_deployment_mask_sha256": sorted(
            referenced_prefix_masks
        ),
        "audit_prefix_deployment_mask_manifest_sha256": sha256_file(
            prefix_mask_store.root / "manifest.json"
        ),
        "audit_prefix_deployment_mask_content_sha256": prefix_mask_manifest[
            "content_sha256"
        ],
        "native_generation_receipt_path": str(native_generation_receipt),
        "native_generation_receipt_sha256": native_generation_coverage[
            "receipt_sha256"
        ],
        "native_generation_results_path": native_generation_coverage[
            "results_path"
        ],
        "native_generation_results_sha256": native_generation_coverage[
            "results_sha256"
        ],
        "source_token_coverage_receipt_path": (
            None
            if source_token_coverage is None
            else source_token_coverage["receipt_path"]
        ),
        "source_token_coverage_receipt_sha256": (
            None
            if source_token_coverage is None
            else source_token_coverage["receipt_sha256"]
        ),
        "source_token_coverage_canonical_sha256": (
            None
            if source_token_coverage is None
            else source_token_coverage["canonical_sha256"]
        ),
    }
    compiler_contract = {
        "kind": COMPILER_KIND,
        "schema_version": 4,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "maximum_rows_per_shard": int(maximum_rows_per_shard),
        "actor_information": "public_only_v1",
        "storage_schema": {
            "grid": GRID_STORAGE,
            "selected_position_mask": POSITION_MASK_STORAGE,
            "ability_position_mask": POSITION_MASK_STORAGE,
        },
        "action_alignment": "source_tick_plus_episode_metadata_offset",
        "entity_identity": "discrete_native_card_token_v1",
        "mask_policy": "full_complete_prefix_visible_hand_partial_fail_closed_v1",
        "components": {
            "compiler_sha256": sha256_file(Path(__file__).resolve()),
            "training_schema_sha256": sha256_file(
                (Path(__file__).parent / "training_v1" / "schema.py").resolve()
            ),
            "deployment_masks_sha256": sha256_file(
                (Path(__file__).parent / "tick_store_v1" / "deployment_masks.py").resolve()
            ),
            "native_coverage_validator_sha256": sha256_file(
                (Path(__file__).parent / "one_click_v1.py").resolve()
            ),
            "token_coverage_validator_sha256": sha256_file(
                (Path(__file__).parent / "token_coverage_v1.py").resolve()
            ),
        },
    }
    input_content_sha256 = _digest(
        {"inputs": input_contract, "compiler": compiler_contract}
    )
    plan = {
        "kind": PLAN_KIND,
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_content_sha256": input_content_sha256,
        "inputs": input_contract,
        "compiler": compiler_contract,
        "tick_store_root": str(tick_store_root),
        "output_root": str(output_root),
        "episodes": len(episodes),
        "estimated_rows": sum(
            value.compiled_tick_count * 2 for value in episodes
        ),
        "split_audit": split_audit,
        "card_vocabulary": card_vocabulary,
        "native_card_id_to_token": {str(key): value for key, value in id_to_card_token.items()},
        "source_card_to_token": source_to_card_token,
        "ability_vocabulary": ability_vocabulary,
        "native_ability_id_to_token": {str(key): value for key, value in id_to_ability_token.items()},
        "capacity_preflight": capacity_reference,
        "shards": [
            {
                **asdict(value),
                "episodes": [asdict(episode) for episode in value.episodes],
            }
            for value in shard_specs
        ],
    }
    plan["plan_content_sha256"] = _digest(plan)
    existing = existing_plan_path
    if existing_authenticated is not None:
        old = existing_authenticated
        normalized_plan = json.loads(_canonical_bytes(plan))
        expected_without_time = {
            key: value
            for key, value in normalized_plan.items()
            if key not in {"created_utc", "plan_content_sha256"}
        }
        old_without_time = {
            key: value
            for key, value in old.items()
            if key not in {"created_utc", "plan_content_sha256"}
        }
        if old_without_time != expected_without_time:
            raise NativeBcCompileError(
                "existing compile-plan differs from deterministic current plan"
            )
        # Keep the original plan byte identity/timestamp on resume.
        return old
    _atomic_json(existing, plan)
    _atomic_bytes(
        output_root / "compile-plan.sha256",
        f"{sha256_file(existing)}  compile-plan.json\n".encode("ascii"),
    )
    return load_compile_plan(existing)


def _mask_array(rows: Sequence[str], *, actor_side: int) -> np.ndarray:
    raw = np.fromiter(
        (cell == "1" for row in rows for cell in row), dtype=np.bool_, count=POSITION_COUNT
    ).reshape(ARENA_ROWS, ARENA_COLUMNS)
    if actor_side == 1:
        raw = raw[::-1, ::-1]
    return raw.reshape(-1)


def _cell(x: int, y: int) -> int:
    if not 0 <= x < 18_000 or not 0 <= y < 32_000:
        raise NativeBcCompileError(f"native coordinate outside arena: {(x, y)}")
    return min(31, y // 1000) * ARENA_COLUMNS + min(17, x // 1000)


def _tower_map(actor: ActorTick) -> dict[tuple[int, int, int], Any]:
    return {(tower.side, tower.role, tower.lane): tower for tower in actor.towers}


def _public_scalars(actor: ActorTick, state: TickState) -> np.ndarray:
    towers = _tower_map(actor)

    def hp(relation: int, role: int, lane: int) -> float:
        tower = towers.get((relation, role, lane))
        if tower is None or tower.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, tower.hp / tower.max_hp))

    own = [entity for entity in actor.entities if entity.relation == 0]
    enemy = [entity for entity in actor.entities if entity.relation == 1]
    return np.asarray(
        [
            # Fixed public 3-minute regulation + 2-minute overtime horizon;
            # using the recorded episode stop here would leak future length.
            min(1.0, actor.tick / 6000.0),
            actor.own_player.elixir_raw / 100_000.0,
            float(bool(state.episode.commands_allowed) and not actor.episode.terminated),
            float(actor.episode.terminated),
            actor.episode.own_crowns / 3.0,
            actor.episode.enemy_crowns / 3.0,
            hp(0, 0, -1), hp(0, 1, 0), hp(0, 1, 1),
            hp(1, 0, -1), hp(1, 1, 0), hp(1, 1, 1),
            math.log1p(len(own)) / math.log(257.0),
            math.log1p(len(enemy)) / math.log(257.0),
            math.log1p(sum(max(0, value.hp) for value in own)) / math.log(1_000_001.0),
            math.log1p(sum(max(0, value.hp) for value in enemy)) / math.log(1_000_001.0),
        ],
        dtype=np.float32,
    )


def _grid(actor: ActorTick) -> np.ndarray:
    result = np.zeros((len(GRID_CHANNELS), ARENA_ROWS, ARENA_COLUMNS), dtype=np.float32)
    for tower in actor.towers:
        index = _cell(tower.x, tower.y)
        row, column = divmod(index, ARENA_COLUMNS)
        offset = 0 if tower.side == 0 else 2
        result[offset, row, column] = 1.0
        result[offset + 1, row, column] = (
            max(0.0, min(1.0, tower.hp / tower.max_hp)) if tower.max_hp > 0 else 0.0
        )
    for entity in actor.entities:
        index = _cell(entity.x, entity.y)
        row, column = divmod(index, ARENA_COLUMNS)
        offset = 4 if entity.relation == 0 else 6
        result[offset, row, column] += 1.0 / 16.0
        if entity.max_hp > 0:
            result[offset + 1, row, column] += max(
                0.0, min(1.0, entity.hp / entity.max_hp)
            ) / 16.0
    return np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


def _sparse_grid_row(actor: ActorTick) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact quantized grid row without allocating 4,608 zeros."""
    accumulated: dict[int, np.float32] = {}

    def flat(channel: int, row: int, column: int) -> int:
        return (channel * ARENA_ROWS + row) * ARENA_COLUMNS + column

    def assign(channel: int, row: int, column: int, value: float) -> None:
        accumulated[flat(channel, row, column)] = np.float32(value)

    def add(channel: int, row: int, column: int, value: float) -> None:
        key = flat(channel, row, column)
        accumulated[key] = np.float32(
            accumulated.get(key, np.float32(0.0)) + np.float32(value)
        )

    for tower in actor.towers:
        index = _cell(tower.x, tower.y)
        row, column = divmod(index, ARENA_COLUMNS)
        offset = 0 if tower.side == 0 else 2
        assign(offset, row, column, 1.0)
        assign(
            offset + 1,
            row,
            column,
            max(0.0, min(1.0, tower.hp / tower.max_hp))
            if tower.max_hp > 0
            else 0.0,
        )
    for entity in actor.entities:
        index = _cell(entity.x, entity.y)
        row, column = divmod(index, ARENA_COLUMNS)
        offset = 4 if entity.relation == 0 else 6
        add(offset, row, column, 1.0 / 16.0)
        if entity.max_hp > 0:
            add(
                offset + 1,
                row,
                column,
                max(0.0, min(1.0, entity.hp / entity.max_hp)) / 16.0,
            )
    indices = np.asarray(sorted(accumulated), dtype=np.uint16)
    raw_values = np.asarray(
        [accumulated[int(index)] for index in indices], dtype=np.float32
    )
    values = np.rint(np.clip(raw_values, 0.0, 1.0) * 255.0).astype(np.uint8)
    nonzero = values != 0
    return indices[nonzero], values[nonzero]


def _event_maps(plan: BattlePlan, offset: int) -> dict[tuple[int, int], tuple[str, Any]]:
    result: dict[tuple[int, int], tuple[str, Any]] = {}
    for action in plan.actions:
        key = (int(action.tick) + offset, int(action.side))
        if key in result:
            raise NativeBcCompileError(f"same-side action collision: {plan.battle_tag}/{key}")
        result[key] = ("deploy", action)
    for action in plan.ability_events:
        key = (int(action.tick) + offset, int(action.side))
        if key in result:
            raise NativeBcCompileError(f"same-side action collision: {plan.battle_tag}/{key}")
        result[key] = ("ability", action)
    return result


def _compile_actor(
    states: Sequence[TickState],
    metadata: Mapping[str, Any],
    plan: BattlePlan,
    *,
    actor_side: int,
    tick_store_root: Path,
    id_to_card_token: Mapping[int, int],
    source_to_card_token: Mapping[str, int],
    id_to_ability_token: Mapping[int, int],
    max_ability_slots: int,
    replay_extent: str = FULL_SUCCESS_EXTENT,
    action_label_tick_stop_exclusive: int | None = None,
    timing_censor_tick_exclusive: int | None = None,
    prefix_ability_evidence: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    if not states:
        raise NativeBcCompileError(f"empty Tick episode: {plan.battle_tag}")
    offset = int(metadata.get(ACTION_EXECUTION_OFFSET_METADATA, -999))
    if offset < 0 or offset > 2:
        raise NativeBcCompileError(f"unsupported action execution offset: {offset}")
    events = _event_maps(plan, offset)
    mask_store = DeploymentMaskStore(tick_store_root, create=False)
    is_prefix = replay_extent == VALID_PREFIX_EXTENT
    if replay_extent not in {FULL_SUCCESS_EXTENT, VALID_PREFIX_EXTENT}:
        raise NativeBcCompileError("unsupported actor replay extent")
    mask_metadata = mask_store.verify_episode_metadata(
        metadata, require_complete=not is_prefix
    )
    mask_entries = {
        (int(value["side"]), int(value["deck_index"])): value
        for value in mask_metadata["entries"]
    }
    sidecars: dict[str, dict[str, Any]] = {}
    for entry in mask_metadata["entries"]:
        for reference in (entry, *entry.get("dynamic_label_variants", [])):
            digest = str(reference["content_sha256"])
            if digest not in sidecars:
                sidecars[digest] = mask_store.load(digest, allow_cached=True)
    dense_mask_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    side_plan = plan.sides[actor_side]
    enemy_plan = plan.sides[1 - actor_side]
    deck_tokens = np.asarray(
        [source_to_card_token[str(card.source_token)] for card in side_plan.deck],
        dtype=np.int16,
    )
    enemy_source_tokens = [str(card.source_token) for card in enemy_plan.deck]
    allowed_ability_base = {value.base_card_id for value in ability_cards(side_plan.deck)}
    allowed_ability_form = {value.native_form_id for value in ability_cards(side_plan.deck)}
    count = len(states)
    arrays: dict[str, np.ndarray] = {
        "public_scalars": np.zeros((count, len(PUBLIC_SCALARS)), dtype=np.float32),
        "own_deck_tokens": np.repeat(deck_tokens[None, :], count, axis=0),
        "hand_tokens": np.zeros((count, 4), dtype=np.int16),
        "next_card_token": np.zeros(count, dtype=np.int16),
        "revealed_enemy_tokens": np.zeros((count, 8), dtype=np.int16),
        "ability_tokens": np.zeros((count, max_ability_slots), dtype=np.int16),
        "delta_ticks": np.ones(count, dtype=np.float32),
        "timing_exposure_ticks": np.ones(count, dtype=np.float32),
        "card_mask": np.zeros((count, 4), dtype=np.uint8),
        "action_kind_mask": np.zeros((count, 2), dtype=np.uint8),
        "ability_mask": np.zeros((count, max_ability_slots), dtype=np.uint8),
        "play_now": np.zeros(count, dtype=np.uint8),
        "action_kind": np.full(count, -100, dtype=np.int16),
        "card_slot": np.full(count, -100, dtype=np.int16),
        "position": np.full(count, -100, dtype=np.int16),
        "ability_slot": np.full(count, -100, dtype=np.int16),
        "ability_position": np.full(count, -100, dtype=np.int16),
        "timing_label_mask": np.zeros(count, dtype=np.uint8),
        "kind_label_mask": np.zeros(count, dtype=np.uint8),
        "card_label_mask": np.zeros(count, dtype=np.uint8),
        "position_label_mask": np.zeros(count, dtype=np.uint8),
        "ability_label_mask": np.zeros(count, dtype=np.uint8),
        "ability_position_label_mask": np.zeros(count, dtype=np.uint8),
        "sample_weight": np.ones(count, dtype=np.float32),
        "replay_extent": np.full(
            count, 1 if is_prefix else 0, dtype=np.uint8
        ),
    }
    revealed_enemy: list[int] = []
    grid_offsets = np.zeros(count + 1, dtype=np.int64)
    grid_index_buffer = array("H")
    grid_value_buffer = bytearray()
    grid_entry_count = 0
    selected_position_masks: list[np.ndarray] = []
    ability_position_masks: list[np.ndarray] = []
    entity_offsets = np.zeros(count + 1, dtype=np.int64)
    entity_tokens: list[int] = []
    entity_positions: list[int] = []
    entity_relations: list[int] = []
    entity_numeric: list[tuple[float, float, float]] = []
    enemy_events = sorted(
        (
            int(action.tick) + offset,
            enemy_source_tokens[int(action.logical_card_index)],
        )
        for action in plan.actions
        if int(action.side) != actor_side
    )
    enemy_cursor = 0
    for row, state in enumerate(states):
        actor = actor_projection(state, actor_side=actor_side)
        while enemy_cursor < len(enemy_events) and enemy_events[enemy_cursor][0] < state.tick:
            token = source_to_card_token[enemy_events[enemy_cursor][1]]
            if token not in revealed_enemy:
                revealed_enemy.append(token)
            enemy_cursor += 1
        arrays["revealed_enemy_tokens"][row, : min(8, len(revealed_enemy))] = revealed_enemy[:8]
        occupied, occupied_values = _sparse_grid_row(actor)
        grid_index_buffer.frombytes(occupied.tobytes())
        grid_value_buffer.extend(occupied_values.tobytes())
        grid_entry_count += len(occupied)
        grid_offsets[row + 1] = grid_entry_count
        arrays["public_scalars"][row] = _public_scalars(actor, state)
        hand_indices = actor.own_player.hand
        if (
            len(hand_indices) != 4
            or any(value < -1 or value >= 8 for value in hand_indices)
            or sum(value == -1 for value in hand_indices) not in (0, 1)
            or len({value for value in hand_indices if value >= 0})
            != sum(value >= 0 for value in hand_indices)
        ):
            raise NativeBcCompileError(
                f"invalid exact native hand/refill transient: "
                f"{plan.battle_tag}/{state.tick}"
            )
        arrays["hand_tokens"][row] = np.asarray(
            [0 if value == -1 else deck_tokens[value] for value in hand_indices],
            dtype=np.int16,
        )
        next_index = actor.own_player.next_deck_index
        if (
            next_index < 0
            or next_index >= 8
            or next_index in hand_indices
        ):
            raise NativeBcCompileError(
                f"invalid exact next-card/refill semantics: "
                f"{plan.battle_tag}/{state.tick}"
            )
        arrays["next_card_token"][row] = deck_tokens[next_index]

        for entity in actor.entities:
            if int(entity.card_id) < 0:
                # Native traces also expose non-card helpers/effects.  They
                # remain represented in the public occupancy/HP grid, but are
                # not forged into the categorical card vocabulary.
                continue
            try:
                token = id_to_card_token[int(entity.card_id)]
            except KeyError as error:
                raise NativeBcCompileError(
                    f"native entity card ID is outside frozen vocabulary: {entity.card_id}"
                ) from error
            entity_tokens.append(token)
            entity_positions.append(_cell(entity.x, entity.y))
            entity_relations.append(entity.relation)
            entity_numeric.append((
                max(0.0, min(1.0, entity.level / 16.0)),
                max(0.0, min(1.0, entity.hp / entity.max_hp)) if entity.max_hp > 0 else 0.0,
                math.log1p(max(0, entity.max_hp)) / math.log(1_000_001.0),
            ))
        entity_offsets[row + 1] = len(entity_tokens)

        command_allowed = bool(state.episode.commands_allowed) and not bool(state.episode.terminated)
        card_position_masks: dict[int, np.ndarray] = {}
        if command_allowed:
            for hand_slot, deck_index in enumerate(hand_indices):
                if int(deck_index) == -1:
                    continue
                reference = mask_entries.get((actor_side, int(deck_index)))
                if reference is None:
                    raise NativeBcCompileError(
                        f"visible native hand slot lacks mask coverage: "
                        f"{plan.battle_tag}/{state.tick}/{actor_side}/{deck_index}"
                    )
                current_event = events.get((state.tick, actor_side))
                if (
                    action_label_tick_stop_exclusive is not None
                    and state.tick >= action_label_tick_stop_exclusive
                ):
                    current_event = None
                card_supervised = current_event is not None and current_event[0] == "deploy"
                selected_reference = resolve_deployment_reference(
                    reference,
                    tick=state.tick,
                    require_dynamic_exact=card_supervised,
                )
                if selected_reference is None:
                    # Dynamic choices have no exact reusable mask away from a
                    # card-supervised Tick.  Mask the slot instead of silently
                    # substituting its first-hand base probe.
                    continue
                content_sha256 = str(selected_reference["content_sha256"])
                sidecar = sidecars[content_sha256]
                rows = mask_store.derive(
                    content_sha256,
                    state,
                    side=actor_side,
                    card_id=int(reference["card_id"]),
                )
                dense_key = (content_sha256, rows)
                position_mask = dense_mask_cache.get(dense_key)
                if position_mask is None:
                    position_mask = _mask_array(rows, actor_side=actor_side)
                    dense_mask_cache[dense_key] = position_mask
                affordable = actor.own_player.elixir_raw >= int(sidecar["card_cost_raw"])
                legal = bool(affordable and position_mask.any())
                arrays["card_mask"][row, hand_slot] = legal
                card_position_masks[hand_slot] = position_mask

        ability_candidates: list[Any] = []
        for entity in sorted(
            (value for value in state.entities if value.side == actor_side),
            key=lambda value: value.key,
        ):
            if entity.ability_slot <= 0:
                continue
            native_id = int(entity.card_id)
            base_id = native_id
            # Form IDs map through the frozen catalog vocabulary; matching an
            # allowed form or base keeps unrelated button-bearing entities out.
            if native_id not in allowed_ability_form and native_id not in allowed_ability_base:
                # Evolution/hero form base lookup is encoded by matching the
                # configured source token's native ID in ``allowed_ability_form``.
                continue
            ability_candidates.append(entity)
        if len(ability_candidates) > max_ability_slots:
            raise NativeBcCompileError(
                f"ability capacity exceeded: {plan.battle_tag}/{state.tick}"
            )
        for slot, entity in enumerate(ability_candidates):
            native_id = int(entity.card_id)
            token = id_to_ability_token.get(native_id)
            if token is None:
                raise NativeBcCompileError(
                    f"ability entity outside frozen vocabulary: {native_id}"
                )
            arrays["ability_tokens"][row, slot] = token
            legal = command_allowed and bool(entity.ability_available)
            arrays["ability_mask"][row, slot] = legal
        arrays["action_kind_mask"][row, 0] = bool(arrays["card_mask"][row].any())
        arrays["action_kind_mask"][row, 1] = bool(arrays["ability_mask"][row].any())
        any_action = bool(arrays["action_kind_mask"][row].any())
        # Timing is a public-state hazard target and does not consume the
        # conditional card/kind masks.  Keeping legal forced-WAIT observations
        # is important for dynamic-choice cards whose exact non-play Tick mask
        # is intentionally unavailable.
        before_censor = (
            timing_censor_tick_exclusive is None
            or state.tick < timing_censor_tick_exclusive
        )
        arrays["timing_label_mask"][row] = command_allowed and before_censor

        event = events.get((state.tick, actor_side))
        if (
            event is not None
            and action_label_tick_stop_exclusive is not None
            and state.tick >= action_label_tick_stop_exclusive
        ):
            event = None
        if event is None:
            continue
        if not bool(arrays["timing_label_mask"][row]):
            raise NativeBcCompileError(
                f"expert action reaches censored/forbidden Tick: "
                f"{plan.battle_tag}/{state.tick}"
            )
        kind, value = event
        prefix_ability = (
            None
            if prefix_ability_evidence is None or kind != "ability"
            else prefix_ability_evidence.get(int(value.source_event_index))
        )
        if is_prefix and kind == "ability" and prefix_ability is None:
            # A failed replay has no authenticated per-ability identity
            # transcript.  Its accepted pre-boundary event still supervises
            # the timing hazard, but never manufactures a conditional token.
            arrays["play_now"][row] = 1
            continue
        if not any_action:
            raise NativeBcCompileError(
                f"expert action has no legal native action kind: {plan.battle_tag}/{state.tick}"
            )
        arrays["play_now"][row] = 1
        arrays["kind_label_mask"][row] = 1
        if kind == "deploy":
            arrays["action_kind"][row] = 0
            deck_index = int(value.logical_card_index)
            try:
                hand_slot = hand_indices.index(deck_index)
            except ValueError as error:
                raise NativeBcCompileError(
                    f"expert deck index absent from exact hand: {plan.battle_tag}/{state.tick}"
                ) from error
            if not arrays["card_mask"][row, hand_slot]:
                raise NativeBcCompileError(
                    f"expert card is masked illegal: {plan.battle_tag}/{state.tick}"
                )
            position = _cell(
                value.x if actor_side == 0 else 17_999 - value.x,
                value.y if actor_side == 0 else 31_999 - value.y,
            )
            position_mask = card_position_masks[hand_slot]
            if not position_mask[position]:
                raise NativeBcCompileError(
                    f"expert position is masked illegal: {plan.battle_tag}/{state.tick}"
                )
            arrays["card_slot"][row] = hand_slot
            arrays["position"][row] = position
            arrays["card_label_mask"][row] = 1
            arrays["position_label_mask"][row] = 1
            selected_position_masks.append(
                np.packbits(position_mask, bitorder="little")
            )
        else:
            arrays["action_kind"][row] = 1
            legal_slots = np.flatnonzero(arrays["ability_mask"][row])
            if prefix_ability is None:
                selected_slots = legal_slots
            else:
                selected_entity_id = int(
                    prefix_ability["selected_entity_id"]
                )
                selected_slots = np.asarray(
                    [
                        slot
                        for slot, entity in enumerate(ability_candidates)
                        if int(entity.key) == selected_entity_id
                        and int(entity.card_id)
                        == int(prefix_ability["selected_native_form_id"])
                        and bool(arrays["ability_mask"][row, slot])
                    ],
                    dtype=np.int64,
                )
            if len(selected_slots) != 1:
                raise NativeBcCompileError(
                    f"expert ability identity is not uniquely authenticated: "
                    f"{plan.battle_tag}/{state.tick}"
                )
            arrays["ability_slot"][row] = int(selected_slots[0])
            arrays["ability_label_mask"][row] = 1
    missing_events = [
        key
        for key in events
        if key[1] == actor_side
        and (
            action_label_tick_stop_exclusive is None
            or key[0] < action_label_tick_stop_exclusive
        )
        and not any(state.tick == key[0] for state in states)
    ]
    if missing_events:
        raise NativeBcCompileError(
            f"expert action Tick is absent from Tick Store: {plan.battle_tag}/{missing_events[:3]}"
        )
    arrays["entity_offsets"] = entity_offsets
    arrays["grid_offsets"] = grid_offsets
    arrays["grid_indices"] = np.asarray(
        grid_index_buffer, dtype=np.uint16
    )
    arrays["grid_values"] = np.frombuffer(
        grid_value_buffer, dtype=np.uint8
    ).copy()
    arrays["selected_position_mask_packed"] = np.asarray(
        selected_position_masks, dtype=np.uint8
    ).reshape(-1, POSITION_MASK_BYTES)
    arrays["ability_position_mask_packed"] = np.asarray(
        ability_position_masks, dtype=np.uint8
    ).reshape(-1, POSITION_MASK_BYTES)
    arrays["entity_tokens"] = np.asarray(entity_tokens, dtype=np.int16)
    arrays["entity_positions"] = np.asarray(entity_positions, dtype=np.int16)
    arrays["entity_relations"] = np.asarray(entity_relations, dtype=np.uint8)
    arrays["entity_numeric"] = np.asarray(
        entity_numeric, dtype=np.float32
    ).reshape(-1, len(ENTITY_NUMERIC_FIELDS))
    return arrays


def _save_npy(path: Path, value: np.ndarray) -> str:
    np.save(path, value, allow_pickle=False)
    return sha256_file(path)


def _verify_existing_shard(
    path: Path, raw_spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    metadata_path = path / "shard.json"
    if not metadata_path.is_file():
        return None
    try:
        raw_metadata = metadata_path.read_bytes()
        value = json.loads(raw_metadata)
    except Exception as error:
        raise NativeBcCompileError(f"existing shard metadata is invalid: {path}") from error
    if not isinstance(value, Mapping) or _canonical_bytes(value) != raw_metadata:
        raise NativeBcCompileError(f"existing shard metadata is not canonical: {path}")
    _require_keys(
        value,
        {
            "kind", "schema_version", "metadata_content_sha256",
            "content_sha256", "split", "shard_index", "rows", "sequences",
            "battles", "max_entities", "max_ability_slots",
            "sequence_identity", "storage_bytes", "file_sha256",
        },
        "compiled shard metadata",
    )
    metadata_content_sha = _require_sha(
        value.get("metadata_content_sha256"),
        "shard.metadata_content_sha256",
    )
    if metadata_content_sha != _digest(
        {
            key: item
            for key, item in value.items()
            if key != "metadata_content_sha256"
        }
    ):
        raise NativeBcCompileError(f"existing shard metadata content SHA changed: {path}")
    if (
        value.get("kind") != SHARD_KIND
        or int(value.get("schema_version", -1)) != SCHEMA_VERSION
        or value.get("content_sha256") != raw_spec.get("content_sha256")
        or value.get("split") != raw_spec.get("split")
        or int(value.get("shard_index", -1)) != int(raw_spec.get("index", -2))
    ):
        raise NativeBcCompileError(f"existing shard belongs to different inputs: {path}")
    hashes = value.get("file_sha256")
    actual_files = {
        file.name for file in path.glob("*.npy") if file.is_file()
    }
    if (
        not isinstance(hashes, Mapping)
        or set(str(name) for name in hashes) != actual_files
    ):
        raise NativeBcCompileError(f"existing shard lacks file hashes: {path}")
    for name, expected in hashes.items():
        file = path / str(name)
        if (
            not file.is_file()
            or not _SHA256_RE.fullmatch(str(expected))
            or sha256_file(file) != str(expected)
        ):
            raise NativeBcCompileError(f"existing shard is corrupt: {file}")
    offsets = np.load(path / "sequence_offsets.npy", mmap_mode="r")
    entity_offsets = np.load(path / "entity_offsets.npy", mmap_mode="r")
    ability_tokens = np.load(path / "ability_tokens.npy", mmap_mode="r")
    expected_identity = [
        {
            "battle_tag": str(episode["battle_tag"]),
            "actor_side": side,
            "source_sha256": str(episode["source_sha256"]),
            "source_group": str(episode["source_group"]),
            "player_tags": list(episode["player_tags"]),
            "replay_extent": str(episode["replay_extent"]),
            "timing_target": str(episode["timing_target"]),
            "terminal_target": str(episode["terminal_target"]),
            "extent_sha256": str(episode["extent_sha256"]),
            "mask_metadata_sha256": str(episode["mask_metadata_sha256"]),
        }
        for episode in raw_spec["episodes"]
        for side in (0, 1)
    ]
    expected_rows = sum(
        int(episode["compiled_tick_count"]) * 2
        for episode in raw_spec["episodes"]
    )
    expected_sequences = len(expected_identity)
    maximum_entities = int(np.diff(entity_offsets).max(initial=0))
    storage_bytes = sum((path / name).stat().st_size for name in actual_files)
    if (
        offsets.ndim != 1
        or len(offsets) != expected_sequences + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != expected_rows
        or np.any(np.diff(offsets) <= 0)
        or int(value.get("rows", -1)) != expected_rows
        or int(value.get("sequences", -1)) != expected_sequences
        or int(value.get("battles", -1)) != len(raw_spec["episodes"])
        or int(value.get("max_entities", -1)) != maximum_entities
        or ability_tokens.ndim != 2
        or int(value.get("max_ability_slots", -1)) != int(ability_tokens.shape[1])
        or value.get("sequence_identity") != expected_identity
        or int(value.get("storage_bytes", -1)) != storage_bytes
    ):
        raise NativeBcCompileError(
            f"existing shard metadata disagrees with plan/arrays: {path}"
        )
    return value


def _compile_output_shard(
    output_root_text: str,
    tick_store_root_text: str,
    raw_spec: Mapping[str, Any],
    raw_plan: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = Path(output_root_text)
    tick_store_root = Path(tick_store_root_text)
    relative = str(raw_spec["relative_path"])
    final = output_root / relative
    existing = _verify_existing_shard(final, raw_spec)
    if existing is not None:
        return {**existing, "resumed": True, "relative_path": relative}
    temporary = final.with_name(final.name + f".{os.getpid()}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    id_to_card_token = {
        int(key): int(value) for key, value in raw_plan["native_card_id_to_token"].items()
    }
    source_to_card_token = {
        str(key): int(value) for key, value in raw_plan["source_card_to_token"].items()
    }
    id_to_ability_token = {
        int(key): int(value) for key, value in raw_plan["native_ability_id_to_token"].items()
    }
    replay_contract = load_native_ingest_contract(
        Path(str(raw_plan["native_contract_path"])).resolve(strict=True)
    )
    compiled: list[dict[str, np.ndarray]] = []
    capacity_episode_estimates: list[dict[str, Any]] = []
    sequence_identity: list[dict[str, Any]] = []
    maximum_entities = 0
    maximum_abilities = MAX_ABILITY_SLOTS
    readers: dict[tuple[str, str], ShardReader] = {}
    reservation_path: Path | None = None
    try:
        for raw_episode in raw_spec["episodes"]:
            episode = EpisodeInput(**raw_episode)
            episode_compiled: list[dict[str, np.ndarray]] = []
            source_path = Path(episode.source_path)
            if sha256_file(source_path) != episode.source_sha256:
                raise NativeBcCompileError(f"source changed during compilation: {episode.battle_tag}")
            source = json.loads(source_path.read_text(encoding="utf-8-sig"))
            plan = compile_battle(
                source, native_ingest_contract=replay_contract
            )
            if plan.source_schema_version != 5 or not plan.native_replay_ready:
                raise NativeBcCompileError(f"schema-v5 plan is not native ready: {episode.battle_tag}")
            key = (episode.tick_data_path, episode.tick_index_path)
            reader = readers.get(key)
            if reader is None:
                reader = ShardReader(Path(key[0]), Path(key[1]))
                readers[key] = reader
            native_episode = reader.episode(episode.battle_tag)
            if hashlib.sha256(native_episode.blob).hexdigest() != episode.tick_payload_sha256:
                raise NativeBcCompileError(f"Tick payload changed: {episode.battle_tag}")
            stored_states = list(native_episode.iter_ticks())
            if len(stored_states) != episode.tick_count:
                raise NativeBcCompileError(
                    f"stored Tick count changed: {episode.battle_tag}"
                )
            metadata = dict(native_episode.metadata)
            extent_contract = _episode_extent_contract(
                metadata,
                tick_count=episode.tick_count,
                replay_extent=episode.replay_extent,
            )
            if any(
                getattr(episode, name) != value
                for name, value in extent_contract.items()
            ):
                raise NativeBcCompileError(
                    f"compiled replay extent differs from plan: {episode.battle_tag}"
                )
            active_tick_store_root = Path(episode.tick_store_root)
            active_mask_store = DeploymentMaskStore(
                active_tick_store_root, create=False
            )
            mask_metadata = active_mask_store.verify_episode_metadata(
                metadata,
                require_complete=episode.replay_extent == FULL_SUCCESS_EXTENT,
            )
            if _digest(dict(mask_metadata)) != episode.mask_metadata_sha256:
                raise NativeBcCompileError(
                    f"compiled mask metadata differs from plan: {episode.battle_tag}"
                )
            if episode.replay_extent == VALID_PREFIX_EXTENT:
                states = [
                    state
                    for state in stored_states
                    if state.tick < episode.timing_censor_tick_exclusive
                ]
                if (
                    len(states) != episode.compiled_tick_count
                    or not states
                    or states[0].tick != episode.observation_tick_start
                    or states[-1].tick + 1
                    != episode.timing_censor_tick_exclusive
                    or any(state.episode.terminated for state in states)
                ):
                    raise NativeBcCompileError(
                        f"audit-prefix retained rows cross censor/terminal boundary: "
                        f"{episode.battle_tag}"
                    )
                coverage = metadata[REPLAY_EXTENT_METADATA_KEY]["mask_coverage"]
                visible_references = sum(
                    value >= 0
                    for state in states
                    for player in state.players
                    for value in player.hand
                )
                empty_actor_ticks = sum(
                    -1 in player.hand
                    for state in states
                    for player in state.players
                )
                safe_deploy_labels = sum(
                    int(action.tick)
                    + int(metadata[ACTION_EXECUTION_OFFSET_METADATA])
                    < episode.action_label_tick_stop_exclusive
                    for action in plan.actions
                )
                if (
                    int(coverage["visible_slot_references"])
                    != visible_references
                    or int(coverage["empty_slot_actor_ticks"])
                    != empty_actor_ticks
                    or int(coverage["safe_deploy_labels"])
                    != safe_deploy_labels
                ):
                    raise NativeBcCompileError(
                        f"audit-prefix mask coverage differs from retained Tick rows: "
                        f"{episode.battle_tag}"
                    )
            else:
                states = stored_states
            maximum_entities = max(
                maximum_entities, max((len(state.entities) for state in states), default=0)
            )
            label_stop = (
                None
                if episode.replay_extent == FULL_SUCCESS_EXTENT
                else episode.action_label_tick_stop_exclusive
            )
            label_audit = verify_deployment_labels(
                states,
                metadata,
                active_mask_store,
                (
                    {
                        "tick": int(action.tick)
                        + int(metadata[ACTION_EXECUTION_OFFSET_METADATA]),
                        "side": int(action.side),
                        "deck_index": int(action.logical_card_index),
                        "x": int(action.x),
                        "y": int(action.y),
                    }
                    for action in plan.actions
                    if label_stop is None
                    or int(action.tick)
                    + int(metadata[ACTION_EXECUTION_OFFSET_METADATA])
                    < label_stop
                ),
                require_complete=episode.replay_extent == FULL_SUCCESS_EXTENT,
            )
            if label_audit.get("all_legal") is not True:
                raise NativeBcCompileError(
                    f"native deployment label audit failed: {episode.battle_tag} "
                    f"{label_audit.get('violations', [])[:3]}"
                )
            if episode.replay_extent == VALID_PREFIX_EXTENT and int(
                metadata[REPLAY_EXTENT_METADATA_KEY]["mask_coverage"][
                    "safe_deploy_labels"
                ]
            ) != int(label_audit["checked"]):
                raise NativeBcCompileError(
                    f"audit-prefix safe deployment count changed: "
                    f"{episode.battle_tag}"
                )
            for actor_side in (0, 1):
                actor_prefix_abilities = {
                    int(value["source_event_index"]): value
                    for value in episode.prefix_ability_evidence
                    if int(value["actor_side"]) == actor_side
                }
                arrays = _compile_actor(
                    states,
                    metadata,
                    plan,
                    actor_side=actor_side,
                    tick_store_root=active_tick_store_root,
                    id_to_card_token=id_to_card_token,
                    source_to_card_token=source_to_card_token,
                    id_to_ability_token=id_to_ability_token,
                    max_ability_slots=maximum_abilities,
                    replay_extent=episode.replay_extent,
                    action_label_tick_stop_exclusive=(
                        None
                        if episode.replay_extent == FULL_SUCCESS_EXTENT
                        else episode.action_label_tick_stop_exclusive
                    ),
                    timing_censor_tick_exclusive=(
                        None
                        if episode.replay_extent == FULL_SUCCESS_EXTENT
                        else episode.timing_censor_tick_exclusive
                    ),
                    prefix_ability_evidence=actor_prefix_abilities,
                )
                compiled.append(arrays)
                episode_compiled.append(arrays)
                sequence_identity.append(
                    {
                        "battle_tag": episode.battle_tag,
                        "actor_side": actor_side,
                        "source_sha256": episode.source_sha256,
                        "source_group": episode.source_group,
                        "player_tags": list(episode.player_tags),
                        "replay_extent": episode.replay_extent,
                        "timing_target": episode.timing_target,
                        "terminal_target": episode.terminal_target,
                        "extent_sha256": episode.extent_sha256,
                        "mask_metadata_sha256": episode.mask_metadata_sha256,
                    }
                )
            actor_rows = sum(len(value["play_now"]) for value in episode_compiled)
            entity_observations = sum(
                int(value["entity_offsets"][-1]) for value in episode_compiled
            )
            maximum_episode_entities = max(
                (
                    int(np.diff(value["entity_offsets"]).max(initial=0))
                    for value in episode_compiled
                ),
                default=0,
            )
            sparse_mask_row_bytes = sum(
                8
                * (
                    len(value["selected_position_mask_packed"])
                    + len(value["ability_position_mask_packed"])
                )
                for value in episode_compiled
            )
            array_payload_bytes = (
                sum(
                    int(array_value.nbytes)
                    for value in episode_compiled
                    for array_value in value.values()
                )
                + sparse_mask_row_bytes
                + 3 * np.dtype(np.int64).itemsize
            )
            capacity_episode_estimates.append({
                "battle_tag": episode.battle_tag,
                "tick_count": int(episode.tick_count),
                "tick_payload_size": int(episode.tick_payload_size),
                "tick_payload_bytes_per_tick": (
                    episode.tick_payload_size / episode.tick_count
                ),
                "actor_rows": actor_rows,
                "array_payload_bytes": array_payload_bytes,
                "array_payload_bytes_per_actor_row": (
                    array_payload_bytes / actor_rows
                ),
                "mean_entities_per_actor_row": (
                    entity_observations / actor_rows
                ),
                "max_entities_per_actor_row": maximum_episode_entities,
            })
            # Actor arrays are now self-contained; release decoded TickState
            # objects before moving to the next battle in this shard.
            del states
        if raw_plan.get("capacity_preflight_path"):
            capacity = _read_capacity_preflight(
                Path(str(raw_plan["capacity_preflight_path"]))
            )
            logical_payload_bytes = sum(
                int(array_value.nbytes)
                for sequence in compiled
                for array_value in sequence.values()
            )
            logical_payload_bytes += sum(
                8
                * (
                    len(sequence["selected_position_mask_packed"])
                    + len(sequence["ability_position_mask_packed"])
                )
                for sequence in compiled
            )
            sampled_upper = math.ceil(
                float(capacity["sample_max_episode_bytes_per_actor_row"])
                * int(raw_spec["estimated_rows"])
                * CAPACITY_SAFETY_FACTOR
            ) + CAPACITY_SHARD_OVERHEAD_BYTES
            exact_upper = logical_payload_bytes + CAPACITY_SHARD_OVERHEAD_BYTES
            reservation_path = _acquire_capacity_reservation(
                output_root,
                relative_path=relative,
                requested_bytes=max(sampled_upper, exact_upper),
            )
        lengths = [len(value["play_now"]) for value in compiled]
        offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        hashes: dict[str, str] = {
            "sequence_offsets.npy": _save_npy(temporary / "sequence_offsets.npy", offsets)
        }
        entity_counts = np.concatenate(
            [np.diff(sequence["entity_offsets"]) for sequence in compiled]
        )
        global_entity_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(entity_counts, dtype=np.int64))
        )
        hashes["entity_offsets.npy"] = _save_npy(
            temporary / "entity_offsets.npy", global_entity_offsets
        )
        grid_counts = np.concatenate(
            [np.diff(sequence["grid_offsets"]) for sequence in compiled]
        )
        global_grid_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(grid_counts, dtype=np.int64))
        )
        hashes["grid_offsets.npy"] = _save_npy(
            temporary / "grid_offsets.npy", global_grid_offsets
        )
        for name in compiled[0]:
            if name in {"entity_offsets", "grid_offsets"}:
                continue
            value = np.concatenate([sequence[name] for sequence in compiled], axis=0)
            hashes[f"{name}.npy"] = _save_npy(temporary / f"{name}.npy", value)
        for label_name, rows_name in (
            ("position_label_mask", "selected_position_mask_rows"),
            ("ability_position_label_mask", "ability_position_mask_rows"),
        ):
            labels = np.concatenate(
                [sequence[label_name] for sequence in compiled], axis=0
            )
            sparse_rows = np.flatnonzero(labels).astype(np.int64, copy=False)
            hashes[f"{rows_name}.npy"] = _save_npy(
                temporary / f"{rows_name}.npy", sparse_rows
            )
        metadata_body = {
            "kind": SHARD_KIND,
            "schema_version": SCHEMA_VERSION,
            "content_sha256": raw_spec["content_sha256"],
            "split": raw_spec["split"],
            "shard_index": int(raw_spec["index"]),
            "rows": int(offsets[-1]),
            "sequences": len(compiled),
            "battles": len(raw_spec["episodes"]),
            "max_entities": maximum_entities,
            "max_ability_slots": maximum_abilities,
            "sequence_identity": sequence_identity,
            "storage_bytes": sum(
                (temporary / name).stat().st_size for name in hashes
            ),
            "file_sha256": hashes,
        }
        metadata = {
            **metadata_body,
            "metadata_content_sha256": _digest(metadata_body),
        }
        _atomic_json(temporary / "shard.json", metadata)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, final)
        except OSError:
            # Another equivalent worker may have won the atomic publish race.
            existing = _verify_existing_shard(final, raw_spec)
            if existing is None:
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            return {**existing, "resumed": True, "relative_path": relative}
        result = {
            **metadata,
            "resumed": False,
            "relative_path": relative,
        }
        if not raw_plan.get("capacity_preflight_path"):
            result["capacity_episode_estimates"] = capacity_episode_estimates
        return result
    finally:
        for reader in readers.values():
            reader.close()
        _release_capacity_reservation(output_root, reservation_path)
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def compile_planned_shards(
    plan: Mapping[str, Any],
    *,
    worker_index: int = 0,
    worker_count: int = 1,
    process_workers: int = 1,
) -> list[dict[str, Any]]:
    """Compile one deterministic worker partition; completed shards are reused."""
    # The authenticated plan is loaded/created once by the CLI.  Per-worker
    # calls repeat all canonical structure checks but avoid re-hashing the
    # entire 100K input corpus; each worker still verifies every source and
    # Tick payload it consumes.
    plan = _authenticate_plan_argument(plan, verify_live_inputs=False)
    capacity = _read_capacity_preflight(
        Path(str(plan["capacity_preflight"]["path"]))
    )
    output_root = Path(str(plan["output_root"]))
    usage = shutil.disk_usage(output_root)
    mandatory_reserve = max(
        CAPACITY_MINIMUM_RESERVE_BYTES,
        math.ceil(
            int(usage.total) * CAPACITY_FILESYSTEM_RESERVE_FRACTION
        ),
    )
    if int(usage.free) < mandatory_reserve:
        raise NativeBcCompileError(
            "capacity reserve is already breached before shard compilation: "
            f"free={usage.free}, mandatory_reserve={mandatory_reserve}"
        )
    recommended_workers = int(capacity["recommended_max_parallel_compile_workers"])
    if worker_count <= 0 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("invalid worker partition")
    specs = [
        value
        for index, value in enumerate(plan["shards"])
        if index % worker_count == worker_index
    ]
    requested_process_workers = max(1, int(process_workers))
    recommended_for_partition = max(1, recommended_workers // worker_count)
    effective_process_workers = min(
        requested_process_workers,
        recommended_for_partition,
        max(1, len(specs)),
    )
    worker_contract = {
        "native_contract_path": plan["inputs"]["native_contract_path"],
        "capacity_preflight_path": plan["capacity_preflight"]["path"],
        "native_card_id_to_token": plan["native_card_id_to_token"],
        "source_card_to_token": plan["source_card_to_token"],
        "native_ability_id_to_token": plan["native_ability_id_to_token"],
    }
    arguments = [
        (
            str(plan["output_root"]),
            str(plan["tick_store_root"]),
            value,
            worker_contract,
        )
        for value in specs
    ]
    if effective_process_workers <= 1:
        completed = [_compile_output_shard(*value) for value in arguments]
    else:
        with ProcessPoolExecutor(max_workers=effective_process_workers) as executor:
            completed = list(executor.map(_compile_output_shard_star, arguments))
    receipt = {
        "kind": "cr_native_bc_compile_worker_receipt_v1",
        "schema_version": 1,
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "requested_process_workers": requested_process_workers,
        "capacity_recommended_total_process_workers": recommended_workers,
        "capacity_recommended_partition_process_workers": (
            recommended_for_partition
        ),
        "effective_process_workers": effective_process_workers,
        "planned_shards": len(specs),
        "completed_shards": len(completed),
        "rows": sum(int(value["rows"]) for value in completed),
    }
    _atomic_json(
        output_root
        / "worker-receipts"
        / f"worker-{worker_index:05d}-of-{worker_count:05d}.json",
        receipt,
    )
    return completed


def _compile_output_shard_star(arguments: Sequence[Any]) -> dict[str, Any]:
    return _compile_output_shard(*arguments)


def _final_compiled_token_coverage(
    plan: Mapping[str, Any],
    compiled_tags: set[str],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Rejoin source, generator evidence and completed BC shards per token."""

    source_receipt_path = plan["inputs"].get(
        "source_token_coverage_receipt_path"
    )
    if source_receipt_path is None:
        raise NativeBcCompileError(
            "production token coverage requires an explicit source receipt"
        )
    source_auth = _authenticate_source_token_coverage_receipt(
        Path(str(source_receipt_path)),
        schema5_manifest=Path(str(plan["inputs"]["schema5_manifest_path"])),
        native_contract=Path(str(plan["inputs"]["native_contract_path"])),
        contract_sha256=str(plan["inputs"]["native_contract_sha256"]),
        contract_file_sha256=str(
            plan["inputs"]["native_contract_file_sha256"]
        ),
    )
    source_coverage = source_auth["source_coverage"]
    source_events_sha = str(
        source_coverage["ability_event_registry"]["source_events_sha256"]
    )
    results_path = Path(
        str(plan["inputs"]["native_generation_results_path"])
    ).resolve(strict=True)
    if sha256_file(results_path) != plan["inputs"][
        "native_generation_results_sha256"
    ]:
        raise NativeBcCompileError("native results changed before token coverage")
    by_tag: dict[str, Mapping[str, Any]] = {}
    for row in _json_lines(results_path):
        tag = str(row.get("battle_tag") or "")
        if not tag or tag in by_tag:
            raise NativeBcCompileError("native token evidence result tags are malformed")
        by_tag[tag] = row
    success_tags = {
        tag for tag, row in by_tag.items()
        if row.get("teacher_forced_success") is True
    }
    full_tags = {
        str(episode["battle_tag"])
        for shard in plan["shards"]
        for episode in shard["episodes"]
        if episode["replay_extent"] == FULL_SUCCESS_EXTENT
    }
    prefix_tags = {
        str(episode["battle_tag"])
        for shard in plan["shards"]
        for episode in shard["episodes"]
        if episode["replay_extent"] == VALID_PREFIX_EXTENT
    }
    failed_prefix_tags = {
        tag
        for tag, row in by_tag.items()
        if row.get("teacher_forced_success") is False
        and isinstance(row.get("audit_prefix_tick_store_entry"), Mapping)
    }
    if (
        success_tags != full_tags
        or failed_prefix_tags != prefix_tags
        or compiled_tags != full_tags | prefix_tags
        or full_tags & prefix_tags
    ):
        raise NativeBcCompileError(
            "compiled full/prefix tags differ from native result union"
        )

    sequence_locations: dict[tuple[str, int], tuple[Path, int]] = {}
    ordered_tags: list[str] = []
    output_root = Path(str(plan["output_root"])).resolve()
    for raw_shard in plan["shards"]:
        shard_path = output_root / str(raw_shard["relative_path"])
        for episode_index, episode in enumerate(raw_shard["episodes"]):
            tag = str(episode["battle_tag"])
            ordered_tags.append(tag)
            for side in (0, 1):
                key = (tag, side)
                if key in sequence_locations:
                    raise NativeBcCompileError(
                        f"compiled token sequence is duplicated: {tag}/{side}"
                    )
                sequence_locations[key] = (
                    shard_path,
                    episode_index * 2 + side,
                )
    if set(ordered_tags) != compiled_tags or len(ordered_tags) != len(compiled_tags):
        raise NativeBcCompileError(
            "compiled token sequence locations differ from admitted tags"
        )

    source_paths: dict[str, Path] = {}
    episode_inputs: dict[str, Mapping[str, Any]] = {}
    for shard in plan["shards"]:
        for episode in shard["episodes"]:
            tag = str(episode["battle_tag"])
            source_paths[tag] = Path(str(episode["source_path"])).resolve(strict=True)
            episode_inputs[tag] = episode
    loaded_contract = load_native_ingest_contract(
        Path(str(plan["inputs"]["native_contract_path"]))
    )
    actor_records: list[dict[str, Any]] = []
    prefix_actor_records: list[dict[str, Any]] = []
    prefix_generator_actor_records: list[dict[str, Any]] = []
    readers: dict[tuple[str, str], ShardReader] = {}
    mask_store = DeploymentMaskStore(
        Path(str(plan["tick_store_root"])), create=False
    )
    prefix_mask_store = DeploymentMaskStore(
        Path(str(plan["inputs"]["audit_prefix_store_root"])), create=False
    )
    compiled_array_names = (
        "sequence_offsets", "play_now", "kind_label_mask", "action_kind",
        "card_label_mask", "position_label_mask", "card_slot", "position",
        "hand_tokens", "card_mask", "ability_label_mask", "ability_slot",
        "ability_tokens", "ability_mask", "ability_position_label_mask",
        "timing_label_mask", "replay_extent",
    )
    current_compiled_path: Path | None = None
    current_compiled_arrays: dict[str, np.ndarray] = {}

    def close_compiled_arrays() -> None:
        nonlocal current_compiled_arrays
        for array_value in current_compiled_arrays.values():
            mapping = getattr(array_value, "_mmap", None)
            if mapping is not None:
                mapping.close()
        current_compiled_arrays = {}

    def compiled_arrays(path: Path) -> Mapping[str, np.ndarray]:
        nonlocal current_compiled_path, current_compiled_arrays
        if current_compiled_path != path:
            close_compiled_arrays()
            current_compiled_arrays = {
                name: np.load(path / f"{name}.npy", mmap_mode="r")
                for name in compiled_array_names
            }
            current_compiled_path = path
        return current_compiled_arrays

    try:
        for tag in ordered_tags:
            result = by_tag[tag]
            source = json.loads(source_paths[tag].read_bytes())
            native_plan = compile_battle(
                source, native_ingest_contract=loaded_contract
            )
            episode_input = episode_inputs[tag]
            reader_key = (
                str(episode_input["tick_data_path"]),
                str(episode_input["tick_index_path"]),
            )
            reader = readers.get(reader_key)
            if reader is None:
                reader = ShardReader(Path(reader_key[0]), Path(reader_key[1]))
                readers[reader_key] = reader
            episode = reader.episode(tag)
            stored_states = tuple(episode.iter_ticks())
            is_prefix = episode_input["replay_extent"] == VALID_PREFIX_EXTENT
            states = (
                tuple(
                    state
                    for state in stored_states
                    if state.tick
                    < int(episode_input["timing_censor_tick_exclusive"])
                )
                if is_prefix
                else stored_states
            )
            states_by_tick = {state.tick: state for state in states}
            if len(states_by_tick) != len(states):
                raise NativeBcCompileError(
                    f"token evidence Tick Store has duplicate Ticks: {tag}"
                )
            active_mask_store = prefix_mask_store if is_prefix else mask_store
            mask_metadata = active_mask_store.verify_episode_metadata(
                episode.metadata, require_complete=not is_prefix
            )
            mask_entries = list(mask_metadata["entries"])
            if is_prefix:
                raw_prefix_actors = result.get(
                    "prefix_token_coverage_actor_evidence"
                )
                if (
                    not isinstance(raw_prefix_actors, list)
                    or len(raw_prefix_actors) != 2
                ):
                    raise NativeBcCompileError(
                        f"compiled prefix lacks generator actor evidence: {tag}"
                    )
                prefix_actors = {
                    int(row.get("actor_side", -1)): row
                    for row in raw_prefix_actors
                    if isinstance(row, Mapping)
                }
                if set(prefix_actors) != {0, 1}:
                    raise NativeBcCompileError(
                        f"compiled prefix actor evidence sides changed: {tag}"
                    )
                for actor in prefix_actors.values():
                    if (
                        actor.get("action_label_tick_stop_exclusive")
                        != episode_input["action_label_tick_stop_exclusive"]
                        or actor.get("timing_target") != PREFIX_TIMING_TARGET
                        or actor.get("replay_extent_sha256")
                        != episode_input["extent_sha256"]
                    ):
                        raise NativeBcCompileError(
                            f"compiled prefix actor extent binding changed: {tag}"
                        )
                    prefix_generator_actor_records.append(dict(actor))
                if len(states) != int(episode_input["compiled_tick_count"]):
                    raise NativeBcCompileError(
                        f"compiled prefix sequence crosses censor boundary: {tag}"
                    )
                offset = int(episode.metadata[ACTION_EXECUTION_OFFSET_METADATA])
                action_stop = int(
                    episode_input["action_label_tick_stop_exclusive"]
                )
                for side in (0, 1):
                    planned_prefix_abilities = {
                        int(value["source_event_index"]): value
                        for value in episode_input["prefix_ability_evidence"]
                        if int(value["actor_side"]) == side
                    }
                    prefix_actor = prefix_actors[side]
                    if prefix_actor.get("deck_tokens") != [
                        card.source_token for card in native_plan.sides[side].deck
                    ] or {
                        int(label.get("source_event_index", -1))
                        for label in prefix_actor.get("ability_labels") or []
                        if isinstance(label, Mapping)
                    } != set(planned_prefix_abilities):
                        raise NativeBcCompileError(
                            f"compiled prefix ability/deck evidence changed: {tag}/{side}"
                        )
                    shard_path, sequence_index = sequence_locations[(tag, side)]
                    arrays = compiled_arrays(shard_path)
                    offsets = arrays["sequence_offsets"]
                    sequence_start = int(offsets[sequence_index])
                    sequence_stop = int(offsets[sequence_index + 1])
                    if (
                        sequence_stop - sequence_start != len(states)
                        or np.any(
                            np.asarray(
                                arrays["replay_extent"][
                                    sequence_start:sequence_stop
                                ]
                            ) != 1
                        )
                    ):
                        raise NativeBcCompileError(
                            f"compiled prefix provenance differs from final shard: "
                            f"{tag}/{side}"
                        )
                    state_start_tick = states[0].tick
                    safe_deploys = [
                        action
                        for action in native_plan.actions
                        if int(action.side) == side
                        and int(action.tick) + offset < action_stop
                    ]
                    safe_abilities = [
                        event
                        for event in native_plan.ability_events
                        if int(event.side) == side
                        and int(event.tick) + offset < action_stop
                    ]
                    deploy_labels: list[dict[str, Any]] = []
                    expected_deploy_rows: set[int] = set()
                    expected_play_rows: set[int] = set()
                    for action in safe_deploys:
                        execution_tick = int(action.tick) + offset
                        row = sequence_start + execution_tick - state_start_tick
                        card = native_plan.sides[side].deck[
                            int(action.logical_card_index)
                        ]
                        matches = [
                            entry
                            for entry in mask_entries
                            if int(entry["side"]) == side
                            and int(entry["deck_index"])
                            == int(action.logical_card_index)
                            and int(entry["card_id"]) == int(card.card_id)
                            and int(entry["form_flags"]) == int(card.form_flags)
                        ]
                        if len(matches) != 1:
                            raise NativeBcCompileError(
                                f"compiled prefix mask identity is ambiguous: "
                                f"{tag}/{side}/{execution_tick}"
                            )
                        reference = resolve_deployment_reference(
                            matches[0],
                            tick=execution_tick,
                            require_dynamic_exact=True,
                        )
                        if reference is None:
                            raise NativeBcCompileError(
                                f"compiled prefix deploy lacks exact mask: "
                                f"{tag}/{side}/{execution_tick}"
                            )
                        payload = active_mask_store.load(
                            str(reference["content_sha256"])
                        )
                        state = states_by_tick.get(execution_tick)
                        if state is None:
                            raise NativeBcCompileError(
                                f"compiled prefix deploy Tick is absent: "
                                f"{tag}/{side}/{execution_tick}"
                            )
                        hand = list(state.players[side].hand)
                        try:
                            expected_slot = hand.index(
                                int(action.logical_card_index)
                            )
                        except ValueError as error:
                            raise NativeBcCompileError(
                                f"compiled prefix deploy selects empty hand: "
                                f"{tag}/{side}/{execution_tick}"
                            ) from error
                        expected_token = int(
                            plan["source_card_to_token"][str(card.source_token)]
                        )
                        expected_position = _cell(
                            int(action.x) if side == 0 else 17_999 - int(action.x),
                            int(action.y) if side == 0 else 31_999 - int(action.y),
                        )
                        if (
                            int(arrays["play_now"][row]) != 1
                            or int(arrays["kind_label_mask"][row]) != 1
                            or int(arrays["action_kind"][row]) != 0
                            or int(arrays["card_label_mask"][row]) != 1
                            or int(arrays["position_label_mask"][row]) != 1
                            or int(arrays["card_slot"][row]) != expected_slot
                            or int(arrays["position"][row]) != expected_position
                            or int(arrays["hand_tokens"][row, expected_slot])
                            != expected_token
                            or int(arrays["card_mask"][row, expected_slot]) != 1
                        ):
                            raise NativeBcCompileError(
                                f"compiled prefix deploy differs from final shard: "
                                f"{tag}/{side}/{execution_tick}"
                            )
                        expected_deploy_rows.add(row)
                        expected_play_rows.add(row)
                        deploy_labels.append({
                            "source_event_index": int(action.source_event_index),
                            "source_token": str(card.source_token),
                            "resolved_native_form_id": int(
                                payload["resolved_data_id"]
                            ),
                            "accepted": True,
                            "mask_legal": True,
                            "compiled": True,
                        })
                    for event in safe_abilities:
                        ability_row = (
                            sequence_start
                            + int(event.tick)
                            + offset
                            - state_start_tick
                        )
                        expected_play_rows.add(ability_row)
                    expected_ability_rows = {
                        sequence_start
                        + int(event.tick)
                        + offset
                        - state_start_tick
                        for event in safe_abilities
                        if int(event.source_event_index)
                        in planned_prefix_abilities
                    }
                    row_slice = slice(sequence_start, sequence_stop)
                    actual = lambda name: {
                        sequence_start + int(value)
                        for value in np.flatnonzero(
                            np.asarray(arrays[name][row_slice])
                        )
                    }
                    if (
                        actual("card_label_mask") != expected_deploy_rows
                        or actual("position_label_mask") != expected_deploy_rows
                        or actual("kind_label_mask")
                        != expected_deploy_rows | expected_ability_rows
                        or actual("ability_label_mask")
                        != expected_ability_rows
                        or actual("ability_position_label_mask")
                        or actual("play_now") != expected_play_rows
                    ):
                        raise NativeBcCompileError(
                            f"compiled prefix label rows differ from censor-safe events: "
                            f"{tag}/{side}"
                        )
                    ability_labels = []
                    for event in safe_abilities:
                        evidence = planned_prefix_abilities.get(
                            int(event.source_event_index)
                        )
                        if evidence is None:
                            continue
                        row = (
                            sequence_start + int(event.tick) + offset
                            - state_start_tick
                        )
                        slot = int(arrays["ability_slot"][row])
                        expected_token = int(
                            plan["native_ability_id_to_token"][
                                str(int(evidence["selected_native_form_id"]))
                            ]
                        )
                        if (
                            int(arrays["kind_label_mask"][row]) != 1
                            or int(arrays["action_kind"][row]) != 1
                            or not 0 <= slot < arrays["ability_tokens"].shape[1]
                            or int(arrays["ability_tokens"][row, slot])
                            != expected_token
                            or int(arrays["ability_mask"][row, slot]) != 1
                        ):
                            raise NativeBcCompileError(
                                f"compiled prefix ability differs from final shard: "
                                f"{tag}/{side}/{int(event.tick) + offset}"
                            )
                        ability_labels.append({
                            "source_event_index": int(event.source_event_index),
                            "resolved_token": str(evidence["resolved_token"]),
                            "resolved_native_form_id": int(
                                evidence["selected_native_form_id"]
                            ),
                            "selected_entity_id": int(
                                evidence["selected_entity_id"]
                            ),
                            "resolution_transcript_sha256": str(
                                evidence["transcript_sha256"]
                            ),
                            "accepted": True,
                            "legal": True,
                            "compiled": True,
                        })
                    prefix_actor_records.append({
                        "battle_tag": tag,
                        "actor_side": side,
                        "full_success": False,
                        "censored_prefix": True,
                        "deck_tokens": [
                            card.source_token
                            for card in native_plan.sides[side].deck
                        ],
                        "deploy_labels": deploy_labels,
                        "ability_labels": ability_labels,
                    })
                continue
            raw_actors = result.get("token_coverage_actor_evidence")
            if not isinstance(raw_actors, list) or len(raw_actors) != 2:
                raise NativeBcCompileError(
                    f"full-success result lacks actor token evidence: {tag}"
                )
            actors = {
                int(row.get("actor_side", -1)): row
                for row in raw_actors if isinstance(row, Mapping)
            }
            if set(actors) != {0, 1}:
                raise NativeBcCompileError(f"actor token evidence sides changed: {tag}")
            for side in (0, 1):
                actor = actors[side]
                shard_path, sequence_index = sequence_locations[(tag, side)]
                arrays = compiled_arrays(shard_path)
                offsets = arrays["sequence_offsets"]
                sequence_start = int(offsets[sequence_index])
                sequence_stop = int(offsets[sequence_index + 1])
                if sequence_stop - sequence_start != len(states):
                    raise NativeBcCompileError(
                        f"compiled actor sequence length differs from Tick Store: {tag}/{side}"
                    )
                expected_deck = [
                    card.source_token for card in native_plan.sides[side].deck
                ]
                if actor.get("deck_tokens") != expected_deck:
                    raise NativeBcCompileError(
                        f"actor token evidence deck differs from source: {tag}/{side}"
                    )
                expected_deploy = []
                for action in native_plan.actions:
                    if int(action.side) != side:
                        continue
                    card = native_plan.sides[side].deck[
                        int(action.logical_card_index)
                    ]
                    matches = [
                        entry for entry in mask_entries
                        if int(entry["side"]) == side
                        and int(entry["card_id"]) == int(card.card_id)
                        and int(entry["form_flags"]) == int(card.form_flags)
                    ]
                    if len(matches) != 1:
                        raise NativeBcCompileError(
                            f"compiled mask identity is ambiguous: {tag}/{side}"
                        )
                    execution_tick = int(action.tick) + int(
                        episode.metadata[ACTION_EXECUTION_OFFSET_METADATA]
                    )
                    reference = resolve_deployment_reference(
                        matches[0], tick=execution_tick, require_dynamic_exact=True
                    )
                    if reference is None:
                        raise NativeBcCompileError(
                            f"compiled deploy lacks exact mask: {tag}/{side}"
                        )
                    payload = mask_store.load(str(reference["content_sha256"]))
                    expected_deploy.append((
                        int(action.source_event_index),
                        int(action.source_marker_index),
                        int(action.tick),
                        str(card.source_token),
                        int(payload["resolved_data_id"]),
                        str(reference["content_sha256"]),
                    ))
                actual_deploy = sorted(
                    (
                        int(label.get("source_event_index", -1)),
                        int(label.get("source_marker_index", -1)),
                        int(label.get("source_tick", -1)),
                        str(label.get("source_token") or ""),
                        int(label.get("resolved_native_form_id", -1)),
                        str(label.get("mask_content_sha256") or ""),
                    )
                    for label in actor.get("deploy_labels") or []
                    if isinstance(label, Mapping)
                )
                expected_ability = sorted(
                    (
                        int(event.source_event_index),
                        int(event.source_marker_index),
                        int(event.tick),
                    )
                    for event in native_plan.ability_events if int(event.side) == side
                )
                actual_ability = sorted(
                    (
                        int(label.get("source_event_index", -1)),
                        int(label.get("source_marker_index", -1)),
                        int(label.get("source_tick", -1)),
                    )
                    for label in actor.get("ability_labels") or []
                    if isinstance(label, Mapping)
                )
                if sorted(expected_deploy) != actual_deploy or expected_ability != actual_ability:
                    raise NativeBcCompileError(
                        f"actor token evidence events differ from source/native: {tag}/{side}"
                    )
                deploy_labels = {
                    int(label["source_event_index"]): label
                    for label in actor.get("deploy_labels") or []
                }
                ability_labels = {
                    int(label["source_event_index"]): label
                    for label in actor.get("ability_labels") or []
                }
                expected_deploy_rows: set[int] = set()
                expected_ability_rows: set[int] = set()
                state_start_tick = states[0].tick
                for action in native_plan.actions:
                    if int(action.side) != side:
                        continue
                    execution_tick = int(action.tick) + int(
                        episode.metadata[ACTION_EXECUTION_OFFSET_METADATA]
                    )
                    local_row = execution_tick - state_start_tick
                    row = sequence_start + local_row
                    label = deploy_labels.get(int(action.source_event_index))
                    state = states_by_tick.get(execution_tick)
                    if (
                        label is None
                        or state is None
                        or not 0 <= local_row < len(states)
                    ):
                        raise NativeBcCompileError(
                            f"compiled deploy row is absent: {tag}/{side}/{execution_tick}"
                        )
                    hand = list(state.players[side].hand)
                    try:
                        expected_slot = hand.index(int(action.logical_card_index))
                    except ValueError as error:
                        raise NativeBcCompileError(
                            f"compiled deploy exact hand differs: {tag}/{side}/{execution_tick}"
                        ) from error
                    expected_token = int(
                        plan["source_card_to_token"][str(label["source_token"])]
                    )
                    expected_position = _cell(
                        int(action.x) if side == 0 else 17_999 - int(action.x),
                        int(action.y) if side == 0 else 31_999 - int(action.y),
                    )
                    if (
                        int(arrays["play_now"][row]) != 1
                        or int(arrays["kind_label_mask"][row]) != 1
                        or int(arrays["action_kind"][row]) != 0
                        or int(arrays["card_label_mask"][row]) != 1
                        or int(arrays["position_label_mask"][row]) != 1
                        or int(arrays["card_slot"][row]) != expected_slot
                        or int(arrays["position"][row]) != expected_position
                        or int(arrays["hand_tokens"][row, expected_slot])
                        != expected_token
                        or int(arrays["card_mask"][row, expected_slot]) != 1
                    ):
                        raise NativeBcCompileError(
                            f"compiled deploy supervision differs from final shard: "
                            f"{tag}/{side}/{execution_tick}"
                        )
                    expected_deploy_rows.add(row)
                for event in native_plan.ability_events:
                    if int(event.side) != side:
                        continue
                    execution_tick = int(event.tick) + int(
                        episode.metadata[ACTION_EXECUTION_OFFSET_METADATA]
                    )
                    local_row = execution_tick - state_start_tick
                    row = sequence_start + local_row
                    label = ability_labels.get(int(event.source_event_index))
                    if label is None or not 0 <= local_row < len(states):
                        raise NativeBcCompileError(
                            f"compiled ability row is absent: {tag}/{side}/{execution_tick}"
                        )
                    slot = int(arrays["ability_slot"][row])
                    expected_token = int(
                        plan["native_ability_id_to_token"][
                            str(int(label["resolved_native_form_id"]))
                        ]
                    )
                    if (
                        int(arrays["play_now"][row]) != 1
                        or int(arrays["kind_label_mask"][row]) != 1
                        or int(arrays["action_kind"][row]) != 1
                        or int(arrays["ability_label_mask"][row]) != 1
                        or not 0 <= slot < arrays["ability_tokens"].shape[1]
                        or int(arrays["ability_tokens"][row, slot])
                        != expected_token
                        or int(arrays["ability_mask"][row, slot]) != 1
                        or int(arrays["ability_position_label_mask"][row]) != 0
                    ):
                        raise NativeBcCompileError(
                            f"compiled ability supervision differs from final shard: "
                            f"{tag}/{side}/{execution_tick}"
                        )
                    expected_ability_rows.add(row)
                row_slice = slice(sequence_start, sequence_stop)
                actual_deploy_rows = set(
                    sequence_start
                    + int(value)
                    for value in np.flatnonzero(
                        np.asarray(arrays["card_label_mask"][row_slice])
                    )
                )
                actual_position_rows = set(
                    sequence_start
                    + int(value)
                    for value in np.flatnonzero(
                        np.asarray(arrays["position_label_mask"][row_slice])
                    )
                )
                actual_ability_rows = set(
                    sequence_start
                    + int(value)
                    for value in np.flatnonzero(
                        np.asarray(arrays["ability_label_mask"][row_slice])
                    )
                )
                actual_kind_rows = set(
                    sequence_start
                    + int(value)
                    for value in np.flatnonzero(
                        np.asarray(arrays["kind_label_mask"][row_slice])
                    )
                )
                actual_play_rows = set(
                    sequence_start
                    + int(value)
                    for value in np.flatnonzero(
                        np.asarray(arrays["play_now"][row_slice])
                    )
                )
                expected_all_rows = expected_deploy_rows | expected_ability_rows
                if (
                    actual_deploy_rows != expected_deploy_rows
                    or actual_position_rows != expected_deploy_rows
                    or actual_ability_rows != expected_ability_rows
                    or actual_kind_rows != expected_all_rows
                    or actual_play_rows != expected_all_rows
                ):
                    raise NativeBcCompileError(
                        f"compiled label-row coverage differs from source events: {tag}/{side}"
                    )
                for label in actor.get("ability_labels") or []:
                    if not isinstance(label, Mapping):
                        raise NativeBcCompileError("ability token evidence is not an object")
                    tick = int(label.get("execution_tick", -1))
                    entity_id = int(label.get("selected_entity_id", -1))
                    state = states_by_tick.get(tick)
                    matches = [] if state is None else [
                        entity for entity in state.entities
                        if entity.key == entity_id
                    ]
                    if (
                        len(matches) != 1
                        or matches[0].side != side
                        or matches[0].ability_slot <= 0
                        or matches[0].ability_available != 1
                        or int(matches[0].card_id)
                        != int(label.get("resolved_native_form_id", -1))
                    ):
                        raise NativeBcCompileError(
                            f"ability token evidence differs from Tick Store: {tag}/{side}/{tick}"
                        )
                actor_records.append(dict(actor))
    finally:
        close_compiled_arrays()
        for reader in readers.values():
            reader.close()

    try:
        authenticated = authenticate_generator_ability_evidence(
            [*actor_records, *prefix_generator_actor_records],
            contract,
            source_coverage,
            expected_source_events_sha256=source_events_sha,
        )
        transcript_by_event = {
            (
                str(row["battle_tag"]),
                int(row["actor_side"]),
                int(row["source_event_index"]),
            ): str(row["transcript_sha256"])
            for row in authenticated["transcripts"]
        }
        compiled_records: list[dict[str, Any]] = []
        for raw_actor in actor_records:
            actor = json.loads(json.dumps(raw_actor))
            for label in actor["deploy_labels"]:
                label["compiled"] = True
            for label in actor["ability_labels"]:
                key = (
                    str(actor["battle_tag"]),
                    int(actor["actor_side"]),
                    int(label["source_event_index"]),
                )
                digest = transcript_by_event.get(key)
                if digest is None:
                    raise NativeBcCompileError(
                        f"compiled ability lacks authenticated transcript: {key}"
                    )
                label["compiled"] = True
                label["resolution_transcript_sha256"] = digest
            compiled_records.append(actor)
        compiled_records.extend(prefix_actor_records)
        success = summarize_success_token_coverage(
            compiled_records,
            contract,
            source=source_coverage,
            authenticated_ability_transcripts=authenticated,
            expected_source_events_sha256=source_events_sha,
            expected_authenticated_transcripts_sha256=str(
                authenticated["authenticated_transcripts_sha256"]
            ),
        )
        receipt = build_token_coverage_receipt(
            source_coverage,
            success,
            source_auth["adaptive_quotas"],
        )
        digest = coverage_receipt_sha256(receipt)
    except NativeBcCompileError:
        raise
    except Exception as error:
        raise NativeBcCompileError(
            f"token coverage authentication failed: {error}"
        ) from error
    return receipt, digest, source_auth["receipt_sha256"]


def finalize_dataset(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every planned shard and atomically expose ``manifest.json`` last."""
    plan = _authenticate_plan_argument(plan, verify_live_inputs=True)
    output = Path(str(plan["output_root"])).resolve()
    source = Path(str(plan["inputs"]["schema5_manifest_path"])).resolve(strict=True)
    if sha256_file(source) != str(plan["inputs"]["schema5_manifest_sha256"]):
        raise NativeBcCompileError("schema-v5 manifest changed before finalization")
    immutable_inputs = (
        (
            Path(str(plan["inputs"]["tick_store_manifest_path"])),
            str(plan["inputs"]["tick_store_manifest_sha256"]),
            "Tick Store manifest",
        ),
        (
            Path(str(plan["inputs"]["native_contract_path"])),
            str(plan["inputs"]["native_contract_file_sha256"]),
            "native ingest contract",
        ),
        (
            Path(str(plan["inputs"]["native_generation_receipt_path"])),
            str(plan["inputs"]["native_generation_receipt_sha256"]),
            "native generation receipt",
        ),
        (
            Path(str(plan["inputs"]["native_generation_results_path"])),
            str(plan["inputs"]["native_generation_results_sha256"]),
            "native generation results",
        ),
        (
            Path(str(plan["tick_store_root"]))
            / "deployment-masks-v1"
            / "manifest.json",
            str(plan["inputs"]["deployment_mask_manifest_sha256"]),
            "deployment-mask manifest",
        ),
        (
            Path(str(plan["inputs"]["audit_prefix_store_manifest_path"])),
            str(plan["inputs"]["audit_prefix_store_manifest_sha256"]),
            "audit-prefix Tick Store manifest",
        ),
        (
            Path(str(plan["inputs"]["audit_prefix_store_root"]))
            / "deployment-masks-v1"
            / "manifest.json",
            str(plan["inputs"]["audit_prefix_deployment_mask_manifest_sha256"]),
            "audit-prefix deployment-mask manifest",
        ),
    )
    if plan["inputs"].get("source_token_coverage_receipt_path") is not None:
        immutable_inputs += ((
            Path(str(plan["inputs"]["source_token_coverage_receipt_path"])),
            str(plan["inputs"]["source_token_coverage_receipt_sha256"]),
            "source token coverage receipt",
        ),)
    for path, expected, label in immutable_inputs:
        if not path.is_file() or sha256_file(path) != expected:
            raise NativeBcCompileError(f"{label} changed before finalization")
    component_paths = {
        "compiler_sha256": Path(__file__).resolve(),
        "training_schema_sha256": (Path(__file__).parent / "training_v1" / "schema.py").resolve(),
        "deployment_masks_sha256": (
            Path(__file__).parent / "tick_store_v1" / "deployment_masks.py"
        ).resolve(),
        "native_coverage_validator_sha256": (
            Path(__file__).parent / "one_click_v1.py"
        ).resolve(),
        "token_coverage_validator_sha256": (
            Path(__file__).parent / "token_coverage_v1.py"
        ).resolve(),
    }
    for name, path in component_paths.items():
        if sha256_file(path) != str(plan["compiler"]["components"][name]):
            raise NativeBcCompileError(f"compiler component changed before finalization: {name}")
    split_paths: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test")}
    split_stats: dict[str, Counter[str]] = {
        name: Counter() for name in ("train", "validation", "test")
    }
    file_hashes: dict[str, str] = {}
    shard_metadata_hashes: dict[str, str] = {}
    maximum_entities = 1
    maximum_abilities = 1
    assignments: list[dict[str, Any]] = []
    seen_tags: dict[str, str] = {}
    player_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    source_file_splits: dict[str, str] = {}
    for raw in plan["shards"]:
        relative = str(raw["relative_path"])
        path = output / relative
        metadata = _verify_existing_shard(path, raw)
        if metadata is None:
            raise NativeBcCompileError(f"planned shard is not complete: {path}")
        split = str(raw["split"])
        split_paths[split].append(relative)
        split_stats[split].update(
            rows=int(metadata["rows"]),
            sequences=int(metadata["sequences"]),
            battles=int(metadata["battles"]),
        )
        maximum_entities = max(maximum_entities, int(metadata["max_entities"]))
        maximum_abilities = max(maximum_abilities, int(metadata["max_ability_slots"]))
        for name, digest in metadata["file_sha256"].items():
            file_hashes[f"{relative}/{name}"] = str(digest)
        shard_metadata_digest = sha256_file(path / "shard.json")
        file_hashes[f"{relative}/shard.json"] = shard_metadata_digest
        shard_metadata_hashes[relative] = shard_metadata_digest
        for episode in raw["episodes"]:
            tag = str(episode["battle_tag"])
            if tag in seen_tags:
                raise NativeBcCompileError(f"battle crosses output shards: {tag}")
            seen_tags[tag] = split
            for player in episode["player_tags"]:
                previous = player_splits.setdefault(str(player), split)
                if previous != split:
                    raise NativeBcCompileError(f"player split leak: {player}")
            source_group = str(episode["source_group"])
            previous_group = source_splits.setdefault(source_group, split)
            if previous_group != split:
                raise NativeBcCompileError(f"source-group split leak: {source_group}")
            source_sha = str(episode["source_sha256"])
            previous_file = source_file_splits.setdefault(source_sha, split)
            if previous_file != split:
                raise NativeBcCompileError(f"source-file split leak: {source_sha}")
            assignments.append(
                {
                    "kind": ASSIGNMENT_KIND,
                    "schema_version": 1,
                    "battle_tag": tag,
                    "split": split,
                    "component_sha256": episode["component_sha256"],
                    "source_sha256": source_sha,
                    "source_group": source_group,
                    "player_tags": episode["player_tags"],
                    "replay_extent": episode["replay_extent"],
                }
            )
    assignment_raw = b"".join(
        _canonical_bytes(value) for value in sorted(assignments, key=lambda value: value["battle_tag"])
    )
    _atomic_bytes(output / "split-assignments.jsonl", assignment_raw)
    total_rows = sum(value["rows"] for value in split_stats.values())
    total_sequences = sum(value["sequences"] for value in split_stats.values())
    planned_episodes = [
        episode for shard in plan["shards"] for episode in shard["episodes"]
    ]
    full_episodes = [
        episode
        for episode in planned_episodes
        if episode["replay_extent"] == FULL_SUCCESS_EXTENT
    ]
    prefix_episodes = [
        episode
        for episode in planned_episodes
        if episode["replay_extent"] == VALID_PREFIX_EXTENT
    ]
    full_rows = sum(int(value["compiled_tick_count"]) * 2 for value in full_episodes)
    prefix_rows = sum(
        int(value["compiled_tick_count"]) * 2 for value in prefix_episodes
    )
    if (
        full_rows + prefix_rows != total_rows
        or len(full_episodes) + len(prefix_episodes) != len(seen_tags)
        or {str(value["battle_tag"]) for value in full_episodes}
        & {str(value["battle_tag"]) for value in prefix_episodes}
    ):
        raise NativeBcCompileError("compiled full/prefix training union is not exact")
    native_generation_coverage = _authenticate_native_generation_receipt(
        Path(str(plan["inputs"]["native_generation_receipt_path"])),
        schema5_manifest=source,
        native_contract=Path(str(plan["inputs"]["native_contract_path"])),
        contract_sha256=str(plan["inputs"]["native_contract_sha256"]),
        contract_file_sha256=str(
            plan["inputs"]["native_contract_file_sha256"]
        ),
        expected_episodes=len(full_episodes),
    )
    token_coverage_manifest: dict[str, Any]
    if plan["inputs"].get("source_token_coverage_receipt_path") is None:
        token_coverage_manifest = {
            "enforced": False,
            "reason": "legacy_direct_test_without_source_token_receipt",
        }
    else:
        token_receipt, token_digest, source_receipt_sha = (
            _final_compiled_token_coverage(plan, set(seen_tags), contract={
                **load_native_ingest_contract(
                    Path(str(plan["inputs"]["native_contract_path"]))
                ).value
            })
        )
        token_path = output / "token-coverage-receipt.json"
        _atomic_json(token_path, token_receipt)
        token_file_sha = sha256_file(token_path)
        _atomic_bytes(
            output / "token-coverage-receipt.sha256",
            f"{token_file_sha}  token-coverage-receipt.json\n".encode("ascii"),
        )
        gate = (token_receipt.get("evaluation") or {}).get("gate") or {}
        token_coverage_manifest = {
            "enforced": True,
            "receipt_kind": TOKEN_COVERAGE_RECEIPT_KIND,
            "receipt_schema_version": TOKEN_COVERAGE_RECEIPT_SCHEMA_VERSION,
            "receipt": "token-coverage-receipt.json",
            "receipt_file_sha256": token_file_sha,
            "receipt_canonical_sha256": token_digest,
            "source_receipt_sha256": source_receipt_sha,
            "gate": gate,
        }
        if gate.get("admitted") is not True:
            deficits = token_receipt.get("evaluation") or {}
            raise NativeBcCompileError(
                "FAILED_COVERAGE: per-token deficits remain; evidence="
                f"{token_path}; deficits="
                + json.dumps(
                    {
                        "hard_floor": deficits.get("hard_floor_deficits"),
                        "adaptive": deficits.get("adaptive_quota_deficits"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    manifest: dict[str, Any] = {
        "kind": DATASET_KIND,
        "schema_version": SCHEMA_VERSION,
        # Stable across repeated finalization of the same authenticated plan.
        "created_utc": str(plan["created_utc"]),
        "production_ready": bool(
            token_coverage_manifest.get("enforced") is True
            and (token_coverage_manifest.get("gate") or {}).get("admitted")
            is True
        ),
        "native_replay_validated": True,
        "observation_mode": OBSERVATION_MODE,
        "timing_target": "native_tick_hazard_with_right_censored_prefix_v1",
        "actor_information": "public_only_v1",
        "storage_schema": {
            "grid": GRID_STORAGE,
            "selected_position_mask": POSITION_MASK_STORAGE,
            "ability_position_mask": POSITION_MASK_STORAGE,
        },
        "source_manifest": {
            "path": str(source),
            "sha256": plan["inputs"]["schema5_manifest_sha256"],
        },
        "source_inputs": plan["inputs"],
        "capacity_preflight": plan["capacity_preflight"],
        "native_generation_coverage": native_generation_coverage,
        "token_coverage": token_coverage_manifest,
        "dataset_content_sha256": _digest(
            {
                "input": plan["input_content_sha256"],
                "shards": file_hashes,
                "assignments": hashlib.sha256(assignment_raw).hexdigest(),
                "token_coverage": token_coverage_manifest,
            }
        ),
        "dimensions": {
            "grid_channels": len(GRID_CHANNELS),
            "public_scalar_size": len(PUBLIC_SCALARS),
            "card_vocab_size": len(plan["card_vocabulary"]),
            "ability_vocab_size": len(plan["ability_vocabulary"]),
            "max_ability_slots": maximum_abilities,
            "max_entities": maximum_entities,
            "entity_numeric_size": len(ENTITY_NUMERIC_FIELDS),
        },
        "feature_schema": {
            "grid_channels": list(GRID_CHANNELS),
            "public_scalars": list(PUBLIC_SCALARS),
            "entity_identity": "categorical_card_vocabulary_v1",
            "entity_numeric": list(ENTITY_NUMERIC_FIELDS),
        },
        "card_vocabulary": plan["card_vocabulary"],
        "ability_vocabulary": plan["ability_vocabulary"],
        "splits": split_paths,
        "split_statistics": {key: dict(value) for key, value in split_stats.items()},
        "split_contract": {
            "battle_tag_disjoint": True,
            "source_file_disjoint": True,
            "player_holdout_test": True,
            "player_disjoint_all_splits": True,
            "source_group_disjoint_all_splits": True,
            "assignment": "connected_components_of_battle_player_source_group_v1",
            **plan["split_audit"],
        },
        "state_provenance": {
            "mode": "native_full_plus_semantic_censored_prefix_v1",
            "authoritative_rows": 0,
            "native_generated_unanchored_rows": total_rows,
            "native_grid_rows": total_rows,
            "every_native_tick_present": True,
            "full_success_rows": full_rows,
            "censored_prefix_rows": prefix_rows,
            "prefix_terminal_target": "unknown_censored",
        },
        "mask_provenance": {
            "kind": "content_addressed_full_and_partial_native_sidecars_v1",
            "full_referenced_sidecar_sha256": plan["inputs"]["referenced_deployment_mask_sha256"],
            "prefix_referenced_sidecar_sha256": plan["inputs"][
                "referenced_audit_prefix_deployment_mask_sha256"
            ],
            "prefix_policy": PREFIX_MASK_PROVENANCE,
            "missing_sidecars": 0,
            "expert_mask_violations": 0,
            "timing_independent_of_conditional_masks": True,
        },
        "quality_gates": {
            "split_collisions": 0,
            "forbidden_actor_features": 0,
            "nonfinite_features": 0,
            "expert_label_mask_violations": 0,
            "native_action_rejections": 0,
            "terminal_mismatches": 0,
            "terminal_validation_unknown": total_rows,
            "player_holdout_leaks": 0,
            "source_group_leaks": 0,
            "missing_mask_sidecars": 0,
            "entity_token_unknowns": 0,
        },
        "coverage": {
            "battles": len(seen_tags),
            "actor_sequences": total_sequences,
            "rows": total_rows,
            "full_success_episodes": len(full_episodes),
            "full_success_actor_sequences": len(full_episodes) * 2,
            "full_success_rows": full_rows,
            "censored_prefix_episodes": len(prefix_episodes),
            "censored_prefix_actor_sequences": len(prefix_episodes) * 2,
            "censored_prefix_rows": prefix_rows,
            "training_episode_union_exact": True,
            "training_episode_union_sha256": _digest({
                "full_success_tags": sorted(
                    str(value["battle_tag"]) for value in full_episodes
                ),
                "censored_prefix_tags": sorted(
                    str(value["battle_tag"]) for value in prefix_episodes
                ),
            }),
        },
        "compiler": {
            **plan["compiler"],
            "input_content_sha256": plan["input_content_sha256"],
            "resumability": "deterministic_atomic_output_shards_v1",
        },
        "shard_file_sha256": file_hashes,
        "shard_metadata_sha256": shard_metadata_hashes,
    }
    validate_manifest(manifest, root=output)
    for split, paths in split_paths.items():
        for relative in paths:
            validate_shard(output / relative, manifest)
    # ``manifest.json`` is the visibility boundary: publish it only after all
    # immutable shards have passed the full training schema validation.
    _atomic_json(output / "manifest.json", manifest)
    manifest_sha = sha256_file(output / "manifest.json")
    _atomic_bytes(
        output / "manifest.sha256",
        f"{manifest_sha}  manifest.json\n".encode("ascii"),
    )
    result = {
        "kind": "cr_native_tick_store_bc_compile_result_v1",
        "output_root": str(output),
        "manifest_sha256": manifest_sha,
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "battles": len(seen_tags),
        "actor_sequences": total_sequences,
        "rows": total_rows,
        "splits": manifest["split_statistics"],
        "token_coverage": token_coverage_manifest,
    }
    _atomic_json(output / "compile-result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile content-addressed native Tick Store episodes into BC shards"
    )
    parser.add_argument("--tick-store-root", type=Path, required=True)
    parser.add_argument("--audit-prefix-store-root", type=Path, required=True)
    parser.add_argument("--schema5-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-contract", type=Path, required=True)
    parser.add_argument("--native-generation-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-token-coverage-receipt", type=Path, required=True
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-rows-per-shard", type=int, default=524_288)
    parser.add_argument("--io-workers", type=int, default=min(32, max(1, (os.cpu_count() or 1) * 2)))
    parser.add_argument("--process-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.output_root.resolve() / "compile-plan.json"
    if args.finalize_only:
        plan = load_compile_plan(plan_path)
        return finalize_dataset(plan)
    plan = create_compile_plan(
        args.tick_store_root,
        args.schema5_manifest,
        args.output_root,
        args.native_contract,
        args.native_generation_receipt,
        args.source_token_coverage_receipt,
        audit_prefix_store_root=args.audit_prefix_store_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        maximum_rows_per_shard=args.maximum_rows_per_shard,
        io_workers=args.io_workers,
    )
    if args.plan_only:
        return {
            "kind": PLAN_KIND,
            "output_root": plan["output_root"],
            "input_content_sha256": plan["input_content_sha256"],
            "episodes": plan["episodes"],
            "estimated_rows": plan["estimated_rows"],
            "shards": len(plan["shards"]),
            "capacity_preflight": plan["capacity_preflight"],
        }
    completed = compile_planned_shards(
        plan,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        process_workers=args.process_workers,
    )
    if args.worker_count == 1:
        return finalize_dataset(plan)
    worker_receipt = json.loads(
        (
            Path(str(plan["output_root"]))
            / "worker-receipts"
            / f"worker-{args.worker_index:05d}-of-{args.worker_count:05d}.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "kind": "cr_native_tick_store_bc_worker_result_v1",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "completed_shards": len(completed),
        "rows": sum(int(value["rows"]) for value in completed),
        "requested_process_workers": worker_receipt[
            "requested_process_workers"
        ],
        "effective_process_workers": worker_receipt[
            "effective_process_workers"
        ],
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
