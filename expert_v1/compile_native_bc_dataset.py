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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .native_capabilities import ability_cards
from .native_ingest_contract import load_native_ingest_contract
from .native_replay_plan import BattlePlan, compile_battle
from .tick_store_v1.deployment_masks import (
    DeploymentMaskStore,
    derive_deployment_rows,
    resolve_deployment_reference,
    verify_deployment_labels,
)
from .tick_store_v1.shard import SHARD_KIND as TICK_SHARD_KIND
from .tick_store_v1.shard import STORE_KIND as TICK_STORE_KIND
from .tick_store_v1.shard import ShardReader, sha256_file
from .tick_store_v1.schema import ActorTick, TickState, actor_projection
from .training_v1.schema import (
    ARENA_COLUMNS,
    ARENA_ROWS,
    DATASET_KIND,
    POSITION_COUNT,
    POSITION_MASK_BYTES,
    SCHEMA_VERSION,
    SHARD_KIND,
    validate_manifest,
    validate_shard,
)


COMPILER_KIND = "cr_native_tick_store_bc_compiler_v1"
PLAN_KIND = "cr_native_tick_store_bc_compile_plan_v1"
ASSIGNMENT_KIND = "cr_native_tick_store_bc_split_assignment_v1"
OBSERVATION_MODE = "native_state_v1"
ACTION_EXECUTION_OFFSET_METADATA = "action_execution_tick_offset"
ENTITY_NUMERIC_FIELDS = ("level_ratio", "hp_ratio", "log_max_hp")
MAX_ABILITY_SLOTS = 16
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
    source_path: str
    source_sha256: str
    source_group: str
    player_tags: tuple[str, ...]
    split: str
    component_sha256: str


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


def _validate_tick_store(root: Path, *, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NativeBcCompileError(f"Tick Store manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        manifest.get("kind") != TICK_STORE_KIND
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("every_native_tick_present") is not True
        or int(manifest.get("tick_hz", -1)) != 20
    ):
        raise NativeBcCompileError("Tick Store manifest contract changed")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
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
        weight = sum(int(value["tick_count"]) * 2 for value in values)
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
            "native_ability_id_to_token", "shards",
        },
        "compile-plan",
    )
    if (
        plan.get("kind") != PLAN_KIND
        or _require_integer(plan.get("schema_version"), "plan.schema_version", minimum=1)
        != 1
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
            "deployment_mask_content_sha256",
        },
        "compile-plan inputs",
    )
    _require_keys(
        compiler,
        {
            "kind", "schema_version", "seed", "validation_fraction",
            "test_fraction", "maximum_rows_per_shard", "actor_information",
            "action_alignment", "entity_identity", "mask_policy", "components",
        },
        "compile-plan compiler",
    )
    if (
        compiler.get("kind") != COMPILER_KIND
        or _require_integer(
            compiler.get("schema_version"), "compiler.schema_version", minimum=1
        )
        != 1
        or compiler.get("actor_information") != "public_only_v1"
        or compiler.get("action_alignment") != "source_tick_plus_episode_metadata_offset"
        or compiler.get("entity_identity") != "discrete_native_card_token_v1"
        or compiler.get("mask_policy")
        != "content_addressed_native_sidecar_fail_closed_v1"
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
        {"compiler_sha256", "training_schema_sha256", "deployment_masks_sha256"},
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
        "deployment_mask_content_sha256",
    ):
        _require_sha(inputs.get(name), f"inputs.{name}")
    referenced_masks = inputs.get("referenced_deployment_mask_sha256")
    if (
        not isinstance(referenced_masks, list)
        or referenced_masks != sorted(set(str(value) for value in referenced_masks))
    ):
        raise NativeBcCompileError("referenced deployment masks are not sorted unique")
    for value in referenced_masks:
        _require_sha(value, "referenced deployment mask")

    tick_root = Path(str(plan.get("tick_store_root") or "")).resolve()
    output_root = Path(str(plan.get("output_root") or "")).resolve()
    if str(tick_root) != str(plan["tick_store_root"]):
        raise NativeBcCompileError("compile-plan Tick Store path is not canonical absolute")
    if str(output_root) != str(plan["output_root"]):
        raise NativeBcCompileError("compile-plan output path is not canonical absolute")
    if Path(str(inputs["tick_store_manifest_path"])).resolve() != tick_root / "manifest.json":
        raise NativeBcCompileError("compile-plan Tick Store manifest path disagrees with root")
    if plan_path is not None and output_root != plan_path.resolve().parent:
        raise NativeBcCompileError("compile-plan output root disagrees with its location")

    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise NativeBcCompileError("compile-plan has no output shards")
    episode_fields = {
        "battle_tag", "tick_data_path", "tick_index_path", "tick_count",
        "tick_payload_sha256", "source_path", "source_sha256", "source_group",
        "player_tags", "split", "component_sha256",
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
            for name in ("tick_payload_sha256", "source_sha256", "component_sha256"):
                _require_sha(item.get(name), f"episode.{name}")
            for name in ("tick_data_path", "tick_index_path", "source_path"):
                raw_path = str(item.get(name) or "")
                if not Path(raw_path).is_absolute() or str(Path(raw_path).resolve()) != raw_path:
                    raise NativeBcCompileError(f"episode.{name} is not canonical absolute")
            if tick_root not in Path(str(item["tick_data_path"])).resolve().parents:
                raise NativeBcCompileError("episode Tick data escapes Tick Store root")
            if tick_root not in Path(str(item["tick_index_path"])).resolve().parents:
                raise NativeBcCompileError("episode Tick index escapes Tick Store root")
            normalized_episodes.append(item)
            all_episodes.append(item)
        if [value["battle_tag"] for value in normalized_episodes] != sorted(
            value["battle_tag"] for value in normalized_episodes
        ):
            raise NativeBcCompileError("compile-plan shard episodes are not deterministic")
        expected_rows = sum(int(value["tick_count"]) * 2 for value in normalized_episodes)
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
            cost = int(value["tick_count"]) * 2
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
    expected_rows = sum(int(value["tick_count"]) * 2 for value in all_episodes)
    if _require_integer(plan.get("estimated_rows"), "plan.estimated_rows") != expected_rows:
        raise NativeBcCompileError("compile-plan estimated row count changed")

    contract_path = Path(str(inputs["native_contract_path"])).resolve()
    if verify_live_inputs:
        files = (
            (Path(str(inputs["tick_store_manifest_path"])), inputs["tick_store_manifest_sha256"], "Tick Store manifest"),
            (Path(str(inputs["schema5_manifest_path"])), inputs["schema5_manifest_sha256"], "Schema5 manifest"),
            (contract_path, inputs["native_contract_file_sha256"], "native contract"),
            (tick_root / "deployment-masks-v1" / "manifest.json", inputs["deployment_mask_manifest_sha256"], "mask manifest"),
        )
        for path, expected, label in files:
            if not path.is_file() or sha256_file(path) != str(expected):
                raise NativeBcCompileError(f"compile-plan {label} changed")
        tick_manifest = json.loads(
            Path(str(inputs["tick_store_manifest_path"])).read_text(encoding="utf-8-sig")
        )
        if tick_manifest.get("content_sha256") != inputs["tick_store_content_sha256"]:
            raise NativeBcCompileError("compile-plan Tick Store content SHA changed")
        contract, contract_file_sha = _load_contract(contract_path)
        if (
            contract_file_sha != inputs["native_contract_file_sha256"]
            or contract.get("contract_sha256") != inputs["native_contract_sha256"]
        ):
            raise NativeBcCompileError("compile-plan native contract identity changed")
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
        # Rebuild the episode join from the immutable inputs.  This prevents a
        # re-signed plan from omitting a battle, swapping a source path, or
        # pointing a tag at a different Tick frame while preserving plausible
        # aggregate counts.
        _tick_manifest, tick_shards = _validate_tick_store(
            tick_root, workers=min(16, max(1, os.cpu_count() or 1))
        )
        tick_episodes = _episode_index(tick_shards)
        planned_by_tag = {
            str(value["battle_tag"]): value for value in all_episodes
        }
        if set(planned_by_tag) != set(tick_episodes):
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
                tick_entry = tick_episodes[tag]
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
                episode_masks = mask_store.verify_episode_metadata(metadata)
                for entry in episode_masks["entries"]:
                    actual_referenced_masks.add(str(entry["content_sha256"]))
                    actual_referenced_masks.update(
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
        component_paths = {
            "compiler_sha256": Path(__file__).resolve(),
            "training_schema_sha256": (
                Path(__file__).parent / "training_v1" / "schema.py"
            ).resolve(),
            "deployment_masks_sha256": (
                Path(__file__).parent / "tick_store_v1" / "deployment_masks.py"
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


def create_compile_plan(
    tick_store_root: Path,
    schema5_manifest: Path,
    output_root: Path,
    native_contract: Path,
    *,
    seed: int = 20260827,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    maximum_rows_per_shard: int = 32_768,
    io_workers: int = 16,
) -> dict[str, Any]:
    """Verify all immutable inputs and atomically publish a deterministic plan."""
    tick_store_root = tick_store_root.resolve(strict=True)
    schema5_manifest = schema5_manifest.resolve(strict=True)
    native_contract = native_contract.resolve(strict=True)
    output_root = output_root.resolve()
    if maximum_rows_per_shard <= 0:
        raise ValueError("maximum_rows_per_shard must be positive")
    contract, contract_file_sha256 = _load_contract(native_contract)
    contract_sha256 = str(contract["contract_sha256"])
    store, shards = _validate_tick_store(tick_store_root, workers=io_workers)
    episode_entries = _episode_index(shards)
    index_rows = _json_lines(schema5_manifest)
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in index_rows:
        tag = str(row.get("battle_tag") or "")
        if not tag:
            raise NativeBcCompileError("schema-v5 index row lacks battle_tag")
        if tag in indexed:
            raise NativeBcCompileError(f"duplicate schema-v5 index battle: {tag}")
        indexed[tag] = row
    missing = sorted(set(episode_entries) - set(indexed))
    if missing:
        raise NativeBcCompileError(
            f"Tick Store episodes missing from schema-v5 index: {missing[:5]}"
        )
    wanted = set(episode_entries)
    with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
        loaded = list(
            executor.map(
                lambda row: _read_source_input(row, schema5_manifest, wanted),
                (indexed[tag] for tag in sorted(wanted)),
            )
        )
    source_rows: list[dict[str, Any]] = []
    mask_store = DeploymentMaskStore(tick_store_root, create=False)
    mask_manifest = mask_store.verify_manifest()
    referenced_masks: set[str] = set()
    for item in loaded:
        assert item is not None
        tag, source, path, source_sha, source_group, players = item
        entry = episode_entries[tag]
        with ShardReader(Path(entry["tick_data_path"]), Path(entry["tick_index_path"])) as reader:
            episode = reader.episode(tag)
            metadata = dict(episode.metadata)
            if str(metadata.get("source_sha256") or "") != source_sha:
                raise NativeBcCompileError(f"Tick/source SHA identity mismatch: {tag}")
            if int(metadata.get("source_schema_version", -1)) != 5:
                raise NativeBcCompileError(f"Tick episode is not schema-v5: {tag}")
            masks = mask_store.verify_episode_metadata(metadata)
            for value in masks["entries"]:
                referenced_masks.add(str(value["content_sha256"]))
                referenced_masks.update(
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
                "tick_data_path": entry["tick_data_path"],
                "tick_index_path": entry["tick_index_path"],
                "tick_count": int(entry["ticks"]),
                "tick_payload_sha256": str(entry["payload_sha256"]),
                "source_path": str(path),
                "source_sha256": source_sha,
                "source_group": source_group,
                "player_tags": players,
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
            cost = episode.tick_count * 2
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
    input_contract = {
        "tick_store_manifest_path": str(tick_store_root / "manifest.json"),
        "tick_store_manifest_sha256": sha256_file(tick_store_root / "manifest.json"),
        "tick_store_content_sha256": store["content_sha256"],
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
    }
    compiler_contract = {
        "kind": COMPILER_KIND,
        "schema_version": 1,
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "maximum_rows_per_shard": int(maximum_rows_per_shard),
        "actor_information": "public_only_v1",
        "action_alignment": "source_tick_plus_episode_metadata_offset",
        "entity_identity": "discrete_native_card_token_v1",
        "mask_policy": "content_addressed_native_sidecar_fail_closed_v1",
        "components": {
            "compiler_sha256": sha256_file(Path(__file__).resolve()),
            "training_schema_sha256": sha256_file(
                (Path(__file__).parent / "training_v1" / "schema.py").resolve()
            ),
            "deployment_masks_sha256": sha256_file(
                (Path(__file__).parent / "tick_store_v1" / "deployment_masks.py").resolve()
            ),
        },
    }
    input_content_sha256 = _digest(
        {"inputs": input_contract, "compiler": compiler_contract}
    )
    plan = {
        "kind": PLAN_KIND,
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_content_sha256": input_content_sha256,
        "inputs": input_contract,
        "compiler": compiler_contract,
        "tick_store_root": str(tick_store_root),
        "output_root": str(output_root),
        "episodes": len(episodes),
        "estimated_rows": sum(value.tick_count * 2 for value in episodes),
        "split_audit": split_audit,
        "card_vocabulary": card_vocabulary,
        "native_card_id_to_token": {str(key): value for key, value in id_to_card_token.items()},
        "source_card_to_token": source_to_card_token,
        "ability_vocabulary": ability_vocabulary,
        "native_ability_id_to_token": {str(key): value for key, value in id_to_ability_token.items()},
        "shards": [
            {
                **asdict(value),
                "episodes": [asdict(episode) for episode in value.episodes],
            }
            for value in shard_specs
        ],
    }
    plan["plan_content_sha256"] = _digest(plan)
    output_root.mkdir(parents=True, exist_ok=True)
    existing = output_root / "compile-plan.json"
    if existing.exists():
        old = load_compile_plan(existing)
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
) -> dict[str, np.ndarray]:
    if not states:
        raise NativeBcCompileError(f"empty Tick episode: {plan.battle_tag}")
    offset = int(metadata.get(ACTION_EXECUTION_OFFSET_METADATA, -999))
    if offset < 0 or offset > 2:
        raise NativeBcCompileError(f"unsupported action execution offset: {offset}")
    events = _event_maps(plan, offset)
    mask_store = DeploymentMaskStore(tick_store_root, create=False)
    mask_metadata = mask_store.verify_episode_metadata(metadata)
    mask_entries = {
        (int(value["side"]), int(value["deck_index"])): value
        for value in mask_metadata["entries"]
    }
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
        "grid": np.zeros((count, len(GRID_CHANNELS), ARENA_ROWS, ARENA_COLUMNS), dtype=np.uint8),
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
        "selected_position_mask_packed": np.zeros((count, POSITION_MASK_BYTES), dtype=np.uint8),
        "ability_position_mask_packed": np.zeros((count, POSITION_MASK_BYTES), dtype=np.uint8),
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
    }
    revealed_enemy: list[int] = []
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
        arrays["grid"][row] = _grid(actor)
        arrays["public_scalars"][row] = _public_scalars(actor, state)
        hand_indices = actor.own_player.hand
        if len(set(hand_indices)) != 4 or any(value < 0 or value >= 8 for value in hand_indices):
            raise NativeBcCompileError(f"invalid exact native hand: {plan.battle_tag}/{state.tick}")
        arrays["hand_tokens"][row] = deck_tokens[list(hand_indices)]
        next_index = actor.own_player.next_deck_index
        if next_index < 0 or next_index >= 8 or next_index in hand_indices:
            raise NativeBcCompileError(f"invalid exact next card: {plan.battle_tag}/{state.tick}")
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
                reference = mask_entries[(actor_side, int(deck_index))]
                current_event = events.get((state.tick, actor_side))
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
                sidecar = mask_store.load(str(selected_reference["content_sha256"]))
                rows = derive_deployment_rows(
                    sidecar, state, side=actor_side, card_id=int(reference["card_id"])
                )
                position_mask = _mask_array(rows, actor_side=actor_side)
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
        arrays["timing_label_mask"][row] = command_allowed

        event = events.get((state.tick, actor_side))
        if event is None:
            continue
        if not any_action:
            raise NativeBcCompileError(
                f"expert action has no legal native action kind: {plan.battle_tag}/{state.tick}"
            )
        kind, value = event
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
            arrays["selected_position_mask_packed"][row] = np.packbits(
                position_mask, bitorder="little"
            )
        else:
            arrays["action_kind"][row] = 1
            legal_slots = np.flatnonzero(arrays["ability_mask"][row])
            if len(legal_slots) != 1:
                raise NativeBcCompileError(
                    f"expert ability is not uniquely legal: {plan.battle_tag}/{state.tick}"
                )
            arrays["ability_slot"][row] = int(legal_slots[0])
            arrays["ability_label_mask"][row] = 1
    missing_events = [key for key in events if key[1] == actor_side and not any(
        state.tick == key[0] for state in states
    )]
    if missing_events:
        raise NativeBcCompileError(
            f"expert action Tick is absent from Tick Store: {plan.battle_tag}/{missing_events[:3]}"
        )
    arrays["entity_offsets"] = entity_offsets
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


def _verify_existing_shard(path: Path, content_sha256: str) -> dict[str, Any] | None:
    metadata_path = path / "shard.json"
    if not metadata_path.is_file():
        return None
    value = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if value.get("kind") != SHARD_KIND or value.get("content_sha256") != content_sha256:
        raise NativeBcCompileError(f"existing shard belongs to different inputs: {path}")
    hashes = value.get("file_sha256")
    if not isinstance(hashes, Mapping):
        raise NativeBcCompileError(f"existing shard lacks file hashes: {path}")
    for name, expected in hashes.items():
        file = path / str(name)
        if not file.is_file() or sha256_file(file) != str(expected):
            raise NativeBcCompileError(f"existing shard is corrupt: {file}")
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
    existing = _verify_existing_shard(final, str(raw_spec["content_sha256"]))
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
    compiled: list[dict[str, np.ndarray]] = []
    sequence_identity: list[dict[str, Any]] = []
    maximum_entities = 0
    maximum_abilities = MAX_ABILITY_SLOTS
    readers: dict[tuple[str, str], ShardReader] = {}
    try:
        for raw_episode in raw_spec["episodes"]:
            episode = EpisodeInput(**raw_episode)
            source_path = Path(episode.source_path)
            if sha256_file(source_path) != episode.source_sha256:
                raise NativeBcCompileError(f"source changed during compilation: {episode.battle_tag}")
            source = json.loads(source_path.read_text(encoding="utf-8-sig"))
            plan = compile_battle(source)
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
            states = list(native_episode.iter_ticks())
            maximum_entities = max(
                maximum_entities, max((len(state.entities) for state in states), default=0)
            )
            metadata = dict(native_episode.metadata)
            label_audit = verify_deployment_labels(
                states,
                metadata,
                DeploymentMaskStore(tick_store_root, create=False),
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
                ),
            )
            if label_audit.get("all_legal") is not True:
                raise NativeBcCompileError(
                    f"native deployment label audit failed: {episode.battle_tag} "
                    f"{label_audit.get('violations', [])[:3]}"
                )
            for actor_side in (0, 1):
                arrays = _compile_actor(
                    states,
                    metadata,
                    plan,
                    actor_side=actor_side,
                    tick_store_root=tick_store_root,
                    id_to_card_token=id_to_card_token,
                    source_to_card_token=source_to_card_token,
                    id_to_ability_token=id_to_ability_token,
                    max_ability_slots=maximum_abilities,
                )
                compiled.append(arrays)
                sequence_identity.append(
                    {
                        "battle_tag": episode.battle_tag,
                        "actor_side": actor_side,
                        "source_sha256": episode.source_sha256,
                        "source_group": episode.source_group,
                        "player_tags": list(episode.player_tags),
                    }
                )
            # Actor arrays are now self-contained; release decoded TickState
            # objects before moving to the next battle in this shard.
            del states
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
        for name in compiled[0]:
            if name == "entity_offsets":
                continue
            value = np.concatenate([sequence[name] for sequence in compiled], axis=0)
            hashes[f"{name}.npy"] = _save_npy(temporary / f"{name}.npy", value)
        metadata = {
            "kind": SHARD_KIND,
            "schema_version": SCHEMA_VERSION,
            "content_sha256": raw_spec["content_sha256"],
            "split": raw_spec["split"],
            "rows": int(offsets[-1]),
            "sequences": len(compiled),
            "battles": len(raw_spec["episodes"]),
            "max_entities": maximum_entities,
            "max_ability_slots": maximum_abilities,
            "sequence_identity": sequence_identity,
            "file_sha256": hashes,
        }
        _atomic_json(temporary / "shard.json", metadata)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, final)
        except OSError:
            # Another equivalent worker may have won the atomic publish race.
            existing = _verify_existing_shard(final, str(raw_spec["content_sha256"]))
            if existing is None:
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            return {**existing, "resumed": True, "relative_path": relative}
        return {**metadata, "resumed": False, "relative_path": relative}
    finally:
        for reader in readers.values():
            reader.close()
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
    if worker_count <= 0 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("invalid worker partition")
    specs = [
        value
        for index, value in enumerate(plan["shards"])
        if index % worker_count == worker_index
    ]
    worker_contract = {
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
    if process_workers <= 1:
        return [_compile_output_shard(*value) for value in arguments]
    with ProcessPoolExecutor(max_workers=process_workers) as executor:
        return list(executor.map(_compile_output_shard_star, arguments))


def _compile_output_shard_star(arguments: Sequence[Any]) -> dict[str, Any]:
    return _compile_output_shard(*arguments)


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
            Path(str(plan["tick_store_root"]))
            / "deployment-masks-v1"
            / "manifest.json",
            str(plan["inputs"]["deployment_mask_manifest_sha256"]),
            "deployment-mask manifest",
        ),
    )
    for path, expected, label in immutable_inputs:
        if not path.is_file() or sha256_file(path) != expected:
            raise NativeBcCompileError(f"{label} changed before finalization")
    component_paths = {
        "compiler_sha256": Path(__file__).resolve(),
        "training_schema_sha256": (Path(__file__).parent / "training_v1" / "schema.py").resolve(),
        "deployment_masks_sha256": (
            Path(__file__).parent / "tick_store_v1" / "deployment_masks.py"
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
        metadata = _verify_existing_shard(path, str(raw["content_sha256"]))
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
                }
            )
    assignment_raw = b"".join(
        _canonical_bytes(value) for value in sorted(assignments, key=lambda value: value["battle_tag"])
    )
    _atomic_bytes(output / "split-assignments.jsonl", assignment_raw)
    total_rows = sum(value["rows"] for value in split_stats.values())
    total_sequences = sum(value["sequences"] for value in split_stats.values())
    manifest: dict[str, Any] = {
        "kind": DATASET_KIND,
        "schema_version": SCHEMA_VERSION,
        # Stable across repeated finalization of the same authenticated plan.
        "created_utc": str(plan["created_utc"]),
        "production_ready": True,
        "native_replay_validated": True,
        "observation_mode": OBSERVATION_MODE,
        "timing_target": "native_tick_hazard_v1",
        "actor_information": "public_only_v1",
        "source_manifest": {
            "path": str(source),
            "sha256": plan["inputs"]["schema5_manifest_sha256"],
        },
        "source_inputs": plan["inputs"],
        "dataset_content_sha256": _digest(
            {
                "input": plan["input_content_sha256"],
                "shards": file_hashes,
                "assignments": hashlib.sha256(assignment_raw).hexdigest(),
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
            "mode": "native_teacher_forced_from_schema5_actions",
            "authoritative_rows": 0,
            "native_generated_unanchored_rows": total_rows,
            "native_grid_rows": total_rows,
            "every_native_tick_present": True,
        },
        "mask_provenance": {
            "kind": "content_addressed_native_deployment_sidecar_v1",
            "referenced_sidecar_sha256": plan["inputs"]["referenced_deployment_mask_sha256"],
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
        },
        "compiler": {
            **plan["compiler"],
            "input_content_sha256": plan["input_content_sha256"],
            "resumability": "deterministic_atomic_output_shards_v1",
        },
        "shard_file_sha256": file_hashes,
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
    }
    _atomic_json(output / "compile-result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile content-addressed native Tick Store episodes into BC shards"
    )
    parser.add_argument("--tick-store-root", type=Path, required=True)
    parser.add_argument("--schema5-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-rows-per-shard", type=int, default=32_768)
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
        }
    completed = compile_planned_shards(
        plan,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        process_workers=args.process_workers,
    )
    if args.worker_count == 1:
        return finalize_dataset(plan)
    return {
        "kind": "cr_native_tick_store_bc_worker_result_v1",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "completed_shards": len(completed),
        "rows": sum(int(value["rows"]) for value in completed),
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
