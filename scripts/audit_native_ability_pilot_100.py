#!/usr/bin/env python3
"""Read-only audit for the fixed 100-battle native ability pilot.

The pilot, task manifest, source JSON files, diagnostics and Tick Store are
immutable inputs.  The only write is the requested machine-readable report.
No Worker is contacted and no replay is rerun.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.tick_store_v1.shard import (  # noqa: E402
    FRAME_HEADER,
    FRAME_MAGIC,
    ShardReader,
    sha256_file,
)


DEFAULT_PILOT_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-ability-pilot-100-data-i-phase-plus1-v1"
)
DEFAULT_TASK_MANIFEST = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-ability-pilot-100-plan\selected.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_PILOT_ROOT / "ability-pilot-100-result-audit.json"
EXPECTED_TASK_SHA256 = (
    "2097f359fda18de4a08bf7e07ec43501d14569cebe36f3352ac1c5cf6666b250"
)
EXPECTED_ACTION_PROVENANCE = (
    "source label is RoyaleAPI time_raw; native execution Tick is "
    "source_tick+1; source label unchanged"
)
EXPECTED_COORDINATE_PROVENANCE = "royaleapi_raw_data_i_to_native_v1"
EXPECTED_COORDINATE_TRANSFORM = (
    "data_i=0:rotate_18000_32000;data_i=1:identity"
)
FAILURE_CODE_RE = re.compile(r"codes_\[([^]]+)\]")
HERO_MEGA_MINION_TAG = "099P9RVLP908"
LOGIC_FREEZE_TAG = "088Y82YV9G9C"
OFFICIAL_HERO_MEGA_MINION_URL = (
    "https://supercell.com/en/games/clashroyale/blog/news/"
    "new-season-midnight-mischief"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object row: {path}")
            values.append(value)
    return values


def _indexed(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        tag = str(row["battle_tag"])
        if tag in values:
            raise RuntimeError(f"duplicate battle tag in {source}: {tag}")
        values[tag] = row
    return values


def _failure_code(row: Mapping[str, Any]) -> int | None:
    match = FAILURE_CODE_RE.search(str(row.get("failure") or ""))
    if match is None:
        return None
    codes = [int(value.strip()) for value in match.group(1).split(",")]
    if len(codes) != 1:
        raise RuntimeError(f"expected one first-failure code: {row.get('battle_tag')}")
    return codes[0]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return numerator / denominator


def _verify_source_inputs(
    task_manifest: Path,
    results_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    task_rows = _jsonl(task_manifest)
    result_rows = _jsonl(results_path)
    tasks = _indexed(task_rows, source=task_manifest)
    results = _indexed(result_rows, source=results_path)
    if len(tasks) != 100 or len(results) != 100 or set(tasks) != set(results):
        raise RuntimeError(
            "expected identical 100-tag task/result sets: "
            f"tasks={len(tasks)} results={len(results)}"
        )
    if sorted(int(row["selection_index"]) for row in task_rows) != list(range(100)):
        raise RuntimeError("task selection_index is not exactly 0..99")
    if sorted(int(row["selection_index"]) for row in result_rows) != list(range(100)):
        raise RuntimeError("result selection_index is not exactly 0..99")

    per_tag = []
    for tag in sorted(tasks):
        task = tasks[tag]
        result = results[tag]
        source_path = Path(str(task["source_path"])).resolve(strict=True)
        source_sha256 = sha256_file(source_path)
        source = _json(source_path)
        source_deployments = len(source.get("card_plays") or [])
        source_abilities = len(source.get("ability_plays") or [])
        valid = (
            source_sha256 == task["source_sha256"]
            and source_sha256 == result["source_sha256"]
            and str(source.get("battle_tag")) == tag
            and int(source.get("schema_version", -1)) == 3
            and source_deployments == int(task["deploy_action_count"])
            and source_deployments == int(result["source_deploy_actions"])
            and source_abilities == int(task["ability_event_count"])
            and source_abilities == int(result["source_ability_events"])
        )
        if not valid:
            raise RuntimeError(f"source/task/result mismatch: {tag}")
        per_tag.append(
            {
                "battle_tag": tag,
                "selection_index": int(task["selection_index"]),
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                "source_schema_version": 3,
                "source_deploy_actions": source_deployments,
                "source_ability_events": source_abilities,
                "teacher_forced_success": bool(result["teacher_forced_success"]),
                "failure_class": result.get("failure_class"),
                "failure": result.get("failure"),
                "accepted_deploy_actions": int(result["accepted_deploy_actions"]),
                "accepted_ability_actions": int(result["accepted_ability_actions"]),
                "reached_ability_markers": len(result.get("ability_resolutions") or []),
                "terminal_diagnostic_status": result.get(
                    "terminal_diagnostic_status"
                ),
                "chosen_seed": result.get("chosen_seed"),
            }
        )
    return tasks, results, per_tag


def _verify_provenance(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    coordinate_totals: Counter[str] = Counter()
    ability_offset_checks = 0
    for tag, result in results.items():
        if int(result.get("action_execution_tick_offset", -1)) != 1:
            raise RuntimeError(f"wrong action execution offset: {tag}")
        if result.get("action_tick_provenance") != EXPECTED_ACTION_PROVENANCE:
            raise RuntimeError(f"wrong action Tick provenance: {tag}")
        if result.get("coordinate_provenance") != EXPECTED_COORDINATE_PROVENANCE:
            raise RuntimeError(f"wrong coordinate provenance: {tag}")
        coordinate = result.get("coordinate_audit") or {}
        if coordinate.get("transform") != EXPECTED_COORDINATE_TRANSFORM:
            raise RuntimeError(f"wrong coordinate transform: {tag}")
        raw = int(coordinate.get("raw_data_i_events", -1))
        zero = int(coordinate.get("data_i_zero_events", -1))
        one = int(coordinate.get("data_i_one_events", -1))
        fallback = int(coordinate.get("legacy_xy_fallback_events", -1))
        if raw != int(result["source_deploy_actions"]) or zero + one != raw:
            raise RuntimeError(f"coordinate event count mismatch: {tag}")
        if fallback != 0:
            raise RuntimeError(f"legacy coordinate fallback used: {tag}")
        coordinate_totals.update(
            {
                "raw_data_i_events": raw,
                "data_i_zero_events": zero,
                "data_i_one_events": one,
                "legacy_xy_fallback_events": fallback,
            }
        )
        for resolution in result.get("ability_resolutions") or []:
            if int(resolution["execution_tick_offset"]) != 1:
                raise RuntimeError(f"ability offset mismatch: {tag}")
            if int(resolution["execution_tick"]) != int(resolution["source_tick"]) + 1:
                raise RuntimeError(f"ability source/execution Tick mismatch: {tag}")
            if resolution.get("action_tick_provenance") != EXPECTED_ACTION_PROVENANCE:
                raise RuntimeError(f"ability provenance mismatch: {tag}")
            ability_offset_checks += 1
    return {
        "result_count_with_verified_coordinate_provenance": len(results),
        "coordinate_provenance": EXPECTED_COORDINATE_PROVENANCE,
        "coordinate_transform": EXPECTED_COORDINATE_TRANSFORM,
        "coordinate_event_totals": dict(sorted(coordinate_totals.items())),
        "result_count_with_verified_action_offset": len(results),
        "action_execution_tick_offset": 1,
        "action_tick_provenance": EXPECTED_ACTION_PROVENANCE,
        "ability_resolution_offset_checks": ability_offset_checks,
        "source_labels_unchanged": True,
    }


def _rejection_evidence(
    pilot_root: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[tuple[str, int]], dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, int]] = Counter()
    mega_minion: dict[str, Any] | None = None
    for tag, result in sorted(results.items()):
        if result.get("failure_class") != "native_action_rejected":
            continue
        diagnostic_path = pilot_root / "diagnostics" / f"{tag}.json"
        diagnostic = _json(diagnostic_path)
        history = diagnostic["native_boundary_snapshot"]["recent_action_history"]
        final = history[-1]
        requests = final["request"]
        actions = final["response"]["actions"]
        if len(requests) != 1 or len(actions) != 1:
            raise RuntimeError(f"unexpected rejected joint-action width: {tag}")
        request = requests[0]
        native = actions[0]["result"]
        action_type = str(request["type"])
        result_code = int(native["result_code"])
        if native.get("accepted") is not False or result_code != _failure_code(result):
            raise RuntimeError(f"rejection evidence mismatch: {tag}")
        counts[(action_type, result_code)] += 1
        item = {
            "battle_tag": tag,
            "action_type": action_type,
            "result_code": result_code,
            "result_reason": native.get("reason", native.get("result_reason")),
            "source_tick": int(str(result["failure"]).split("source_tick_")[1].split("_")[0]),
            "execution_tick": int(native["tick"]),
        }
        rejected.append(item)

        if tag == HERO_MEGA_MINION_TAG:
            state = diagnostic["native_boundary_snapshot"]["latest_state"]
            entity_id = int(request["entity_id"])
            entities = [
                entity for entity in state["entities"]
                if int(entity["entity_id"]) == entity_id
            ]
            if len(entities) != 1:
                raise RuntimeError("Hero Mega Minion entity evidence changed")
            entity = entities[0]
            player = state["players"][int(request["side"])]
            enemy_live_non_towers = [
                {
                    "entity_id": int(candidate["entity_id"]),
                    "name": candidate.get("name"),
                    "hp": int(candidate["hp"]),
                    "max_hp": int(candidate["max_hp"]),
                }
                for candidate in state["entities"]
                if int(candidate.get("side", -1)) != int(request["side"])
                and candidate.get("name") is not None
                and int(candidate.get("hp", 0)) > 0
            ]
            mega_minion = {
                "battle_tag": tag,
                "source_tick": item["source_tick"],
                "execution_tick": item["execution_tick"],
                "request": request,
                "native_result": {
                    "accepted": False,
                    "result_code": result_code,
                    "reason": native.get("reason"),
                    "native_mana_cost": int(native["native_mana_cost"]),
                    "elixir_before_integer": int(native["elixir_before"]),
                    "elixir_after_integer": int(native["elixir_after"]),
                },
                "entity_snapshot": {
                    "entity_id": entity_id,
                    "base_card_id": int(entity["base_card_id"]),
                    "native_card_id": int(entity["native_card_id"]),
                    "name": entity["name"],
                    "card_form": entity["card_form"],
                    "side": int(entity["side"]),
                    "behavior_state": int(entity["behavior_state"]),
                    "ability_available": bool(entity["ability_available"]),
                    "ability_state_code": int(entity["ability_state_code"]),
                    "ability_state_name": entity["ability_state_name"],
                    "ability_charges_remaining": int(
                        entity["ability_charges_remaining"]
                    ),
                    "ability_cooldown_remaining_ms": int(
                        entity["ability_cooldown_remaining_ms"]
                    ),
                    "ability_mana_cost": int(entity["ability_mana_cost"]),
                },
                "player_snapshot": {
                    "side": int(player["side"]),
                    "elixir": int(player["elixir"]),
                    "elixir_exact": float(player["elixir_exact"]),
                    "elixir_raw": int(player["elixir_raw"]),
                },
                "episode_snapshot": {
                    "commands_allowed": bool(state["episode"]["commands_allowed"]),
                    "command_gate_code": int(state["episode"]["command_gate_code"]),
                    "terminated": bool(state["episode"]["terminated"]),
                },
                "observable_live_enemy_non_tower_entities": enemy_live_non_towers,
                "official_semantics": {
                    "source": OFFICIAL_HERO_MEGA_MINION_URL,
                    "summary": (
                        "Wounding Warp selects the enemy with the lowest maximum "
                        "HP anywhere in the Arena"
                    ),
                },
                "classification": "unknown_context_specific_native_rejection",
                "interpretation": (
                    "Generic compact fields report ready, one charge, zero cooldown, "
                    "sufficient Elixir and an open episode command gate.  The official "
                    "ability is target-dependent, but these fields do not prove that "
                    "libg resolved an eligible target or that every hidden contextual "
                    "precondition was met.  Result code 1013 therefore remains Unknown."
                ),
            }
    if mega_minion is None:
        raise RuntimeError("missing Hero Mega Minion code1013 evidence")
    return rejected, counts, mega_minion


def _verify_tick_store(
    store_root: Path,
    *,
    successful_tags: set[str],
    results: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    store_root = store_root.resolve(strict=True)
    manifest = _json(store_root / "manifest.json")
    manifest_sha256 = sha256_file(store_root / "manifest.json")
    sidecar = (store_root / "manifest.sha256").read_text(encoding="ascii").split()
    if sidecar != [manifest_sha256, "manifest.json"]:
        raise RuntimeError("Tick Store manifest sidecar mismatch")

    digest_input: dict[str, dict[str, str]] = {}
    decoded_tags: set[str] = set()
    decoded_ticks = 0
    shard_reports = []
    metadata_provenance_count = 0
    for shard in manifest["shards"]:
        name = str(shard["name"])
        local_manifest = _json(store_root / f"{name}.manifest.json")
        if local_manifest != shard:
            raise RuntimeError(f"global/local shard manifest mismatch: {name}")
        data_path = store_root / str(shard["data_file"])
        index_path = store_root / str(shard["index_file"])
        data_sha256 = sha256_file(data_path)
        index_sha256 = sha256_file(index_path)
        if data_sha256 != shard["data_sha256"]:
            raise RuntimeError(f"data SHA-256 mismatch: {name}")
        if index_sha256 != shard["index_sha256"]:
            raise RuntimeError(f"index SHA-256 mismatch: {name}")
        if data_path.stat().st_size != int(shard["bytes"]):
            raise RuntimeError(f"data byte count mismatch: {name}")
        digest_input[name] = {"data": data_sha256, "index": index_sha256}

        shard_episodes = 0
        shard_ticks = 0
        with ShardReader(data_path, index_path) as reader:
            for tag, entry in sorted(reader.entries.items()):
                if tag in decoded_tags or tag not in successful_tags:
                    raise RuntimeError(f"unexpected or duplicate stored tag: {tag}")
                result_entry = results[tag].get("tick_store_entry")
                if not isinstance(result_entry, Mapping):
                    raise RuntimeError(f"success result lacks Tick Store entry: {tag}")
                for key in (
                    "offset",
                    "payload_size",
                    "ticks",
                    "tick_start",
                    "tick_stop",
                    "payload_sha256",
                ):
                    if result_entry.get(key) != entry.get(key):
                        raise RuntimeError(f"result/index {key} mismatch: {tag}")

                with data_path.open("rb") as handle:
                    handle.seek(int(entry["offset"]))
                    raw_header = handle.read(FRAME_HEADER.size)
                    magic, payload_size, _crc, _tag_hash, ticks, _reserved = (
                        FRAME_HEADER.unpack(raw_header)
                    )
                    if magic != FRAME_MAGIC or payload_size != int(
                        entry["payload_size"]
                    ) or ticks != int(entry["ticks"]):
                        raise RuntimeError(f"frame header mismatch: {tag}")
                    payload = handle.read(payload_size)
                if hashlib.sha256(payload).hexdigest() != entry["payload_sha256"]:
                    raise RuntimeError(f"episode payload SHA-256 mismatch: {tag}")

                episode = reader.episode(tag)
                metadata = episode.metadata
                task = tasks[tag]
                if (
                    metadata.get("source_sha256") != task["source_sha256"]
                    or int(metadata.get("action_execution_tick_offset", -1)) != 1
                    or metadata.get("action_tick_provenance")
                    != EXPECTED_ACTION_PROVENANCE
                    or metadata.get("coordinate_provenance")
                    != EXPECTED_COORDINATE_PROVENANCE
                    or metadata.get("coordinate_audit", {}).get("transform")
                    != EXPECTED_COORDINATE_TRANSFORM
                    or metadata.get("every_native_tick_present") is not True
                    or int(metadata.get("native_tick_hz", -1)) != 20
                ):
                    raise RuntimeError(f"episode provenance mismatch: {tag}")
                metadata_provenance_count += 1

                count = 0
                first_tick: int | None = None
                previous_tick: int | None = None
                for state in episode.iter_ticks():
                    if first_tick is None:
                        first_tick = state.tick
                    if previous_tick is not None and state.tick != previous_tick + 1:
                        raise RuntimeError(
                            f"non-consecutive Tick in {tag}: {previous_tick}->{state.tick}"
                        )
                    previous_tick = state.tick
                    count += 1
                if (
                    count != int(entry["ticks"])
                    or first_tick != int(entry["tick_start"])
                    or previous_tick is None
                    or previous_tick + 1 != int(entry["tick_stop"])
                    or count != int(results[tag]["tick_trace_complete_frames"])
                ):
                    raise RuntimeError(f"decoded Tick range/count mismatch: {tag}")
                decoded_tags.add(tag)
                decoded_ticks += count
                shard_episodes += 1
                shard_ticks += count

        if shard_episodes != int(shard["episode_count"]):
            raise RuntimeError(f"shard episode count mismatch: {name}")
        if shard_ticks != int(shard["tick_count"]):
            raise RuntimeError(f"shard Tick count mismatch: {name}")
        shard_reports.append(
            {
                "name": name,
                "episode_count": shard_episodes,
                "tick_count": shard_ticks,
                "bytes": int(shard["bytes"]),
                "data_sha256": data_sha256,
                "index_sha256": index_sha256,
                "all_episodes_fully_decoded": True,
                "all_ticks_consecutive": True,
                "all_episode_payload_sha256_match": True,
                "all_episode_provenance_verified": True,
            }
        )

    content_sha256 = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if content_sha256 != manifest["content_sha256"]:
        raise RuntimeError("global Tick Store content SHA-256 mismatch")
    if decoded_tags != successful_tags:
        raise RuntimeError("stored tags do not equal teacher-forced success tags")
    if decoded_ticks != int(manifest["tick_count"]):
        raise RuntimeError("global decoded Tick count mismatch")
    if len(decoded_tags) != int(manifest["episode_count"]):
        raise RuntimeError("global decoded episode count mismatch")
    if metadata_provenance_count != len(decoded_tags):
        raise RuntimeError("not every stored episode provenance was verified")

    return {
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar_matches": True,
        "content_sha256": content_sha256,
        "source_manifest_sha256": manifest["source_manifest"]["sha256"],
        "physical_shard_count": len(shard_reports),
        "episode_count": len(decoded_tags),
        "tick_count": decoded_ticks,
        "total_bytes": int(manifest["total_bytes"]),
        "bytes_per_tick": int(manifest["total_bytes"]) / decoded_ticks,
        "all_manifest_data_index_sha256_match": True,
        "all_episode_payload_sha256_match": True,
        "all_episodes_fully_decoded": True,
        "all_ticks_consecutive": True,
        "all_episode_provenance_verified": True,
        "stored_tags_equal_successful_tags": True,
        "shards": shard_reports,
    }


def audit(pilot_root: Path, task_manifest: Path) -> dict[str, Any]:
    pilot_root = pilot_root.resolve(strict=True)
    task_manifest = task_manifest.resolve(strict=True)
    results_path = pilot_root / "results.jsonl"
    summary_path = pilot_root / "summary.json"
    summary = _json(summary_path)
    task_sha256 = sha256_file(task_manifest)
    if task_sha256 != EXPECTED_TASK_SHA256:
        raise RuntimeError(f"unexpected fixed task manifest SHA-256: {task_sha256}")
    if summary.get("task_manifest_sha256") != task_sha256:
        raise RuntimeError("summary/task manifest SHA-256 mismatch")

    tasks, results, per_tag = _verify_source_inputs(task_manifest, results_path)
    provenance = _verify_provenance(results)
    successes = {tag for tag, row in results.items() if row["teacher_forced_success"]}
    failures = set(results) - successes
    failure_classes = Counter(
        str(results[tag].get("failure_class")) for tag in failures
    )
    resolution_counts = Counter(
        str(resolution["status"])
        for row in results.values()
        for resolution in row.get("ability_resolutions") or []
    )
    rejected, rejection_counts, mega_minion = _rejection_evidence(
        pilot_root, results
    )

    source_deployments = sum(int(row["source_deploy_actions"]) for row in results.values())
    accepted_deployments = sum(
        int(row["accepted_deploy_actions"]) for row in results.values()
    )
    rejected_deployments = sum(
        count for (action_type, _code), count in rejection_counts.items()
        if action_type == "play"
    )
    attempted_deployments = accepted_deployments + rejected_deployments
    source_abilities = sum(int(row["source_ability_events"]) for row in results.values())
    accepted_abilities = sum(
        int(row["accepted_ability_actions"]) for row in results.values()
    )
    reached_ability_markers = sum(resolution_counts.values())
    dispatched_abilities = resolution_counts["unique"]
    rejected_abilities = rejection_counts[("ability", 1013)]

    freeze = results[LOGIC_FREEZE_TAG]
    if (
        freeze.get("failure_class") != "teacher_forced_failure"
        or "native_tick_mismatch_3681_expected_execution_tick_3744" not in str(
            freeze.get("failure")
        )
    ):
        raise RuntimeError("logic-freeze evidence changed")

    tick_store = _verify_tick_store(
        pilot_root / "shards",
        successful_tags=successes,
        results=results,
        tasks=tasks,
    )
    terminal_all = Counter(
        str(row.get("terminal_diagnostic_status")) for row in results.values()
    )
    terminal_success = Counter(
        str(results[tag].get("terminal_diagnostic_status")) for tag in successes
    )
    successful_source_deployments = sum(
        int(results[tag]["source_deploy_actions"]) for tag in successes
    )
    successful_source_abilities = sum(
        int(results[tag]["source_ability_events"]) for tag in successes
    )
    unique_card_ids = sorted(
        {
            int(resolution["candidate_card_ids"][0])
            for row in results.values()
            for resolution in row.get("ability_resolutions") or []
            if resolution.get("status") == "unique"
            and len(resolution.get("candidate_card_ids") or []) == 1
        }
    )

    assertions = {
        "fixed_task_manifest_sha256": task_sha256 == EXPECTED_TASK_SHA256,
        "task_result_tag_set_exact_100": len(tasks) == len(results) == 100
        and set(tasks) == set(results),
        "actual_source_sha256_verified_100": len(per_tag) == 100,
        "source_schema3_and_action_counts_verified_100": len(per_tag) == 100,
        "teacher_forced_outcome_partition_58_42": len(successes) == 58
        and len(failures) == 42,
        "failure_partition_5_13_23_1": failure_classes
        == Counter(
            {
                "ability_branch_required": 5,
                "ability_entity_missing": 13,
                "native_action_rejected": 23,
                "teacher_forced_failure": 1,
            }
        ),
        "native_rejection_partition_18_4_1": rejection_counts
        == Counter({("play", 4): 18, ("play", 13): 4, ("ability", 1013): 1}),
        "ability_source_reached_dispatched_accepted_376_318_300_299": (
            source_abilities,
            reached_ability_markers,
            dispatched_abilities,
            accepted_abilities,
        )
        == (376, 318, 300, 299),
        "ability_resolution_partition_300_13_5": resolution_counts
        == Counter(
            {"unique": 300, "no_legal_matching_entity": 13, "branch_required": 5}
        ),
        "deployment_source_attempted_accepted_7174_5981_5959": (
            source_deployments,
            attempted_deployments,
            accepted_deployments,
        )
        == (7174, 5981, 5959),
        "all_58_success_actions_complete": successful_source_deployments == 4168
        and successful_source_abilities == 221
        and sum(int(results[tag]["accepted_deploy_actions"]) for tag in successes)
        == successful_source_deployments
        and sum(int(results[tag]["accepted_ability_actions"]) for tag in successes)
        == successful_source_abilities,
        "coordinate_and_action_provenance_verified_100": (
            provenance["result_count_with_verified_coordinate_provenance"] == 100
            and provenance["result_count_with_verified_action_offset"] == 100
            and provenance["ability_resolution_offset_checks"] == 318
        ),
        "hero_mega_minion_1013_state_evidence_verified": (
            mega_minion["entity_snapshot"]["name"] == "MegaMinion"
            and mega_minion["entity_snapshot"]["card_form"] == "hero"
            and mega_minion["entity_snapshot"]["ability_state_name"] == "ready"
            and mega_minion["entity_snapshot"]["ability_charges_remaining"] == 1
            and mega_minion["player_snapshot"]["elixir_raw"] == 85175
            and mega_minion["native_result"]["result_code"] == 1013
        ),
        "logic_freeze_separate_from_action_rejection": LOGIC_FREEZE_TAG in failures
        and _failure_code(freeze) is None,
        "tick_store_58_episodes_269233_ticks": tick_store["episode_count"] == 58
        and tick_store["tick_count"] == 269233,
        "tick_store_full_decode_sha_and_provenance": (
            tick_store["all_episodes_fully_decoded"]
            and tick_store["all_ticks_consecutive"]
            and tick_store["all_manifest_data_index_sha256_match"]
            and tick_store["all_episode_payload_sha256_match"]
            and tick_store["all_episode_provenance_verified"]
            and tick_store["stored_tags_equal_successful_tags"]
        ),
    }
    if not all(assertions.values()):
        failed = [name for name, value in assertions.items() if not value]
        raise RuntimeError(f"ability pilot audit assertions failed: {failed}")

    for item in per_tag:
        tag = item["battle_tag"]
        item["failure_bucket"] = (
            "success" if tag in successes else str(results[tag]["failure_class"])
        )
        item["first_native_rejection_code"] = _failure_code(results[tag])
        item["tick_store_episode_present"] = tag in successes

    return {
        "schema_version": 1,
        "kind": "native_ability_pilot_100_result_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pilot_root": str(pilot_root),
            "task_manifest": str(task_manifest),
            "task_manifest_sha256": task_sha256,
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "diagnostic_count": len(list((pilot_root / "diagnostics").glob("*.json"))),
        },
        "fixed_input_validation": {
            "task_rows": len(tasks),
            "result_rows": len(results),
            "unique_task_tags": len(tasks),
            "unique_result_tags": len(results),
            "tag_sets_equal": True,
            "selection_indexes_exact_0_to_99": True,
            "actual_source_sha256_verified_count": 100,
            "source_schema3_verified_count": 100,
            "source_action_counts_verified_count": 100,
        },
        "outcomes": {
            "teacher_forced_success": len(successes),
            "teacher_forced_failure": len(failures),
            "success_rate": _ratio(len(successes), len(results)),
            "failure_class_counts": dict(sorted(failure_classes.items())),
        },
        "action_accounting": {
            "deployment": {
                "source_events": source_deployments,
                "reached_native_attempts": attempted_deployments,
                "accepted": accepted_deployments,
                "rejected": rejected_deployments,
                "not_reached_after_first_fail_closed_stop": (
                    source_deployments - attempted_deployments
                ),
                "conditional_native_acceptance_rate": _ratio(
                    accepted_deployments, attempted_deployments
                ),
                "warning": (
                    "accepted/source is coverage, not an acceptance rate; later "
                    "source events are intentionally not attempted after first failure"
                ),
            },
            "ability": {
                "source_events": source_abilities,
                "reached_resolution_markers": reached_ability_markers,
                "not_reached_after_first_fail_closed_stop": (
                    source_abilities - reached_ability_markers
                ),
                "resolution_status_counts": dict(sorted(resolution_counts.items())),
                "unique_entity_native_dispatch_attempts": dispatched_abilities,
                "accepted_native_dispatches": accepted_abilities,
                "rejected_native_dispatches": rejected_abilities,
                "conditional_native_acceptance_rate": _ratio(
                    accepted_abilities, dispatched_abilities
                ),
                "accepted_per_reached_marker_rate": _ratio(
                    accepted_abilities, reached_ability_markers
                ),
                "unique_candidate_base_card_ids": unique_card_ids,
                "unique_candidate_base_card_id_count": len(unique_card_ids),
                "warning": (
                    "299/376 is not an attempt acceptance rate. Only 300 ability "
                    "commands were dispatched; 13 reached markers had no legal "
                    "matching entity, 5 required an explicit branch, and 58 later "
                    "source markers were never reached after an earlier stop."
                ),
            },
            "successful_episodes": {
                "episodes": 58,
                "source_and_accepted_deployments": successful_source_deployments,
                "source_and_accepted_abilities": successful_source_abilities,
            },
        },
        "native_rejections": {
            "counts": {
                "deployment_code4": rejection_counts[("play", 4)],
                "deployment_code13": rejection_counts[("play", 13)],
                "ability_code1013": rejection_counts[("ability", 1013)],
            },
            "events": rejected,
            "hero_mega_minion_code1013": mega_minion,
        },
        "failure_interpretation": {
            "interface_capability": (
                "The external entity-targeted ability command path is demonstrated "
                "by 299 accepted native ability commands across 23 base card IDs."
            ),
            "ability_branch_required_5": (
                "The source marker lacks entity identity while more than one legal "
                "live candidate exists. This is label ambiguity; fail-closed is correct."
            ),
            "ability_entity_missing_13": (
                "At the exact marker the generated native state has no legal matching "
                "live entity. This is generated-state/liveness/legality divergence, "
                "not evidence that the ability RPC is absent."
            ),
            "deployment_code4_18": (
                "The generated state has closed the battle command gate before the "
                "source action. This is downstream state/terminal drift."
            ),
            "deployment_code13_4": (
                "The generated native resource state reports insufficient Elixir. "
                "This is exact-Tick resource-state divergence."
            ),
            "ability_code1013_1": (
                "The command reaches libg for a uniquely resolved Hero Mega Minion, "
                "but libg rejects it. Generic readiness/resource fields pass; because "
                "Wounding Warp is target-dependent and hidden contextual guards are "
                "not decoded, code1013 remains Unknown and is not assigned a name."
            ),
            "logic_freeze_1": (
                "libg stops advancing at Tick 3681 before the requested execution "
                "boundary 3744. No action was attempted there; it is separate from "
                "native action rejection."
            ),
        },
        "terminal_diagnostic": {
            "all_100": dict(sorted(terminal_all.items())),
            "teacher_forced_success_58": dict(sorted(terminal_success.items())),
            "exact_terminal_evaluated_successes": (
                terminal_success["match"] + terminal_success["crowns_mismatch"]
            ),
            "exact_terminal_match_rate": _ratio(
                terminal_success["match"],
                terminal_success["match"] + terminal_success["crowns_mismatch"],
            ),
            "interpretation": (
                "Terminal crowns are a generated-state diagnostic. They do not "
                "retroactively invalidate a fully accepted teacher-forced action path, "
                "and they are not original hidden server-state truth."
            ),
        },
        "provenance_validation": provenance,
        "tick_store_validation": tick_store,
        "assertions": assertions,
        "all_assertions_passed": True,
        "per_tag": per_tag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = audit(arguments.pilot_root, arguments.task_manifest)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "task_manifest_sha256": report["inputs"]["task_manifest_sha256"],
                "teacher_forced_success": report["outcomes"][
                    "teacher_forced_success"
                ],
                "ability_dispatch_attempts": report["action_accounting"]["ability"][
                    "unique_entity_native_dispatch_attempts"
                ],
                "ability_dispatches_accepted": report["action_accounting"]["ability"][
                    "accepted_native_dispatches"
                ],
                "decoded_episodes": report["tick_store_validation"]["episode_count"],
                "decoded_ticks": report["tick_store_validation"]["tick_count"],
                "all_assertions_passed": report["all_assertions_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
