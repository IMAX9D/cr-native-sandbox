#!/usr/bin/env python3
"""Final read-only audit of the fixed-coordinate native Tick phase A/B.

The pilot roots are immutable inputs.  The only write is the requested JSON
report.  One known battle (089Y82CPYYY9) reaches a non-terminal native logic
freeze at Tick 3681: offset 0 observes native hard-gate code 3 at that Tick,
while offset 1 cannot advance to Tick 3682.  It is therefore reported as a
non-phase-comparable diagnostic, not silently counted as an offset-1 reject.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


DEFAULT_V9 = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-data-i-v9"
)
DEFAULT_V10 = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-data-i-phase-plus1-v10"
)
DEFAULT_OUTPUT = DEFAULT_V10 / "coordinate-phase-final-v9-v10.json"
LOGIC_FREEZE_TAG = "089Y82CPYYY9"
FAILURE_CODE_RE = re.compile(r"codes_\[([^]]+)\]")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _indexed(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        tag = str(row["battle_tag"])
        if tag in values:
            raise RuntimeError(f"duplicate battle tag in {path}: {tag}")
        values[tag] = row
    return values


def _failure_codes(row: Mapping[str, Any]) -> list[int]:
    rejection = row.get("first_rejection")
    if isinstance(rejection, Mapping) and isinstance(
        rejection.get("result_codes"), list
    ):
        return [int(value) for value in rejection["result_codes"]]
    match = FAILURE_CODE_RE.search(str(row.get("failure") or ""))
    if match is None:
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]


def _category(row: Mapping[str, Any], battle_tag: str) -> str:
    if battle_tag == LOGIC_FREEZE_TAG:
        return "logic_freeze_excluded"
    if row.get("teacher_forced_success") is True:
        return "success"
    codes = _failure_codes(row)
    if 13 in codes:
        return "code13"
    if 4 in codes:
        return "code4"
    failure = str(row.get("failure") or "")
    if failure.startswith("native_terminal_before_"):
        return "terminal_before"
    return "other"


def _compact(row: Mapping[str, Any], battle_tag: str) -> dict[str, Any]:
    rejection = row.get("first_rejection")
    rejection = rejection if isinstance(rejection, Mapping) else {}
    events = rejection.get("events")
    events = events if isinstance(events, list) else []
    return {
        "category": _category(row, battle_tag),
        "teacher_forced_success": bool(row.get("teacher_forced_success")),
        "failure": row.get("failure"),
        "failure_codes": _failure_codes(row),
        "first_rejection_source_tick": rejection.get("source_tick"),
        "first_rejection_execution_tick": rejection.get(
            "execution_tick", rejection.get("tick")
        ),
        "rejected_cards": sorted(
            {
                str(event["base_token"])
                for event in events
                if isinstance(event, Mapping) and event.get("base_token") is not None
            }
        ),
        "chosen_seed": row.get("chosen_seed", row.get("seed")),
        "source_sha256": row.get("source_sha256"),
        "accepted_deployment_actions": int(
            row.get("accepted_deployment_actions") or 0
        ),
        "source_deployment_actions": int(row.get("source_deployment_actions") or 0),
        "stored_tick_count": int(row.get("stored_tick_count") or 0),
        "terminal_status": row.get("terminal_status"),
        "source_crowns": row.get("source_crowns"),
        "observed_crowns": row.get("observed_crowns"),
        "logical_training_state_sha256": row.get(
            "logical_training_state_sha256"
        ),
    }


def _verify_manifest_sidecar(root: Path) -> str:
    digest = sha256_file(root / "manifest.json")
    fields = (root / "manifest.sha256").read_text(encoding="ascii").split()
    if fields != [digest, "manifest.json"]:
        raise RuntimeError(f"manifest SHA-256 sidecar mismatch: {root}")
    return digest


def _verify_tick_store(
    root: Path,
    *,
    successful_tags: set[str],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = _json(root / "manifest.json")
    manifest_sha256 = _verify_manifest_sidecar(root)
    decoded_tags: set[str] = set()
    decoded_ticks = 0
    digest_input: dict[str, dict[str, str]] = {}
    shard_reports: list[dict[str, Any]] = []

    for shard in manifest["shards"]:
        name = str(shard["name"])
        local_manifest = _json(root / f"{name}.manifest.json")
        if local_manifest != shard:
            raise RuntimeError(f"global/local shard manifest mismatch: {name}")
        data_path = root / str(shard["data_file"])
        index_path = root / str(shard["index_file"])
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
                if tag in decoded_tags:
                    raise RuntimeError(f"duplicate episode across shards: {tag}")
                if tag not in successful_tags:
                    raise RuntimeError(f"stored non-success episode: {tag}")
                result_entry = results[tag].get("store_entry")
                if not isinstance(result_entry, Mapping):
                    raise RuntimeError(f"success result lacks store entry: {tag}")
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
                    ):
                        raise RuntimeError(f"frame header mismatch: {tag}")
                    if ticks != int(entry["ticks"]):
                        raise RuntimeError(f"frame Tick count mismatch: {tag}")
                    payload = handle.read(payload_size)
                if hashlib.sha256(payload).hexdigest() != entry["payload_sha256"]:
                    raise RuntimeError(f"episode payload SHA-256 mismatch: {tag}")

                episode = reader.episode(tag)
                count = 0
                first_tick: int | None = None
                previous_tick: int | None = None
                for state in episode.iter_ticks():
                    if first_tick is None:
                        first_tick = state.tick
                    if previous_tick is not None and state.tick != previous_tick + 1:
                        raise RuntimeError(
                            f"non-consecutive Tick in {tag}: "
                            f"{previous_tick}->{state.tick}"
                        )
                    previous_tick = state.tick
                    count += 1
                if count != int(entry["ticks"]):
                    raise RuntimeError(f"decoded Tick count mismatch: {tag}")
                if first_tick != int(entry["tick_start"]):
                    raise RuntimeError(f"first Tick mismatch: {tag}")
                if previous_tick is None or previous_tick + 1 != int(
                    entry["tick_stop"]
                ):
                    raise RuntimeError(f"last Tick mismatch: {tag}")
                if count != int(results[tag]["stored_tick_count"]):
                    raise RuntimeError(f"result Tick count mismatch: {tag}")
                decoded_tags.add(tag)
                decoded_ticks += count
                shard_ticks += count
                shard_episodes += 1

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
            }
        )

    content_sha256 = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if content_sha256 != manifest["content_sha256"]:
        raise RuntimeError("global store content SHA-256 mismatch")
    if decoded_tags != successful_tags:
        raise RuntimeError(
            "stored/success tag mismatch: "
            f"missing={sorted(successful_tags - decoded_tags)}, "
            f"extra={sorted(decoded_tags - successful_tags)}"
        )
    if decoded_ticks != int(manifest["tick_count"]):
        raise RuntimeError("global decoded Tick count mismatch")
    if len(decoded_tags) != int(manifest["episode_count"]):
        raise RuntimeError("global decoded episode count mismatch")

    return {
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar_matches": True,
        "content_sha256": content_sha256,
        "source_selection_sha256": manifest["source_manifest"]["sha256"],
        "physical_shard_count": len(shard_reports),
        "episode_count": len(decoded_tags),
        "tick_count": decoded_ticks,
        "total_bytes": int(manifest["total_bytes"]),
        "all_manifest_data_index_sha256_match": True,
        "all_episode_payload_sha256_match": True,
        "all_episodes_fully_decoded": True,
        "all_ticks_consecutive": True,
        "stored_tags_equal_successful_tags": True,
        "shards": shard_reports,
    }


def _terminal_counts(
    rows: Mapping[str, Mapping[str, Any]], tags: set[str]
) -> dict[str, int]:
    return dict(
        sorted(Counter(str(rows[tag].get("terminal_status")) for tag in tags).items())
    )


def audit(v9_root: Path, v10_root: Path) -> dict[str, Any]:
    v9_root = v9_root.resolve(strict=True)
    v10_root = v10_root.resolve(strict=True)
    v9_selection_path = v9_root / "selection.jsonl"
    v10_selection_path = v10_root / "selection.jsonl"
    selection_bytes_equal = (
        v9_selection_path.read_bytes() == v10_selection_path.read_bytes()
    )
    selection_sha256 = sha256_file(v9_selection_path)
    if not selection_bytes_equal:
        raise RuntimeError("v9/v10 selections are not byte-identical")

    selections = _indexed(v9_selection_path)
    v9 = _indexed(v9_root / "results.jsonl")
    v10 = _indexed(v10_root / "results.jsonl")
    if set(v9) != set(v10) or set(v9) != set(selections):
        raise RuntimeError("selection/result tag sets differ")
    tags = set(v9)
    if len(tags) != 100:
        raise RuntimeError(f"expected fixed 100 tags, got {len(tags)}")
    if LOGIC_FREEZE_TAG not in tags:
        raise RuntimeError(f"missing known freeze tag: {LOGIC_FREEZE_TAG}")

    v9_summary = _json(v9_root / "summary.json")
    v10_summary = _json(v10_root / "summary.json")
    v9_offset = int(v9_summary["configuration"]["action_execution_tick_offset"])
    v10_offset = int(v10_summary["configuration"]["action_execution_tick_offset"])
    if (v9_offset, v10_offset) != (0, 1):
        raise RuntimeError(f"expected offsets 0/1, got {v9_offset}/{v10_offset}")
    comparable_configuration = {
        key: value
        for key, value in v9_summary["configuration"].items()
        if key not in {"action_execution_tick_offset", "action_tick_provenance"}
    }
    v10_comparable_configuration = {
        key: value
        for key, value in v10_summary["configuration"].items()
        if key not in {"action_execution_tick_offset", "action_tick_provenance"}
    }
    if comparable_configuration != v10_comparable_configuration:
        raise RuntimeError("A/B configuration differs beyond the execution offset")

    directly_comparable_tags = tags - {LOGIC_FREEZE_TAG}
    direct_source_equal = []
    chosen_seed_equal = []
    per_tag: list[dict[str, Any]] = []
    transitions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tag in sorted(tags):
        selection_source = str(selections[tag]["source_sha256"])
        v9_source = v9[tag].get("source_sha256")
        v10_source = v10[tag].get("source_sha256")
        source_equal = (
            v9_source == v10_source == selection_source
            if v10_source is not None
            else v9_source == selection_source and tag == LOGIC_FREEZE_TAG
        )
        if not source_equal:
            raise RuntimeError(f"source SHA-256 mismatch: {tag}")
        if tag in directly_comparable_tags:
            direct_source_equal.append(tag)
            if v9[tag].get("chosen_seed") != v10[tag].get("chosen_seed"):
                raise RuntimeError(f"chosen seed changed: {tag}")
            chosen_seed_equal.append(tag)
        before = _category(v9[tag], tag)
        after = _category(v10[tag], tag)
        transitions[(before, after)].append(tag)
        per_tag.append(
            {
                "battle_tag": tag,
                "phase_comparable": tag in directly_comparable_tags,
                "transition": f"{before}->{after}",
                "selection_source_sha256": selection_source,
                "source_sha256_equal": source_equal,
                "chosen_seed_evidence": (
                    "recorded_equal"
                    if tag in directly_comparable_tags
                    else "inferred_equal_seed_resolution_precedes_phase_offset"
                ),
                "v9": _compact(v9[tag], tag),
                "v10": _compact(v10[tag], tag),
            }
        )

    freeze_v9 = v9[LOGIC_FREEZE_TAG]
    freeze_v10 = v10[LOGIC_FREEZE_TAG]
    freeze_rejection = freeze_v9.get("first_rejection") or {}
    freeze_events = freeze_rejection.get("events") or []
    freeze_native_result = (
        freeze_events[0].get("native_result", {}) if freeze_events else {}
    )
    freeze_error = str(freeze_v10.get("failure") or "")
    freeze_valid = (
        _failure_codes(freeze_v9) == [3]
        and int(freeze_rejection.get("execution_tick", -1)) == 3681
        and freeze_native_result.get("result_reason") == "battle_command_hard_gate"
        and "state_tick=3681" in freeze_error
        and "previous_tick=3681" in freeze_error
        and "terminal=False" in freeze_error
    )
    if not freeze_valid:
        raise RuntimeError("known logic-freeze evidence changed")

    v9_success = {tag for tag in tags if v9[tag].get("teacher_forced_success")}
    v10_success = {tag for tag in tags if v10[tag].get("teacher_forced_success")}
    shared_success = v9_success & v10_success
    new_success = v10_success - v9_success
    comparable_v9_categories = Counter(
        _category(v9[tag], tag) for tag in directly_comparable_tags
    )
    comparable_v10_categories = Counter(
        _category(v10[tag], tag) for tag in directly_comparable_tags
    )
    v9_code4 = {
        tag
        for tag in directly_comparable_tags
        if _category(v9[tag], tag) == "code4"
    }
    v10_code4 = {
        tag
        for tag in directly_comparable_tags
        if _category(v10[tag], tag) == "code4"
    }
    v9_code13 = {
        tag
        for tag in directly_comparable_tags
        if _category(v9[tag], tag) == "code13"
    }

    v9_store = _verify_tick_store(
        v9_root / "shards", successful_tags=v9_success, results=v9
    )
    v10_store = _verify_tick_store(
        v10_root / "shards", successful_tags=v10_success, results=v10
    )

    shared_terminal_transitions: Counter[str] = Counter()
    for tag in shared_success:
        shared_terminal_transitions[
            f"{v9[tag].get('terminal_status')}->{v10[tag].get('terminal_status')}"
        ] += 1
    new_success_terminal = Counter(
        str(v10[tag].get("terminal_status")) for tag in new_success
    )
    v9_exact_terminal = sum(
        v9[tag].get("terminal_status") in {"match", "mismatch"}
        for tag in v9_success
    )
    v10_exact_terminal = sum(
        v10[tag].get("terminal_status") in {"match", "mismatch"}
        for tag in v10_success
    )

    assertions = {
        "selection_byte_identical_100": selection_bytes_equal and len(tags) == 100,
        "source_sha256_equal_100": all(
            item["source_sha256_equal"] for item in per_tag
        ),
        "chosen_seed_recorded_equal_99": len(chosen_seed_equal) == 99,
        "freeze_seed_structurally_same": (
            v9_summary["configuration"]["seed"]
            == v10_summary["configuration"]["seed"]
            and v9_summary["configuration"]["maximum_seeds_to_test"]
            == v10_summary["configuration"]["maximum_seeds_to_test"]
            and comparable_configuration == v10_comparable_configuration
        ),
        "logic_freeze_excluded_from_phase_denominator": freeze_valid
        and len(directly_comparable_tags) == 99,
        "all_83_successes_preserved": len(v9_success) == 83
        and v9_success <= v10_success,
        "all_6_code13_become_success": len(v9_code13) == 6
        and v9_code13 == new_success,
        "code4_10_tag_set_stable": v9_code4 == v10_code4 and len(v9_code4) == 10,
        "phase_comparable_success_83_to_89": comparable_v9_categories["success"]
        == 83
        and comparable_v10_categories["success"] == 89,
        "v9_tick_store_full_decode_and_sha": v9_store[
            "all_episodes_fully_decoded"
        ]
        and v9_store["all_manifest_data_index_sha256_match"],
        "v10_tick_store_full_decode_and_sha": v10_store[
            "all_episodes_fully_decoded"
        ]
        and v10_store["all_manifest_data_index_sha256_match"],
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise RuntimeError(f"final coordinate phase assertions failed: {failed}")

    return {
        "schema_version": 1,
        "kind": "native_coordinate_phase_final_100_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v9_root": str(v9_root),
            "v10_root": str(v10_root),
            "v9_results_sha256": sha256_file(v9_root / "results.jsonl"),
            "v10_results_sha256": sha256_file(v10_root / "results.jsonl"),
            "v9_summary_sha256": sha256_file(v9_root / "summary.json"),
            "v10_summary_sha256": sha256_file(v10_root / "summary.json"),
        },
        "fixed_inputs": {
            "selection_sha256": selection_sha256,
            "selection_files_byte_identical": True,
            "battle_tags_equal": True,
            "battle_count": len(tags),
            "source_sha256_equal_count": 100,
            "chosen_seed": {
                "recorded_equal_count": len(chosen_seed_equal),
                "recorded_comparable_count": len(directly_comparable_tags),
                "freeze_tag": LOGIC_FREEZE_TAG,
                "freeze_v9_recorded_seed": freeze_v9.get("chosen_seed"),
                "freeze_v10_recorded_seed": freeze_v10.get("chosen_seed"),
                "freeze_seed_evidence": (
                    "v10 error serialization omitted chosen_seed; seed resolution "
                    "runs before action_execution_tick_offset is applied and all "
                    "other resolver inputs are identical, so equality is structural "
                    "rather than directly recorded"
                ),
            },
            "configuration_equal_except_phase_offset": True,
            "v9_execution_tick_offset": v9_offset,
            "v10_execution_tick_offset": v10_offset,
            "source_labels_unchanged": True,
        },
        "denominators": {
            "fixed_selection": 100,
            "phase_comparable": 99,
            "excluded_logic_freeze": 1,
            "rule": (
                "089Y82CPYYY9 is excluded only from phase acceptance rates; "
                "it remains in the raw fixed-selection accounting"
            ),
        },
        "logic_freeze_diagnostic": {
            "battle_tag": LOGIC_FREEZE_TAG,
            "phase_comparable": False,
            "v9": {
                "execution_tick": freeze_rejection.get("execution_tick"),
                "result_codes": _failure_codes(freeze_v9),
                "result_reason": freeze_native_result.get("result_reason"),
                "hard_gate": (freeze_native_result.get("guard_before") or {}).get(
                    "hard_gate"
                ),
                "logic_end_counter_198": (
                    freeze_native_result.get("guard_before") or {}
                ).get("logic_end_counter_198"),
            },
            "v10": {
                "requested_execution_tick": 3682,
                "frozen_state_tick": 3681,
                "terminal_flag": False,
                "error": freeze_v10.get("failure"),
            },
            "interpretation": (
                "offset 1 cannot reach its command boundary because libg has "
                "already stopped advancing; this is neither a code13 regression "
                "nor evidence against T+1"
            ),
        },
        "outcomes": {
            "raw_fixed_100": {
                "v9_success": len(v9_success),
                "v9_failure": 100 - len(v9_success),
                "v10_success": len(v10_success),
                "v10_failure": 100 - len(v10_success),
            },
            "phase_comparable_99": {
                "v9_categories": dict(sorted(comparable_v9_categories.items())),
                "v10_categories": dict(sorted(comparable_v10_categories.items())),
                "v9_success_rate": comparable_v9_categories["success"] / 99,
                "v10_success_rate": comparable_v10_categories["success"] / 99,
                "success_rate_delta_percentage_points": (
                    comparable_v10_categories["success"]
                    - comparable_v9_categories["success"]
                )
                / 99
                * 100,
            },
            "preserved_success_count": len(shared_success),
            "new_success_count": len(new_success),
            "new_success_tags": sorted(new_success),
            "stable_code4_tags": sorted(v9_code4),
            "transitions": [
                {
                    "from": before,
                    "to": after,
                    "count": len(transition_tags),
                    "battle_tags": transition_tags,
                }
                for (before, after), transition_tags in sorted(transitions.items())
            ],
        },
        "terminal_diagnostic": {
            "raw_summary_v9": v9_summary["terminal_diagnostic"],
            "raw_summary_v10": v10_summary["terminal_diagnostic"],
            "phase_comparable_v9": _terminal_counts(v9, directly_comparable_tags),
            "phase_comparable_v10": _terminal_counts(v10, directly_comparable_tags),
            "successful_v9": _terminal_counts(v9, v9_success),
            "successful_v10": _terminal_counts(v10, v10_success),
            "shared_83_success_transitions": dict(
                sorted(shared_terminal_transitions.items())
            ),
            "new_6_success_terminal_counts": dict(
                sorted(new_success_terminal.items())
            ),
            "exact_terminal_evaluated": {
                "v9_count": v9_exact_terminal,
                "v9_match": sum(
                    v9[tag].get("terminal_status") == "match" for tag in v9_success
                ),
                "v9_mismatch": sum(
                    v9[tag].get("terminal_status") == "mismatch"
                    for tag in v9_success
                ),
                "v10_count": v10_exact_terminal,
                "v10_match": sum(
                    v10[tag].get("terminal_status") == "match"
                    for tag in v10_success
                ),
                "v10_mismatch": sum(
                    v10[tag].get("terminal_status") == "mismatch"
                    for tag in v10_success
                ),
            },
            "interpretation": (
                "successful match/mismatch counts change 65/2 -> 68/3; "
                "the six newly accepted episodes contribute 4 match, 1 mismatch, "
                "and 1 missing. Generated terminal agreement is secondary evidence, "
                "not a substitute for original hidden per-Tick state truth."
            ),
        },
        "tick_store_validation": {"v9": v9_store, "v10": v10_store},
        "recommendation": {
            "promote_native_execution_boundary_to_source_tick_plus_1": True,
            "modify_source_labels": False,
            "runner_default_modified_by_this_audit": False,
            "required_configuration_change": {
                "action_execution_tick_offset": 1,
                "provenance": (
                    "source time_raw remains T; execute native action at T+1"
                ),
            },
            "reason": (
                "On 99 phase-comparable fixed inputs, T+1 preserves all 83 prior "
                "successes, converts all 6/6 exact-Tick code13 failures to success, "
                "and leaves the 10-tag code4 set unchanged. The lone excluded "
                "battle is a native logic freeze before the T+1 boundary, not an "
                "offset-1 action rejection. Terminal diagnostics add 3 matches and "
                "1 mismatch overall and do not outweigh the monotonic command-phase "
                "evidence."
            ),
        },
        "assertions": assertions,
        "all_assertions_passed": True,
        "per_tag": per_tag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9)
    parser.add_argument("--v10-root", type=Path, default=DEFAULT_V10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = audit(arguments.v9_root, arguments.v10_root)
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
                "selection_sha256": report["fixed_inputs"]["selection_sha256"],
                "phase_comparable": report["denominators"]["phase_comparable"],
                "v9_success": report["outcomes"]["phase_comparable_99"][
                    "v9_categories"
                ]["success"],
                "v10_success": report["outcomes"]["phase_comparable_99"][
                    "v10_categories"
                ]["success"],
                "v9_decoded_ticks": report["tick_store_validation"]["v9"][
                    "tick_count"
                ],
                "v10_decoded_ticks": report["tick_store_validation"]["v10"][
                    "tick_count"
                ],
                "recommend_offset_1": report["recommendation"][
                    "promote_native_execution_boundary_to_source_tick_plus_1"
                ],
                "all_assertions_passed": report["all_assertions_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
