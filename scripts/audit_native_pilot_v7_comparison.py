#!/usr/bin/env python3
"""Audit the fixed 100-battle native teacher-forced v6 -> v7 comparison.

This is deliberately read-only for both pilot directories.  The only write is
the requested machine-readable comparison report.
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
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.tick_store_v1.shard import (
    FRAME_HEADER,
    FRAME_MAGIC,
    ShardReader,
    sha256_file,
)


V6_PROVENANCE = {
    "runtime_baseline_head": "7a713452e2aad31ba1fff159ee7174dfdad4f718",
    "compact_tick_trace": "1152dd1d6d7f08691d43f4c7bc1918bfdf59d377",
    "native_guard_diagnostics": "5d90e5651ae72582838fd32df895796dc340ebc3",
}
V7_PROVENANCE = {
    "runtime_head": "411541791d6ea70b537115a593f6f6893d3d603e",
    "bounded_native_seed_search": "77223a0204469153b6de1bdcdec6b7739a974ba4",
    "spirit_empress_native_form_selection": "411541791d6ea70b537115a593f6f6893d3d603e",
}
FAILURE_CODE_RE = re.compile(r"codes_\[([^]]+)\]")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _indexed(rows: list[dict[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        tag = str(row["battle_tag"])
        if tag in result:
            raise RuntimeError(f"duplicate battle tag in {path}: {tag}")
        result[tag] = row
    return result


def _failure_codes(row: dict[str, Any]) -> list[int]:
    failure = str(row.get("failure") or "")
    match = FAILURE_CODE_RE.search(failure)
    if not match:
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]


def _category(row: dict[str, Any]) -> str:
    if bool(row.get("teacher_forced_success")):
        return "success"
    failure = str(row.get("failure") or "")
    codes = _failure_codes(row)
    if failure.startswith("native_shuffle_layout"):
        return "layout"
    if 4 in codes:
        return "code4"
    if 13 in codes:
        return "code13"
    if failure.startswith("native_terminal_before_source_tick"):
        return "terminal_before"
    return "other"


def _compact_result(row: dict[str, Any]) -> dict[str, Any]:
    rejection = row.get("first_rejection") or {}
    rejected_tokens = sorted(
        {
            str(event.get("base_token"))
            for event in rejection.get("events", [])
            if event.get("base_token") is not None
        }
    )
    return {
        "category": _category(row),
        "teacher_forced_success": bool(row.get("teacher_forced_success")),
        "failure": row.get("failure"),
        "failure_codes": _failure_codes(row),
        "rejected_tokens": rejected_tokens,
        "accepted_deployment_actions": int(row.get("accepted_deployment_actions", 0)),
        "source_deployment_actions": int(row.get("source_deployment_actions", 0)),
        "stored_tick_count": int(row.get("stored_tick_count", 0)),
        "terminal_status": row.get("terminal_status"),
        "source_crowns": row.get("source_crowns"),
        "observed_crowns": row.get("observed_crowns"),
        "seed": row.get("seed"),
        "chosen_seed": row.get("chosen_seed"),
        "layout_calibration_attempts": int(row.get("layout_calibration_attempts", 0)),
        "source_sha256": row.get("source_sha256"),
        "logical_training_state_sha256": row.get("logical_training_state_sha256"),
    }


def _verify_manifest_checksum(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    checksum_path = root / "manifest.sha256"
    manifest_digest = sha256_file(manifest_path)
    fields = checksum_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != "manifest.json":
        raise RuntimeError(f"invalid manifest.sha256 format: {checksum_path}")
    if fields[0].lower() != manifest_digest:
        raise RuntimeError("global manifest SHA-256 sidecar mismatch")
    return {
        "manifest_sha256": manifest_digest,
        "manifest_sidecar_matches": True,
    }


def _verify_tick_store(
    shard_root: Path,
    *,
    successful_tags: set[str],
    result_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = _json(shard_root / "manifest.json")
    checksum = _verify_manifest_checksum(shard_root)
    decoded_tags: set[str] = set()
    decoded_ticks = 0
    shard_reports: list[dict[str, Any]] = []
    content_digest_input: dict[str, dict[str, str]] = {}

    for shard in manifest["shards"]:
        name = str(shard["name"])
        data_path = shard_root / str(shard["data_file"])
        index_path = shard_root / str(shard["index_file"])
        shard_manifest_path = shard_root / f"{name}.manifest.json"
        on_disk_shard_manifest = _json(shard_manifest_path)
        if on_disk_shard_manifest != shard:
            raise RuntimeError(f"global/local shard manifest mismatch: {name}")

        data_sha256 = sha256_file(data_path)
        index_sha256 = sha256_file(index_path)
        if data_sha256 != shard["data_sha256"]:
            raise RuntimeError(f"data SHA-256 mismatch: {name}")
        if index_sha256 != shard["index_sha256"]:
            raise RuntimeError(f"index SHA-256 mismatch: {name}")
        if data_path.stat().st_size != int(shard["bytes"]):
            raise RuntimeError(f"data byte count mismatch: {name}")
        content_digest_input[name] = {
            "data": data_sha256,
            "index": index_sha256,
        }

        shard_ticks = 0
        shard_episodes = 0
        with ShardReader(data_path, index_path) as reader:
            for tag, entry in sorted(reader.entries.items()):
                if tag in decoded_tags:
                    raise RuntimeError(f"duplicate episode across shards: {tag}")
                if tag not in successful_tags:
                    raise RuntimeError(f"stored episode was not successful: {tag}")
                result_entry = result_rows[tag].get("store_entry")
                if result_entry is None:
                    raise RuntimeError(f"successful result lacks store entry: {tag}")
                for key in (
                    "offset",
                    "payload_size",
                    "ticks",
                    "tick_start",
                    "tick_stop",
                    "payload_sha256",
                ):
                    if result_entry.get(key) != entry.get(key):
                        raise RuntimeError(f"result/index {key} mismatch for {tag}")

                with data_path.open("rb") as handle:
                    handle.seek(int(entry["offset"]))
                    raw_header = handle.read(FRAME_HEADER.size)
                    magic, payload_size, _crc, _tag_hash, ticks, _reserved = FRAME_HEADER.unpack(
                        raw_header
                    )
                    if magic != FRAME_MAGIC or payload_size != int(entry["payload_size"]):
                        raise RuntimeError(f"frame header mismatch: {tag}")
                    if ticks != int(entry["ticks"]):
                        raise RuntimeError(f"frame Tick count mismatch: {tag}")
                    payload = handle.read(payload_size)
                if hashlib.sha256(payload).hexdigest() != entry["payload_sha256"]:
                    raise RuntimeError(f"episode payload SHA-256 mismatch: {tag}")

                episode = reader.episode(tag)
                count = 0
                previous_tick: int | None = None
                first_tick: int | None = None
                last_tick: int | None = None
                for state in episode.iter_ticks():
                    if first_tick is None:
                        first_tick = state.tick
                    if previous_tick is not None and state.tick != previous_tick + 1:
                        raise RuntimeError(
                            f"non-consecutive Tick in {tag}: {previous_tick} -> {state.tick}"
                        )
                    previous_tick = state.tick
                    last_tick = state.tick
                    count += 1
                if count != int(entry["ticks"]):
                    raise RuntimeError(f"decoded Tick count mismatch: {tag}")
                if first_tick != int(entry["tick_start"]):
                    raise RuntimeError(f"first Tick mismatch: {tag}")
                if last_tick is None or last_tick + 1 != int(entry["tick_stop"]):
                    raise RuntimeError(f"last Tick mismatch: {tag}")
                if count != int(result_rows[tag]["stored_tick_count"]):
                    raise RuntimeError(f"result decoded Tick count mismatch: {tag}")

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
        json.dumps(
            content_digest_input, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if content_sha256 != manifest["content_sha256"]:
        raise RuntimeError("global store content SHA-256 mismatch")
    if decoded_tags != successful_tags:
        missing = sorted(successful_tags - decoded_tags)
        extra = sorted(decoded_tags - successful_tags)
        raise RuntimeError(f"stored/success tag mismatch: missing={missing}, extra={extra}")
    if decoded_ticks != int(manifest["tick_count"]):
        raise RuntimeError("global decoded Tick count mismatch")
    if len(decoded_tags) != int(manifest["episode_count"]):
        raise RuntimeError("global decoded episode count mismatch")

    return {
        **checksum,
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


def audit(v6_root: Path, v7_root: Path) -> dict[str, Any]:
    v6_root = v6_root.resolve(strict=True)
    v7_root = v7_root.resolve(strict=True)
    v6_selection = v6_root / "selection.jsonl"
    v7_selection = v7_root / "selection.jsonl"
    v6_selection_sha = sha256_file(v6_selection)
    v7_selection_sha = sha256_file(v7_selection)
    selection_bytes_equal = v6_selection.read_bytes() == v7_selection.read_bytes()

    v6_results = _indexed(_jsonl(v6_root / "results.jsonl"), path=v6_root)
    v7_results = _indexed(_jsonl(v7_root / "results.jsonl"), path=v7_root)
    if set(v6_results) != set(v7_results):
        raise RuntimeError("v6/v7 result tag sets differ")
    tags = sorted(v6_results)
    if len(tags) != 100:
        raise RuntimeError(f"fixed comparison must contain 100 tags, got {len(tags)}")
    if not selection_bytes_equal or v6_selection_sha != v7_selection_sha:
        raise RuntimeError("v6/v7 selection files are not identical")
    for tag in tags:
        if v6_results[tag].get("source_sha256") != v7_results[tag].get("source_sha256"):
            raise RuntimeError(f"source SHA-256 changed for {tag}")

    v6_summary = _json(v6_root / "summary.json")
    v7_summary = _json(v7_root / "summary.json")
    transitions: dict[tuple[str, str], list[str]] = defaultdict(list)
    per_tag: list[dict[str, Any]] = []
    for tag in tags:
        before = _category(v6_results[tag])
        after = _category(v7_results[tag])
        transitions[(before, after)].append(tag)
        per_tag.append(
            {
                "battle_tag": tag,
                "transition": f"{before}->{after}",
                "source_sha256_equal": True,
                "v6": _compact_result(v6_results[tag]),
                "v7": _compact_result(v7_results[tag]),
            }
        )

    transition_rows = [
        {
            "from": before,
            "to": after,
            "count": len(transition_tags),
            "battle_tags": transition_tags,
        }
        for (before, after), transition_tags in sorted(transitions.items())
    ]
    v6_categories = Counter(_category(row) for row in v6_results.values())
    v7_categories = Counter(_category(row) for row in v7_results.values())
    v6_success = {tag for tag, row in v6_results.items() if row["teacher_forced_success"]}
    v7_success = {tag for tag, row in v7_results.items() if row["teacher_forced_success"]}
    v6_code4 = {tag for tag, row in v6_results.items() if _category(row) == "code4"}
    v7_code4 = {tag for tag, row in v7_results.items() if _category(row) == "code4"}
    v6_terminal = {
        tag for tag, row in v6_results.items() if _category(row) == "terminal_before"
    }
    v7_terminal = {
        tag for tag, row in v7_results.items() if _category(row) == "terminal_before"
    }
    spirit_tags = sorted(
        tag
        for tag, row in v6_results.items()
        if "spirit-empress" in _compact_result(row)["rejected_tokens"]
        and _category(row) == "code13"
        and _category(v7_results[tag]) == "success"
    )
    seed_probe_rows = _jsonl(v7_root / "seed-probe.jsonl")
    seed_equal = sum(bool(row.get("logical_training_state_equal")) for row in seed_probe_rows)

    tick_store = _verify_tick_store(
        v7_root / "shards",
        successful_tags=v7_success,
        result_rows=v7_results,
    )
    assertions = {
        "selection_identical_100_tags": selection_bytes_equal and len(tags) == 100,
        "all_source_sha256_equal": all(item["source_sha256_equal"] for item in per_tag),
        "success_37_to_40": len(v6_success) == 37 and len(v7_success) == 40,
        "layout_6_to_0": v6_categories["layout"] == 6 and v7_categories["layout"] == 0,
        "layout_migration_4_to_code13_2_to_success": len(transitions[("layout", "code13")]) == 4
        and len(transitions[("layout", "success")]) == 2,
        "spirit_empress_code13_to_success": len(spirit_tags) == 1,
        "code4_episode_set_stable": v6_code4 == v7_code4 and len(v6_code4) == 41,
        "terminal_before_episode_set_stable": v6_terminal == v7_terminal
        and len(v6_terminal) == 8,
        "terminal_match_23_to_26": int(v6_summary["terminal_diagnostic"]["match"]) == 23
        and int(v7_summary["terminal_diagnostic"]["match"]) == 26,
        "seed_probe_4_of_4_equal": len(seed_probe_rows) == 4 and seed_equal == 4,
        "v7_tick_store_40_episodes_164723_ticks": tick_store["episode_count"] == 40
        and tick_store["tick_count"] == 164723,
        "v7_tick_store_full_decode_and_sha": tick_store["all_episodes_fully_decoded"]
        and tick_store["all_manifest_data_index_sha256_match"]
        and tick_store["all_episode_payload_sha256_match"],
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise RuntimeError(f"fixed v7 comparison assertions failed: {failed}")

    return {
        "schema_version": 1,
        "kind": "native_teacher_forced_pilot_v7_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_provenance": {
            "v6": V6_PROVENANCE,
            "v7": V7_PROVENANCE,
            "semantic_delta_under_audit": [
                V7_PROVENANCE["bounded_native_seed_search"],
                V7_PROVENANCE["spirit_empress_native_form_selection"],
            ],
        },
        "inputs": {
            "v6_root": str(v6_root),
            "v7_root": str(v7_root),
            "v6_results_sha256": sha256_file(v6_root / "results.jsonl"),
            "v7_results_sha256": sha256_file(v7_root / "results.jsonl"),
            "v6_summary_sha256": sha256_file(v6_root / "summary.json"),
            "v7_summary_sha256": sha256_file(v7_root / "summary.json"),
        },
        "selection": {
            "battle_count": len(tags),
            "tag_sets_equal": True,
            "files_byte_identical": selection_bytes_equal,
            "v6_sha256": v6_selection_sha,
            "v7_sha256": v7_selection_sha,
        },
        "outcomes": {
            "v6_categories": dict(sorted(v6_categories.items())),
            "v7_categories": dict(sorted(v7_categories.items())),
            "v6_successful_episodes": len(v6_success),
            "v7_successful_episodes": len(v7_success),
            "success_delta": len(v7_success) - len(v6_success),
            "new_success_tags": sorted(v7_success - v6_success),
        },
        "transitions": transition_rows,
        "focused_migrations": {
            "spirit_empress_code13_to_success": spirit_tags,
            "layout_to_code13": transitions[("layout", "code13")],
            "layout_to_success": transitions[("layout", "success")],
            "stable_code4": sorted(v6_code4),
            "stable_terminal_before": sorted(v6_terminal),
        },
        "native_rejections": {
            "v6_code_counts": v6_summary["teacher_forced"]["native_rejection_code_counts"],
            "v7_code_counts": v7_summary["teacher_forced"]["native_rejection_code_counts"],
            "code4_episode_count": len(v7_code4),
            "code4_rejected_action_count": int(
                v7_summary["teacher_forced"]["native_rejection_code_counts"]["4"]
            ),
        },
        "terminal_diagnostic": {
            "v6": v6_summary["terminal_diagnostic"],
            "v7": v7_summary["terminal_diagnostic"],
        },
        "seed_diagnostic": {
            "probed": len(seed_probe_rows),
            "logical_training_state_equal": seed_equal,
            "all_equal": seed_equal == len(seed_probe_rows),
            "probes": seed_probe_rows,
        },
        "v7_tick_store_validation": tick_store,
        "assertions": assertions,
        "all_assertions_passed": True,
        "per_tag": per_tag,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("v6_root", type=Path)
    parser.add_argument("v7_root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.v6_root, arguments.v7_root)
    output = arguments.output or arguments.v7_root / "comparison-v6-v7.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output.resolve()),
        "selection_sha256": report["selection"]["v7_sha256"],
        "v6_success": report["outcomes"]["v6_successful_episodes"],
        "v7_success": report["outcomes"]["v7_successful_episodes"],
        "decoded_episodes": report["v7_tick_store_validation"]["episode_count"],
        "decoded_ticks": report["v7_tick_store_validation"]["tick_count"],
        "all_assertions_passed": report["all_assertions_passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
