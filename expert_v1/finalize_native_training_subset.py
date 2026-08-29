"""Publish an auditable Full+Prefix training subset after native generation.

This module never calls an Android worker and never fabricates a Tick.  It is
the narrow continuation path for a completed native run whose final result set
contains a small number of attempts without either a Full frame or a verified
Prefix frame.  Those unframed battles are written to an immutable exclusion
receipt; every remaining battle must form an exact candidate/result/store
union before a production compiler receipt is emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .native_dataset_generator import (
    AUDIT_PREFIX_DIRECTORY,
    COORDINATE_PROVENANCE,
    DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS,
    GENERATOR_KIND,
    NATIVE_PREFLIGHT_CONTRACT_VERSION,
    NATIVE_PREFLIGHT_MODE,
    SEMANTIC_SEED_AUDIT_KIND,
    SEMANTIC_SEED_AUDIT_SCHEMA_VERSION,
)
from .one_click_v1 import (
    evaluate_ability_positive_coverage,
    file_fingerprint,
    native_contract_binding,
    validate_native_result_records,
)
from .tick_store_v1.deployment_masks import (
    DYNAMIC_RULE,
    MASK_STORE_DIRECTORY,
    DeploymentMaskStore,
)
from .tick_store_v1.shard import (
    AUDIT_PREFIX_STORE_KIND,
    AppendOnlyShardWriter,
    build_store_manifest,
    sha256_file,
)


SUBSET_SCHEMA_VERSION = 1
SUBSET_KIND = "cr_expert_native_unframed_exclusion_v1"
AUTHORIZATION = "user_authorized_no_reprocess_20260829"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_filtered_results(
    source: Path,
    destination: Path,
    *,
    expected_rows: int,
) -> tuple[set[str], list[dict[str, str]], dict[str, int], dict[str, int]]:
    """Stream 2+ GiB of results without retaining result objects in RAM."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    usable: set[str] = set()
    seen: set[str] = set()
    excluded: list[dict[str, str]] = []
    failure_classes: Counter[str] = Counter()
    failure_domains: Counter[str] = Counter()
    with (
        source.resolve(strict=True).open("rb") as rows,
        temporary.open("wb") as output,
    ):
        for line_number, line in enumerate(rows, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as error:
                raise RuntimeError(
                    f"invalid native result JSON at line {line_number}"
                ) from error
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"native result is not an object at line {line_number}"
                )
            tag = str(row.get("battle_tag") or "")
            if not tag or tag in seen:
                raise RuntimeError(
                    f"missing/duplicate native result tag at line {line_number}"
                )
            seen.add(tag)
            full = bool(
                row.get("teacher_forced_success") is True
                and isinstance(row.get("tick_store_entry"), Mapping)
            )
            prefix = bool(
                row.get("teacher_forced_success") is False
                and isinstance(row.get("audit_prefix_tick_store_entry"), Mapping)
            )
            if full or prefix:
                usable.add(tag)
                output.write(line if line.endswith(b"\n") else line + b"\n")
                if not full:
                    failure_classes[str(row.get("failure_class") or "unknown")] += 1
                    failure_domains[str(row.get("failure_domain") or "unknown")] += 1
            else:
                excluded.append({
                    "battle_tag": tag,
                    "failure_class": str(row.get("failure_class") or "unknown"),
                    "failure_domain": str(row.get("failure_domain") or "unknown"),
                })
        output.flush()
        os.fsync(output.fileno())
    if len(seen) != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"native result cardinality changed: {len(seen)}/{expected_rows}"
        )
    if len(usable) + len(excluded) != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("native Full/Prefix/excluded partition is not exact")
    os.replace(temporary, destination)
    return (
        usable,
        sorted(excluded, key=lambda value: value["battle_tag"]),
        dict(sorted(failure_classes.items())),
        dict(sorted(failure_domains.items())),
    )


def _write_filtered_candidates(
    source: Path,
    destination: Path,
    *,
    usable: set[str],
    expected_original_rows: int,
) -> dict[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    seen: set[str] = set()
    selected: set[str] = set()
    ability_positive = 0
    with (
        source.resolve(strict=True).open("rb") as rows,
        temporary.open("wb") as output,
    ):
        for line_number, line in enumerate(rows, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as error:
                raise RuntimeError(
                    f"invalid candidate JSON at line {line_number}"
                ) from error
            tag = str(row.get("battle_tag") or "")
            if not tag or tag in seen:
                raise RuntimeError(
                    f"missing/duplicate candidate tag at line {line_number}"
                )
            seen.add(tag)
            if tag in usable:
                output.write(line if line.endswith(b"\n") else line + b"\n")
                selected.add(tag)
                ability_positive += int(
                    int(row.get("ability_events_observed") or 0) > 0
                )
        output.flush()
        os.fsync(output.fileno())
    if len(seen) != expected_original_rows or selected != usable:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "filtered candidate queue is not the exact Full+Prefix tag set"
        )
    os.replace(temporary, destination)
    return {
        "rows": len(selected),
        "ability_positive": ability_positive,
        "ability_zero": len(selected) - ability_positive,
    }


def _count_frozen_tags(path: Path) -> set[str]:
    tags: set[str] = set()
    with path.resolve(strict=True).open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            tag = str(row.get("battle_tag") or "")
            if not tag or tag in tags:
                raise RuntimeError(
                    f"missing/duplicate frozen tag at line {line_number}"
                )
            tags.add(tag)
    return tags


def _build_store_manifests(
    native_root: Path,
    *,
    selection_manifest: Path,
    run_contract: Mapping[str, Any],
    full_episodes: int,
    full_ticks: int,
    prefix_episodes: int,
    prefix_ticks: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    full_root = native_root / "shards"
    prefix_root = native_root / AUDIT_PREFIX_DIRECTORY
    _finalize_existing_partial_shards(full_root)
    _finalize_existing_partial_shards(prefix_root)
    full_mask = DeploymentMaskStore(full_root, create=False).build_manifest()
    full_mask_path = full_root / MASK_STORE_DIRECTORY / "manifest.json"
    full_store = build_store_manifest(
        full_root,
        source_manifest=selection_manifest,
        expected_episodes=full_episodes,
        expected_ticks=full_ticks,
        store_metadata={
            "generator_kind": GENERATOR_KIND,
            "native_teacher_forced_profile": run_contract[
                "native_teacher_forced_profile"
            ],
            "coordinate_provenance": COORDINATE_PROVENANCE,
            "ability_branch_policy": run_contract["ability_branch_policy"],
            "source_json_copied": False,
            "native_deployment_masks": {
                "required": True,
                "schema_version": 1,
                "dynamic_rule": DYNAMIC_RULE,
                "manifest": f"{MASK_STORE_DIRECTORY}/manifest.json",
                "manifest_sha256": sha256_file(full_mask_path),
                "sidecars": int(full_mask["sidecars"]),
            },
        },
    )
    prefix_mask = DeploymentMaskStore(prefix_root, create=False).build_manifest()
    prefix_mask_path = prefix_root / MASK_STORE_DIRECTORY / "manifest.json"
    prefix_store = build_store_manifest(
        prefix_root,
        source_manifest=selection_manifest,
        expected_episodes=prefix_episodes,
        expected_ticks=prefix_ticks,
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
                "manifest_sha256": sha256_file(prefix_mask_path),
                "sidecars": int(prefix_mask["sidecars"]),
            },
        },
    )
    return full_store, prefix_store, full_mask_path, prefix_mask_path


def _finalize_existing_partial_shards(root: Path) -> list[dict[str, Any]]:
    """Seal already-fsynced frames left by an interrupted supervisor.

    ``AppendOnlyShardWriter.finalize`` performs a non-truncating frame scan,
    rebuilds the index from checksummed payloads, hashes both files, and only
    then atomically renames the partial data file.  No native execution occurs.
    """

    finalized: list[dict[str, Any]] = []
    for partial in sorted(root.glob("*.crts.partial")):
        stem = partial.name.removesuffix(".crts.partial")
        if (root / f"{stem}.crts").exists() or (
            root / f"{stem}.manifest.json"
        ).exists():
            raise RuntimeError(f"partial/final shard collision: {stem}")
        writer = AppendOnlyShardWriter(
            root,
            stem,
            anchor_interval=256,
            compression_level=1,
        )
        if writer.episode_count == 0:
            writer.close()
            writer.partial_path.unlink(missing_ok=True)
            continue
        finalized.append(writer.finalize())
    return finalized


def finalize_native_training_subset(
    data_root: Path,
    *,
    authorization: str = AUTHORIZATION,
    expected_original_rows: int = 100_000,
    minimum_ability_positive_success_count: int = 1,
    minimum_ability_positive_success_rate: float = 0.10,
) -> dict[str, Any]:
    data_root = data_root.resolve(strict=True)
    native_root = data_root / "native-authoritative-ticks-v1"
    receipts_root = data_root / "receipts"
    summary_path = native_root / "summary.json"
    original_results = native_root / "results.jsonl"
    original_candidates = (
        data_root / "eligibility" / "native-eligibility-v1" / "queues"
        / "authoritative-native-full.jsonl"
    )
    frozen_manifest = data_root / "manifests" / "schema5-v3-100k.jsonl"
    source_token_receipt = receipts_root / "source-token-coverage-v1.json"
    run_contract_path = native_root / "run-contract.json"
    selection_manifest = native_root / "selection.jsonl"
    filtered_results = native_root / "results.trainable-full-prefix.jsonl"
    filtered_candidates = (
        original_candidates.parent / "authoritative-native-trainable-full-prefix.jsonl"
    )
    exclusion_path = receipts_root / "native-unframed-exclusion-v1.json"
    coverage_path = receipts_root / "native-generation-coverage.json"

    summary = _load_object(summary_path)
    run_contract = _load_object(run_contract_path)
    if (
        int(summary.get("selected_battles", -1)) != expected_original_rows
        or int(summary.get("processed_battles", -1)) != expected_original_rows
        or summary.get("missing_result_tags") != []
        or summary.get("unexpected_result_tags") != []
        or summary.get("worker_errors") != []
    ):
        raise RuntimeError("native run is not a closed 100k final-attempt set")

    usable, excluded, failure_classes, failure_domains = _write_filtered_results(
        original_results,
        filtered_results,
        expected_rows=expected_original_rows,
    )
    frozen_tags = _count_frozen_tags(frozen_manifest)
    excluded_tags = {row["battle_tag"] for row in excluded}
    if (
        len(frozen_tags) != expected_original_rows
        or usable & excluded_tags
        or usable | excluded_tags != frozen_tags
    ):
        raise RuntimeError("Full+Prefix/exclusion set is not the frozen 100k set")
    queue_summary = _write_filtered_candidates(
        original_candidates,
        filtered_candidates,
        usable=usable,
        expected_original_rows=expected_original_rows,
    )
    admitted_rows = len(usable)
    full_episodes = int(summary["stored_episodes"])
    prefix_episodes = int(summary["audit_prefix_episodes"])
    if (
        admitted_rows != full_episodes + prefix_episodes
        or len(excluded) != int(summary["unframed_episodes"])
    ):
        raise RuntimeError("summary Full/Prefix/unframed counts changed")

    exclusion_body: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "kind": SUBSET_KIND,
        "created_utc": utc_now(),
        "authorization": authorization,
        "policy": {
            "native_reprocessing": False,
            "synthetic_ticks": False,
            "excluded_tags_trainable": False,
            "admission": "exact_physical_full_or_verified_prefix_v1",
        },
        "frozen_manifest": file_fingerprint(frozen_manifest),
        "original_candidate_queue": file_fingerprint(original_candidates),
        "original_results": file_fingerprint(original_results),
        "original_summary": file_fingerprint(summary_path),
        "original_battles": expected_original_rows,
        "training_battles": admitted_rows,
        "excluded_unframed_battles": len(excluded),
        "excluded_failure_class_counts": dict(sorted(Counter(
            row["failure_class"] for row in excluded
        ).items())),
        "excluded_failure_domain_counts": dict(sorted(Counter(
            row["failure_domain"] for row in excluded
        ).items())),
        "excluded": excluded,
    }
    exclusion_body["canonical_sha256"] = hashlib.sha256(
        _canonical_bytes(exclusion_body)
    ).hexdigest()
    _atomic_json(exclusion_path, exclusion_body)

    full_store, prefix_store, full_mask_path, prefix_mask_path = (
        _build_store_manifests(
            native_root,
            selection_manifest=selection_manifest,
            run_contract=run_contract,
            full_episodes=full_episodes,
            full_ticks=int(summary["stored_ticks"]),
            prefix_episodes=prefix_episodes,
            prefix_ticks=int(summary["audit_prefix_ticks"]),
        )
    )
    result_audit = validate_native_result_records(
        filtered_results,
        filtered_candidates,
        expected_rows=admitted_rows,
        require_token_evidence=True,
        verify_physical_mask_invalid_proof=True,
    )
    if (
        result_audit["unframed_tags"] != []
        or len(result_audit["success_tags"]) != full_episodes
        or len(result_audit["audit_prefix_tags"]) != prefix_episodes
        or int(result_audit["token_coverage_actor_evidence_records"])
        != admitted_rows * 2
    ):
        raise RuntimeError("filtered result audit is not an exact Full+Prefix union")
    ability_coverage = evaluate_ability_positive_coverage(
        queue_summary,
        result_audit,
        minimum_success_count=minimum_ability_positive_success_count,
        minimum_success_rate=minimum_ability_positive_success_rate,
        waived=False,
        waiver_reason=None,
    )
    if (ability_coverage.get("gate") or {}).get("admitted") is not True:
        raise RuntimeError("ability-positive evidence gate rejected training subset")

    coverage = {
        "schema_version": 2,
        "kind": "cr_expert_native_generation_coverage_v2",
        "created_utc": utc_now(),
        "frozen_manifest": file_fingerprint(frozen_manifest),
        "candidate_queue": file_fingerprint(filtered_candidates),
        "results": file_fingerprint(filtered_results),
        "native_contract": native_contract_binding(
            Path(str(run_contract["native_ingest_contract"]["path"]))
        ),
        "native_execution_pipeline": {
            "contract_version": NATIVE_PREFLIGHT_CONTRACT_VERSION,
            "mode": NATIVE_PREFLIGHT_MODE,
            "semantic_seed_audit_schema_version": (
                SEMANTIC_SEED_AUDIT_SCHEMA_VERSION
            ),
            "semantic_seed_audit_kind": SEMANTIC_SEED_AUDIT_KIND,
            "layout_compatible_candidate_limit": (
                DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
            ),
        },
        "source_token_coverage": file_fingerprint(source_token_receipt),
        "unframed_exclusion": file_fingerprint(exclusion_path),
        "original_target_battles": expected_original_rows,
        "target_battles": admitted_rows,
        "selected_battles": admitted_rows,
        "processed_battles": admitted_rows,
        "teacher_forced_successes": full_episodes,
        "teacher_forced_failures": prefix_episodes,
        "stored_episodes": full_episodes,
        "audit_prefix_episodes": prefix_episodes,
        "audit_tick_episodes": admitted_rows,
        "unframed_episodes": 0,
        "audit_tick_coverage_rate": 1.0,
        "audit_prefix_store": file_fingerprint(
            native_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
        ),
        "success_rate": full_episodes / admitted_rows,
        "full_success_rate_semantics": "diagnostic_only",
        "ability_coverage": ability_coverage,
        "failure_class_counts": failure_classes,
        "failure_domain_counts": failure_domains,
        "terminal_diagnostic_counts": summary.get(
            "terminal_diagnostic_counts"
        ) or {},
        "queue_counts": {
            "full": full_episodes,
            "prefix": prefix_episodes,
            "excluded_unframed": len(excluded),
        },
        "native_actions_attempted": int(summary.get("native_actions_attempted", 0)),
        "native_actions_accepted": int(summary.get("native_actions_accepted", 0)),
        "token_coverage_actor_evidence_records": int(
            result_audit["token_coverage_actor_evidence_records"]
        ),
    }
    _atomic_json(coverage_path, coverage)

    native_manifest = {
        "schema_version": 1,
        "kind": "expert_authoritative_native_tick_dataset_manifest_v1",
        "created_utc": utc_now(),
        "status": "complete_with_user_authorized_unframed_exclusion",
        "source": {
            "frozen_manifest": file_fingerprint(frozen_manifest),
            "original_candidate_queue": file_fingerprint(original_candidates),
            "training_candidate_queue": file_fingerprint(filtered_candidates),
            "selection_manifest": file_fingerprint(selection_manifest),
            "source_json_copied": False,
        },
        "semantics": {
            "battle_core": "original libg.so",
            "tick_hz": 20,
            "first_difference_policy": "fail_closed",
            "training_admission": "exact_physical_full_or_verified_prefix_v1",
            "unframed_exclusion": file_fingerprint(exclusion_path),
        },
        "counts": {
            "original_battles": expected_original_rows,
            "training_battles": admitted_rows,
            "excluded_unframed_battles": len(excluded),
            "stored_episodes": full_episodes,
            "stored_ticks": int(full_store["tick_count"]),
            "audit_prefix_episodes": prefix_episodes,
            "audit_prefix_ticks": int(prefix_store["tick_count"]),
        },
        "content": {
            "results": filtered_results.name,
            "results_sha256": sha256_file(filtered_results),
            "original_results": original_results.name,
            "original_results_sha256": sha256_file(original_results),
            "summary": summary_path.name,
            "summary_sha256": sha256_file(summary_path),
            "tick_store_manifest": "shards/manifest.json",
            "tick_store_manifest_sha256": sha256_file(
                native_root / "shards" / "manifest.json"
            ),
            "deployment_mask_store_manifest": (
                f"shards/{MASK_STORE_DIRECTORY}/manifest.json"
            ),
            "deployment_mask_store_manifest_sha256": sha256_file(full_mask_path),
            "audit_prefix_tick_store_manifest": (
                f"{AUDIT_PREFIX_DIRECTORY}/manifest.json"
            ),
            "audit_prefix_tick_store_manifest_sha256": sha256_file(
                native_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
            ),
            "audit_prefix_deployment_mask_store_manifest": (
                f"{AUDIT_PREFIX_DIRECTORY}/{MASK_STORE_DIRECTORY}/manifest.json"
            ),
            "audit_prefix_deployment_mask_store_manifest_sha256": sha256_file(
                prefix_mask_path
            ),
        },
    }
    _atomic_json(native_root / "manifest.json", native_manifest)
    _atomic_bytes(
        native_root / "manifest.sha256",
        (
            f"{sha256_file(native_root / 'manifest.json')}  manifest.json\n"
        ).encode("ascii"),
    )

    result = {
        "kind": "cr_expert_native_training_subset_finalized_v1",
        "schema_version": 1,
        "created_utc": utc_now(),
        "original_battles": expected_original_rows,
        "training_battles": admitted_rows,
        "full_episodes": full_episodes,
        "prefix_episodes": prefix_episodes,
        "excluded_unframed_battles": len(excluded),
        "coverage_receipt": file_fingerprint(coverage_path),
        "exclusion_receipt": file_fingerprint(exclusion_path),
        "native_manifest": file_fingerprint(native_root / "manifest.json"),
        "tick_store_manifest": file_fingerprint(
            native_root / "shards" / "manifest.json"
        ),
        "audit_prefix_store_manifest": file_fingerprint(
            native_root / AUDIT_PREFIX_DIRECTORY / "manifest.json"
        ),
    }
    _atomic_json(receipts_root / "native-training-subset-finalized-v1.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-original-rows", type=int, default=100_000)
    parser.add_argument("--authorization", default=AUTHORIZATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_native_training_subset(
        args.data_root,
        authorization=args.authorization,
        expected_original_rows=args.expected_original_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
