"""Read-only eligibility audit for the frozen expert-v1 100k corpus.

The audit deliberately does not connect to a Worker or call libg.  It applies
the same ``compile_battle`` and native capability mappings as the replay
runner, then writes reference-only queue manifests.  Source battle JSON files
are opened read-only and are never copied into the output tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .native_capabilities import ability_log_tier
from .native_replay_plan import ReplayPlanError, compile_battle, split_card_token


AUDIT_KIND = "expert_100k_native_eligibility_v1"
DEFAULT_MANIFEST = Path(
    r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
    r"\version-window-20260804\accepted-cycle-clean.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\native-eligibility-v1"
)


@dataclass(frozen=True)
class PilotEvidence:
    successes: int
    trials: int
    ticks_per_wall_second: float
    bytes_per_tick: float
    source: str


DEPLOYMENT_PILOT = PilotEvidence(
    successes=89,
    trials=100,
    ticks_per_wall_second=1448.1381173543393,
    bytes_per_tick=25.059714019028796,
    source=(
        r"D:\AI_data\cr-native-core\expert-v1"
        r"\native-teacher-forced-pilot-100-data-i-phase-plus1-v10\summary.json"
    ),
)
ABILITY_PILOT = PilotEvidence(
    successes=58,
    trials=100,
    ticks_per_wall_second=1035.1159561590086,
    bytes_per_tick=26.211192535833273,
    source=(
        r"D:\AI_data\cr-native-core\expert-v1"
        r"\native-ability-pilot-100-data-i-phase-plus1-v1\summary.json"
    ),
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _ability_count(value: Mapping[str, Any]) -> int:
    total = 0
    for side in ("team", "opponent"):
        try:
            total += int(value["elixir_stats"][side]["Ability"]["count"] or 0)
        except (KeyError, TypeError, ValueError):
            pass
    return total


def _raw_coordinate_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    plays = value.get("card_plays")
    if not isinstance(plays, list):
        plays = []
    exact = zero = one = partial = 0
    for event in plays:
        if not isinstance(event, Mapping):
            partial += 1
            continue
        raw_x, raw_y, data_i = event.get("x_raw"), event.get("y_raw"), event.get("data_i")
        valid = (
            isinstance(raw_x, int) and not isinstance(raw_x, bool)
            and isinstance(raw_y, int) and not isinstance(raw_y, bool)
            and isinstance(data_i, int) and not isinstance(data_i, bool)
            and data_i in (0, 1)
        )
        if valid:
            exact += 1
            zero += int(data_i == 0)
            one += int(data_i == 1)
        else:
            partial += 1
    if plays and exact == len(plays):
        tier = "all_card_events_raw_data_i"
    elif exact:
        tier = "partial_raw_data_i"
    else:
        tier = "legacy_xy_only"
    return {
        "coordinate_tier": tier,
        "card_events": len(plays),
        "raw_data_i_events": exact,
        "data_i_zero_events": zero,
        "data_i_one_events": one,
        "legacy_or_invalid_events": partial,
    }


def _metadata_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    schema = int(value.get("schema_version") or 1)
    rounds = value.get("rounds")
    deck_complete = levels_complete = forms_complete = towers_complete = True
    tower_levels_complete = schema == 5
    tower_tokens: list[str | None] = []
    if schema < 2 or not isinstance(rounds, list) or len(rounds) != 1:
        deck_complete = levels_complete = forms_complete = towers_complete = False
    else:
        round_zero = rounds[0] if isinstance(rounds[0], Mapping) else {}
        for side in ("team", "opponent"):
            players = round_zero.get(side) if isinstance(round_zero, Mapping) else None
            if not isinstance(players, list) or len(players) != 1 or not isinstance(players[0], Mapping):
                deck_complete = levels_complete = forms_complete = towers_complete = False
                tower_tokens.append(None)
                continue
            player = players[0]
            tokens = player.get("full_deck")
            levels = player.get("card_levels")
            complete = player.get("complete") is True
            valid_tokens = isinstance(tokens, list) and len(tokens) == 8
            if valid_tokens:
                try:
                    bases = [split_card_token(str(token))[0] for token in tokens]
                    valid_tokens = len(set(bases)) == 8
                except ReplayPlanError:
                    valid_tokens = False
            deck_complete &= bool(complete and valid_tokens)
            forms_complete &= bool(complete and valid_tokens)
            levels_complete &= bool(
                complete and valid_tokens and isinstance(levels, Mapping)
                and all(str(token) in levels for token in tokens)
            )
            tower = str(player.get("tower_troop") or "") or None
            tower_tokens.append(tower)
            towers_complete &= tower is not None
            if schema == 5:
                tower_level = player.get("tower_troop_level")
                tower_levels_complete &= (
                    isinstance(tower_level, int)
                    and not isinstance(tower_level, bool)
                    and tower_level >= 1
                )
    terminal = value.get("final_tower_hp")
    final_tower_hp_complete = False
    if schema == 5 and isinstance(terminal, Mapping):
        final_tower_hp_complete = (
            terminal.get("provenance") == "list_hp_both_popup"
            and terminal.get("slot_mapping_provenance") == "source_slots_unmapped"
        )
        for side in ("team", "opponent"):
            hp = terminal.get(side)
            keys = ("king", "princess0", "princess1", "total")
            valid = isinstance(hp, Mapping) and all(
                isinstance(hp.get(key), int)
                and not isinstance(hp.get(key), bool)
                and int(hp[key]) >= 0
                for key in keys
            )
            final_tower_hp_complete &= bool(
                valid and int(hp["total"]) == sum(int(hp[key]) for key in keys[:3])
            )
    contract_stamp = value.get("authoritative_native_contract")
    schema5_contract_stamp_complete = bool(
        schema == 5
        and isinstance(contract_stamp, Mapping)
        and str(contract_stamp.get("game_version") or "")
        and len(str(contract_stamp.get("contract_sha256") or "")) == 64
        and len(str(contract_stamp.get("contract_file_sha256") or "")) == 64
    )
    return {
        "eight_card_decks_complete": deck_complete,
        "card_levels_complete": levels_complete,
        "card_forms_complete_by_schema_contract": forms_complete,
        "tower_troops_complete": towers_complete,
        "tower_troops": tower_tokens,
        "tower_troop_levels_complete": tower_levels_complete,
        "final_tower_hp_complete": final_tower_hp_complete,
        "schema5_contract_stamp_complete": schema5_contract_stamp_complete,
    }


def _failure_class(message: str) -> str:
    value = message.lower()
    if "absent from frozen libg catalog" in value:
        return "card_catalog_mapping_missing"
    if "unmapped native tower troop" in value:
        return "tower_troop_mapping_missing"
    if "runtime" in value and "mapping" in value:
        return "native_mapping_missing"
    if "cycle" in value:
        return "eight_card_cycle_rejected"
    if "coordinate" in value or "arena" in value or "data_i" in value:
        return "coordinate_rejected"
    if "multiple deploy/ability" in value:
        return "same_tick_deploy_ability_collision"
    if "multiple actions for side" in value:
        return "same_tick_multiple_deployments"
    if "deck" in value or "card level" in value or "player" in value:
        return "deck_metadata_rejected"
    if "ability" in value:
        return "ability_log_rejected"
    return "other_compile_rejected"


def audit_one(
    manifest_row: Mapping[str, Any], *, native_ingest_contract: Any | None = None,
) -> dict[str, Any]:
    source = Path(str(manifest_row["source_path"]))
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    schema = int(value.get("schema_version") or manifest_row.get("schema_version") or 1)
    duration = value.get("duration_seconds")
    source_ticks = (
        int(round(float(duration) * 20.0))
        if isinstance(duration, (int, float)) else None
    )
    coordinates = _raw_coordinate_audit(value)
    metadata = _metadata_audit(value)
    ability_count = _ability_count(value)
    ability_tier = ability_log_tier(value)
    base: dict[str, Any] = {
        "battle_tag": str(value.get("battle_tag") or manifest_row.get("battle_tag") or ""),
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_schema_version": schema,
        "duration_ticks": source_ticks,
        "deployment_actions": coordinates["card_events"],
        "ability_count_reported": ability_count,
        "ability_events_observed": len(value.get("ability_plays") or [])
            if isinstance(value.get("ability_plays"), list) else 0,
        "ability_positive": ability_count > 0,
        "ability_log_tier": ability_tier,
        **coordinates,
        **metadata,
    }
    try:
        crowns = (
            int(manifest_row["team_crowns"]), int(manifest_row["opponent_crowns"])
        ) if "team_crowns" in manifest_row and "opponent_crowns" in manifest_row else None
        plan = compile_battle(
            value, terminal_crowns=crowns,
            native_ingest_contract=native_ingest_contract,
        )
    except (ReplayPlanError, KeyError, TypeError, ValueError) as error:
        return {
            **base,
            "compile_ok": False,
            "compile_rejection_class": _failure_class(str(error)),
            "compile_rejection": str(error),
            "compiler_native_replay_ready": False,
            "authoritative_native_full_candidate": False,
            "eligibility_tier": "compile_rejected",
        }

    mapping_limitations = [
        item for item in plan.limitations
        if item.startswith((
            "runtime_card_forms_unsupported:",
            "tower_troop_runtime_mapping_missing:",
            "ability_card_runtime_mapping_missing_side_",
        ))
    ]
    all_metadata = all(bool(metadata[key]) for key in (
        "eight_card_decks_complete", "card_levels_complete",
        "card_forms_complete_by_schema_contract", "tower_troops_complete",
    ))
    exact_coordinates = coordinates["coordinate_tier"] == "all_card_events_raw_data_i"
    exact_ability = ability_tier in {
        "source_reports_zero", "observed_ticks_identity_runtime_resolved",
    }
    schema5_contract_verified = bool(
        schema == 5
        and plan.authoritative_contract_provenance
        == "schema5_authoritative_native_contract_verified"
    )
    schema_specific_complete = bool(
        schema == 3
        or (
            schema == 5
            and metadata["tower_troop_levels_complete"]
            and metadata["final_tower_hp_complete"]
            and metadata["schema5_contract_stamp_complete"]
            and schema5_contract_verified
        )
    )
    authoritative = bool(
        plan.native_replay_ready and schema in {3, 5} and exact_coordinates
        and all_metadata and exact_ability and not mapping_limitations
        and schema_specific_complete
    )
    if authoritative and ability_count == 0:
        tier = "authoritative_native_deployment_only"
    elif authoritative:
        tier = "authoritative_native_ability_exact"
    elif plan.native_replay_ready and schema == 2 and ability_count == 0:
        tier = "schema2_native_ready_legacy_coordinate_approximate"
    elif schema == 2 and ability_tier == "count_only_missing_ticks":
        tier = "schema2_deployment_only_missing_ability_ticks_approximate"
    elif schema == 1:
        tier = "schema1_sequence_only_missing_native_metadata"
    elif mapping_limitations:
        tier = "native_mapping_rejected"
    elif ability_tier == "count_only_missing_ticks":
        tier = "deployment_only_missing_ability_ticks"
    else:
        tier = "action_sequence_only_other"
    return {
        **base,
        "compile_ok": True,
        "compile_rejection_class": None,
        "compile_rejection": None,
        "compiler_native_replay_ready": bool(plan.native_replay_ready),
        "authoritative_native_full_candidate": authoritative,
        "eligibility_tier": tier,
        "replay_tier": plan.replay_tier,
        "mapping_complete": not mapping_limitations,
        "schema5_authoritative_contract_verified": schema5_contract_verified,
        "numeric_game_mode_id": plan.numeric_game_mode_id,
        "battle_index": plan.battle_index,
        "terminal_provenance": plan.terminal_provenance,
        "mapping_limitations": mapping_limitations,
        "all_limitations": list(plan.limitations),
        "compiled_deployment_actions": len(plan.actions),
        "compiled_ability_events": len(plan.ability_events),
        "compatible_initial_state_counts": [
            side.cycle.compatible_initial_state_count for side in plan.sides
        ],
    }


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str, int]:
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical(row))
            count += 1
    return count, _sha256(path), path.stat().st_size


def _estimate(count: int, ticks: int, pilot: PilotEvidence) -> dict[str, Any]:
    low, high = _wilson(pilot.successes, pilot.trials)
    point = pilot.successes / pilot.trials
    return {
        "candidate_battles": count,
        "candidate_source_ticks": ticks,
        "pilot_successes": pilot.successes,
        "pilot_trials": pilot.trials,
        "pilot_success_rate": point,
        "pilot_wilson_95": [low, high],
        "estimated_successful_battles": {
            "point": round(count * point),
            "wilson_95_floor_ceil": [math.floor(count * low), math.ceil(count * high)],
        },
        "estimated_successful_ticks_duration_neutral_assumption": {
            "point": round(ticks * point),
            "wilson_95_floor_ceil": [math.floor(ticks * low), math.ceil(ticks * high)],
        },
        "measured_four_worker_ticks_per_wall_second": pilot.ticks_per_wall_second,
        "measured_bytes_per_stored_tick": pilot.bytes_per_tick,
        "pilot_source": pilot.source,
    }


def run_audit(
    manifest: Path = DEFAULT_MANIFEST,
    output: Path = DEFAULT_OUTPUT,
    *,
    workers: int = 8,
    shard_rows: int = 5_000,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    output = output.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if output.exists():
        # The output is fully derived and versioned.  Refuse broad or source-
        # adjacent targets even when a caller supplies custom CLI arguments.
        if output == Path(output.anchor) or len(output.parts) < 4:
            raise ValueError(f"refusing unsafe audit output replacement: {output}")
        if output == manifest or output in manifest.parents or manifest in output.parents:
            raise ValueError("audit output must not overlap the source manifest")
        existing_manifest = output / "manifest.json"
        if existing_manifest.exists():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if existing.get("kind") != "expert_100k_native_eligibility_manifest_v1":
                raise ValueError(f"output is not an eligibility audit tree: {output}")
        elif output.name != "native-eligibility-v1":
            raise ValueError(f"refusing to replace unmarked output tree: {output}")
        shutil.rmtree(output)
    shards_dir = output / "shards"
    queues_dir = output / "queues"
    shards_dir.mkdir(parents=True)
    queues_dir.mkdir(parents=True)

    counters: dict[str, Counter[Any]] = defaultdict(Counter)
    tick_sums: Counter[str] = Counter()
    event_sums: Counter[str] = Counter()
    queue_handles: dict[str, Any] = {}
    queue_counts: Counter[str] = Counter()
    queue_hashers: dict[str, Any] = {}
    queue_sizes: Counter[str] = Counter()
    shard_entries: list[dict[str, Any]] = []
    current_shard = None
    current_hasher = hashlib.sha256()
    current_count = current_size = 0
    shard_index = -1

    queue_names = (
        "authoritative-native-full",
        "authoritative-deployment-only",
        "authoritative-ability-exact",
        "compiler-native-ready",
        "old-schema-approximate",
        "sequence-only",
        "compile-rejected",
    )
    for name in queue_names:
        queue_handles[name] = (queues_dir / f"{name}.jsonl").open("wb")
        queue_hashers[name] = hashlib.sha256()

    def open_shard(index: int) -> Any:
        nonlocal current_hasher, current_count, current_size, shard_index
        shard_index = index
        current_hasher = hashlib.sha256()
        current_count = current_size = 0
        return (shards_dir / f"part-{index:05d}.jsonl").open("wb")

    def close_shard(handle: Any) -> None:
        if handle is None:
            return
        handle.close()
        path = shards_dir / f"part-{shard_index:05d}.jsonl"
        shard_entries.append({
            "path": str(path.relative_to(output)),
            "rows": current_count,
            "bytes": current_size,
            "sha256": current_hasher.hexdigest(),
        })

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for index, audited in enumerate(executor.map(audit_one, rows, chunksize=32)):
                if index % shard_rows == 0:
                    close_shard(current_shard)
                    current_shard = open_shard(index // shard_rows)
                encoded = _canonical(audited)
                current_shard.write(encoded)
                current_hasher.update(encoded)
                current_count += 1
                current_size += len(encoded)

                schema = audited["source_schema_version"]
                ability = audited["ability_log_tier"]
                tier = audited["eligibility_tier"]
                coord = audited["coordinate_tier"]
                counters["schema"][schema] += 1
                counters["ability_log_tier"][ability] += 1
                counters["eligibility_tier"][tier] += 1
                counters["coordinate_tier"][coord] += 1
                counters["compile_ok"][str(bool(audited["compile_ok"])).lower()] += 1
                counters["compiler_native_replay_ready"][str(bool(audited["compiler_native_replay_ready"])).lower()] += 1
                counters["authoritative_native_full_candidate"][str(bool(audited["authoritative_native_full_candidate"])).lower()] += 1
                counters["ability_positive"][str(bool(audited["ability_positive"])).lower()] += 1
                for metadata_key in (
                    "eight_card_decks_complete", "card_levels_complete",
                    "card_forms_complete_by_schema_contract", "tower_troops_complete",
                    "tower_troop_levels_complete", "final_tower_hp_complete",
                    "schema5_contract_stamp_complete",
                ):
                    counters[metadata_key][str(bool(audited[metadata_key])).lower()] += 1
                mapping_value = audited.get("mapping_complete")
                counters["mapping_complete"][
                    "not_compiled" if mapping_value is None else str(bool(mapping_value)).lower()
                ] += 1
                counters["schema_x_ability_log_tier"][f"schema{schema}:{ability}"] += 1
                counters["schema_x_eligibility_tier"][f"schema{schema}:{tier}"] += 1
                if audited.get("compile_rejection_class"):
                    counters["compile_rejection_class"][audited["compile_rejection_class"]] += 1
                ticks = int(audited.get("duration_ticks") or 0)
                tick_sums["all"] += ticks
                tick_sums[f"schema_{schema}"] += ticks
                tick_sums[f"tier:{tier}"] += ticks
                event_sums["deployment_actions"] += int(audited["deployment_actions"])
                event_sums["ability_count_reported"] += int(audited["ability_count_reported"])
                event_sums["ability_events_observed"] += int(audited["ability_events_observed"])
                event_sums["raw_data_i_events"] += int(audited["raw_data_i_events"])
                event_sums["data_i_zero_events"] += int(audited["data_i_zero_events"])
                event_sums["data_i_one_events"] += int(audited["data_i_one_events"])

                selected_queues: list[str] = []
                if audited["authoritative_native_full_candidate"]:
                    selected_queues.append("authoritative-native-full")
                    selected_queues.append(
                        "authoritative-ability-exact"
                        if audited["ability_positive"]
                        else "authoritative-deployment-only"
                    )
                if audited["compiler_native_replay_ready"]:
                    selected_queues.append("compiler-native-ready")
                if str(tier).startswith("schema2_"):
                    selected_queues.append("old-schema-approximate")
                if tier == "schema1_sequence_only_missing_native_metadata":
                    selected_queues.append("sequence-only")
                if not audited["compile_ok"]:
                    selected_queues.append("compile-rejected")
                queue_row = {
                    key: audited[key] for key in (
                        "battle_tag", "source_path", "source_sha256",
                        "source_schema_version", "duration_ticks", "deployment_actions",
                        "ability_count_reported", "ability_events_observed", "ability_log_tier",
                        "coordinate_tier", "eligibility_tier",
                        "compiler_native_replay_ready", "authoritative_native_full_candidate",
                    )
                }
                if schema == 5:
                    queue_row.update({
                        "schema5_authoritative_contract_verified": bool(
                            audited.get("schema5_authoritative_contract_verified")
                        ),
                        "tower_troop_levels_complete": bool(
                            audited.get("tower_troop_levels_complete")
                        ),
                        "final_tower_hp_complete": bool(
                            audited.get("final_tower_hp_complete")
                        ),
                        "numeric_game_mode_id": audited.get("numeric_game_mode_id"),
                        "battle_index": audited.get("battle_index"),
                    })
                queue_encoded = _canonical(queue_row)
                for name in selected_queues:
                    queue_handles[name].write(queue_encoded)
                    queue_hashers[name].update(queue_encoded)
                    queue_counts[name] += 1
                    queue_sizes[name] += len(queue_encoded)
    finally:
        close_shard(current_shard)
        for handle in queue_handles.values():
            handle.close()

    queue_entries = [
        {
            "path": f"queues/{name}.jsonl",
            "rows": queue_counts[name],
            "bytes": queue_sizes[name],
            "sha256": queue_hashers[name].hexdigest(),
        }
        for name in queue_names
    ]
    deployment_tier = "authoritative_native_deployment_only"
    ability_tier_name = "authoritative_native_ability_exact"
    deployment_count = counters["eligibility_tier"][deployment_tier]
    ability_count = counters["eligibility_tier"][ability_tier_name]
    deployment_ticks = tick_sums[f"tier:{deployment_tier}"]
    ability_ticks = tick_sums[f"tier:{ability_tier_name}"]
    deployment_estimate = _estimate(deployment_count, deployment_ticks, DEPLOYMENT_PILOT)
    ability_estimate = _estimate(ability_count, ability_ticks, ABILITY_PILOT)
    dep_tick_interval = deployment_estimate["estimated_successful_ticks_duration_neutral_assumption"]["wilson_95_floor_ceil"]
    ability_tick_interval = ability_estimate["estimated_successful_ticks_duration_neutral_assumption"]["wilson_95_floor_ceil"]
    expected_bytes_point = (
        deployment_estimate["estimated_successful_ticks_duration_neutral_assumption"]["point"] * DEPLOYMENT_PILOT.bytes_per_tick
        + ability_estimate["estimated_successful_ticks_duration_neutral_assumption"]["point"] * ABILITY_PILOT.bytes_per_tick
    )
    expected_bytes_interval = [
        dep_tick_interval[0] * DEPLOYMENT_PILOT.bytes_per_tick
        + ability_tick_interval[0] * ABILITY_PILOT.bytes_per_tick,
        dep_tick_interval[1] * DEPLOYMENT_PILOT.bytes_per_tick
        + ability_tick_interval[1] * ABILITY_PILOT.bytes_per_tick,
    ]
    full_candidate_wall = (
        deployment_ticks / DEPLOYMENT_PILOT.ticks_per_wall_second
        + ability_ticks / ABILITY_PILOT.ticks_per_wall_second
    )
    successful_only_wall_interval = [
        dep_tick_interval[0] / DEPLOYMENT_PILOT.ticks_per_wall_second
        + ability_tick_interval[0] / ABILITY_PILOT.ticks_per_wall_second,
        dep_tick_interval[1] / DEPLOYMENT_PILOT.ticks_per_wall_second
        + ability_tick_interval[1] / ABILITY_PILOT.ticks_per_wall_second,
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "rows": len(rows),
        },
        "semantics": {
            "does_not_call_libg": True,
            "does_not_copy_source_battles": True,
            "authoritative_native_full_candidate": (
                "schema3 legacy-exact or contract-verified schema5 + every deployment has raw x/y/data_i "
                "+ complete 8-card/level/form/tower metadata + current card/form/tower/ability mapping "
                "+ zero ability or exact ability ticks; schema5 additionally requires numeric mode, battle index, "
                "tower troop levels and final six-tower HP"
            ),
            "compiler_native_replay_ready_warning": (
                "includes schema2 zero-ability rows whose legacy x/y lacks raw data_i and is therefore approximate"
            ),
        },
        "counts": {name: dict(sorted(counter.items(), key=lambda item: str(item[0])))
                   for name, counter in counters.items()},
        "tick_sums": dict(sorted(tick_sums.items())),
        "event_sums": dict(sorted(event_sums.items())),
        "pilot_based_success_estimate_not_100k_measurement": {
            "confidence_method": "two-sided Wilson score interval, 95%",
            "deployment_only": deployment_estimate,
            "ability_positive_exact": ability_estimate,
            "combined_estimated_successful_battles": {
                "point": (
                    deployment_estimate["estimated_successful_battles"]["point"]
                    + ability_estimate["estimated_successful_battles"]["point"]
                ),
                "conservative_sum_of_stratum_wilson_bounds": [
                    deployment_estimate["estimated_successful_battles"]["wilson_95_floor_ceil"][0]
                    + ability_estimate["estimated_successful_battles"]["wilson_95_floor_ceil"][0],
                    deployment_estimate["estimated_successful_battles"]["wilson_95_floor_ceil"][1]
                    + ability_estimate["estimated_successful_battles"]["wilson_95_floor_ceil"][1],
                ],
            },
            "combined_estimated_successful_ticks_duration_neutral_assumption": {
                "point": (
                    deployment_estimate["estimated_successful_ticks_duration_neutral_assumption"]["point"]
                    + ability_estimate["estimated_successful_ticks_duration_neutral_assumption"]["point"]
                ),
                "conservative_sum_of_stratum_wilson_bounds": [
                    dep_tick_interval[0] + ability_tick_interval[0],
                    dep_tick_interval[1] + ability_tick_interval[1],
                ],
            },
            "estimated_tick_store_bytes": {
                "point": round(expected_bytes_point),
                "conservative_sum_of_stratum_wilson_bounds": [
                    math.floor(expected_bytes_interval[0]), math.ceil(expected_bytes_interval[1])
                ],
            },
            "four_worker_wall_seconds": {
                "successful_episodes_only_lower_bound_range": [
                    successful_only_wall_interval[0], successful_only_wall_interval[1]
                ],
                "all_static_candidates_run_to_source_duration_upper_bound": full_candidate_wall,
                "warning": "real failed episodes consume a non-zero prefix, so actual wall time lies above successful-only work",
            },
        },
    }
    inventory = {
        "schema_version": 1,
        "kind": "expert_100k_native_eligibility_manifest_v1",
        "source_manifest": summary["source_manifest"],
        "rows": len(rows),
        "shards": shard_entries,
        "queues": queue_entries,
    }
    (output / "summary.json").write_bytes(_canonical(summary))
    (output / "manifest.json").write_bytes(_canonical(inventory))
    return summary
