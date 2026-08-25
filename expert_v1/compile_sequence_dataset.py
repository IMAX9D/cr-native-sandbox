"""Compile public replay histories into mmap sequence-only BC shards.

This compiler is intentionally not a native-state compiler.  It consumes the
cycle proof produced by :mod:`expert_v1.upgrade_base_cycles`, starts each side
at its first exact post-burn-in hand, and stores only information visible from
the action log.  It never creates a placeholder libg grid or a guessed native
legality mask.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .training_v1.schema import (
    ARENA_COLUMNS,
    ARENA_ROWS,
    DATASET_KIND,
    OBSERVATION_SEQUENCE,
    POSITION_COUNT,
    SCHEMA_VERSION,
    sha256_file,
    validate_manifest,
    validate_shard,
)
from .upgrade_base_cycles import process_battle

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


PUBLIC_SCALARS = [
    "elapsed_fraction_of_5_minutes",
    "regular_time_fraction",
    "overtime_fraction",
    "interval_seconds_clipped_10s",
    "own_deploy_count_fraction_32",
    "enemy_deploy_count_fraction_32",
    "revealed_enemy_fraction",
]

SEQUENCE_DTYPES: dict[str, np.dtype[Any]] = {
    "public_scalars": np.dtype(np.float32),
    "own_deck_tokens": np.dtype(np.int32),
    "hand_tokens": np.dtype(np.int32),
    "next_card_token": np.dtype(np.int32),
    "revealed_enemy_tokens": np.dtype(np.int32),
    "previous_event_card_token": np.dtype(np.int32),
    "previous_event_side": np.dtype(np.uint8),
    "previous_event_position": np.dtype(np.uint16),
    "delta_ticks": np.dtype(np.int32),
    "timing_exposure_ticks": np.dtype(np.int32),
    "card_mask": np.dtype(np.bool_),
    "play_now": np.dtype(np.bool_),
    "card_slot": np.dtype(np.int16),
    "position": np.dtype(np.int16),
    "timing_label_mask": np.dtype(np.bool_),
    "card_label_mask": np.dtype(np.bool_),
    "position_label_mask": np.dtype(np.bool_),
    "sample_weight": np.dtype(np.float32),
}


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON line is not an object")
    return value


def dumps(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _player_tags(record: Mapping[str, Any]) -> tuple[str, str] | None:
    team = [str(tag) for tag in (record.get("team_tags") or []) if str(tag)]
    opponent = [str(tag) for tag in (record.get("opponent_tags") or []) if str(tag)]
    if len(team) != 1 or len(opponent) != 1:
        return None
    return team[0], opponent[0]


def _bucket(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    players = sorted(
        {
            player
            for record in records
            for player in (_player_tags(record) or ())
        }
    )
    test_players = {
        player for player in players if _bucket(seed, "test", player) < test_fraction
    }
    validation_players = {
        player
        for player in players
        if player not in test_players
        and _bucket(seed, "validation", player) < validation_fraction
    }
    assignments: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    split_players: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for record in records:
        tag = str(record["battle_tag"])
        pair = _player_tags(record)
        if pair is None:
            counts["missing_player_metadata"] += 1
            # Missing identities cannot enter a claimed player-aware corpus.
            continue
        if any(player in test_players for player in pair):
            split = "test"
        elif any(player in validation_players for player in pair):
            split = "validation"
        else:
            split = "train"
        assignments[tag] = split
        source_path = str(record.get("source_path") or "")
        if not source_path:
            raise RuntimeError(f"accepted battle lacks source path: {tag}")
        previous_split = source_splits.setdefault(source_path, split)
        if previous_split != split:
            raise RuntimeError(
                f"source file crosses splits: {source_path} ({previous_split}/{split})"
            )
        split_players[split].update(pair)
        counts[f"{split}_battles"] += 1
    leaks = test_players & split_players["train"]
    validation_leaks = validation_players & split_players["train"]
    if leaks or validation_leaks:
        raise RuntimeError("player holdout leaked into training split")
    if any(counts[f"{split}_battles"] <= 0 for split in ("train", "validation", "test")):
        raise RuntimeError(
            "player-aware split produced an empty split; increase the sample or holdout fractions"
        )
    audit = {
        **counts,
        "unique_players": len(players),
        "test_holdout_players": len(test_players),
        "validation_holdout_players": len(validation_players),
        "test_holdout_sha256": hashlib.sha256(
            "\n".join(sorted(test_players)).encode()
        ).hexdigest(),
        "validation_holdout_sha256": hashlib.sha256(
            "\n".join(sorted(validation_players)).encode()
        ).hexdigest(),
        "player_holdout_leaks": 0,
        "source_file_collisions": 0,
    }
    return assignments, audit


def read_accepted(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = loads(raw)
            tag = str(record.get("battle_tag") or "")
            if not tag or tag in seen:
                raise ValueError(f"missing/duplicate accepted battle tag: {tag!r}")
            seen.add(tag)
            records.append(record)
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError("accepted manifest is empty")
    return records


def discover_valid_sides(root: Path) -> list[Path]:
    paths = sorted(root.glob("schema*-base-cycle-*/valid-sides.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no valid-sides.jsonl below {root}")
    return paths


def open_side_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE sides (
            battle_tag TEXT NOT NULL,
            side TEXT NOT NULL,
            source_schema_version INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (battle_tag, side)
        ) WITHOUT ROWID
        """
    )
    return connection


def upsert_side(connection: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO sides(battle_tag, side, source_schema_version, payload)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(battle_tag, side) DO UPDATE SET
            source_schema_version=excluded.source_schema_version,
            payload=excluded.payload
        WHERE excluded.source_schema_version >= sides.source_schema_version
        """,
        (
            str(record["battle_tag"]),
            str(record["side"]),
            int(record.get("source_schema_version") or 1),
            dumps(record),
        ),
    )


def stage_sides(
    connection: sqlite3.Connection,
    accepted: list[dict[str, Any]],
    paths: list[Path],
    *,
    derive_missing: bool,
    workers: int,
) -> tuple[list[str], dict[str, int]]:
    accepted_tags = {str(record["battle_tag"]) for record in accepted}
    vocab: set[str] = set()
    counters: Counter[str] = Counter()
    for path in paths:
        with path.open("rb") as handle:
            for raw in handle:
                record = loads(raw)
                if str(record.get("battle_tag") or "") not in accepted_tags:
                    continue
                upsert_side(connection, record)
                vocab.update(str(card) for card in (record.get("base_deck") or []))
                counters["staged_side_records"] += 1
    connection.commit()
    available = {
        str(row[0])
        for row in connection.execute(
            "SELECT battle_tag FROM sides GROUP BY battle_tag HAVING COUNT(*)=2"
        )
    }
    missing = [record for record in accepted if str(record["battle_tag"]) not in available]
    counters["missing_battles_before_derivation"] = len(missing)
    if missing and derive_missing:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for sides in executor.map(process_battle, missing):
                for side in sides:
                    if not bool(side.get("cycle_valid")):
                        continue
                    upsert_side(connection, side)
                    vocab.update(str(card) for card in (side.get("base_deck") or []))
                    counters["derived_side_records"] += 1
        connection.commit()
    available_after = {
        str(row[0])
        for row in connection.execute(
            "SELECT battle_tag FROM sides GROUP BY battle_tag HAVING COUNT(*)=2"
        )
    }
    counters["missing_battles_after_derivation"] = sum(
        str(record["battle_tag"]) not in available_after for record in accepted
    )
    return sorted(vocab), dict(counters)


def fetch_sides(
    connection: sqlite3.Connection, battle_tag: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    values = {
        str(side): loads(bytes(payload))
        for side, payload in connection.execute(
            "SELECT side, payload FROM sides WHERE battle_tag=?", (battle_tag,)
        )
    }
    if set(values) != {"team", "opponent"}:
        return None
    return values["team"], values["opponent"]


def _events(record: Mapping[str, Any], *, enemy_for_actor: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, tick, card_index, x, y, event_index in zip(
        range(int(record["action_count"])),
        record["ticks"],
        record["card_indices"],
        record["actor_x"],
        record["actor_y"],
        record["event_indices"],
    ):
        x = int(x)
        y = int(y)
        if enemy_for_actor:
            x = 17_999 - x
            y = 31_999 - y
        output.append(
            {
                "action_index": index,
                "tick": int(tick),
                "card_index": int(card_index),
                "card": str(record["base_deck"][int(card_index)]),
                "position": min(ARENA_ROWS - 1, max(0, y // 1000)) * ARENA_COLUMNS
                + min(ARENA_COLUMNS - 1, max(0, x // 1000)),
                "event_index": int(event_index),
                "enemy": enemy_for_actor,
                "deploy": True,
            }
        )
    return output


def _ability_events(
    record: Mapping[str, Any], *, enemy_for_actor: bool
) -> list[dict[str, Any]]:
    return [
        {
            "action_index": -1,
            "tick": int(tick),
            "card_index": -1,
            "card": None,
            "position": POSITION_COUNT,
            "event_index": int(event_index),
            "enemy": enemy_for_actor,
            "deploy": False,
        }
        for tick, event_index in zip(
            record.get("ability_ticks") or [],
            record.get("ability_event_indices") or [],
        )
    ]


def compile_side_sequence(
    own: Mapping[str, Any],
    enemy: Mapping[str, Any],
    vocabulary: Mapping[str, int],
) -> tuple[dict[str, np.ndarray[Any, Any]] | None, str | None]:
    first_exact = own.get("first_exact_action_index")
    if first_exact is None or int(first_exact) <= 0:
        return None, "no_exact_hand_suffix"
    first_exact = int(first_exact)
    own_events = _events(own, enemy_for_actor=False)
    enemy_events = _events(enemy, enemy_for_actor=True)
    public_ability_events = _ability_events(
        own, enemy_for_actor=False
    ) + _ability_events(enemy, enemy_for_actor=True)
    if first_exact >= len(own_events):
        return None, "no_exact_action"
    anchor_tick = int(own_events[first_exact - 1]["tick"])
    last_tick = int(own_events[-1]["tick"])
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in own_events + enemy_events + public_ability_events:
        groups[int(event["tick"])].append(event)
    for values in groups.values():
        values.sort(key=lambda value: int(value["event_index"]))
    exact_groups = [
        (tick, groups[tick])
        for tick in sorted(groups)
        if anchor_tick < tick <= last_tick
    ]
    if not exact_groups:
        return None, "no_public_intervals"
    if any(
        sum((not event["enemy"]) and event["deploy"] for event in values) > 1
        for _, values in exact_groups
    ):
        return None, "multiple_own_deploys_same_tick"

    # Public state at the exact-cycle boundary.  Same-tick actions are joint;
    # they become visible only to later tick groups, never to each other.
    prior_events = [
        event
        for tick, values in groups.items()
        if tick <= anchor_tick
        for event in values
    ]
    prior_events.sort(key=lambda value: (int(value["tick"]), int(value["event_index"])))
    previous = prior_events[-1] if prior_events else None
    revealed: list[str] = []
    for event in prior_events:
        if event["enemy"] and event["deploy"] and event["card"] not in revealed:
            revealed.append(str(event["card"]))
    own_count = sum((not event["enemy"]) and event["deploy"] for event in prior_events)
    enemy_count = sum(bool(event["enemy"]) and event["deploy"] for event in prior_events)
    previous_tick = anchor_tick
    timings_complete = bool(own.get("ability_events_complete")) and bool(
        enemy.get("ability_events_complete")
    )

    rows: dict[str, list[Any]] = {name: [] for name in SEQUENCE_DTYPES}
    deck_tokens = [vocabulary[str(card)] for card in own["base_deck"]]
    canonical_deck_tokens = sorted(deck_tokens)
    for tick, events in exact_groups:
        next_own = next(
            (
                event
                for event in own_events[first_exact:]
                if int(event["tick"]) >= tick
            ),
            None,
        )
        if next_own is None:
            break
        action_index = int(next_own["action_index"])
        hand_mask = own["hand_masks_before"][action_index]
        next_index = own["next_card_indices_before"][action_index]
        if hand_mask is None or next_index is None:
            return None, "lost_exact_cycle_suffix"
        hand_indices = [index for index in range(8) if int(hand_mask) & (1 << index)]
        if len(hand_indices) != 4:
            return None, "bad_exact_hand_width"
        # Cycle inference proves a set, not the replay client's UI slot order.
        # Canonical token ordering prevents accidental dependence on the
        # first-seen/card_counts insertion order and is reproducible online.
        hand_tokens = sorted(deck_tokens[index] for index in hand_indices)
        next_token = deck_tokens[int(next_index)]
        if next_token in hand_tokens:
            return None, "next_card_in_hand"

        own_at_tick = [
            event for event in events if (not event["enemy"]) and event["deploy"]
        ]
        play_now = len(own_at_tick) == 1
        expert = own_at_tick[0] if play_now else None
        card_slot = -100
        position = -100
        if expert is not None:
            played_token = vocabulary[str(expert["card"])]
            if played_token not in hand_tokens:
                return None, "expert_card_outside_exact_hand"
            card_slot = hand_tokens.index(played_token)
            position = int(expert["position"])

        interval = tick - previous_tick
        if interval <= 0:
            return None, "nonpositive_event_interval"
        revealed_tokens = [vocabulary[card] for card in revealed[:8]]
        revealed_tokens.extend([0] * (8 - len(revealed_tokens)))
        previous_card = (
            vocabulary[str(previous["card"])]
            if previous is not None and previous["deploy"]
            else 0
        )
        previous_side = 0 if previous is None else (2 if previous["enemy"] else 1)
        previous_position = (
            POSITION_COUNT
            if previous is None or not previous["deploy"]
            else int(previous["position"])
        )
        rows["public_scalars"].append(
            [
                min(tick / 6000.0, 1.5),
                min(tick / 3600.0, 1.0),
                min(max(tick - 3600, 0) / 2400.0, 1.0),
                min(interval / 200.0, 1.0),
                min(own_count / 32.0, 1.5),
                min(enemy_count / 32.0, 1.5),
                len(revealed) / 8.0,
            ]
        )
        rows["own_deck_tokens"].append(canonical_deck_tokens)
        rows["hand_tokens"].append(hand_tokens)
        rows["next_card_token"].append(next_token)
        rows["revealed_enemy_tokens"].append(revealed_tokens)
        rows["previous_event_card_token"].append(previous_card)
        rows["previous_event_side"].append(previous_side)
        rows["previous_event_position"].append(previous_position)
        rows["delta_ticks"].append(interval)
        rows["timing_exposure_ticks"].append(interval)
        rows["card_mask"].append([True, True, True, True])
        rows["play_now"].append(play_now)
        rows["card_slot"].append(card_slot)
        rows["position"].append(position)
        rows["timing_label_mask"].append(timings_complete)
        rows["card_label_mask"].append(play_now)
        rows["position_label_mask"].append(play_now)
        rows["sample_weight"].append(1.0)

        for event in events:
            if event["enemy"] and event["deploy"]:
                enemy_count += 1
                if event["card"] not in revealed:
                    revealed.append(str(event["card"]))
            elif (not event["enemy"]) and event["deploy"]:
                own_count += 1
        previous = events[-1]
        previous_tick = tick

    if not rows["play_now"]:
        return None, "empty_sequence"
    return {
        name: np.asarray(values, dtype=SEQUENCE_DTYPES[name])
        for name, values in rows.items()
    }, None


class ShardWriter:
    def __init__(self, root: Path, split: str, sequences_per_shard: int) -> None:
        self.root = root
        self.split = split
        self.sequences_per_shard = sequences_per_shard
        self.pending: list[dict[str, np.ndarray[Any, Any]]] = []
        self.paths: list[str] = []
        self.file_hashes: dict[str, str] = {}
        self.sequences = 0
        self.rows = 0

    def add(self, sequence: dict[str, np.ndarray[Any, Any]]) -> None:
        self.pending.append(sequence)
        if len(self.pending) >= self.sequences_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        index = len(self.paths)
        relative = f"shards/{self.split}-{index:05d}"
        path = self.root / relative
        path.mkdir(parents=True)
        lengths = [len(sequence["play_now"]) for sequence in self.pending]
        offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        np.save(path / "sequence_offsets.npy", offsets, allow_pickle=False)
        for name in SEQUENCE_DTYPES:
            value = np.concatenate([sequence[name] for sequence in self.pending], axis=0)
            np.save(path / f"{name}.npy", value, allow_pickle=False)
        self.paths.append(relative)
        self.sequences += len(self.pending)
        self.rows += int(offsets[-1])
        for file in sorted(path.glob("*.npy")):
            key = f"{relative}/{file.name}"
            self.file_hashes[key] = sha256_file(file)
        self.pending.clear()


def _replace_directory(building: Path, output: Path, replace: bool) -> None:
    if output.exists() and not replace:
        raise FileExistsError(output)
    previous = output.with_name(output.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if output.exists():
        os.replace(output, previous)
    try:
        os.replace(building, output)
    except Exception:
        if previous.exists() and not output.exists():
            os.replace(previous, output)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.accepted_manifest.resolve(strict=True)
    source_digest_before = sha256_file(source)
    output = args.output_root.resolve()
    building = output.with_name(output.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    accepted = read_accepted(source, args.limit_battles)
    valid_paths = [path.resolve(strict=True) for path in args.valid_sides]
    if not valid_paths:
        valid_paths = discover_valid_sides(args.local_upgrades_root.resolve(strict=True))
    input_stats_before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [source, *valid_paths]
    }

    started = datetime.now(timezone.utc)
    database = open_side_database(building / "staging.sqlite3")
    try:
        cards, staging = stage_sides(
            database,
            accepted,
            valid_paths,
            derive_missing=not args.no_derive_missing,
            workers=args.workers,
        )
        vocabulary = {card: index + 1 for index, card in enumerate(cards)}
        assignments, split_audit = assign_splits(
            accepted,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
        writers = {
            split: ShardWriter(building, split, args.sequences_per_shard)
            for split in ("train", "validation", "test")
        }
        counters: Counter[str] = Counter()
        rejection_reasons: Counter[str] = Counter()
        assignment_path = building / "split-assignments.jsonl"
        with assignment_path.open("wb") as assignment_handle:
            for index, record in enumerate(accepted, 1):
                tag = str(record["battle_tag"])
                split = assignments.get(tag)
                if split is None:
                    counters["battles_missing_split"] += 1
                    continue
                pair = fetch_sides(database, tag)
                if pair is None:
                    counters["battles_missing_cycle_sides"] += 1
                    continue
                team, opponent = pair
                compiled_sides = 0
                for own, enemy in ((team, opponent), (opponent, team)):
                    sequence, reason = compile_side_sequence(own, enemy, vocabulary)
                    if sequence is None:
                        rejection_reasons[str(reason)] += 1
                        continue
                    writers[split].add(sequence)
                    compiled_sides += 1
                    counters["rows"] += len(sequence["play_now"])
                    counters["timing_rows"] += int(sequence["timing_label_mask"].sum())
                    counters["play_rows"] += int(sequence["play_now"].sum())
                    counters["sequence_only_sides"] += 1
                if compiled_sides:
                    counters["compiled_battles"] += 1
                    assignment_handle.write(
                        dumps(
                            {
                                "battle_tag": tag,
                                "split": split,
                                "compiled_sides": compiled_sides,
                            }
                        )
                        + b"\n"
                    )
                if args.progress_every and index % args.progress_every == 0:
                    print(
                        f"compiled {index}/{len(accepted)} battles; rows={counters['rows']}",
                        flush=True,
                    )
        for writer in writers.values():
            writer.flush()
        if any(not writer.paths for writer in writers.values()):
            raise RuntimeError("every split must produce at least one shard")
    finally:
        database.close()
        (building / "staging.sqlite3").unlink(missing_ok=True)

    all_file_hashes = {
        key: digest
        for writer in writers.values()
        for key, digest in writer.file_hashes.items()
    }
    source_hashes = {
        str(path): sha256_file(path) for path in [source, *valid_paths]
    }
    input_stats_after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [source, *valid_paths]
    }
    if input_stats_after != input_stats_before or source_hashes[str(source)] != source_digest_before:
        raise RuntimeError("source manifest/cycle input changed during compilation")
    content_digest = hashlib.sha256(
        json.dumps(
            {"source": source_hashes, "shards": all_file_hashes},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest: dict[str, Any] = {
        "kind": DATASET_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_utc": started.isoformat(),
        "production_ready": True,
        "native_replay_validated": False,
        "observation_mode": OBSERVATION_SEQUENCE,
        "timing_target": "piecewise_exponential_event_v1",
        "actor_information": "public_only_v1",
        "source_manifest": {"path": str(source), "sha256": sha256_file(source)},
        "source_inputs": source_hashes,
        "dataset_content_sha256": content_digest,
        "dimensions": {
            "grid_channels": 0,
            "public_scalar_size": len(PUBLIC_SCALARS),
            "card_vocab_size": len(vocabulary) + 1,
            "ability_vocab_size": 1,
            "max_ability_slots": 1,
        },
        "feature_schema": {"grid_channels": [], "public_scalars": PUBLIC_SCALARS},
        "card_vocabulary": ["<PAD>", *cards],
        "splits": {split: writer.paths for split, writer in writers.items()},
        "split_statistics": {
            split: {"sequences": writer.sequences, "rows": writer.rows}
            for split, writer in writers.items()
        },
        "split_contract": {
            "battle_tag_disjoint": True,
            "source_file_disjoint": True,
            "player_holdout_test": True,
            "assignment": "battle_if_either_player_is_deterministic_holdout_v1",
            **split_audit,
        },
        "state_provenance": {
            "mode": "sequence_only",
            "sequence_only_rows": int(counters["rows"]),
            "native_grid_rows": 0,
            "authoritative_rows": 0,
            "native_generated_unanchored_rows": 0,
            "notes": "public action history and exact base-card cycle only",
        },
        "mask_provenance": {
            "card_mask": "exact_hand_membership_not_elixir_legality",
            "position_mask": None,
            "native_legality_claimed": False,
        },
        "quality_gates": {
            "split_collisions": 0,
            "forbidden_actor_features": 0,
            "nonfinite_features": 0,
            "expert_label_mask_violations": 0,
            "fabricated_native_grid_rows": 0,
            "player_holdout_leaks": 0,
        },
        "coverage": {
            "accepted_battles": len(accepted),
            **staging,
            **counters,
            "rejected_sides": sum(rejection_reasons.values()),
            "rejection_reasons": dict(rejection_reasons),
        },
        "compiler": {
            "kind": "expert_sequence_only_compiler_v1",
            "seed": args.seed,
            "sequences_per_shard": args.sequences_per_shard,
            "valid_side_inputs": [str(path) for path in valid_paths],
            "future_schema_support": "derive missing schema-N cycles from source_path",
        },
        "shard_file_sha256": all_file_hashes,
    }
    (building / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_manifest(manifest, root=building)
    for split, paths in manifest["splits"].items():
        for relative in paths:
            validate_shard(building / relative, manifest)
    manifest_hash = sha256_file(building / "manifest.json")
    (building / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n", encoding="ascii"
    )
    result = {
        "output_root": str(output),
        "manifest_sha256": manifest_hash,
        "dataset_content_sha256": content_digest,
        "accepted_battles": len(accepted),
        "compiled_battles": int(counters["compiled_battles"]),
        "sequences": int(counters["sequence_only_sides"]),
        "rows": int(counters["rows"]),
        "timing_rows": int(counters["timing_rows"]),
        "play_rows": int(counters["play_rows"]),
        "rejected_sides": sum(rejection_reasons.values()),
        "split_statistics": manifest["split_statistics"],
    }
    (building / "compile-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _replace_directory(building, output, args.replace)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-manifest",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
            r"\version-window-20260804\accepted-cycle-clean.jsonl"
        ),
    )
    parser.add_argument(
        "--local-upgrades-root",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\training-dataset\local-upgrades"
        ),
    )
    parser.add_argument("--valid-sides", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\compiled\sequence-only-bc-v1"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sequences-per-shard", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--limit-battles", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--no-derive-missing", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("workers must be in 1..64")
    if args.sequences_per_shard <= 0:
        raise ValueError("sequences-per-shard must be positive")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("holdout fractions must be in (0, 0.5)")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
