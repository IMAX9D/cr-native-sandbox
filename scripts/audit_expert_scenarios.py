"""Read-only, mmap-based coverage audit for expert native-state scenarios."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SCENARIOS = (
    "all_valid",
    "enemy_in_own_half",
    "defense_pressure",
    "critical_tower_pressure",
    "contested_own_half",
    "counterpush",
    "calm_board",
    "ability_available",
)
LATENCY_EDGES = (0, 10, 20, 40, 80, 160, 320, 640)


def _row_counts(offsets: np.ndarray, selected_entities: np.ndarray) -> np.ndarray:
    prefix = np.zeros(len(selected_entities) + 1, dtype=np.int32)
    np.cumsum(selected_entities, dtype=np.int32, out=prefix[1:])
    return prefix[offsets[1:]] - prefix[offsets[:-1]]


def _latency_bucket(ticks: int) -> str:
    for edge in LATENCY_EDGES[1:]:
        if ticks < edge:
            return f"lt_{edge}_ticks"
    return "ge_640_ticks"


def _card_classes(card_vocabulary: list[str]) -> np.ndarray:
    values = np.zeros(len(card_vocabulary), dtype=np.uint8)
    parsed: list[tuple[str, int]] = []
    base_types: dict[str, int] = {}
    for raw in card_vocabulary:
        name, _, suffix = str(raw).partition("@")
        try:
            card_id = int(suffix)
        except ValueError:
            card_id = 0
        parsed.append((name, card_id))
        kind = card_id // 1_000_000
        if kind in (26, 27, 28):
            base_types[name] = {26: 1, 27: 2, 28: 3}[kind]
    for token, (name, card_id) in enumerate(parsed):
        kind = card_id // 1_000_000
        if kind in (26, 27, 28):
            values[token] = {26: 1, 27: 2, 28: 3}[kind]
        else:
            base = name.removesuffix("-ev1").removesuffix("-hero")
            values[token] = base_types.get(base, 4 if token else 0)
    return values


def audit_shard(task: tuple[str, str, list[str]]) -> dict[str, Any]:
    raw_path, split, vocabulary = task
    path = Path(raw_path)
    play = np.load(path / "play_now.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    timing = np.load(path / "timing_label_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    weights = np.load(path / "sample_weight.npy", mmap_mode="r", allow_pickle=False)
    valid = timing & np.isfinite(weights) & (weights > 0)
    rows = len(play)
    offsets = np.load(path / "entity_offsets.npy", mmap_mode="r", allow_pickle=False)
    relations = np.load(path / "entity_relations.npy", mmap_mode="r", allow_pickle=False)
    positions = np.load(path / "entity_positions.npy", mmap_mode="r", allow_pickle=False)
    entity_rows = positions.astype(np.int32) // 18
    enemy = relations == 1
    own = relations == 0
    enemy_half = _row_counts(offsets, enemy & (entity_rows < 16))
    enemy_close = _row_counts(offsets, enemy & (entity_rows <= 10))
    enemy_critical = _row_counts(offsets, enemy & (entity_rows <= 7))
    own_half = _row_counts(offsets, own & (entity_rows < 16))
    own_push = _row_counts(offsets, own & (entity_rows >= 16))
    ability = np.load(path / "ability_mask.npy", mmap_mode="r", allow_pickle=False).any(axis=1)
    scenes = {
        "all_valid": valid,
        "enemy_in_own_half": valid & (enemy_half > 0),
        "defense_pressure": valid & (enemy_close > 0),
        "critical_tower_pressure": valid & (enemy_critical > 0),
        "contested_own_half": valid & (enemy_half > 0) & (own_half > 0),
        "counterpush": valid & (own_push > 0) & (enemy_close == 0),
        "calm_board": valid & (enemy_half == 0) & (own_push == 0),
        "ability_available": valid & ability,
    }
    extent = np.load(path / "replay_extent.npy", mmap_mode="r", allow_pickle=False)
    label = np.load(path / "card_label_mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    slots = np.load(path / "card_slot.npy", mmap_mode="r", allow_pickle=False).astype(np.int64)
    hand = np.load(path / "hand_tokens.npy", mmap_mode="r", allow_pickle=False)
    card_tokens = np.zeros(rows, dtype=np.int64)
    selected = label & (slots >= 0) & (slots < 4)
    row_indices = np.flatnonzero(selected)
    card_tokens[row_indices] = hand[row_indices, slots[row_indices]]
    class_map = _card_classes(vocabulary)
    card_class = class_map[np.clip(card_tokens, 0, len(class_map) - 1)]
    positions_out = np.load(path / "position.npy", mmap_mode="r", allow_pickle=False).astype(np.int32)
    result: dict[str, Any] = {"split": split, "shard": path.name, "rows": rows, "scenes": {}, "latency": {}}
    for name, mask in scenes.items():
        actions = mask & play
        cards = mask & label
        pos_rows = positions_out[cards] // 18
        result["scenes"][name] = {
            "rows": int(mask.sum()),
            "actions": int(actions.sum()),
            "card_labels": int(cards.sum()),
            "prefix_rows": int((mask & (extent == 1)).sum()),
            "full_rows": int((mask & (extent == 0)).sum()),
            "troop_labels": int((cards & (card_class == 1)).sum()),
            "building_labels": int((cards & (card_class == 2)).sum()),
            "spell_labels": int((cards & (card_class == 3)).sum()),
            "position_own_back": int((pos_rows <= 7).sum()),
            "position_own_front": int(((pos_rows >= 8) & (pos_rows <= 15)).sum()),
            "position_enemy_front": int(((pos_rows >= 16) & (pos_rows <= 23)).sum()),
            "position_enemy_back": int((pos_rows >= 24).sum()),
            "card_token_counts": {str(key): int(value) for key, value in
                                  zip(*np.unique(card_tokens[cards], return_counts=True))},
        }
    sequence_offsets = np.load(path / "sequence_offsets.npy", mmap_mode="r", allow_pickle=False)
    for name in ("enemy_in_own_half", "defense_pressure", "critical_tower_pressure"):
        counter = Counter()
        mask = scenes[name]
        for start, stop in zip(sequence_offsets[:-1], sequence_offsets[1:]):
            start, stop = int(start), int(stop)
            local = mask[start:stop]
            if not local.any():
                continue
            onset = np.flatnonzero(local & ~np.r_[False, local[:-1]]) + start
            action_ticks = np.flatnonzero(play[start:stop]) + start
            for tick in onset:
                index = int(np.searchsorted(action_ticks, tick, side="left"))
                if index >= len(action_ticks):
                    counter["unresolved_prefix" if int(extent[tick]) == 1 else "unresolved_full"] += 1
                else:
                    delay = int(action_ticks[index] - tick)
                    counter[_latency_bucket(delay)] += 1
            counter["onsets"] += len(onset)
        result["latency"][name] = dict(counter)
    return result


def _merge(target: dict[str, Any], shard: dict[str, Any]) -> None:
    split = target.setdefault(shard["split"], {"rows": 0, "shards": 0, "scenes": {}, "latency": {}})
    split["rows"] += shard["rows"]
    split["shards"] += 1
    for name, values in shard["scenes"].items():
        merged = split["scenes"].setdefault(name, defaultdict(int))
        for key, value in values.items():
            if key == "card_token_counts":
                cards = merged.setdefault(key, defaultdict(int))
                for token, count in value.items():
                    cards[token] += int(count)
            else:
                merged[key] += int(value)
    for name, values in shard["latency"].items():
        merged = split["latency"].setdefault(name, defaultdict(int))
        for key, value in values.items():
            merged[key] += int(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-shards-per-split", type=int, default=0)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    tasks = []
    for split in args.splits:
        paths = manifest["splits"][split]
        if args.max_shards_per_split:
            paths = paths[: args.max_shards_per_split]
        tasks.extend((str((root / path).resolve()), split, manifest["card_vocabulary"]) for path in paths)
    began = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "kind": "expert_scenario_coverage_audit_v1",
        "dataset_manifest_sha256": __import__("hashlib").sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "read_only": True,
        "definitions": {
            "enemy_in_own_half": "at least one visible enemy card entity in canonical rows 0..15",
            "defense_pressure": "at least one visible enemy card entity in canonical rows 0..10",
            "critical_tower_pressure": "at least one visible enemy card entity in canonical rows 0..7",
            "counterpush": "own visible card entity in enemy half and no enemy entity in rows 0..10",
            "latency": "native ticks from pressure onset to the next recorded expert action; unresolved prefixes are censored, not failures",
        },
        "splits": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, shard in enumerate(pool.map(audit_shard, tasks), start=1):
            _merge(result["splits"], shard)
            if done % 25 == 0 or done == len(tasks):
                progress = {"phase": "scenario_audit", "completed": done, "total": len(tasks),
                            "percent": done * 100 / max(1, len(tasks)), "last_split": shard["split"]}
                (args.output.parent / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    for split in result["splits"].values():
        for scenario in split["scenes"].values():
            scenario["row_fraction"] = scenario["rows"] / max(1, split["rows"])
            scenario["action_rate_per_row"] = scenario["actions"] / max(1, scenario["rows"])
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["wall_seconds"] = (datetime.now(timezone.utc) - began).total_seconds()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "scenario_audit_complete", "output": str(args.output), "wall_seconds": result["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
