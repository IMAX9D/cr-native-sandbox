"""Synthetic codec benchmark calibrated to compact native observation fields."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

from .codec import EpisodeReader, encode_episode
from .schema import EntityState, EpisodeState, PlayerPrivate, TickState, TowerState


def synthetic_ticks(count: int, entities: int) -> list[TickState]:
    towers = tuple(
        TowerState(side * 3 + slot, side, 0 if slot == 0 else 1, -1 if slot == 0 else slot - 1,
                   9000 if slot == 0 else 3500 if slot == 1 else 14500,
                   3000 if side == 0 else 29000, 4824 if slot == 0 else 3052,
                   4824 if slot == 0 else 3052)
        for side in (0, 1) for slot in (0, 1, 2)
    )
    active = tuple(
        EntityState(5_001_000 + index, index % 2, 500 + (index * 997) % 17000,
                    3000 + (index * 1231) % 26000, 26_000_000 + index % 100,
                    15, 2000, 2000, 1, 0, -1, 0, -1, -1, -1, -1)
        for index in range(entities)
    )
    result = []
    for tick in range(count):
        active = tuple(
            replace(item, x=(item.x + (17 if item.side == 0 else -17)) % 18000,
                    y=max(0, min(32000, item.y + (23 if item.side == 0 else -23))),
                    hp=max(0, item.hp - (1 if tick % 11 == 0 else 0)))
            for item in active
        )
        result.append(TickState(
            100 + tick,
            (PlayerPrivate(0, min(100000, 50000 + tick * 35), (0,1,2,3), 4),
             PlayerPrivate(1, min(100000, 50000 + tick * 35), (4,5,6,7), 0)),
            towers, active, EpisodeState(1,0,1,0,0,0,0,0,0),
        ))
    return result


def run(ticks: int, entities: int, total_ticks: int) -> dict[str, float | int]:
    states = synthetic_ticks(ticks, entities)
    native_like = [
        {"tick": state.tick, "players": [player.values() for player in state.players],
         "towers": [tower.values() for tower in state.towers],
         "entities": [entity.values() for entity in state.entities]}
        for state in states
    ]
    json_bytes = sum(len(json.dumps(value, separators=(",", ":"))) + 1 for value in native_like)
    started = time.perf_counter()
    blob, stats = encode_episode(states, {"battle_tag": "BENCH"}, anchor_interval=256)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decoded = sum(1 for _ in EpisodeReader(blob).iter_ticks())
    decode_seconds = time.perf_counter() - started
    if decoded != ticks:
        raise RuntimeError("benchmark decode lost Ticks")
    dense_bytes = ticks * 10 * 32 * 18
    return {
        "sample_ticks": ticks,
        "entities_per_tick": entities,
        "json_bytes_per_tick": json_bytes / ticks,
        "dense_grid_bytes_per_tick": dense_bytes / ticks,
        "tick_store_bytes_per_tick": len(blob) / ticks,
        "compression_vs_json": json_bytes / len(blob),
        "compression_vs_dense_grid": dense_bytes / len(blob),
        "encode_ticks_per_second": ticks / encode_seconds,
        "decode_ticks_per_second": ticks / decode_seconds,
        "estimated_tick_store_gib": len(blob) / ticks * total_ticks / 2**30,
        "estimated_dense_grid_tib": dense_bytes / ticks * total_ticks / 2**40,
        "estimated_json_tib": json_bytes / ticks * total_ticks / 2**40,
        "total_ticks_for_estimate": total_ticks,
        "stored_bytes": stats["stored_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=10_000)
    parser.add_argument("--entities", type=int, default=24)
    parser.add_argument("--total-ticks", type=int, default=471_450_949)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.ticks, args.entities, args.total_ticks)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
