"""Reproducible frozen-libg evidence for Ranked -> uncapped 1v1 execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.client import JsonLineClient  # noqa: E402


SOURCE_RANKED_MODES = (72_000_323, 72_000_450, 72_000_464)
EXECUTION_MODE = 72_000_006
KING_HP = {
    1: 2400, 2: 2568, 3: 2736, 4: 2904,
    5: 3096, 6: 3312, 7: 3528, 8: 3768,
    9: 4008, 10: 4392, 11: 4824, 12: 5304,
    13: 5832, 14: 6408, 15: 7032, 16: 7728,
}
PRINCESS_HP = {
    1: 1400, 2: 1512, 3: 1624, 4: 1750,
    5: 1890, 6: 2030, 7: 2184, 8: 2352,
    9: 2534, 10: 2786, 11: 3052, 12: 3346,
    13: 3668, 14: 4032, 15: 4424, 16: 4858,
}


def _replay(template: dict[str, Any], mode: int, level: int) -> dict[str, Any]:
    value = deepcopy(template)
    battle = value["battle"]
    battle["gamemode"] = mode
    for side in range(2):
        for row in battle[f"deck{side}"]["sp"]:
            row["l"] = level - 1
        for row in battle[f"deck{side}"]["sc"]:
            row["l"] = level - 1
        battle[f"avatar{side}"]["expLevel"] = level
        battle[f"avatar{side}"]["kt"] = level
        battle["hbd"][side]["kt"] = level
    return value


def _tower_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    towers = state["episode"]["crown_towers"]
    return {
        "king": sorted({int(row["max_hp"]) for row in towers if row["type"] == "king"}),
        "princess": sorted({
            int(row["max_hp"]) for row in towers if row["type"] == "princess"
        }),
        "entity_levels": sorted({
            int(row["level"]) for row in state["entities"]
            if int(row["kind"]) in (12, 13)
        }),
    }


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def run(port: int) -> dict[str, Any]:
    template = json.loads((
        PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
    ).read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ranked_mode_normalization_libg_evidence_v1",
        "runtime": "15.535.29",
        "source_ranked_modes": list(SOURCE_RANKED_MODES),
        "native_execution_mode": EXECUTION_MODE,
    }
    with JsonLineClient(port=port, timeout=60) as client:
        tower_table: dict[str, Any] = {}
        for level in range(1, 17):
            state = client.request({
                "op": "reset", "replay": _replay(template, EXECUTION_MODE, level)
            })["state"]
            observed = _tower_snapshot(state)
            expected = {
                "king": [KING_HP[level]],
                "princess": [PRINCESS_HP[level]],
                "entity_levels": [level],
            }
            if observed != expected:
                raise RuntimeError(
                    f"level {level} tower probe mismatch: {observed} != {expected}"
                )
            tower_table[str(level)] = {**observed, "state_hash": state["state_hash"]}
        result["native_level_table"] = tower_table

        direct_cap: dict[str, Any] = {}
        for mode in SOURCE_RANKED_MODES[1:]:
            state = client.request({
                "op": "reset", "replay": _replay(template, mode, 16)
            })["state"]
            observed = _tower_snapshot(state)
            if observed != {
                "king": [KING_HP[11]],
                "princess": [PRINCESS_HP[11]],
                "entity_levels": [11],
            }:
                raise RuntimeError(f"Ranked fallback cap changed for {mode}: {observed}")
            direct_cap[str(mode)] = observed
        mapped = _tower_snapshot(client.request({
            "op": "reset", "replay": _replay(template, EXECUTION_MODE, 16)
        })["state"])
        if mapped != {
            "king": [KING_HP[16]],
            "princess": [PRINCESS_HP[16]],
            "entity_levels": [16],
        }:
            raise RuntimeError(f"uncapped execution did not preserve level 16: {mapped}")
        result["direct_ranked_level16_fallback"] = direct_cap
        result["mapped_execution_level16"] = mapped

        parity: dict[str, Any] = {}
        for mode in SOURCE_RANKED_MODES:
            replay = _replay(template, mode, 11)
            state = client.request({"op": "reset", "replay": replay})["state"]
            checkpoints: list[dict[str, Any]] = []
            current_tick = int(state["tick"])
            for target in (10, 100, 3600, 6000):
                if target > current_tick:
                    client.request({"op": "step", "steps": target - current_tick})
                state = client.request({"op": "observe"})["state"]
                current_tick = int(state["tick"])
                episode = state["episode"]
                checkpoints.append({
                    "tick": current_tick,
                    "state_hash": state["state_hash"],
                    "elixir_raw": [int(row["elixir_raw"]) for row in state["players"]],
                    "outcome": episode["outcome"],
                    "crowns": episode["crowns"],
                    "commands_allowed": episode["commands_allowed"],
                    "native_phase": episode["native_phase"],
                })

            state = client.request({"op": "reset", "replay": replay})["state"]
            deck_index = int(state["players"][0]["hand_deck_indices"][0])
            grid = client.request({
                "op": "probe_grid",
                "action": {
                    "side": 0, "deck_index": deck_index,
                    "account_hi": 1, "account_lo": 1,
                },
            })["result"]["rows"]

            client.request({"op": "reset", "replay": replay})
            client.request({"op": "step", "steps": 90})
            action = client.request({
                "op": "act",
                "action": {
                    "side": 0, "deck_index": deck_index, "x": 9000, "y": 6000,
                    "account_hi": 1, "account_lo": 1,
                },
            })
            if not action.get("ok") or not action["result"].get("accepted"):
                raise RuntimeError(f"parity deployment rejected for mode {mode}: {action}")
            client.request({"op": "step", "steps": 400})
            state = client.request({"op": "observe"})["state"]
            units = [
                {
                    key: row.get(key)
                    for key in (
                        "generation_key", "kind", "side", "x", "y", "hp", "max_hp",
                        "behavior_state", "target_previous_x", "target_previous_y",
                        "movement_direction_x", "movement_direction_y", "collision_count",
                        "path_segment_direction_x", "path_segment_direction_y",
                        "path_node_consumed",
                    )
                }
                for row in state["entities"] if int(row["kind"]) not in (12, 13)
            ]
            parity[str(mode)] = {
                "checkpoints": checkpoints,
                "grid_sha256": _sha(grid),
                "grid_legal_cells": sum(bool(cell) for row in grid for cell in row),
                "deployment_tick500_state_hash": state["state_hash"],
                "deployment_units_sha256": _sha(units),
                "deployment_units": units,
            }

        signatures = {
            _sha({
                "checkpoints": row["checkpoints"],
                "grid": row["grid_sha256"],
                "legal": row["grid_legal_cells"],
                "state": row["deployment_tick500_state_hash"],
                "units": row["deployment_units_sha256"],
            })
            for row in parity.values()
        }
        if len(signatures) != 1:
            raise RuntimeError(f"Ranked logic parity mismatch: {parity}")
        result["ranked_mode_logic_parity"] = parity
        result["ranked_mode_logic_parity_sha256"] = next(iter(signatures))
        result["passed"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=38031)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\ranked-mode-normalization-v2.json"
        ),
    )
    args = parser.parse_args()
    value = run(args.port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "passed": True,
        "output": str(args.output.resolve()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "parity_sha256": value["ranked_mode_logic_parity_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
