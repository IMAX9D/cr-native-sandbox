"""Stable observation, action, reward and on-disk schemas for self-play."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np


CARD_IDS = (
    26000000,  # Knight
    26000001,  # Archers
    26000003,  # Giant
    26000010,  # Skeletons
    26000014,  # Musketeer
    26000021,  # Hog Rider
    27000000,  # Cannon
    28000001,  # Arrows
)
CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
CARD_COSTS = {
    26000000: 3,
    26000001: 3,
    26000003: 5,
    26000010: 1,
    26000014: 4,
    26000021: 4,
    27000000: 3,
    28000001: 3,
}

ARENA_COLUMNS = 18
ARENA_ROWS = 32
GRID_CHANNELS = 10
SCALAR_SIZE = 64
PRIVILEGED_SIZE = 33
POCKET_DEPTH_CELLS = 5
LANE_SPLIT_COLUMN = 9


def _tower_role(entity: Mapping[str, Any]) -> str | None:
    if int(entity.get("card_id", -1)) != -1:
        return None
    slot = int(entity.get("category", -1)) - 5_000_000
    if 0 <= slot <= 5:
        return "king" if slot % 3 == 0 else "princess"
    x = int(entity.get("x", -1))
    if x == 9000:
        return "king"
    if 0 <= x < 18000:
        return "princess"
    return None


def deployment_mask(
    native_rows: list[str],
    state: Mapping[str, Any],
    *,
    side: int,
    card_id: int,
) -> list[str]:
    """Apply server-observed ownership/tower rules to libg's terrain mask."""
    if len(native_rows) != ARENA_ROWS or any(
        len(row) != ARENA_COLUMNS for row in native_rows
    ):
        raise ValueError("native deployment mask must be 18x32")
    if card_id // 1_000_000 == 28:  # spells retain the native full target mask
        return list(native_rows)
    result = [
        "".join(
            "1"
            if all(
                native_rows[source_row][source_column] == "1"
                for source_row, source_column in (
                    (row, column),
                    (row, 17 - column),
                    (31 - row, column),
                    (31 - row, 17 - column),
                )
            )
            else "0"
            for column in range(ARENA_COLUMNS)
        )
        for row in range(ARENA_ROWS)
    ]
    entities = state.get("entities", [])
    for entity in entities:
        role = _tower_role(entity)
        if role is None or int(entity.get("hp", 0)) <= 0:
            continue
        footprint = 4 if role == "king" else 3
        half_extent = footprint * 1000 // 2
        x = int(entity["x"])
        y = int(entity["y"])
        column_start = max(0, (x - half_extent) // 1000)
        column_stop = min(18, (x + half_extent + 999) // 1000)
        row_start = max(0, (y - half_extent) // 1000)
        row_stop = min(32, (y + half_extent + 999) // 1000)
        for row in range(row_start, row_stop):
            cells = list(result[row])
            for column in range(column_start, column_stop):
                cells[column] = "0"
            result[row] = "".join(cells)

    living_enemy_princesses = [
        entity
        for entity in entities
        if int(entity.get("side", -1)) != side
        and _tower_role(entity) == "princess"
        and int(entity.get("hp", 0)) > 0
    ]
    left_alive = any(int(entity.get("x", 0)) < 9000 for entity in living_enemy_princesses)
    right_alive = any(int(entity.get("x", 0)) >= 9000 for entity in living_enemy_princesses)
    allowed = [[False] * ARENA_COLUMNS for _ in range(ARENA_ROWS)]
    own_rows = range(0, 15) if side == 0 else range(17, 32)
    for row in own_rows:
        allowed[row] = [True] * ARENA_COLUMNS
    pocket_rows = (
        range(17, 17 + POCKET_DEPTH_CELLS)
        if side == 0
        else range(15 - POCKET_DEPTH_CELLS, 15)
    )
    if not left_alive:
        for row in pocket_rows:
            for column in range(0, LANE_SPLIT_COLUMN):
                allowed[row][column] = True
    if not right_alive:
        for row in pocket_rows:
            for column in range(LANE_SPLIT_COLUMN, ARENA_COLUMNS):
                allowed[row][column] = True
    return [
        "".join(
            "1" if allowed[row][column] and result[row][column] == "1" else "0"
            for column in range(ARENA_COLUMNS)
        )
        for row in range(ARENA_ROWS)
    ]


class ActionMaskCache:
    """Cache static native terrain and tower-state-dependent common masks."""

    def __init__(self) -> None:
        self._static: dict[tuple[int, int, int, tuple[str, ...]], np.ndarray] = {}
        self._dynamic: dict[tuple[int, tuple[tuple[int, ...], ...]], np.ndarray] = {}

    @staticmethod
    def _tower_signature(state: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
        values = []
        for entity in state.get("entities", []):
            role = _tower_role(entity)
            if role is None:
                continue
            values.append((
                int(entity.get("side", -1)),
                1 if role == "king" else 0,
                int(entity.get("x", 0)),
                int(entity.get("y", 0)),
                1 if int(entity.get("hp", 0)) > 0 else 0,
            ))
        return tuple(sorted(values))

    def _static_mask(
        self,
        native_rows: list[str],
        *,
        side: int,
        deck_index: int,
        card_id: int,
    ) -> np.ndarray:
        rows_key = tuple(native_rows)
        key = (side, deck_index, card_id, rows_key)
        cached = self._static.get(key)
        if cached is not None:
            return cached
        raw = np.asarray(
            [[cell == "1" for cell in row] for row in native_rows],
            dtype=np.bool_,
        )
        if raw.shape != (ARENA_ROWS, ARENA_COLUMNS):
            raise ValueError("native deployment mask must be 18x32")
        if card_id // 1_000_000 != 28:
            raw = raw & raw[:, ::-1] & raw[::-1, :] & raw[::-1, ::-1]
        result = np.ascontiguousarray(raw.reshape(-1))
        result.setflags(write=False)
        self._static[key] = result
        return result

    def _dynamic_mask(
        self, state: Mapping[str, Any], *, side: int
    ) -> np.ndarray:
        signature = self._tower_signature(state)
        key = (side, signature)
        cached = self._dynamic.get(key)
        if cached is not None:
            return cached
        allowed = np.zeros((ARENA_ROWS, ARENA_COLUMNS), dtype=np.bool_)
        own_rows = slice(0, 15) if side == 0 else slice(17, 32)
        allowed[own_rows, :] = True
        living_enemy_princesses = [
            entity
            for entity in state.get("entities", [])
            if int(entity.get("side", -1)) != side
            and _tower_role(entity) == "princess"
            and int(entity.get("hp", 0)) > 0
        ]
        left_alive = any(
            int(entity.get("x", 0)) < 9000
            for entity in living_enemy_princesses
        )
        right_alive = any(
            int(entity.get("x", 0)) >= 9000
            for entity in living_enemy_princesses
        )
        pocket_rows = (
            slice(17, 17 + POCKET_DEPTH_CELLS)
            if side == 0
            else slice(15 - POCKET_DEPTH_CELLS, 15)
        )
        if not left_alive:
            allowed[pocket_rows, :LANE_SPLIT_COLUMN] = True
        if not right_alive:
            allowed[pocket_rows, LANE_SPLIT_COLUMN:] = True
        for entity in state.get("entities", []):
            role = _tower_role(entity)
            if role is None or int(entity.get("hp", 0)) <= 0:
                continue
            footprint = 4 if role == "king" else 3
            half_extent = footprint * 1000 // 2
            x, y = int(entity["x"]), int(entity["y"])
            column_start = max(0, (x - half_extent) // 1000)
            column_stop = min(18, (x + half_extent + 999) // 1000)
            row_start = max(0, (y - half_extent) // 1000)
            row_stop = min(32, (y + half_extent + 999) // 1000)
            allowed[row_start:row_stop, column_start:column_stop] = False
        result = np.ascontiguousarray(allowed.reshape(-1))
        result.setflags(write=False)
        self._dynamic[key] = result
        return result

    def position_mask(
        self,
        native_rows: list[str],
        state: Mapping[str, Any],
        *,
        side: int,
        deck_index: int,
        card_id: int,
    ) -> np.ndarray:
        static = self._static_mask(
            native_rows,
            side=side,
            deck_index=deck_index,
            card_id=card_id,
        )
        if card_id // 1_000_000 == 28:
            return static
        return static & self._dynamic_mask(state, side=side)


def build_action_masks(
    state: Mapping[str, Any],
    *,
    side: int,
    native_masks: Mapping[tuple[int, int], list[str]],
    decks: list[list[Mapping[str, int]]],
    cache: ActionMaskCache | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return card mask, four position masks and current hand deck indices."""
    player = next(
        item for item in state.get("players", []) if int(item["side"]) == side
    )
    hand = [int(value) for value in player["hand_deck_indices"]]
    card_mask = np.zeros(5, dtype=np.bool_)
    card_mask[0] = True  # wait is always legal
    position_masks = np.zeros((4, ARENA_ROWS * ARENA_COLUMNS), dtype=np.bool_)
    elixir = int(player["elixir"])
    deployment_open = int(state.get("tick", 0)) >= 100
    for hand_index, deck_index in enumerate(hand[:4]):
        if deck_index < 0:
            continue
        card = decks[side][deck_index]
        card_id = int(card["card_id"])
        if cache is None:
            rows = deployment_mask(
                native_masks[(side, deck_index)], state, side=side, card_id=card_id
            )
            flat = np.fromiter(
                (cell == "1" for row in rows for cell in row),
                dtype=np.bool_,
                count=ARENA_ROWS * ARENA_COLUMNS,
            )
        else:
            flat = cache.position_mask(
                native_masks[(side, deck_index)], state,
                side=side, deck_index=deck_index, card_id=card_id,
            )
        position_masks[hand_index] = flat
        card_mask[hand_index + 1] = (
            deployment_open
            and elixir >= CARD_COSTS[card_id]
            and bool(flat.any())
        )
    return card_mask, position_masks, hand


class ObservationEncoder:
    """Encode either side into one rotationally canonical tensor view."""

    grid_channels = GRID_CHANNELS
    scalar_size = SCALAR_SIZE
    privileged_size = PRIVILEGED_SIZE

    @staticmethod
    def _cell(x: int, y: int, side: int) -> tuple[int, int]:
        column = min(17, max(0, x // 1000))
        row = min(31, max(0, y // 1000))
        if side == 1:
            column = 17 - column
            row = 31 - row
        return row, column

    def encode(
        self,
        state: Mapping[str, Any],
        *,
        side: int,
        public_actions: Mapping[int, Mapping[str, int] | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid = np.zeros((GRID_CHANNELS, ARENA_ROWS, ARENA_COLUMNS), dtype=np.float32)
        entity_counts = [0, 0]
        for entity in state.get("entities", []):
            absolute_side = int(entity.get("side", -1))
            if absolute_side not in (0, 1):
                continue
            relation = 0 if absolute_side == side else 1
            row, column = self._cell(int(entity["x"]), int(entity["y"]), side)
            hp = max(0, int(entity.get("hp", 0)))
            maximum = max(1, int(entity.get("max_hp", 1)))
            hp_ratio = min(1.0, hp / maximum)
            card_id = int(entity.get("card_id", -1))
            if _tower_role(entity) is not None:
                channel = relation
            elif card_id // 1_000_000 == 27:
                channel = 4 + relation
            else:
                channel = 2 + relation
            grid[channel, row, column] = max(grid[channel, row, column], hp_ratio)
            card_index = CARD_INDEX.get(card_id, -1)
            if card_index >= 0:
                grid[6 + relation, row, column] = max(
                    grid[6 + relation, row, column], (card_index + 1) / 8.0
                )
            grid[8 + relation, row, column] = min(
                1.0,
                grid[8 + relation, row, column]
                + min(1.0, max(0, int(entity.get("behavior_state", 0))) / 16.0),
            )
            entity_counts[relation] += 1

        players = {int(item["side"]): item for item in state.get("players", [])}
        own = players[side]
        scalars: list[float] = [
            min(1.5, int(state["tick"]) / 6000.0),
            int(own["elixir"]) / 10.0,
            min(1.0, entity_counts[0] / 64.0),
            min(1.0, entity_counts[1] / 64.0),
        ]
        towers = state.get("episode", {}).get("crown_towers", [])
        for owner in (side, 1 - side):
            owned = [tower for tower in towers if int(tower["side"]) == owner]
            ordered = sorted(
                owned,
                key=lambda tower: (
                    0 if tower.get("type") == "king" else 1,
                    int(tower.get("x", 0)) if side == 0 else -int(tower.get("x", 0)),
                ),
            )
            for tower in ordered[:3]:
                scalars.append(
                    max(0.0, int(tower["hp"]) / max(1, int(tower["max_hp"])))
                )
        while len(scalars) < 10:
            scalars.append(0.0)
        hand_features = [0.0] * 32
        hand_by_index = {
            int(item["hand_index"]): item for item in own.get("hand", [])
        }
        for hand_index, deck_index in enumerate(own["hand_deck_indices"][:4]):
            if deck_index < 0:
                continue
            card_id = int(hand_by_index[hand_index]["card_id"])
            card_index = CARD_INDEX[card_id]
            hand_features[hand_index * 8 + card_index] = 1.0
        scalars.extend(hand_features)
        actions = public_actions or {}
        for actor in (side, 1 - side):
            action = actions.get(actor)
            event = [0.0] * 11
            if action is not None:
                event[0] = 1.0
                card_id = int(action["card_id"])
                event[1 + CARD_INDEX[card_id]] = 1.0
                x = int(action["x"])
                y = int(action["y"])
                if side == 1:
                    x = 17999 - x
                    y = 31999 - y
                event[9] = np.clip(x / 18000.0, 0.0, 1.0)
                event[10] = np.clip(y / 32000.0, 0.0, 1.0)
            scalars.extend(event)
        if len(scalars) != SCALAR_SIZE:
            raise AssertionError(f"scalar schema drift: {len(scalars)}")
        return grid, np.asarray(scalars, dtype=np.float32)

    def privileged(self, state: Mapping[str, Any], *, side: int) -> np.ndarray:
        """Critic-only hidden information; never pass this tensor to Actor."""
        players = {int(item["side"]): item for item in state.get("players", [])}
        enemy = players[1 - side]
        values = [int(enemy["elixir"]) / 10.0] + [0.0] * 32
        for hand_index, item in enumerate(enemy.get("hand", [])[:4]):
            card_index = CARD_INDEX[int(item["card_id"])]
            values[1 + hand_index * 8 + card_index] = 1.0
        return np.asarray(values, dtype=np.float32)


class PotentialReward:
    """Zero-sum potential shaping; terminal winner remains the only objective."""

    def __init__(self, *, gamma: float = 0.997, shaping_scale: float = 0.20) -> None:
        self.gamma = gamma
        self.shaping_scale = shaping_scale

    @staticmethod
    def potential(state: Mapping[str, Any], side: int) -> float:
        towers = state.get("episode", {}).get("crown_towers", [])
        tower_value = [0.0, 0.0]
        for tower in towers:
            owner = int(tower["side"])
            weight = 2.0 if tower.get("type") == "king" else 1.0
            tower_value[owner] += weight * max(
                0.0, int(tower["hp"]) / max(1, int(tower["max_hp"]))
            )
        tower_advantage = (tower_value[side] - tower_value[1 - side]) / 4.0
        players = {int(item["side"]): item for item in state.get("players", [])}
        elixir_advantage = 0.0
        if side in players and 1 - side in players:
            elixir_advantage = (
                int(players[side]["elixir"]) - int(players[1 - side]["elixir"])
            ) / 10.0
        board_value = [0.0, 0.0]
        for entity in state.get("entities", []):
            owner = int(entity.get("side", -1))
            card_id = int(entity.get("card_id", -1))
            if owner not in (0, 1) or card_id not in CARD_COSTS:
                continue
            hp_ratio = max(0.0, int(entity.get("hp", 0))) / max(
                1, int(entity.get("max_hp", 1))
            )
            board_value[owner] += CARD_COSTS[card_id] * min(1.0, hp_ratio)
        board_advantage = np.clip(
            (board_value[side] - board_value[1 - side]) / 20.0, -1.0, 1.0
        )
        return float(np.clip(
            0.75 * tower_advantage
            + 0.10 * elixir_advantage
            + 0.15 * board_advantage,
            -1.0,
            1.0,
        ))

    def transition(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any] | None,
        *,
        terminal_rewards: Mapping[int, float] | None = None,
        done: bool = False,
    ) -> dict[int, float]:
        terminal_rewards = terminal_rewards or {0: 0.0, 1: 0.0}
        phi_previous = self.potential(previous, 0)
        phi_current = 0.0 if done or current is None else self.potential(current, 0)
        shaped = self.shaping_scale * (
            self.gamma * phi_current - phi_previous
        )
        reward0 = float(terminal_rewards.get(0, 0.0)) + shaped
        # All features and terminal rewards are antisymmetric by construction.
        return {0: reward0, 1: -reward0}


@dataclass(frozen=True)
class TrainingPaths:
    root: Path
    runs: Path
    trajectories: Path
    checkpoints: Path
    logs: Path
    evaluations: Path


class RunStore:
    """Create atomic, self-describing training runs under D:\\AI_data."""

    def __init__(self, root: Path = Path(r"D:\AI_data")) -> None:
        self.root = root.resolve()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def create(self, config: Mapping[str, Any], *, run_id: str | None = None) -> TrainingPaths:
        if run_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"native8-{stamp}-{uuid.uuid4().hex[:8]}"
        run = self.root / "runs" / run_id
        paths = TrainingPaths(
            root=run,
            runs=self.root / "runs",
            trajectories=run / "trajectories",
            checkpoints=run / "checkpoints",
            logs=run / "logs",
            evaluations=run / "evaluations",
        )
        for path in asdict(paths).values():
            Path(path).mkdir(parents=True, exist_ok=True)
        self._atomic_json(
            run / "manifest.json",
            {
                "schema_version": 1,
                "kind": "native_eight_card_selfplay_run",
                "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_root": str(self.root),
                "config": dict(config),
            },
        )
        return paths
