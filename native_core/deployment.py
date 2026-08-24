"""Arena deployment rules layered on libg's raw 18x32 terrain mask."""

from __future__ import annotations

from typing import Any, Mapping


ARENA_COLUMNS = 18
ARENA_ROWS = 32
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
    """Apply observed ownership/tower rules to libg's raw terrain mask.

    Spells keep their native full targeting mask. Unit/building masks are made
    symmetric, remove the 4x4 King and 3x3 Princess tower footprints, restrict
    normal deployment to the friendly half, and open the matching five-cell
    enemy pocket after a Princess tower is destroyed.
    """
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    if len(native_rows) != ARENA_ROWS or any(
        len(row) != ARENA_COLUMNS or set(row) - {"0", "1"}
        for row in native_rows
    ):
        raise ValueError("native deployment mask must be 18x32 binary rows")
    if card_id // 1_000_000 == 28:
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
        x, y = int(entity["x"]), int(entity["y"])
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
    left_alive = any(
        int(entity.get("x", 0)) < 9000
        for entity in living_enemy_princesses
    )
    right_alive = any(
        int(entity.get("x", 0)) >= 9000
        for entity in living_enemy_princesses
    )
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
