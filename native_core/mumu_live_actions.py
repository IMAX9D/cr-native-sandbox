"""Ordinary Android touch deployment and read-back receipts (no libg writes)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .mumu_live_protocol import adb_run

# Native ticks can advance underneath the entrance/loading animation before
# the four card buttons accept input. This is a UI guard, not training timing.
UI_READY_MIN_TICK = 150


@dataclass(frozen=True)
class ScreenLayout:
    width: int
    height: int
    viewport_left: float
    viewport_width: float
    arena_left: float
    arena_right: float
    arena_top: float
    arena_bottom: float
    hand_y: float
    hand_x: tuple[float, float, float, float]

    @classmethod
    def from_size(cls, width: int, height: int) -> 'ScreenLayout':
        if width <= 0 or height <= 0:
            raise ValueError('invalid Android display size')
        viewport_width = min(float(width), float(height) * 9 / 16) if width / height > 0.8 else float(width)
        left = (width - viewport_width) / 2
        return cls(width, height, left, viewport_width,
            left + viewport_width * .055, left + viewport_width * .945,
            # Live stationary-Cannon read-back: previous arena origin was
            # one native tile too low. Preserve scale, shift both endpoints.
            height * (.105 - .685 / 32), height * (.790 - .685 / 32), height * .890,
            tuple(left + viewport_width * x for x in (.31, .50, .69, .88)))

    def deployment_point(self, canonical_position: int, side: int = 0) -> tuple[int, int]:
        if type(canonical_position) is not int or not 0 <= canonical_position < 576:
            raise ValueError('placement must be a cell in the 18x32 grid')
        row, column = divmod(canonical_position, 18)
        if type(side) is not int or side not in (0, 1):
            raise ValueError('side must be 0 or 1')
        x_fraction = (column + .5) / 18
        # The actor canonicalizes side 1 with a 180-degree rotation, while
        # the online camera preserves native X when putting self at the bottom.
        if side == 1:
            x_fraction = 1 - x_fraction
        return (round(self.arena_left + x_fraction * (self.arena_right - self.arena_left)),
                round(self.arena_bottom - (row + .5) / 32 * (self.arena_bottom - self.arena_top)))

    def hand_point(self, slot: int) -> tuple[int, int]:
        if type(slot) is not int or slot not in range(4):
            raise ValueError('hand slot must be 0..3')
        return round(self.hand_x[slot]), round(self.hand_y)


def send_card_taps(adb: Path, serial: str, layout: ScreenLayout, slot: int, cell: int, *, side: int = 0) -> dict:
    hand, target = layout.hand_point(slot), layout.deployment_point(cell, side)
    for x, y in (hand, target):
        if not 0 <= x < layout.width or not 0 <= y < layout.height:
            raise ValueError('touch outside Android display')
    # One Android-side command keeps the 50ms inter-tap wait off the host/RPC
    # critical path. All interpolated arguments are validated integers.
    command = f'input tap {hand[0]} {hand[1]}; sleep 0.05; input tap {target[0]} {target[1]}'
    adb_run(adb, serial, 'shell', command, timeout=5)
    return {'hand_screen': hand, 'target_screen': target, 'inter_tap_sleep_ms': 50}


def own_player(frame: dict, side: int) -> dict | None:
    return next((p for p in frame.get('players', []) if p.get('side') == side), None)


def card_receipt(before: dict, after: dict, *, side: int, slot: int, card_id: int) -> dict:
    identity = lambda f: (f.get('pid'), (f.get('chain') or {}).get('battle'), (f.get('chain') or {}).get('player_state'))
    if identity(before) != identity(after):
        return {'accepted': False, 'reason': 'battle_changed'}
    if not after.get('coherent') or after.get('game_tick', -1) <= before.get('game_tick', -1):
        return {'accepted': False, 'reason': 'awaiting_new_frame'}
    old, new = own_player(before, side), own_player(after, side)
    if not old or not new or slot not in range(4):
        return {'accepted': False, 'reason': 'player_unavailable'}
    old_hand, new_hand = old.get('hand_deck_indices', []), new.get('hand_deck_indices', [])
    if len(old_hand) != 4 or len(new_hand) != 4:
        return {'accepted': False, 'reason': 'invalid_hand'}
    changed = old_hand[slot] != new_hand[slot] and new_hand[slot] in range(8)
    decrease = old['elixir_raw'] - new['elixir_raw']
    old_keys = {(e['address'], e['category']) for e in before.get('entities', [])}
    matching = [{k: e[k] for k in ('address', 'category', 'card_id', 'side', 'x', 'y', 'hp')}
        for e in after.get('entities', []) if e.get('side') == side and e.get('card_id') == card_id
        and (e['address'], e['category']) not in old_keys]
    return {'accepted': changed and decrease > 0,
        'reason': 'selected_slot_rotated_and_elixir_decreased' if changed and decrease > 0 else 'awaiting_receipt',
        'slot_changed': changed, 'elixir_decrease_raw': decrease,
        'hand_before': old_hand, 'hand_after': new_hand, 'matching_new_entities': matching,
        'tick_before': before['game_tick'], 'tick_after': after['game_tick']}
