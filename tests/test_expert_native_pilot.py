from __future__ import annotations

from dataclasses import replace
import unittest

from expert_v1.native_pilot import _logical_state_digest
from expert_v1.tick_store_v1.schema import (
    EntityState,
    EpisodeState,
    PlayerPrivate,
    TickState,
)


class ExpertNativePilotTest(unittest.TestCase):
    def test_logical_digest_removes_seed_specific_native_deck_slots(self) -> None:
        mapping_a = ((0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
        mapping_b = ((7, 6, 5, 4, 3, 2, 1, 0), (4, 5, 6, 7, 0, 1, 2, 3))
        base = TickState(
            tick=10,
            players=(
                PlayerPrivate(0, 50_000, (0, 1, 2, 3), 4),
                PlayerPrivate(1, 60_000, (4, 5, 6, 7), 0),
            ),
            towers=(),
            entities=(
                EntityState(
                    5_000_010, 0, 9000, 8000, 26_000_000, 15,
                    100, 100, 1, 0, 1, 0, -1, -1, -1, -1,
                ),
            ),
            episode=EpisodeState(1, 0, 1, 0, 0, 0, 0, 0, 0),
        )
        permuted = replace(
            base,
            players=(
                PlayerPrivate(0, 50_000, (7, 6, 5, 4), 3),
                PlayerPrivate(1, 60_000, (0, 1, 2, 3), 4),
            ),
        )
        self.assertEqual(
            _logical_state_digest((base,), mapping_a),
            _logical_state_digest((permuted,), mapping_b),
        )


if __name__ == "__main__":
    unittest.main()
