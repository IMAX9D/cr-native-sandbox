from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.tick_store_v1.codec import EpisodeReader, encode_episode
from expert_v1.tick_store_v1.schema import (
    EntityState,
    EpisodeState,
    PlayerPrivate,
    TickState,
    TowerState,
    actor_projection,
)
from expert_v1.tick_store_v1.shard import (
    AppendOnlyShardWriter,
    ShardReader,
    build_store_manifest,
)
from expert_v1.tick_store_v1.work_queue import TickStoreWorkQueue


def frames(count: int = 600) -> list[TickState]:
    towers = (
        TowerState(1, 0, 1, 0, 3500, 6500, 3052, 3052),
        TowerState(2, 0, 1, 1, 14500, 6500, 3052, 3052),
        TowerState(4, 1, 1, 0, 3500, 25500, 3052, 3052),
        TowerState(5, 1, 1, 1, 14500, 25500, 3052, 3052),
    )
    output = []
    entities: tuple[EntityState, ...] = ()
    for index in range(count):
        if index == 10:
            entities = (
                EntityState(
                    5_000_101, 0, 9000, 5000, 26_000_000, 15,
                    2000, 2000, 1, 0, -1, 0, -1, -1, -1, -1,
                ),
            )
        elif 10 < index < 300:
            entity = entities[0]
            entities = (replace(entity, y=entity.y + 40, hp=max(0, entity.hp - (1 if index % 7 == 0 else 0))),)
        elif index == 300:
            entities = ()
        current_towers = towers
        if index >= 100:
            current_towers = towers[:2] + (replace(towers[2], hp=3052 - min(1000, index - 100)), towers[3])
        output.append(
            TickState(
                tick=100 + index,
                players=(
                    PlayerPrivate(0, min(100_000, 50_000 + index * 35), (0, 1, 2, 3), 4),
                    PlayerPrivate(1, min(100_000, 50_000 + index * 35), (4, 5, 6, 7), 0),
                ),
                towers=current_towers,
                entities=entities,
                episode=EpisodeState(1, 0, 1, 0, 0, 0, 0, 0, 0),
            )
        )
    return output


class TickStoreV1Test(unittest.TestCase):
    def test_anchor_delta_roundtrip_and_random_access(self) -> None:
        source = frames()
        blob, stats = encode_episode(
            source, {"battle_tag": "ROUNDTRIP"}, anchor_interval=128
        )
        decoded = list(EpisodeReader(blob).iter_ticks())
        self.assertEqual(source, decoded)
        self.assertEqual(EpisodeReader(blob).read_tick(477), source[377])
        self.assertEqual(stats["ticks"], len(source))
        self.assertLess(stats["stored_bytes"], 30 * len(source))

    def test_actor_projection_drops_opponent_private_state(self) -> None:
        projection = actor_projection(frames(1)[0], actor_side=0)
        encoded = json.dumps(
            {
                "own": projection.own_player.values(),
                "entities": [
                    (item.key, item.relation, item.x, item.y, item.card_id,
                     item.level, item.hp, item.max_hp, item.own_ability_slot,
                     item.own_ability_available)
                    for item in projection.entities
                ],
            }
        )
        self.assertNotIn("50000, 4, 5, 6, 7", encoded)
        self.assertEqual(projection.own_player.hand, (0, 1, 2, 3))
        self.assertFalse(hasattr(projection, "opponent_player"))
        self.assertFalse(hasattr(projection.episode, "logic_state"))

    def test_refill_empty_slot_and_timer_roundtrip_losslessly(self) -> None:
        source = frames(20)
        transient = replace(
            source[1],
            players=(
                PlayerPrivate(0, 42_000, (0, 1, -1, 3), 4, 600),
                source[1].players[1],
            ),
        )
        source[1] = transient
        blob, _ = encode_episode(
            source, {"battle_tag": "REFILL"}, anchor_interval=16
        )
        decoded = list(EpisodeReader(blob).iter_ticks())
        self.assertEqual(decoded, source)
        self.assertEqual(decoded[1].players[0].hand, (0, 1, -1, 3))
        self.assertEqual(decoded[1].players[0].refill_timer, 600)

    def test_shard_recovers_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AppendOnlyShardWriter(root, "worker-00000")
            writer.append("A", frames(80), {})
            writer.close()
            with writer.partial_path.open("ab") as handle:
                handle.write(b"EPS1broken-tail")
            writer = AppendOnlyShardWriter(root, "worker-00000")
            self.assertEqual(writer.episode_count, 1)
            writer.append("B", frames(60), {})
            manifest = writer.finalize()
            self.assertEqual(manifest["episode_count"], 2)
            with ShardReader(root / "worker-00000.crts") as reader:
                self.assertEqual(len(list(reader.actor_ticks("B", actor_side=1))), 60)
            source_manifest = root / "selection.jsonl"
            source_manifest.write_text("{}\n", encoding="utf-8")
            global_manifest = build_store_manifest(
                root,
                source_manifest=source_manifest,
                expected_episodes=2,
                expected_ticks=140,
                store_metadata={
                    "native_teacher_forced_profile": {
                        "name": "royaleapi_native_teacher_forced",
                        "version": 1,
                    }
                },
            )
            self.assertEqual(
                global_manifest["metadata"]["native_teacher_forced_profile"],
                {"name": "royaleapi_native_teacher_forced", "version": 1},
            )

    def test_lease_expiry_allows_work_stealing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.sqlite3"
            with TickStoreWorkQueue(path) as queue:
                queue.add_tasks([
                    {"battle_tag": "A", "source_path": "a.json", "source_sha256": "aa"},
                    {"battle_tag": "B", "source_path": "b.json", "source_sha256": "bb"},
                ])
                first = queue.claim("worker-1", limit=1, lease_seconds=100)
                self.assertEqual(len(first), 1)
                queue.connection.execute(
                    "UPDATE tasks SET lease_until=0 WHERE battle_tag=?",
                    (first[0].battle_tag,),
                )
                stolen = queue.claim("worker-2", limit=2, lease_seconds=100)
                self.assertEqual({item.battle_tag for item in stolen}, {"A", "B"})
                self.assertTrue(all(item.attempts >= 1 for item in stolen))


if __name__ == "__main__":
    unittest.main()
