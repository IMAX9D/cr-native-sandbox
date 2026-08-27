from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import expert_v1.tick_store_v1.deployment_masks as deployment_masks_module

from expert_v1.native_dataset_generator import (
    StagedTickSink,
    StoredFrameRegistry,
    verify_published_tick_store,
)
from expert_v1.tick_store_v1.deployment_masks import (
    EPISODE_METADATA_KEY,
    DeploymentMaskContractError,
    DeploymentMaskStore,
    NativeDeploymentMaskCapture,
    deployment_label_is_legal,
    derive_deployment_rows,
    normalize_native_probe,
    resolve_deployment_reference,
    validate_episode_mask_metadata,
    verify_deployment_labels,
)
from expert_v1.tick_store_v1.schema import (
    EpisodeState,
    PlayerPrivate,
    TickState,
    TowerState,
)
from expert_v1.tick_store_v1.shard import (
    ShardReader,
    WorkerShardSink,
    build_store_manifest,
    sha256_file,
)
from training.schema import deployment_mask


def probe(rows: list[str] | None = None) -> dict:
    rows = rows or ["1" * 18 for _ in range(32)]
    return {
        "width": 18,
        "height": 32,
        "cell_size": 1000,
        "valid_cells": sum(row.count("1") for row in rows),
        "resolved_data_id": 26_000_000,
        "packed_selection": 3 << 28,
        "card_cost": 3,
        "card_cost_raw": 30_000,
        "selection_form_index": -1,
        "selection_strategy": "canonical",
        "selection_builder_rva": "0xd5b770",
        "selection_root_vtable_rva": "0x1234",
        "rows": rows,
    }


def deck_slots() -> list[dict]:
    return [
        {
            "side": side,
            "deck_index": deck_index,
            "card_id": 26_000_000 + deck_index,
            "level": 11,
            "form_flags": 0,
            "source_token": f"card-{deck_index}",
            "base_token": f"card-{deck_index}",
        }
        for side in (0, 1)
        for deck_index in range(8)
    ]


def state(tick: int, hand: list[int]) -> dict:
    return {
        "tick": tick,
        "players": [
            {"side": side, "hand_deck_indices": list(hand)}
            for side in (0, 1)
        ],
    }


class ProbeEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        self.calls.append((side, deck_index))
        value = probe()
        value["resolved_data_id"] = 26_000_000 + deck_index
        return value


class DynamicProbeEnv(ProbeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.high_elixir = False

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        self.calls.append((side, deck_index))
        value = probe()
        value["resolved_data_id"] = 26_000_000 + deck_index
        if deck_index == 0:
            value.update({
                "resolved_data_id": (
                    26_000_105 if self.high_elixir else 26_000_104
                ),
                "card_cost": 6 if self.high_elixir else 3,
                "card_cost_raw": 60_000 if self.high_elixir else 30_000,
                "selection_form_index": 1 if self.high_elixir else 0,
                "selection_strategy": "native_dynamic_choice",
                "selection_builder_rva": "0xd71800",
            })
        return value


class MirrorProbeEnv(ProbeEnv):
    def __init__(self, slots: list[dict]) -> None:
        super().__init__()
        self.card_ids = {
            (int(slot["side"]), int(slot["deck_index"])): int(slot["card_id"])
            for slot in slots
        }
        self.mirror_resolved_data_id = 26_000_000
        self.mirror_cost = 4

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        self.calls.append((side, deck_index))
        value = probe()
        if (side, deck_index) == (0, 0):
            value["resolved_data_id"] = self.mirror_resolved_data_id
            value["card_cost"] = self.mirror_cost
            value["card_cost_raw"] = self.mirror_cost * 10_000
        else:
            value["resolved_data_id"] = self.card_ids[(side, deck_index)]
        # Mirror remains on libg's canonical selector path.
        value["selection_strategy"] = "canonical"
        return value


def tower_mapping(*, destroy_enemy_left: bool = False) -> dict:
    towers = [
        (0, "king", 9_000, 3_000, 4_824, 5_000_000),
        (0, "princess", 3_500, 6_500, 3_052, 5_000_001),
        (0, "princess", 14_500, 6_500, 3_052, 5_000_002),
        (1, "king", 9_000, 29_000, 4_824, 5_000_003),
        (
            1, "princess", 3_500, 25_500,
            0 if destroy_enemy_left else 3_052, 5_000_004,
        ),
        (1, "princess", 14_500, 25_500, 3_052, 5_000_005),
    ]
    return {
        "entities": [
            {
                "side": side, "category": category, "card_id": -1,
                "x": x, "y": y, "hp": hp,
            }
            for side, _role, x, y, hp, category in towers
        ],
        "episode": {
            "crown_towers": [
                {
                    "side": side, "type": role, "x": x, "y": y,
                    "hp": hp, "max_hp": 4_824 if role == "king" else 3_052,
                }
                for side, role, x, y, hp, _category in towers
            ]
        },
    }


def tick_state(tick: int = 100) -> TickState:
    towers = (
        TowerState(0, 0, 0, -1, 9_000, 3_000, 4_824, 4_824),
        TowerState(1, 0, 1, 0, 3_500, 6_500, 3_052, 3_052),
        TowerState(2, 0, 1, 1, 14_500, 6_500, 3_052, 3_052),
        TowerState(3, 1, 0, -1, 9_000, 29_000, 4_824, 4_824),
        TowerState(4, 1, 1, 0, 3_500, 25_500, 3_052, 3_052),
        TowerState(5, 1, 1, 1, 14_500, 25_500, 3_052, 3_052),
    )
    return TickState(
        tick=tick,
        players=(
            PlayerPrivate(0, 100_000, (0, 1, 2, 3), 4),
            PlayerPrivate(1, 100_000, (0, 1, 2, 3), 4),
        ),
        towers=towers,
        entities=(),
        episode=EpisodeState(1, 0, 1, 0, 0, 0, 0, 0, 0),
    )


class ExpertDeploymentMaskTests(unittest.TestCase):
    def test_each_side_deck_slot_is_probed_once_not_per_tick(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = ProbeEnv()
        self.assertEqual(capture.capture_available(env, state(10, [0, 1, 2, 3])), 8)
        for tick in range(11, 100):
            self.assertEqual(
                capture.capture_available(env, state(tick, [0, 1, 2, 3])), 0
            )
        self.assertEqual(capture.capture_available(env, state(100, [4, 5, 6, 7])), 8)
        self.assertTrue(capture.complete)
        self.assertEqual(capture.probe_rpc_count, 16)
        self.assertEqual(len(env.calls), 16)
        self.assertEqual(len(set(env.calls)), 16)
        metadata = capture.metadata()
        self.assertEqual(metadata["captured_slots"], 16)
        # Two sides share one payload per resolved card, while all 16 native
        # deck slots remain independently validated.
        self.assertEqual(len(capture.payloads), 8)

    def test_incomplete_or_malformed_capture_fails_closed(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        capture.capture_available(ProbeEnv(), state(10, [0, 1, 2, 3]))
        with self.assertRaisesRegex(
            DeploymentMaskContractError, "capture is incomplete"
        ):
            capture.metadata()
        broken = probe()
        broken["valid_cells"] -= 1
        with self.assertRaisesRegex(
            DeploymentMaskContractError, "valid_cells disagrees"
        ):
            normalize_native_probe(broken)

    def test_transient_refill_sentinel_does_not_abort_mask_capture(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = ProbeEnv()
        transient = state(10, [0, 1, -1, 3])
        self.assertEqual(capture.capture_available(env, transient), 6)
        self.assertEqual(set(env.calls), {
            (side, deck_index)
            for side in (0, 1)
            for deck_index in (0, 1, 3)
        })
        # The actor-side label validator also ignores the other side's pending
        # refill sentinel while retaining exact membership checks.
        capture.capture_label_variants(
            env, transient, [{"side": 0, "deck_index": 0}]
        )
        for invalid in (
            [0, 0, -1, 3],
            [0, 1, -2, 3],
            [-1, -1, -1, -1],
        ):
            with self.assertRaisesRegex(
                DeploymentMaskContractError, "deck indices are invalid"
            ):
                NativeDeploymentMaskCapture(deck_slots()).capture_available(
                    ProbeEnv(), state(11, invalid)
                )

    def test_dynamic_choice_is_reprobed_only_at_exact_expert_play_tick(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = DynamicProbeEnv()
        capture.capture_available(env, state(10, [0, 1, 2, 3]))
        capture.capture_available(env, state(20, [4, 5, 6, 7]))
        self.assertEqual(capture.probe_rpc_count, 16)
        env.high_elixir = True
        calls = capture.capture_label_variants(
            env,
            state(100, [0, 1, 2, 3]),
            [
                {"side": 0, "deck_index": 1},
                {"side": 1, "deck_index": 2},
            ],
        )
        self.assertEqual(calls, 2)
        self.assertEqual(capture.probe_rpc_count, 18)
        self.assertEqual(capture.dynamic_label_probe_rpc_count, 2)
        # Re-validating the same label Tick is cache-only.
        self.assertEqual(
            capture.capture_label_variants(
                env,
                state(100, [0, 1, 2, 3]),
                [{"side": 0, "deck_index": 3}],
            ),
            0,
        )
        metadata = capture.metadata()
        self.assertEqual(metadata["base_probe_rpc_count"], 16)
        self.assertEqual(metadata["probe_rpc_count"], 18)
        normalized = validate_episode_mask_metadata(metadata)
        entry = next(
            item for item in normalized["entries"]
            if item["side"] == 0 and item["deck_index"] == 0
        )
        self.assertEqual(
            capture.payloads[entry["content_sha256"]]["card_cost"], 3
        )
        self.assertEqual(entry["dynamic_label_variants"][0]["tick"], 100)
        self.assertEqual(
            capture.payloads[
                entry["dynamic_label_variants"][0]["content_sha256"]
            ]["card_cost"],
            6,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = DeploymentMaskStore(Path(directory))
            store.publish_many(capture.payloads)
            episode_metadata = {EPISODE_METADATA_KEY: metadata}
            store.verify_episode_metadata(episode_metadata)
            base_state = tick_state(100)
            low_elixir_state = TickState(
                tick=base_state.tick,
                players=(
                    PlayerPrivate(0, 50_000, (0, 1, 2, 3), 4),
                    base_state.players[1],
                ),
                towers=base_state.towers,
                entities=base_state.entities,
                episode=base_state.episode,
            )
            audit = verify_deployment_labels(
                [low_elixir_state], episode_metadata, store,
                [{
                    "tick": 100, "side": 0, "deck_index": 0,
                    "x": 500, "y": 500,
                }],
            )
            self.assertEqual(
                audit["violations"][0]["reasons"],
                ["insufficient_native_elixir"],
            )

    def test_mirror_uses_exact_tick_unit_building_spell_selection_and_cost(self) -> None:
        slots = deck_slots()
        slots[0] = {
            **slots[0],
            "card_id": 28_000_006,
            "source_token": "mirror",
            "base_token": "mirror",
        }
        capture = NativeDeploymentMaskCapture(slots)
        env = MirrorProbeEnv(slots)
        capture.capture_available(env, state(10, [0, 1, 2, 3]))
        capture.capture_available(env, state(20, [4, 5, 6, 7]))
        for tick, resolved_data_id, cost in (
            (100, 26_000_000, 4),  # mirrored Knight
            (200, 27_000_000, 4),  # mirrored Cannon
            (300, 28_000_011, 3),  # mirrored Log
            (400, 26_000_032, 3),  # mirrored Miner (global deploy)
        ):
            env.mirror_resolved_data_id = resolved_data_id
            env.mirror_cost = cost
            capture.capture_label_variants(
                env,
                state(tick, [0, 1, 2, 3]),
                [{"side": 0, "deck_index": 1}],
            )
        self.assertEqual(capture.probe_rpc_count, 20)
        self.assertEqual(capture.dynamic_label_probe_rpc_count, 4)
        normalized = validate_episode_mask_metadata(capture.metadata())
        entry = next(
            item for item in normalized["entries"]
            if item["side"] == 0 and item["deck_index"] == 0
        )
        self.assertEqual(entry["card_id"], 28_000_006)
        self.assertEqual(entry["native_selection_strategy"], "canonical")
        self.assertTrue(entry["tick_variant_required"])
        self.assertEqual(
            [variant["tick"] for variant in entry["dynamic_label_variants"]],
            [100, 200, 300, 400],
        )
        with tempfile.TemporaryDirectory() as directory:
            store = DeploymentMaskStore(Path(directory))
            store.publish_many(capture.payloads)
            store.verify_episode_metadata(
                {EPISODE_METADATA_KEY: capture.metadata()}
            )
            expected = {
                100: (26_000_000, 4, False),
                200: (27_000_000, 4, False),
                300: (28_000_011, 3, True),
                400: (26_000_032, 3, True),
            }
            for tick, (resolved_data_id, cost, enemy_half_legal) in expected.items():
                reference = resolve_deployment_reference(entry, tick=tick)
                self.assertIsNotNone(reference)
                payload = store.load(str(reference["content_sha256"]))  # type: ignore[index]
                self.assertEqual(payload["resolved_data_id"], resolved_data_id)
                self.assertEqual(payload["card_cost"], cost)
                rows = derive_deployment_rows(
                    payload,
                    tower_mapping(),
                    side=0,
                    card_id=28_000_006,
                )
                self.assertEqual(rows[25][0] == "1", enemy_half_legal)
            exact_states = []
            for tick, elixir_raw in (
                (100, 40_000), (200, 30_000),
                (300, 30_000), (400, 30_000),
            ):
                base_state = tick_state(tick)
                exact_states.append(TickState(
                    tick=tick,
                    players=(
                        PlayerPrivate(0, elixir_raw, (0, 1, 2, 3), 4),
                        base_state.players[1],
                    ),
                    towers=base_state.towers,
                    entities=base_state.entities,
                    episode=base_state.episode,
                ))
            cost_audit = verify_deployment_labels(
                exact_states,
                {EPISODE_METADATA_KEY: capture.metadata()},
                store,
                [
                    {"tick": 100, "side": 0, "deck_index": 0, "x": 500, "y": 500},
                    {"tick": 200, "side": 0, "deck_index": 0, "x": 500, "y": 500},
                    {"tick": 300, "side": 0, "deck_index": 0, "x": 500, "y": 25_500},
                    {"tick": 400, "side": 0, "deck_index": 0, "x": 500, "y": 25_500},
                ],
            )
            self.assertEqual(cost_audit["legal"], 3)
            self.assertEqual(cost_audit["violations"][0]["tick"], 200)
            self.assertEqual(
                cost_audit["violations"][0]["reasons"],
                ["insufficient_native_elixir"],
            )
            with self.assertRaisesRegex(
                DeploymentMaskContractError, "lacks one exact",
            ):
                resolve_deployment_reference(entry, tick=500)

    def test_offline_dynamic_projection_is_bit_exact_with_online_cache_rule(self) -> None:
        sidecar = normalize_native_probe(probe())
        for destroyed in (False, True):
            native_state = tower_mapping(destroy_enemy_left=destroyed)
            for side in (0, 1):
                expected = deployment_mask(
                    list(sidecar["rows"]), native_state,
                    side=side, card_id=26_000_000,
                )
                actual = derive_deployment_rows(
                    sidecar, native_state, side=side, card_id=26_000_000
                )
                self.assertEqual(actual, tuple(expected))
        spell_probe = probe()
        spell_probe["resolved_data_id"] = 28_000_001
        spell_sidecar = normalize_native_probe(spell_probe)
        spell = derive_deployment_rows(
            spell_sidecar, tower_mapping(), side=0, card_id=28_000_001
        )
        self.assertEqual(spell, tuple(spell_sidecar["rows"]))

    def test_spell_base_namespace_survives_hero_form_resolution(self) -> None:
        barrel_probe = probe()
        barrel_probe["resolved_data_id"] = 203_000_107
        barrel_sidecar = normalize_native_probe(barrel_probe)
        barrel = derive_deployment_rows(
            barrel_sidecar,
            tower_mapping(),
            side=0,
            card_id=28_000_015,
        )
        self.assertEqual(barrel, tuple(barrel_sidecar["rows"]))
        self.assertEqual(sum(row.count("1") for row in barrel), 576)

        troop_sidecar = normalize_native_probe(probe())
        troop = derive_deployment_rows(
            troop_sidecar,
            tower_mapping(),
            side=0,
            card_id=26_000_000,
        )
        self.assertNotEqual(troop, tuple(troop_sidecar["rows"]))
        self.assertLess(sum(row.count("1") for row in troop), 576)

    def test_smoke_regressions_global_miner_and_destroyed_tower_edge(self) -> None:
        sidecar = normalize_native_probe(probe())
        miner = derive_deployment_rows(
            sidecar, tower_mapping(), side=0, card_id=26_000_032
        )
        self.assertTrue(
            deployment_label_is_legal(miner, x=7_500, y=17_500),
            "Miner is a native global-deploy troop",
        )
        drill = derive_deployment_rows(
            sidecar, tower_mapping(), side=0, card_id=27_000_013
        )
        self.assertTrue(deployment_label_is_legal(drill, x=7_500, y=17_500))
        ordinary = derive_deployment_rows(
            sidecar, tower_mapping(), side=0, card_id=26_000_000
        )
        self.assertFalse(
            deployment_label_is_legal(ordinary, x=7_500, y=17_500),
            "ordinary troops remain ownership-limited",
        )
        skeletons = derive_deployment_rows(
            sidecar,
            tower_mapping(destroy_enemy_left=True),
            side=0,
            card_id=26_000_010,
        )
        self.assertTrue(
            deployment_label_is_legal(skeletons, x=3_500, y=16_501),
            "destroyed-left-tower pocket includes native river-edge row",
        )
        self.assertFalse(
            deployment_label_is_legal(skeletons, x=13_500, y=16_501),
            "destroying the left tower does not unlock the right pocket",
        )
        mirrored = derive_deployment_rows(
            sidecar,
            tower_mapping(destroy_enemy_left=True),
            side=1,
            card_id=26_000_010,
        )
        self.assertFalse(
            deployment_label_is_legal(mirrored, x=3_500, y=15_499),
            "side-1 cannot use a pocket opened by destruction on its own side",
        )

    def test_sidecar_is_content_addressed_and_offline_labels_are_auditable(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = ProbeEnv()
        capture.capture_available(env, state(10, [0, 1, 2, 3]))
        capture.capture_available(env, state(20, [4, 5, 6, 7]))
        metadata = {EPISODE_METADATA_KEY: capture.metadata()}
        with tempfile.TemporaryDirectory() as directory:
            store = DeploymentMaskStore(Path(directory))
            published = store.publish_many(capture.payloads)
            self.assertEqual(len(published), 8)
            store.verify_episode_metadata(metadata)
            manifest = store.build_manifest()
            self.assertEqual(manifest["sidecars"], 8)
            store.verify_manifest()
            audit = verify_deployment_labels(
                [tick_state()], metadata, store,
                [
                    {"tick": 100, "side": 0, "deck_index": 0, "x": 500, "y": 500},
                    {"tick": 100, "side": 0, "deck_index": 0, "x": 500, "y": 20_500},
                ],
            )
            self.assertEqual(audit["checked"], 2)
            self.assertEqual(audit["legal"], 1)
            self.assertFalse(audit["all_legal"])
            digest = next(iter(published))
            path = store.path_for(digest)
            raw = bytearray(path.read_bytes())
            raw[-2] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(
                DeploymentMaskContractError, "sidecar SHA changed"
            ):
                store.load(digest)

    def test_sidecars_commit_before_episode_and_survive_frame_readback(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = ProbeEnv()
        capture.capture_available(env, state(10, [0, 1, 2, 3]))
        capture.capture_available(env, state(20, [4, 5, 6, 7]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = StagedTickSink()
            staged.stage_deployment_masks(capture)
            staged.append(
                "MASK-COMMIT",
                [tick_state(100), tick_state(101)],
                {EPISODE_METADATA_KEY: capture.metadata()},
            )
            sink = WorkerShardSink(root, "worker", episodes_per_shard=1)
            registry = StoredFrameRegistry(root)
            entry = registry.commit_or_reuse(sink, staged.episode)  # type: ignore[arg-type]
            sink.finalize()
            self.assertEqual(entry["ticks"], 2)
            data = next(root.glob("worker-*.crts"))
            with ShardReader(data) as reader:
                episode = reader.episode("MASK-COMMIT")
                verified = registry.deployment_mask_store.verify_episode_metadata(
                    episode.metadata
                )
            self.assertEqual(len(verified["entries"]), 16)
            mask_manifest = registry.deployment_mask_store.build_manifest()
            selection = root / "selection.jsonl"
            selection.write_text("{}\n", encoding="utf-8")
            mask_manifest_path = (
                root / "deployment-masks-v1" / "manifest.json"
            )
            build_store_manifest(
                root,
                source_manifest=selection,
                expected_episodes=1,
                expected_ticks=2,
                store_metadata={
                    "native_deployment_masks": {
                        "required": True,
                        "manifest": "deployment-masks-v1/manifest.json",
                        "manifest_sha256": sha256_file(mask_manifest_path),
                        "sidecars": mask_manifest["sidecars"],
                    }
                },
            )
            physical = verify_published_tick_store(root)
            self.assertEqual(physical["episodes"], 1)
            self.assertEqual(physical["deployment_mask_sidecars_referenced"], 8)

    def test_process_cache_authenticates_digest_once_and_derive_is_bit_exact(self) -> None:
        capture = NativeDeploymentMaskCapture(deck_slots())
        env = ProbeEnv()
        capture.capture_available(env, state(10, [0, 1, 2, 3]))
        capture.capture_available(env, state(20, [4, 5, 6, 7]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = DeploymentMaskStore(root)
            published = publisher.publish_many(capture.payloads)
            digest = next(iter(published))
            process_key = (str(publisher.root), digest)
            with deployment_masks_module._PROCESS_CACHE_LOCK:
                deployment_masks_module._PROCESS_PAYLOAD_CACHE.pop(process_key, None)
                stale = [
                    key
                    for key in deployment_masks_module._PROCESS_DERIVED_CACHE
                    if key[:2] == process_key
                ]
                for key in stale:
                    deployment_masks_module._PROCESS_DERIVED_CACHE.pop(key, None)

            original_read_bytes = Path.read_bytes
            reads = 0

            def counted(path: Path) -> bytes:
                nonlocal reads
                if path == publisher.path_for(digest):
                    reads += 1
                return original_read_bytes(path)

            first = DeploymentMaskStore(root, create=False)
            second = DeploymentMaskStore(root, create=False)
            with patch.object(Path, "read_bytes", counted):
                payload = first.load(digest, allow_cached=True)
                self.assertEqual(second.load(digest, allow_cached=True), payload)
                cached_rows = first.derive(
                    digest, tick_state(), side=0, card_id=28_000_000
                )
                self.assertIs(
                    cached_rows,
                    second.derive(
                        digest, tick_state(), side=0, card_id=28_000_000
                    ),
                )
            self.assertEqual(reads, 1)
            direct_rows = derive_deployment_rows(
                payload, tick_state(), side=0, card_id=28_000_000
            )
            self.assertEqual(cached_rows, direct_rows)


if __name__ == "__main__":
    unittest.main()
