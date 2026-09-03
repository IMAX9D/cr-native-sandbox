from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import torch

from expert_selfplay_v1.native_observation import (
    EncodedNativeBatch,
    NativeActorFrame,
    NativeObservationContractError,
    NativeObservationEncoder,
    RevealedEnemyTracker,
    native_id_token_map,
)
from expert_v1.compile_native_bc_dataset import _grid, _public_scalars
from expert_v1.tick_store_v1.schema import actor_projection, normalize_native_state
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy


HERO_KNIGHT = 203_000_000
CARD_IDS = (
    HERO_KNIGHT,
    26_000_001,
    26_000_002,
    26_000_003,
    26_000_004,
    26_000_010,
    26_000_011,
    26_000_074,
)
CARD_MAP = {card_id: index + 1 for index, card_id in enumerate(CARD_IDS)}
ABILITY_MAP = {HERO_KNIGHT: 1, 26_000_074: 2}
DECK = (
    {"card_id": 26_000_000, "form_flags": 2},
    *({"card_id": card_id} for card_id in CARD_IDS[1:]),
)


def _tower(
    side: int,
    role: str,
    lane: str | None,
    x: int,
    y: int,
    hp: int,
) -> dict:
    result = {
        "side": side,
        "type": role,
        "x": x,
        "y": y,
        "hp": hp,
        "max_hp": 4000 if role == "king" else 3000,
    }
    if lane is not None:
        result["lane"] = lane
    return result


def state() -> dict:
    return {
        "schema_version": 1,
        "kind": "libg_native_train_state_v1",
        "coherent": True,
        "tick": 1200,
        "entity_count": 2,
        "players": [
            {
                "side": 0,
                "elixir_raw": 70_000,
                "hand_deck_indices": [0, 1, 2, 3],
                "next_deck_index": 4,
                "refill_timer": 0,
            },
            {
                "side": 1,
                "elixir_raw": 40_000,
                "hand_deck_indices": [4, 5, 6, 7],
                "next_deck_index": 0,
                "refill_timer": 0,
            },
        ],
        "entities": [
            {
                "category": 5_000_001,
                "side": 0,
                "x": 4000,
                "y": 8000,
                "card_id": HERO_KNIGHT,
                "level": 12,
                "hp": 1000,
                "max_hp": 2000,
                "behavior_state": 1,
                "ability_slot": 1,
                "ability_state_code": 0,
                "ability_available": True,
                "ability_cooldown_remaining_ms": 0,
                "ability_charges_remaining": 1,
                "ability_pending_ms": 0,
                "ability_mana_cost": 2,
            },
            {
                "category": 5_000_002,
                "side": 1,
                "x": 13_999,
                "y": 23_999,
                "card_id": 26_000_003,
                "level": 10,
                "hp": 1500,
                "max_hp": 3000,
                "behavior_state": 2,
                "ability_slot": 0,
                "ability_state_code": -1,
                "ability_available": False,
                "ability_cooldown_remaining_ms": -1,
                "ability_charges_remaining": -1,
                "ability_pending_ms": -1,
                "ability_mana_cost": -1,
            },
        ],
        "episode": {
            "terminated": False,
            "crowns": [1, 0],
            "commands_allowed": True,
            "command_gate_code": 0,
            "native_phase": {
                "battle": 1,
                "logic": 0,
                "logic_substate": 0,
                "flag_1e9": 0,
            },
            "crown_towers": [
                _tower(0, "king", None, 9000, 3000, 3600),
                _tower(0, "princess", "left", 3500, 6500, 2500),
                _tower(0, "princess", "right", 14_500, 6500, 3000),
                _tower(1, "king", None, 8999, 28_999, 4000),
                _tower(1, "princess", "left", 3500, 25_499, 2800),
                _tower(1, "princess", "right", 14_499, 25_499, 2700),
            ],
        },
    }


def encoder(*, require_scene: bool = True) -> NativeObservationEncoder:
    return NativeObservationEncoder(
        card_id_to_token=CARD_MAP,
        ability_id_to_token=ABILITY_MAP,
        max_ability_slots=4,
        card_vocab_size=9,
        ability_vocab_size=3,
        require_nonempty_public_scene=require_scene,
    )


class NativeObservationEncoderTest(unittest.TestCase):
    def test_one_frame_has_exact_hand_next_deck_and_public_scene(self) -> None:
        batch = encoder().encode_one(
            state(), actor_side=0, own_deck=DECK,
            revealed_enemy_card_ids=[26_000_003], delta_ticks=2,
        )
        self.assertIsInstance(batch, EncodedNativeBatch)
        self.assertEqual(tuple(batch["grid"].shape), (1, 1, 8, 32, 18))
        self.assertEqual(batch["own_deck_tokens"][0, 0].tolist(), list(range(1, 9)))
        self.assertEqual(batch["hand_tokens"][0, 0].tolist(), [1, 2, 3, 4])
        self.assertEqual(int(batch["next_card_token"][0, 0]), 5)
        self.assertEqual(batch["revealed_enemy_tokens"][0, 0].tolist(), [4, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(float(batch["delta_ticks"][0, 0]), 2.0)
        self.assertGreater(int(torch.count_nonzero(batch["grid"])), 0)
        self.assertEqual(batch.native_entity_counts, (2,))
        self.assertEqual(batch.encoded_entity_counts, (2,))

    def test_encoding_matches_training_compiler_grid_and_scalars(self) -> None:
        raw = state()
        normalized = normalize_native_state(raw)
        actor = actor_projection(normalized, actor_side=0)
        batch = encoder().encode_one(raw, actor_side=0, own_deck=DECK)
        np.testing.assert_array_equal(
            np.rint(batch["grid"][0, 0].numpy() * 255).astype(np.uint8),
            _grid(actor),
        )
        np.testing.assert_array_equal(
            batch["public_scalars"][0, 0].numpy(),
            _public_scalars(actor, normalized),
        )

    def test_both_sides_are_canonical_and_private_opponent_player_is_absent(self) -> None:
        original = state()
        changed = deepcopy(original)
        changed["players"][1].update(
            elixir_raw=100_000,
            hand_deck_indices=[0, 2, 4, 6],
            next_deck_index=1,
        )
        side0 = encoder().encode_one(original, actor_side=0, own_deck=DECK)
        side0_changed = encoder().encode_one(changed, actor_side=0, own_deck=DECK)
        for name in side0:
            self.assertTrue(torch.equal(side0[name], side0_changed[name]), name)

        side1 = encoder().encode_one(original, actor_side=1, own_deck=DECK)
        self.assertEqual(side1["hand_tokens"][0, 0].tolist(), [5, 6, 7, 8])
        self.assertEqual(int(side1["next_card_token"][0, 0]), 1)
        # Side-1's entity at (13999,23999) rotates to the same canonical cell
        # as side-0's entity at (4000,8000).
        self.assertEqual(
            int(side1["entity_positions"][0, 0, 1]),
            int(side0["entity_positions"][0, 0, 0]),
        )
        self.assertEqual(side1["entity_relations"][0, 0].tolist(), [1, 0])

    def test_batch_pads_ragged_entities_and_runs_actor_directly(self) -> None:
        second = deepcopy(state())
        second["entities"] = second["entities"][:1]
        second["entity_count"] = 1
        batch = encoder().encode_batch(
            [
                NativeActorFrame(state(), 0, DECK),
                NativeActorFrame(second, 1, DECK, delta_ticks=3),
            ]
        )
        self.assertEqual(tuple(batch["entity_tokens"].shape), (2, 1, 2))
        self.assertEqual(batch["entity_mask"][:, 0].tolist(), [[True, True], [True, False]])
        config = ExpertPolicyConfig(
            grid_channels=8,
            public_scalar_size=16,
            card_vocab_size=9,
            ability_vocab_size=3,
            max_ability_slots=4,
            hidden_size=32,
            card_embedding_size=16,
            spatial_size=16,
        )
        model = RecurrentExpertPolicy(config)
        encoder().assert_compatible(config)
        with torch.inference_mode():
            output = model.forward_sequence(**batch)
        self.assertEqual(tuple(output.rate_logits.shape), (2, 1))
        self.assertEqual(tuple(output.position_logits.shape), (2, 1, 4, 576))
        self.assertTrue(torch.isfinite(output.rate_logits).all())

    def test_ability_tokens_keys_and_legality_are_actor_side_specific(self) -> None:
        raw = state()
        batch = encoder().encode_batch(
            [
                NativeActorFrame(raw, 0, DECK),
                NativeActorFrame(raw, 1, DECK),
            ]
        )
        self.assertEqual(batch.ability_entity_keys, ((5_000_001,), ()))
        self.assertEqual(int(batch["ability_tokens"][0, 0, 0]), 1)
        self.assertTrue(bool(batch.ability_mask[0, 0, 0]))
        self.assertFalse(bool(batch.ability_mask[1].any()))

        blocked = deepcopy(raw)
        blocked["episode"]["commands_allowed"] = False
        encoded = encoder().encode_one(blocked, actor_side=0, own_deck=DECK)
        self.assertEqual(encoded.ability_entity_keys, ((5_000_001,),))
        self.assertFalse(bool(encoded.ability_mask.any()))

    def test_native_nonempty_entity_cannot_be_silently_dropped(self) -> None:
        raw = state()
        raw["entities"][0]["card_id"] = 999_999_999
        with self.assertRaisesRegex(
            NativeObservationContractError, "outside frozen card vocabulary"
        ):
            encoder().encode_one(raw, actor_side=0, own_deck=DECK)

        contradictory = state()
        contradictory["entities"] = []
        contradictory["entity_count"] = 2
        with self.assertRaisesRegex(
            NativeObservationContractError, "reports dynamic entities"
        ):
            encoder().encode_one(contradictory, actor_side=0, own_deck=DECK)

    def test_tower_only_opening_is_valid_but_empty_public_scene_is_not(self) -> None:
        opening = state()
        opening["entities"] = []
        opening["entity_count"] = 0
        batch = encoder().encode_one(opening, actor_side=0, own_deck=DECK)
        self.assertEqual(tuple(batch["entity_tokens"].shape), (1, 1, 1))
        self.assertFalse(bool(batch["entity_mask"].any()))
        self.assertGreater(int(torch.count_nonzero(batch["grid"])), 0)

        opening["episode"]["crown_towers"] = []
        with self.assertRaisesRegex(NativeObservationContractError, "empty public"):
            encoder().encode_one(opening, actor_side=0, own_deck=DECK)

    def test_revealed_enemy_tracker_uses_only_public_plays(self) -> None:
        tracker = RevealedEnemyTracker(encoder())
        tracker.record_play(played_side=1, card={"card_id": 26_000_003})
        tracker.record_play(played_side=1, card={"card_id": 26_000_003})
        tracker.record_play(
            played_side=0, card={"card_id": 26_000_000, "form_flags": 2}
        )
        self.assertEqual(tracker.tokens_for(0), (4,))
        self.assertEqual(tracker.tokens_for(1), (1,))
        tracker.reset()
        self.assertEqual(tracker.tokens_for(0), ())

    def test_manifest_factory_schema_hash_and_config_guard(self) -> None:
        card_vocabulary = ["<PAD>"] + [
            f"card-{index}@{card_id}" for index, card_id in enumerate(CARD_IDS)
        ]
        ability_vocabulary = [
            "<PAD>", f"hero-knight@{HERO_KNIGHT}", "golden-knight@26000074"
        ]
        value = NativeObservationEncoder.from_manifest(
            {
                "dimensions": {
                    "grid_channels": 8,
                    "public_scalar_size": 16,
                    "entity_numeric_size": 3,
                    "card_vocab_size": len(card_vocabulary),
                    "ability_vocab_size": len(ability_vocabulary),
                    "max_ability_slots": 4,
                },
                "card_vocabulary": card_vocabulary,
                "ability_vocabulary": ability_vocabulary,
            }
        )
        self.assertEqual(value.card_id_to_token, CARD_MAP)
        self.assertEqual(len(value.schema_sha256()), 64)
        self.assertEqual(
            native_id_token_map(card_vocabulary, name="card"), CARD_MAP
        )
        bad = ExpertPolicyConfig(
            grid_channels=8,
            public_scalar_size=16,
            card_vocab_size=10,
            ability_vocab_size=3,
            max_ability_slots=4,
        )
        with self.assertRaisesRegex(NativeObservationContractError, "dimensions differ"):
            value.assert_compatible(bad)

    def test_invalid_deck_and_reveal_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(NativeObservationContractError, "exactly eight"):
            encoder().encode_one(state(), actor_side=0, own_deck=DECK[:-1])
        with self.assertRaisesRegex(NativeObservationContractError, "not both"):
            encoder().encode_one(
                state(), actor_side=0, own_deck=DECK,
                revealed_enemy_card_ids=[26_000_003], revealed_enemy_tokens=[4],
            )


if __name__ == "__main__":
    unittest.main()
