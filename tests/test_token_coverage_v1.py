from __future__ import annotations

from copy import deepcopy
import math
import unittest

from expert_v1.native_ingest_contract import build_native_ingest_contract
from expert_v1.token_coverage_v1 import (
    TokenCoverageError,
    build_adaptive_token_quotas,
    build_token_coverage_receipt,
    canonical_coverage_receipt_bytes,
    coverage_receipt_sha256,
    evaluate_token_coverage,
    freeze_source_token_coverage,
    summarize_success_token_coverage,
    verify_coverage_receipt_sha256,
)


TEAM_DECK = [
    "electro-dragon",
    "barbarian-hut",
    "rune-giant",
    "berserker-hero",
    "archers-ev1",
    "knight-hero",
    "mirror",
    "arrows",
]
OPPONENT_DECK = [
    "little-prince",
    "ice-spirit-ev1",
    "fireball",
    "hog-rider",
    "cannon",
    "skeletons",
    "bats",
    "knight",
]


def _parts(token: str) -> tuple[str, str]:
    if token.endswith("-ev1"):
        return token[:-4], "ev1"
    if token.endswith("-hero"):
        return token[:-5], "hero"
    return token, "base"


def _player(deck: list[str]) -> dict[str, object]:
    cards = []
    for slot, token in enumerate(deck):
        base, form = _parts(token)
        cards.append({
            "slot": slot,
            "slug": token,
            "base_slug": base,
            "form": form,
            "level": 16,
        })
    return {
        "complete": True,
        "full_deck": list(deck),
        "deck_cards": cards,
        "card_levels": {token: 16 for token in deck},
    }


def _battle(tag: str = "COVERAGE001") -> dict[str, object]:
    card_plays = []
    marker = 0
    tick = 100
    for side, deck in (("team", TEAM_DECK), ("opponent", OPPONENT_DECK)):
        for token in deck:
            card_plays.append({
                "side": side,
                "time_raw": tick,
                "marker_index": marker,
                "card": _parts(token)[0],
                "card_base": _parts(token)[0],
                "card_form": token,
            })
            marker += 1
            tick += 1
    ability_plays = []
    for side in ("team", "team", "team", "opponent"):
        ability_plays.append({
            "side": side,
            "time_raw": tick,
            "marker_index": marker,
            "ability_id": None,
            "resolution_status": "unresolved",
        })
        marker += 1
        tick += 1
    return {
        "battle_tag": tag,
        "rounds": [{
            "team": [_player(TEAM_DECK)],
            "opponent": [_player(OPPONENT_DECK)],
        }],
        "team_deck": list(TEAM_DECK),
        "opponent_deck": list(OPPONENT_DECK),
        "card_plays": card_plays,
        "ability_plays": ability_plays,
        "elixir_stats": {
            "team": {"Ability": {"count": 3}},
            "opponent": {"Ability": {"count": 1}},
        },
    }


def _native_ids(contract: dict[str, object]) -> tuple[
    dict[str, int], dict[str, tuple[int, int]]
]:
    cards: dict[str, int] = {}
    for raw in contract["cards"]:  # type: ignore[index]
        row = dict(raw)
        base_id = int(row["card_id"])
        for token in row["allowed_tokens"]:
            _, form = _parts(token)
            if form == "ev1":
                cards[token] = int(row["evolution"]["native_form_id"])
            elif form == "hero":
                cards[token] = int(row["hero"]["native_form_id"])
            else:
                cards[token] = base_id
    abilities = {
        str(row["token"]): (int(row["base_card_id"]), int(row["native_form_id"]))
        for row in contract["ability_sources"]  # type: ignore[index]
    }
    return cards, abilities


def _deploy_labels(
    deck: list[str], card_ids: dict[str, int], *, start: int
) -> list[dict[str, object]]:
    result = []
    for offset, token in enumerate(deck):
        row: dict[str, object] = {
            "source_event_index": start + offset,
            "source_token": token,
            "resolved_native_form_id": card_ids[token],
            "accepted": True,
            "mask_legal": True,
            "compiled": True,
        }
        if token == "mirror":
            row["resolved_native_form_id"] = card_ids["arrows"]
            row["identity_provenance"] = "libg_dynamic_choice_exact_v1"
        result.append(row)
    return result


def _ability_label(
    token: str,
    ability_ids: dict[str, tuple[int, int]],
    event_index: int,
    *,
    provenance: str = "libg_live_entity_unique_v1",
) -> dict[str, object]:
    row: dict[str, object] = {
        "source_event_index": event_index,
        "resolved_token": token,
        "resolved_native_form_id": ability_ids[token][1],
        "selected_entity_id": 10_000 + event_index,
        "identity_provenance": provenance,
        "accepted": True,
        "legal": True,
        "compiled": True,
    }
    if provenance == "libg_live_entity_explicit_branch_v1":
        row["branch_verified"] = True
    return row


class TokenCoverageV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_native_ingest_contract()
        cls.card_ids, cls.ability_ids = _native_ids(cls.contract)

    def _successful_records(self) -> list[dict[str, object]]:
        return [
            {
                "battle_tag": "COVERAGE001",
                "actor_side": 0,
                "full_success": True,
                "deck_tokens": list(TEAM_DECK),
                "deploy_labels": _deploy_labels(
                    TEAM_DECK, self.card_ids, start=0
                ),
                "ability_labels": [
                    _ability_label("rune-giant", self.ability_ids, 100),
                    _ability_label("berserker-hero", self.ability_ids, 101),
                    _ability_label("knight-hero", self.ability_ids, 102),
                ],
            },
            {
                "battle_tag": "COVERAGE001",
                "actor_side": 1,
                "full_success": True,
                "deck_tokens": list(OPPONENT_DECK),
                "deploy_labels": _deploy_labels(
                    OPPONENT_DECK, self.card_ids, start=20
                ),
                "ability_labels": [
                    _ability_label("little-prince", self.ability_ids, 120)
                ],
            },
        ]

    def test_freezes_exact_card_form_and_candidate_counts(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        self.assertEqual(
            source["contract_token_counts"],
            {"cards": 180, "evolution": 42, "hero": 16, "ability": 25},
        )
        self.assertEqual(source["card_tokens"]["electro-dragon"]["deck_sides"], 1)
        self.assertEqual(source["card_tokens"]["electro-dragon"]["play_labels"], 1)
        self.assertIn("archers-ev1", source["observed_form_tokens"])
        self.assertIn("knight-hero", source["observed_form_tokens"])

        rune = source["ability_tokens"]["rune-giant"]
        berserker = source["ability_tokens"]["berserker-hero"]
        little = source["ability_tokens"]["little-prince"]
        self.assertEqual(rune["candidate_event_upper_bound"], 3)
        self.assertEqual(rune["singleton_candidate_events"], 0)
        self.assertEqual(berserker["candidate_event_upper_bound"], 3)
        self.assertEqual(little["candidate_event_upper_bound"], 1)
        self.assertEqual(little["singleton_candidate_events"], 1)
        self.assertEqual(rune["identity_semantics"], "candidate_only_not_resolved_identity")

    def test_adaptive_quota_caps_and_rare_floor(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        quotas = build_adaptive_token_quotas(source)
        self.assertEqual(
            quotas["card_tokens"]["electro-dragon"],
            {"full_success_episodes": 1, "deploy_labels": 1},
        )
        modified = deepcopy(source)
        modified["card_tokens"]["electro-dragon"].update({
            "deck_sides": 400,
            "play_labels": 2_000,
        })
        modified["form_tokens"]["archers-ev1"].update({
            "deck_sides": 400,
            "play_labels": 2_000,
        })
        modified["ability_tokens"]["rune-giant"].update({
            "candidate_sides": 400,
            "candidate_event_upper_bound": 2_000,
        })
        quotas = build_adaptive_token_quotas(modified)
        self.assertEqual(
            quotas["card_tokens"]["electro-dragon"],
            {"full_success_episodes": 16, "deploy_labels": 64},
        )
        self.assertEqual(
            quotas["form_tokens"]["archers-ev1"],
            {"resolved_form_episodes": 8, "resolved_form_labels": 16},
        )
        self.assertEqual(
            quotas["ability_tokens"]["rune-giant"],
            {"resolved_ability_episodes": 8, "resolved_ability_labels": 32},
        )

    def test_multi_candidate_never_becomes_offline_identity(self) -> None:
        record = self._successful_records()[0]
        record["ability_labels"] = [{
            "source_event_index": 100,
            "candidate_tokens": ["rune-giant", "berserker-hero"],
            "identity_provenance": "schema5_candidate_association",
            "accepted": True,
            "legal": True,
            "compiled": True,
        }]
        with self.assertRaisesRegex(
            TokenCoverageError, "offline candidate association is not identity"
        ):
            summarize_success_token_coverage([record], self.contract)

        valid = self._successful_records()[0]
        valid["ability_labels"] = [
            _ability_label("rune-giant", self.ability_ids, 100)
        ]
        success = summarize_success_token_coverage([valid], self.contract)
        self.assertEqual(
            success["ability_tokens"]["rune-giant"]["resolved_ability_labels"],
            1,
        )
        self.assertEqual(
            success["ability_tokens"]["berserker-hero"]["resolved_ability_labels"],
            0,
        )

    def test_explicit_branch_must_be_verified(self) -> None:
        record = self._successful_records()[0]
        branch = _ability_label(
            "rune-giant",
            self.ability_ids,
            100,
            provenance="libg_live_entity_explicit_branch_v1",
        )
        del branch["branch_verified"]
        record["ability_labels"] = [branch]
        with self.assertRaisesRegex(TokenCoverageError, "branch is not verified"):
            summarize_success_token_coverage([record], self.contract)

    def test_base_cycle_does_not_forge_evolution_form_coverage(self) -> None:
        record = self._successful_records()[0]
        archers = next(
            row
            for row in record["deploy_labels"]
            if row["source_token"] == "archers-ev1"
        )
        archers_card = next(
            row
            for row in self.contract["cards"]
            if "archers-ev1" in row["allowed_tokens"]
        )
        archers["resolved_native_form_id"] = int(archers_card["card_id"])
        success = summarize_success_token_coverage([record], self.contract)
        self.assertEqual(success["card_tokens"]["archers-ev1"]["deploy_labels"], 1)
        self.assertEqual(
            success["form_tokens"]["archers-ev1"]["resolved_form_labels"], 0
        )

    def test_full_success_receipt_admits_only_resolved_compiled_labels(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        success = summarize_success_token_coverage(
            self._successful_records(), self.contract
        )
        receipt = build_token_coverage_receipt(source, success)
        self.assertTrue(receipt["evaluation"]["gate"]["admitted"])
        self.assertFalse(
            any(receipt["evaluation"]["hard_floor_deficits"].values())
        )

        missing = deepcopy(success)
        missing["ability_tokens"]["rune-giant"].update({
            "resolved_ability_episodes": 0,
            "resolved_ability_labels": 0,
        })
        evaluation = evaluate_token_coverage(source, missing)
        self.assertFalse(evaluation["gate"]["admitted"])
        rune_deficits = [
            row
            for row in evaluation["hard_floor_deficits"]["ability_tokens"]
            if row["token"] == "rune-giant"
        ]
        self.assertEqual(
            {row["metric"] for row in rune_deficits},
            {"resolved_ability_episodes", "resolved_ability_labels"},
        )

    def test_token_removal_quota_lowering_and_count_inflation_fail_closed(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        success = summarize_success_token_coverage(
            self._successful_records(), self.contract
        )

        removed = deepcopy(source)
        removed["observed_card_tokens"].remove("electro-dragon")
        with self.assertRaisesRegex(TokenCoverageError, "observed token sets"):
            build_adaptive_token_quotas(removed)

        quotas = build_adaptive_token_quotas(source)
        quotas["card_tokens"]["electro-dragon"]["deploy_labels"] = 0
        with self.assertRaisesRegex(TokenCoverageError, "quotas were changed"):
            evaluate_token_coverage(source, success, quotas)

        inflated = deepcopy(success)
        inflated["card_tokens"]["electro-dragon"]["deploy_labels"] = 2
        with self.assertRaisesRegex(TokenCoverageError, "exceeds source opportunities"):
            evaluate_token_coverage(source, inflated)

    def test_duplicate_actor_or_event_cannot_inflate_counts(self) -> None:
        record = self._successful_records()[0]
        with self.assertRaisesRegex(TokenCoverageError, "duplicate actor"):
            summarize_success_token_coverage([record, deepcopy(record)], self.contract)
        duplicated = deepcopy(record)
        duplicated["deploy_labels"][1]["source_event_index"] = (
            duplicated["deploy_labels"][0]["source_event_index"]
        )
        with self.assertRaisesRegex(TokenCoverageError, "duplicate successful event"):
            summarize_success_token_coverage([duplicated], self.contract)

    def test_canonical_hash_detects_tampering_and_rejects_nan(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        success = summarize_success_token_coverage(
            self._successful_records(), self.contract
        )
        receipt = build_token_coverage_receipt(source, success)
        digest = coverage_receipt_sha256(receipt)
        self.assertTrue(verify_coverage_receipt_sha256(receipt, digest))

        reordered = {key: receipt[key] for key in reversed(tuple(receipt))}
        self.assertEqual(
            canonical_coverage_receipt_bytes(receipt),
            canonical_coverage_receipt_bytes(reordered),
        )
        tampered = deepcopy(receipt)
        tampered["success"]["totals"]["deploy_labels"] += 1
        self.assertNotEqual(digest, coverage_receipt_sha256(tampered))
        self.assertFalse(verify_coverage_receipt_sha256(tampered, digest))

        invalid = deepcopy(receipt)
        invalid["success"]["totals"]["deploy_labels"] = math.nan
        with self.assertRaisesRegex(TokenCoverageError, "not canonical JSON"):
            canonical_coverage_receipt_bytes(invalid)


if __name__ == "__main__":
    unittest.main()
