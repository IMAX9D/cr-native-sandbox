from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import unittest

from expert_v1.native_ingest_contract import (
    build_native_ingest_contract,
    contract_payload_sha256,
)
from expert_v1.token_coverage_v1 import (
    ABILITY_TRANSCRIPT_KIND,
    TokenCoverageError,
    authenticate_ability_resolution_transcripts,
    authenticate_generator_ability_evidence,
    build_adaptive_token_quotas,
    build_token_coverage_receipt,
    canonical_coverage_receipt_bytes,
    canonical_json_bytes,
    coverage_receipt_sha256,
    evaluate_token_coverage,
    freeze_source_token_coverage,
    seal_ability_resolution_transcript,
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
    transcript: dict[str, object],
) -> dict[str, object]:
    return {
        "source_event_index": transcript["source_event_index"],
        "resolved_token": token,
        "resolved_native_form_id": transcript["selected_native_form_id"],
        "selected_entity_id": transcript["selected_entity_id"],
        "resolution_transcript_sha256": transcript["transcript_sha256"],
        "accepted": True,
        "legal": True,
        "compiled": True,
    }


def _transcript(
    contract: dict[str, object],
    ability_ids: dict[str, tuple[int, int]],
    token: str,
    *,
    actor_side: int,
    source_event_index: int,
    source_marker_index: int,
    source_tick: int,
    entity_id: int,
    candidates: list[tuple[int, int]] | None = None,
    selected_entity_id: int | None = None,
    selected_native_form_id: int | None = None,
) -> dict[str, object]:
    native_form_id = ability_ids[token][1]
    candidate_values = candidates or [(entity_id, ability_ids[token][0])]
    selected = entity_id if selected_entity_id is None else selected_entity_id
    selected_form = (
        native_form_id
        if selected_native_form_id is None
        else selected_native_form_id
    )
    branched = len(candidate_values) > 1
    payload = {
        "schema_version": 1,
        "kind": ABILITY_TRANSCRIPT_KIND,
        "contract_sha256": contract["contract_sha256"],
        "runtime_libg_sha256": contract["runtime"]["libg_sha256"],
        "battle_tag": "COVERAGE001",
        "actor_side": actor_side,
        "source_event_index": source_event_index,
        "source_marker_index": source_marker_index,
        "source_tick": source_tick,
        "execution_tick": source_tick + 1,
        "execution_tick_offset": 1,
        "generator_actor_evidence_sha256": hashlib.sha256(
            f"actor/{actor_side}/{source_event_index}".encode()
        ).hexdigest(),
        "generator_ability_evidence_sha256": hashlib.sha256(
            f"ability/{actor_side}/{source_event_index}".encode()
        ).hexdigest(),
        "resolution_status": "branch_required" if branched else "unique",
        "candidate_entity_ids": [item[0] for item in candidate_values],
        "candidate_card_ids": [item[1] for item in candidate_values],
        "selected_entity_id": selected,
        "selected_native_form_id": selected_form,
        "resolved_token": token,
        "execution": (
            "explicit_branch_executed" if branched else "unique_executed"
        ),
        "branch_verified": branched,
        "action_accepted": True,
    }
    return seal_ability_resolution_transcript(payload)


def _reseal_transcript(
    transcript: dict[str, object], **updates: object
) -> dict[str, object]:
    payload = {
        key: deepcopy(value)
        for key, value in transcript.items()
        if key != "transcript_sha256"
    }
    payload.update(updates)
    return seal_ability_resolution_transcript(payload)


def _rewrite_content_hash(value: dict[str, object], field: str) -> None:
    payload = {key: item for key, item in value.items() if key != field}
    value[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _add_native_evidence_hash(value: dict[str, object]) -> dict[str, object]:
    result = deepcopy(value)
    result["native_evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


class TokenCoverageV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_native_ingest_contract()
        cls.card_ids, cls.ability_ids = _native_ids(cls.contract)

    def _source_and_transcripts(
        self,
    ) -> tuple[dict[str, object], dict[str, object], dict[tuple[int, int], dict[str, object]]]:
        source = freeze_source_token_coverage([_battle()], self.contract)
        rows = [
            _transcript(
                self.contract,
                self.ability_ids,
                "rune-giant",
                actor_side=0,
                source_event_index=0,
                source_marker_index=16,
                source_tick=116,
                entity_id=10_000,
            ),
            _transcript(
                self.contract,
                self.ability_ids,
                "berserker-hero",
                actor_side=0,
                source_event_index=1,
                source_marker_index=17,
                source_tick=117,
                entity_id=10_001,
            ),
            _transcript(
                self.contract,
                self.ability_ids,
                "knight-hero",
                actor_side=0,
                source_event_index=2,
                source_marker_index=18,
                source_tick=118,
                entity_id=10_002,
            ),
            _transcript(
                self.contract,
                self.ability_ids,
                "little-prince",
                actor_side=1,
                source_event_index=3,
                source_marker_index=19,
                source_tick=119,
                entity_id=10_003,
            ),
        ]
        bundle = authenticate_ability_resolution_transcripts(
            rows,
            self.contract,
            source,
            expected_source_events_sha256=(
                source["ability_event_registry"]["source_events_sha256"]
            ),
        )
        by_event = {
            (int(row["actor_side"]), int(row["source_event_index"])): row
            for row in rows
        }
        return source, bundle, by_event

    def _successful_records(
        self, transcripts: dict[tuple[int, int], dict[str, object]]
    ) -> list[dict[str, object]]:
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
                    _ability_label("rune-giant", transcripts[(0, 0)]),
                    _ability_label("berserker-hero", transcripts[(0, 1)]),
                    _ability_label("knight-hero", transcripts[(0, 2)]),
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
                    _ability_label("little-prince", transcripts[(1, 3)])
                ],
            },
        ]

    def _authenticate(
        self,
        transcripts: list[dict[str, object]],
        source: dict[str, object],
        *,
        expected_source_events_sha256: str | None = None,
    ) -> dict[str, object]:
        anchor = (
            source["ability_event_registry"]["source_events_sha256"]
            if expected_source_events_sha256 is None
            else expected_source_events_sha256
        )
        return authenticate_ability_resolution_transcripts(
            transcripts,
            self.contract,
            source,
            expected_source_events_sha256=anchor,
        )

    def _valid_fixture(self) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[tuple[int, int], dict[str, object]],
        list[dict[str, object]],
    ]:
        source, bundle, transcripts = self._source_and_transcripts()
        return source, bundle, transcripts, self._successful_records(transcripts)

    def _summarize(
        self,
        records: list[dict[str, object]],
        source: dict[str, object],
        bundle: dict[str, object],
    ) -> dict[str, object]:
        return summarize_success_token_coverage(
            records,
            self.contract,
            source=source,
            authenticated_ability_transcripts=bundle,
            expected_source_events_sha256=(
                source["ability_event_registry"]["source_events_sha256"]
            ),
            expected_authenticated_transcripts_sha256=(
                bundle["authenticated_transcripts_sha256"]
            ),
        )

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
            {"admitted_training_episodes": 1, "deploy_labels": 1},
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
            {"admitted_training_episodes": 16, "deploy_labels": 64},
        )
        self.assertEqual(
            quotas["form_tokens"]["archers-ev1"],
            {"resolved_form_episodes": 8, "resolved_form_labels": 16},
        )
        self.assertEqual(
            quotas["ability_tokens"]["rune-giant"],
            {"resolved_ability_episodes": 8, "resolved_ability_labels": 32},
        )

    def test_censored_prefix_is_admitted_but_never_counted_as_full(self) -> None:
        source, bundle, _transcripts, records = self._valid_fixture()
        records[0]["full_success"] = False
        records[0]["censored_prefix"] = True
        success = self._summarize(records, source, bundle)
        electro = success["card_tokens"]["electro-dragon"]
        self.assertEqual(electro["full_success_episodes"], 0)
        self.assertEqual(electro["admitted_training_episodes"], 1)
        receipt = build_token_coverage_receipt(source, success)
        self.assertTrue(receipt["evaluation"]["gate"]["admitted"])

    def test_legacy_v1_aggregate_artifacts_are_rejected(self) -> None:
        source, bundle, _transcripts, records = self._valid_fixture()
        success = self._summarize(records, source, bundle)
        quotas = build_adaptive_token_quotas(source)
        receipt = build_token_coverage_receipt(source, success, quotas)
        self.assertEqual(success["schema_version"], 2)
        self.assertEqual(quotas["schema_version"], 2)
        self.assertEqual(receipt["schema_version"], 2)

        legacy_success = deepcopy(success)
        legacy_success.update(
            schema_version=1,
            kind="cr_expert_success_token_coverage_v1",
        )
        with self.assertRaisesRegex(TokenCoverageError, "contract binding"):
            evaluate_token_coverage(source, legacy_success, quotas)

        legacy_quotas = deepcopy(quotas)
        legacy_quotas.update(
            schema_version=1,
            kind="cr_expert_adaptive_token_quota_v1",
        )
        with self.assertRaisesRegex(TokenCoverageError, "quotas were changed"):
            evaluate_token_coverage(source, success, legacy_quotas)

        legacy_receipt = deepcopy(receipt)
        legacy_receipt.update(
            schema_version=1,
            kind="cr_expert_token_coverage_receipt_v1",
        )
        with self.assertRaisesRegex(TokenCoverageError, "kind/schema"):
            canonical_coverage_receipt_bytes(legacy_receipt)

        repacked_legacy = deepcopy(receipt)
        repacked_legacy["success"] = legacy_success
        with self.assertRaisesRegex(TokenCoverageError, "contract binding"):
            canonical_coverage_receipt_bytes(repacked_legacy)

    def test_multi_candidate_never_becomes_offline_identity(self) -> None:
        source, bundle, transcripts, records = self._valid_fixture()
        record = records[0]
        record["ability_labels"] = [{
            "source_event_index": 0,
            "candidate_tokens": ["rune-giant", "berserker-hero"],
            "identity_provenance": "schema5_candidate_association",
            "accepted": True,
            "legal": True,
            "compiled": True,
        }]
        with self.assertRaisesRegex(
            TokenCoverageError, "lacks an authenticated libg resolution transcript"
        ):
            self._summarize([record], source, bundle)

        valid = self._successful_records(transcripts)[0]
        valid["ability_labels"] = [
            _ability_label("rune-giant", transcripts[(0, 0)])
        ]
        success = self._summarize([valid], source, bundle)
        self.assertEqual(
            success["ability_tokens"]["rune-giant"]["resolved_ability_labels"],
            1,
        )
        self.assertEqual(
            success["ability_tokens"]["berserker-hero"]["resolved_ability_labels"],
            0,
        )

    def test_explicit_branch_must_be_verified(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        transcript = _transcript(
            self.contract,
            self.ability_ids,
            "rune-giant",
            actor_side=0,
            source_event_index=0,
            source_marker_index=16,
            source_tick=116,
            entity_id=10_000,
            candidates=[
                (10_000, self.ability_ids["rune-giant"][0]),
                (10_001, self.ability_ids["berserker-hero"][0]),
            ],
        )
        payload = {key: value for key, value in transcript.items() if key != "transcript_sha256"}
        payload["branch_verified"] = False
        invalid = seal_ability_resolution_transcript(payload)
        with self.assertRaisesRegex(TokenCoverageError, "branched.*unverified"):
            self._authenticate([invalid], source)

    def test_contract_payload_is_recomputed_before_coverage_use(self) -> None:
        tampered = deepcopy(self.contract)
        tampered["cards"][0]["internal_name"] = "forged-card-name"
        self.assertNotEqual(
            tampered["contract_sha256"], contract_payload_sha256(tampered)
        )
        with self.assertRaisesRegex(
            TokenCoverageError, "contract canonical SHA-256 mismatch"
        ):
            freeze_source_token_coverage([_battle()], tampered)

    def test_generator_libg_resolution_shape_authenticates_without_adapter(self) -> None:
        source = freeze_source_token_coverage([_battle()], self.contract)
        token = "rune-giant"
        base_id, native_form_id = self.ability_ids[token]
        label = _add_native_evidence_hash({
            "source_event_index": 0,
            "source_marker_index": 16,
            "source_tick": 116,
            "execution_tick": 117,
            "resolved_token": token,
            "resolved_native_form_id": native_form_id,
            "selected_entity_id": 55_001,
            "identity_provenance": "for_display_only",
            "branch_verified": False,
            "accepted": True,
            "legal": True,
            "compiled": False,
            "libg_resolution": {
                "status": "unique",
                "execution": "unique_executed",
                "side": 0,
                "source_tick": 116,
                "execution_tick": 117,
                "source_event_index": 0,
                "source_marker_index": 16,
                "candidate_entity_ids": [55_001],
                "candidate_card_ids": [base_id],
                "selected_entity_id": 55_001,
                "selected_native_form_id": native_form_id,
            },
        })
        actor = _add_native_evidence_hash({
            "schema_version": 1,
            "kind": "cr_native_full_success_actor_token_evidence_v1",
            "battle_tag": "COVERAGE001",
            "actor_side": 0,
            "full_success": True,
            "prefix_admission": False,
            "deck_tokens": list(TEAM_DECK),
            "deploy_labels": [],
            "ability_labels": [label],
        })
        bundle = authenticate_generator_ability_evidence(
            [actor],
            self.contract,
            source,
            expected_source_events_sha256=(
                source["ability_event_registry"]["source_events_sha256"]
            ),
        )
        self.assertEqual(bundle["transcript_count"], 1)
        transcript = bundle["transcripts"][0]
        self.assertEqual(transcript["candidate_entity_ids"], [55_001])
        self.assertEqual(transcript["candidate_card_ids"], [base_id])
        self.assertEqual(transcript["selected_native_form_id"], native_form_id)

    def test_whitelisted_provenance_string_is_not_authentication(self) -> None:
        source, bundle, transcripts, records = self._valid_fixture()
        transcript = transcripts[(0, 0)]
        forged = _ability_label("rune-giant", transcript)
        del forged["resolution_transcript_sha256"]
        forged["identity_provenance"] = "libg_live_entity_unique_v1"
        records[0]["ability_labels"] = [forged]
        with self.assertRaisesRegex(
            TokenCoverageError, "lacks an authenticated libg resolution transcript"
        ):
            self._summarize([records[0]], source, bundle)

    def test_ability_event_entity_and_tick_are_strongly_bound(self) -> None:
        source, bundle, transcripts, records = self._valid_fixture()
        valid_label = _ability_label("rune-giant", transcripts[(0, 0)])

        wrong_event = deepcopy(valid_label)
        wrong_event["source_event_index"] = 1
        records[0]["ability_labels"] = [wrong_event]
        with self.assertRaisesRegex(TokenCoverageError, "source event mismatch"):
            self._summarize([records[0]], source, bundle)

        wrong_entity = deepcopy(valid_label)
        wrong_entity["selected_entity_id"] = 999_999
        records[0]["ability_labels"] = [wrong_entity]
        with self.assertRaisesRegex(
            TokenCoverageError, "normalized identity disagrees"
        ):
            self._summarize([records[0]], source, bundle)

        wrong_tick_transcript = _reseal_transcript(
            transcripts[(0, 0)], source_tick=117, execution_tick=118
        )
        with self.assertRaisesRegex(TokenCoverageError, "event/tick mismatch"):
            self._authenticate([wrong_tick_transcript], source)

        wrong_selected_entity = _reseal_transcript(
            transcripts[(0, 0)], selected_entity_id=999_999
        )
        with self.assertRaisesRegex(TokenCoverageError, "selected entity is not"):
            self._authenticate([wrong_selected_entity], source)

        tampered_source = deepcopy(source)
        tampered_source["ability_event_registry"]["events"][0]["source_tick"] += 1
        with self.assertRaisesRegex(TokenCoverageError, "registry SHA-256 mismatch"):
            self._authenticate([transcripts[(0, 0)]], tampered_source)

    def test_duplicate_transcript_entity_or_event_is_rejected(self) -> None:
        source, _, transcripts = self._source_and_transcripts()
        duplicate_entity_candidates = deepcopy(
            transcripts[(0, 0)]["candidate_entity_ids"]
        )
        duplicate_entity_candidates.append(
            deepcopy(duplicate_entity_candidates[0])
        )
        duplicate_entity = _reseal_transcript(
            transcripts[(0, 0)], candidate_entity_ids=duplicate_entity_candidates,
            candidate_card_ids=(
                list(transcripts[(0, 0)]["candidate_card_ids"])
                + [transcripts[(0, 0)]["candidate_card_ids"][0]]
            ),
        )
        with self.assertRaisesRegex(TokenCoverageError, "duplicate entity"):
            self._authenticate([duplicate_entity], source)

        same_event_different_digest = _reseal_transcript(
            transcripts[(0, 0)],
            generator_ability_evidence_sha256=hashlib.sha256(
                b"different ability evidence"
            ).hexdigest(),
        )
        with self.assertRaisesRegex(TokenCoverageError, "duplicate.*source event"):
            self._authenticate(
                [transcripts[(0, 0)], same_event_different_digest],
                source,
            )

    def test_external_anchors_reject_rehashed_source_or_transcript_bundle(self) -> None:
        source, bundle, transcripts, records = self._valid_fixture()
        source_anchor = source["ability_event_registry"]["source_events_sha256"]
        bundle_anchor = bundle["authenticated_transcripts_sha256"]

        rehashed_source = deepcopy(source)
        rehashed_source["ability_event_registry"]["events"][0]["source_tick"] += 1
        _rewrite_content_hash(
            rehashed_source["ability_event_registry"], "source_events_sha256"
        )
        with self.assertRaisesRegex(TokenCoverageError, "trusted anchor"):
            self._authenticate(
                [transcripts[(0, 0)]],
                rehashed_source,
                expected_source_events_sha256=source_anchor,
            )

        rehashed_bundle = deepcopy(bundle)
        rehashed_bundle["transcript_count"] += 1
        _rewrite_content_hash(
            rehashed_bundle, "authenticated_transcripts_sha256"
        )
        with self.assertRaisesRegex(TokenCoverageError, "trusted anchor"):
            summarize_success_token_coverage(
                [records[0]],
                self.contract,
                source=source,
                authenticated_ability_transcripts=rehashed_bundle,
                expected_source_events_sha256=source_anchor,
                expected_authenticated_transcripts_sha256=bundle_anchor,
            )

    def test_hero_ability_rejects_base_native_id(self) -> None:
        source, bundle, transcripts, records = self._valid_fixture()
        token = "berserker-hero"
        base_id, _ = self.ability_ids[token]
        hero_base_transcript = _transcript(
            self.contract,
            self.ability_ids,
            token,
            actor_side=0,
            source_event_index=1,
            source_marker_index=17,
            source_tick=117,
            entity_id=20_001,
            candidates=[(20_001, base_id)],
            selected_native_form_id=base_id,
        )
        with self.assertRaisesRegex(TokenCoverageError, "native form does not uniquely"):
            self._authenticate([hero_base_transcript], source)

        normalized_base = _ability_label(token, transcripts[(0, 1)])
        normalized_base["resolved_native_form_id"] = base_id
        records[0]["ability_labels"] = [normalized_base]
        with self.assertRaisesRegex(
            TokenCoverageError, "normalized identity disagrees"
        ):
            self._summarize([records[0]], source, bundle)

    def test_base_cycle_does_not_forge_evolution_form_coverage(self) -> None:
        source, bundle, _, records = self._valid_fixture()
        record = records[0]
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
        success = self._summarize([record], source, bundle)
        self.assertEqual(success["card_tokens"]["archers-ev1"]["deploy_labels"], 1)
        self.assertEqual(
            success["form_tokens"]["archers-ev1"]["resolved_form_labels"], 0
        )

    def test_full_success_receipt_admits_only_resolved_compiled_labels(self) -> None:
        source, bundle, _, records = self._valid_fixture()
        success = self._summarize(records, source, bundle)
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
        source, bundle, _, records = self._valid_fixture()
        success = self._summarize(records, source, bundle)

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
        source, bundle, _, records = self._valid_fixture()
        record = records[0]
        with self.assertRaisesRegex(TokenCoverageError, "duplicate actor"):
            self._summarize([record, deepcopy(record)], source, bundle)
        duplicated = deepcopy(record)
        duplicated["deploy_labels"][1]["source_event_index"] = (
            duplicated["deploy_labels"][0]["source_event_index"]
        )
        with self.assertRaisesRegex(TokenCoverageError, "duplicate successful event"):
            self._summarize([duplicated], source, bundle)

    def test_canonical_hash_detects_tampering_and_rejects_nan(self) -> None:
        source, bundle, _, records = self._valid_fixture()
        success = self._summarize(records, source, bundle)
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
