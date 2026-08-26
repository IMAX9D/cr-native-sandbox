"""Pure, fail-closed per-token coverage accounting for expert data.

This module deliberately knows nothing about files, workers, AVDs, or the
one-click orchestrator.  Its inputs are already authenticated contract/source
objects and normalized native/compiler result records; its outputs are plain
canonicalizable mappings.

The distinction between *source candidates* and *resolved supervision* is a
hard boundary:

* Schema5 ability markers expose only ``side + time_raw + marker_index``.
  Candidate-token association is useful for scheduling, but never establishes
  the identity of the pressed ability.
* An ability label is counted only when frozen libg selected a live entity and
  the normalized record carries one of the explicit native identity
  provenances accepted below.

Normalized successful actor record contract
--------------------------------------------

Each record represents one actor side of one native replay::

    {
      "battle_tag": "...", "actor_side": 0, "full_success": true,
      "deck_tokens": [eight exact contract tokens],
      "deploy_labels": [{
        "source_event_index": 7,
        "source_token": "archers-ev1",
        "resolved_native_form_id": 13000001,
        "accepted": true, "mask_legal": true, "compiled": true
      }],
      "ability_labels": [{
        "source_event_index": 9,
        "resolved_token": "rune-giant",
        "resolved_native_form_id": 26000101,
        "selected_entity_id": 123,
        "identity_provenance": "libg_live_entity_unique_v1",
        "accepted": true, "legal": true, "compiled": true
      }]
    }

Failed records are allowed but contribute no successful coverage.  Successful
records and event keys must be unique, preventing retries or duplicated
compiler rows from inflating coverage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import hmac
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


COVERAGE_SCHEMA_VERSION = 1
SOURCE_KIND = "cr_expert_source_token_coverage_v1"
SUCCESS_KIND = "cr_expert_success_token_coverage_v1"
QUOTA_KIND = "cr_expert_adaptive_token_quota_v1"
RECEIPT_KIND = "cr_expert_token_coverage_receipt_v1"

EXPECTED_CARD_TOKENS = 180
EXPECTED_EVOLUTION_TOKENS = 42
EXPECTED_HERO_TOKENS = 16
EXPECTED_ABILITY_TOKENS = 25

NATIVE_ABILITY_IDENTITY_PROVENANCE = frozenset({
    "libg_live_entity_unique_v1",
    "libg_live_entity_explicit_branch_v1",
})
EXPLICIT_BRANCH_PROVENANCE = "libg_live_entity_explicit_branch_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIDE_NAMES = ("team", "opponent")


class TokenCoverageError(ValueError):
    """An input cannot support an auditable token-coverage claim."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenCoverageError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TokenCoverageError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TokenCoverageError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _token_parts(token: str) -> tuple[str, str]:
    if token.endswith("-ev1"):
        return token[:-4], "ev1"
    if token.endswith("-hero"):
        return token[:-5], "hero"
    return token, "base"


def _unique_strings(value: Any, label: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in _sequence(value, label))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise TokenCoverageError(f"{label} must contain unique non-empty strings")
    return result


def _contract_index(contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(contract, "native contract")
    contract_sha = str(contract.get("contract_sha256") or "")
    if not _SHA256_RE.fullmatch(contract_sha):
        raise TokenCoverageError("native contract canonical SHA-256 is invalid")
    allowed = _unique_strings(
        contract.get("allowed_card_tokens"), "allowed_card_tokens"
    )
    ability = _unique_strings(
        contract.get("ability_source_tokens"), "ability_source_tokens"
    )
    allowed_set = set(allowed)
    if not set(ability) <= allowed_set:
        raise TokenCoverageError("ability tokens are outside allowed-card tokens")
    evolution = tuple(token for token in allowed if token.endswith("-ev1"))
    hero = tuple(token for token in allowed if token.endswith("-hero"))
    if (
        len(allowed) != EXPECTED_CARD_TOKENS
        or len(evolution) != EXPECTED_EVOLUTION_TOKENS
        or len(hero) != EXPECTED_HERO_TOKENS
        or len(ability) != EXPECTED_ABILITY_TOKENS
    ):
        raise TokenCoverageError(
            "native contract token shape changed: "
            f"cards={len(allowed)}, evo={len(evolution)}, "
            f"hero={len(hero)}, ability={len(ability)}"
        )

    token_specs: dict[str, dict[str, Any]] = {}
    all_native_card_ids: set[int] = set()
    for raw_card in _sequence(contract.get("cards"), "native contract cards"):
        card = _mapping(raw_card, "native contract card")
        card_id = _integer(card.get("card_id"), "card_id", minimum=1)
        all_native_card_ids.add(card_id)
        tokens = _unique_strings(card.get("allowed_tokens"), "card allowed_tokens")
        evolution_row = _mapping(card.get("evolution"), "card evolution")
        hero_row = _mapping(card.get("hero"), "card hero")
        evo_id = evolution_row.get("native_form_id")
        hero_id = hero_row.get("native_form_id")
        if evo_id is not None:
            evo_id = _integer(evo_id, "evolution native_form_id", minimum=1)
            all_native_card_ids.add(evo_id)
        if hero_id is not None:
            hero_id = _integer(hero_id, "hero native_form_id", minimum=1)
            all_native_card_ids.add(hero_id)
        for token in tokens:
            if token not in allowed_set or token in token_specs:
                raise TokenCoverageError(
                    f"native contract has invalid/duplicate token row: {token}"
                )
            base, form = _token_parts(token)
            expected_form_id = (
                evo_id if form == "ev1" else hero_id if form == "hero" else card_id
            )
            if expected_form_id is None:
                raise TokenCoverageError(
                    f"native contract token lacks its form ID: {token}"
                )
            allowed_resolved = {card_id, int(expected_form_id)}
            token_specs[token] = {
                "base_token": base,
                "form": form,
                "base_card_id": card_id,
                "expected_native_form_id": int(expected_form_id),
                "allowed_resolved_ids": frozenset(allowed_resolved),
            }
    if set(token_specs) != allowed_set:
        raise TokenCoverageError("native contract card rows do not cover exact tokens")

    ability_specs: dict[str, dict[str, Any]] = {}
    for raw_row in _sequence(
        contract.get("ability_sources"), "native contract ability_sources"
    ):
        row = _mapping(raw_row, "native contract ability source")
        token = str(row.get("token") or "")
        if token not in ability or token in ability_specs:
            raise TokenCoverageError(
                f"native contract has invalid/duplicate ability source: {token}"
            )
        base_id = _integer(row.get("base_card_id"), "ability base_card_id", minimum=1)
        native_id = _integer(
            row.get("native_form_id"), "ability native_form_id", minimum=1
        )
        ability_specs[token] = {
            "base_card_id": base_id,
            "native_form_id": native_id,
            "allowed_resolved_ids": frozenset({base_id, native_id}),
        }
    if set(ability_specs) != set(ability):
        raise TokenCoverageError("native contract ability rows are incomplete")

    return {
        "contract_sha256": contract_sha,
        "allowed": allowed,
        "allowed_set": allowed_set,
        "evolution": evolution,
        "hero": hero,
        "form_tokens": evolution + hero,
        "ability": ability,
        "ability_set": set(ability),
        "token_specs": token_specs,
        "ability_specs": ability_specs,
        "all_native_card_ids": frozenset(all_native_card_ids),
    }


def _source_side_deck(
    battle: Mapping[str, Any], side: str, allowed: set[str]
) -> tuple[str, ...]:
    rounds = _sequence(battle.get("rounds"), "source rounds")
    if len(rounds) != 1:
        raise TokenCoverageError("source battle must contain exactly one round")
    players = _sequence(
        _mapping(rounds[0], "source round").get(side), f"source {side} players"
    )
    if len(players) != 1:
        raise TokenCoverageError(f"source {side} must contain one player")
    player = _mapping(players[0], f"source {side} player")
    if player.get("complete") is not True:
        raise TokenCoverageError(f"source {side} deck is not complete")
    cards = _sequence(player.get("deck_cards"), f"source {side} deck_cards")
    if len(cards) != 8:
        raise TokenCoverageError(f"source {side} deck must contain eight cards")
    slugs: list[str] = []
    for expected_slot, raw_card in enumerate(cards):
        card = _mapping(raw_card, f"source {side} deck card")
        if _integer(card.get("slot"), "source card slot") != expected_slot:
            raise TokenCoverageError(f"source {side} deck slots are not canonical")
        token = str(card.get("slug") or "")
        base, form = _token_parts(token)
        if (
            token not in allowed
            or str(card.get("base_slug") or "") != base
            or str(card.get("form") or "") != form
        ):
            raise TokenCoverageError(
                f"source {side} deck token/form is invalid: {token}"
            )
        slugs.append(token)
    if len(set(slugs)) != 8:
        raise TokenCoverageError(f"source {side} deck contains duplicate tokens")
    for label, raw in (
        (f"source {side}_deck", battle.get(f"{side}_deck")),
        (f"source {side} full_deck", player.get("full_deck")),
    ):
        if tuple(str(item) for item in _sequence(raw, label)) != tuple(slugs):
            raise TokenCoverageError(f"{label} disagrees with deck_cards")
    return tuple(slugs)


def freeze_source_token_coverage(
    battles: Iterable[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze exact source-side token/candidate statistics.

    Ability event counts in this result are explicitly named *upper bounds*;
    they must never be joined to a token as identity supervision.
    """

    index = _contract_index(contract)
    deck_battles: dict[str, set[str]] = defaultdict(set)
    deck_sides: Counter[str] = Counter()
    play_battles: dict[str, set[str]] = defaultdict(set)
    play_labels: Counter[str] = Counter()
    ability_candidate_battles: dict[str, set[str]] = defaultdict(set)
    ability_candidate_sides: Counter[str] = Counter()
    ability_event_battles: dict[str, set[str]] = defaultdict(set)
    ability_event_upper_bound: Counter[str] = Counter()
    ability_singleton_event_battles: dict[str, set[str]] = defaultdict(set)
    ability_singleton_events: Counter[str] = Counter()
    seen_battles: set[str] = set()
    total_ability_events = 0
    ambiguous_ability_events = 0

    for raw_battle in battles:
        battle = _mapping(raw_battle, "source battle")
        tag = str(battle.get("battle_tag") or "")
        if not tag or tag in seen_battles:
            raise TokenCoverageError(f"source battle tag is missing/duplicate: {tag}")
        seen_battles.add(tag)
        side_decks = {
            side: _source_side_deck(battle, side, index["allowed_set"])
            for side in _SIDE_NAMES
        }
        for side, deck in side_decks.items():
            for token in deck:
                deck_battles[token].add(tag)
                deck_sides[token] += 1
            candidates = set(deck) & index["ability_set"]
            for token in candidates:
                ability_candidate_battles[token].add(tag)
                ability_candidate_sides[token] += 1

        events: list[tuple[int, int, str, str]] = []
        side_ability_counts = Counter[str]()
        for event_kind, field in (("deploy", "card_plays"), ("ability", "ability_plays")):
            for raw_event in _sequence(
                battle.get(field), f"source battle {field}"
            ):
                event = _mapping(raw_event, f"source {event_kind} event")
                side = str(event.get("side") or "")
                if side not in _SIDE_NAMES:
                    raise TokenCoverageError(f"source {event_kind} side is invalid")
                tick = _integer(event.get("time_raw"), f"source {event_kind} tick")
                marker = _integer(
                    event.get("marker_index"), f"source {event_kind} marker"
                )
                events.append((marker, tick, side, event_kind))
                if event_kind == "deploy":
                    token = str(event.get("card_form") or "")
                    if token not in side_decks[side]:
                        raise TokenCoverageError(
                            f"source deploy token is absent from {side} deck: {token}"
                        )
                    play_labels[token] += 1
                    play_battles[token].add(tag)
                    continue
                total_ability_events += 1
                side_ability_counts[side] += 1
                candidates = set(side_decks[side]) & index["ability_set"]
                if not candidates:
                    raise TokenCoverageError(
                        "source ability event has no contract ability candidate"
                    )
                if len(candidates) > 1:
                    ambiguous_ability_events += 1
                for token in candidates:
                    # Scheduling upper bound only.  This is never consumed by
                    # summarize_success_token_coverage as resolved identity.
                    ability_event_upper_bound[token] += 1
                    ability_event_battles[token].add(tag)
                if len(candidates) == 1:
                    token = next(iter(candidates))
                    ability_singleton_events[token] += 1
                    ability_singleton_event_battles[token].add(tag)

        markers = [item[0] for item in events]
        if sorted(markers) != list(range(len(markers))):
            raise TokenCoverageError(
                f"source battle markers are not one global contiguous sequence: {tag}"
            )
        command_keys = [(tick, side) for _, tick, side, _ in events]
        if len(set(command_keys)) != len(command_keys):
            raise TokenCoverageError(
                f"source battle has a same-side same-Tick command collision: {tag}"
            )
        elixir = _mapping(battle.get("elixir_stats"), "source elixir_stats")
        for side in _SIDE_NAMES:
            stats = _mapping(elixir.get(side), f"source {side} elixir stats")
            ability_stats = _mapping(
                stats.get("Ability"), f"source {side} ability stats"
            )
            observed_count = _integer(
                ability_stats.get("count"), f"source {side} ability count"
            )
            if observed_count != side_ability_counts[side]:
                raise TokenCoverageError(
                    f"source {side} ability event/count mismatch: {tag}"
                )

    def card_row(token: str) -> dict[str, int]:
        return {
            "deck_battles": len(deck_battles[token]),
            "deck_sides": int(deck_sides[token]),
            "play_battles": len(play_battles[token]),
            "play_labels": int(play_labels[token]),
        }

    def ability_row(token: str) -> dict[str, int | str]:
        return {
            "candidate_battles": len(ability_candidate_battles[token]),
            "candidate_sides": int(ability_candidate_sides[token]),
            "candidate_event_battles": len(ability_event_battles[token]),
            "candidate_event_upper_bound": int(ability_event_upper_bound[token]),
            "singleton_candidate_event_battles": len(
                ability_singleton_event_battles[token]
            ),
            "singleton_candidate_events": int(ability_singleton_events[token]),
            "identity_semantics": "candidate_only_not_resolved_identity",
        }

    cards = {token: card_row(token) for token in index["allowed"]}
    forms = {
        token: {
            **card_row(token),
            "form": index["token_specs"][token]["form"],
            "expected_native_form_id": index["token_specs"][token][
                "expected_native_form_id"
            ],
        }
        for token in index["form_tokens"]
    }
    abilities = {token: ability_row(token) for token in index["ability"]}
    observed_cards = [token for token in index["allowed"] if deck_sides[token] > 0]
    observed_forms = [token for token in index["form_tokens"] if deck_sides[token] > 0]
    observed_abilities = [
        token for token in index["ability"] if ability_candidate_sides[token] > 0
    ]
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "contract_sha256": index["contract_sha256"],
        "source_battles": len(seen_battles),
        "contract_token_counts": {
            "cards": len(index["allowed"]),
            "evolution": len(index["evolution"]),
            "hero": len(index["hero"]),
            "ability": len(index["ability"]),
        },
        "observed_card_tokens": observed_cards,
        "observed_form_tokens": observed_forms,
        "observed_ability_tokens": observed_abilities,
        "card_tokens": cards,
        "form_tokens": forms,
        "ability_tokens": abilities,
        "totals": {
            "play_labels": sum(play_labels.values()),
            "ability_events": total_ability_events,
            "ambiguous_ability_events": ambiguous_ability_events,
        },
    }


def _successful_deploy_label(
    raw_label: Any,
    *,
    deck: set[str],
    index: Mapping[str, Any],
) -> tuple[int, str, int]:
    label = _mapping(raw_label, "successful deploy label")
    event_index = _integer(
        label.get("source_event_index"), "deploy source_event_index"
    )
    token = str(label.get("source_token") or "")
    if token not in deck or token not in index["allowed_set"]:
        raise TokenCoverageError(
            f"successful deploy token is absent from actor deck: {token}"
        )
    if any(label.get(field) is not True for field in (
        "accepted", "mask_legal", "compiled"
    )):
        raise TokenCoverageError(
            f"successful deploy label is not accepted/mask-legal/compiled: {token}"
        )
    resolved_id = _integer(
        label.get("resolved_native_form_id"),
        "deploy resolved_native_form_id",
        minimum=1,
    )
    spec = index["token_specs"][token]
    if token == "mirror":
        if (
            resolved_id not in index["all_native_card_ids"]
            or label.get("identity_provenance")
            != "libg_dynamic_choice_exact_v1"
        ):
            raise TokenCoverageError(
                "Mirror deployment requires an exact libg dynamic-choice identity"
            )
    elif resolved_id not in spec["allowed_resolved_ids"]:
        raise TokenCoverageError(
            f"deploy resolved native ID does not belong to token {token}"
        )
    return event_index, token, resolved_id


def _successful_ability_label(
    raw_label: Any,
    *,
    deck: set[str],
    index: Mapping[str, Any],
) -> tuple[int, str, int]:
    label = _mapping(raw_label, "successful ability label")
    event_index = _integer(
        label.get("source_event_index"), "ability source_event_index"
    )
    token = str(label.get("resolved_token") or "")
    if token not in deck or token not in index["ability_set"]:
        raise TokenCoverageError(
            "ability identity is missing or is not a deck ability token; "
            "offline candidate association is not identity"
        )
    provenance = str(label.get("identity_provenance") or "")
    if provenance not in NATIVE_ABILITY_IDENTITY_PROVENANCE:
        raise TokenCoverageError(
            f"ability token lacks accepted libg identity provenance: {token}"
        )
    if provenance == EXPLICIT_BRANCH_PROVENANCE and (
        label.get("branch_verified") is not True
    ):
        raise TokenCoverageError(
            f"explicit ability branch is not verified: {token}"
        )
    if any(label.get(field) is not True for field in (
        "accepted", "legal", "compiled"
    )):
        raise TokenCoverageError(
            f"successful ability label is not accepted/legal/compiled: {token}"
        )
    _integer(label.get("selected_entity_id"), "ability selected_entity_id", minimum=1)
    resolved_id = _integer(
        label.get("resolved_native_form_id"),
        "ability resolved_native_form_id",
        minimum=1,
    )
    if resolved_id not in index["ability_specs"][token]["allowed_resolved_ids"]:
        raise TokenCoverageError(
            f"ability native form does not belong to resolved token {token}"
        )
    return event_index, token, resolved_id


def summarize_success_token_coverage(
    records: Iterable[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Count only full-success, accepted, compiled native supervision."""

    index = _contract_index(contract)
    card_episodes: dict[str, set[tuple[str, int]]] = defaultdict(set)
    card_label_episodes: dict[str, set[tuple[str, int]]] = defaultdict(set)
    card_labels: Counter[str] = Counter()
    form_episodes: dict[str, set[tuple[str, int]]] = defaultdict(set)
    form_labels: Counter[str] = Counter()
    ability_episodes: dict[str, set[tuple[str, int]]] = defaultdict(set)
    ability_labels: Counter[str] = Counter()
    resolved_form_ids: dict[str, Counter[int]] = defaultdict(Counter)
    resolved_ability_ids: dict[str, Counter[int]] = defaultdict(Counter)
    seen_records: set[tuple[str, int]] = set()
    total_records = full_success_records = ignored_failed_records = 0

    for raw_record in records:
        total_records += 1
        record = _mapping(raw_record, "native/compiler actor record")
        tag = str(record.get("battle_tag") or "")
        side = _integer(record.get("actor_side"), "actor_side")
        if not tag or side not in (0, 1):
            raise TokenCoverageError("actor record identity is invalid")
        key = (tag, side)
        if key in seen_records:
            raise TokenCoverageError(f"duplicate actor coverage record: {tag}/{side}")
        seen_records.add(key)
        full_success = record.get("full_success")
        if not isinstance(full_success, bool):
            raise TokenCoverageError("actor full_success must be a boolean")
        deploy_rows = _sequence(record.get("deploy_labels", []), "deploy_labels")
        ability_rows = _sequence(record.get("ability_labels", []), "ability_labels")
        if not full_success:
            ignored_failed_records += 1
            continue
        full_success_records += 1
        deck = set(_unique_strings(record.get("deck_tokens"), "actor deck_tokens"))
        if len(deck) != 8 or not deck <= index["allowed_set"]:
            raise TokenCoverageError("successful actor deck must contain eight tokens")
        for token in deck:
            card_episodes[token].add(key)

        seen_events: set[int] = set()
        for raw_label in deploy_rows:
            event_index, token, resolved_id = _successful_deploy_label(
                raw_label, deck=deck, index=index
            )
            if event_index in seen_events:
                raise TokenCoverageError(
                    f"duplicate successful event index: {tag}/{side}/{event_index}"
                )
            seen_events.add(event_index)
            card_labels[token] += 1
            card_label_episodes[token].add(key)
            resolved_form_ids[token][resolved_id] += 1
            spec = index["token_specs"][token]
            if (
                token in index["form_tokens"]
                and resolved_id == spec["expected_native_form_id"]
            ):
                form_labels[token] += 1
                form_episodes[token].add(key)
        for raw_label in ability_rows:
            event_index, token, resolved_id = _successful_ability_label(
                raw_label, deck=deck, index=index
            )
            if event_index in seen_events:
                raise TokenCoverageError(
                    f"duplicate successful event index: {tag}/{side}/{event_index}"
                )
            seen_events.add(event_index)
            ability_labels[token] += 1
            ability_episodes[token].add(key)
            resolved_ability_ids[token][resolved_id] += 1

    cards = {
        token: {
            "full_success_episodes": len(card_episodes[token]),
            "deploy_label_episodes": len(card_label_episodes[token]),
            "deploy_labels": int(card_labels[token]),
            "resolved_native_id_counts": {
                str(native_id): count
                for native_id, count in sorted(resolved_form_ids[token].items())
            },
        }
        for token in index["allowed"]
    }
    forms = {
        token: {
            "resolved_form_episodes": len(form_episodes[token]),
            "resolved_form_labels": int(form_labels[token]),
            "expected_native_form_id": index["token_specs"][token][
                "expected_native_form_id"
            ],
        }
        for token in index["form_tokens"]
    }
    abilities = {
        token: {
            "resolved_ability_episodes": len(ability_episodes[token]),
            "resolved_ability_labels": int(ability_labels[token]),
            "resolved_native_id_counts": {
                str(native_id): count
                for native_id, count in sorted(resolved_ability_ids[token].items())
            },
            "identity_semantics": "libg_live_entity_resolved_only",
        }
        for token in index["ability"]
    }
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "kind": SUCCESS_KIND,
        "contract_sha256": index["contract_sha256"],
        "records": total_records,
        "full_success_records": full_success_records,
        "ignored_failed_records": ignored_failed_records,
        "card_tokens": cards,
        "form_tokens": forms,
        "ability_tokens": abilities,
        "totals": {
            "deploy_labels": sum(card_labels.values()),
            "resolved_form_labels": sum(form_labels.values()),
            "resolved_ability_labels": sum(ability_labels.values()),
        },
    }


def _adaptive_requirement(observed: int, *, divisor: int, cap: int) -> int:
    observed = _integer(observed, "source quota observation")
    return max(1, min(cap, math.floor(observed / divisor)))


def build_adaptive_token_quotas(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build bounded quotas from the immutable source opportunity counts."""

    source = _mapping(source, "source coverage")
    if source.get("kind") != SOURCE_KIND or source.get("schema_version") != 1:
        raise TokenCoverageError("source coverage kind/schema is invalid")
    _validate_observed_sets(source)
    cards = _mapping(source.get("card_tokens"), "source card token stats")
    forms = _mapping(source.get("form_tokens"), "source form token stats")
    abilities = _mapping(source.get("ability_tokens"), "source ability token stats")
    observed_cards = _unique_strings(
        source.get("observed_card_tokens"), "observed_card_tokens"
    )
    observed_forms = _unique_strings(
        source.get("observed_form_tokens"), "observed_form_tokens"
    )
    observed_abilities = _unique_strings(
        source.get("observed_ability_tokens"), "observed_ability_tokens"
    )
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "kind": QUOTA_KIND,
        "contract_sha256": str(source.get("contract_sha256") or ""),
        "formula": {
            "card_episodes": "max(1,min(16,floor(source_deck_sides/4)))",
            "card_labels": "max(1,min(64,floor(source_play_labels/4)))",
            "form_episodes": "max(1,min(8,floor(source_deck_sides/4)))",
            "form_labels": "max(1,min(16,floor(source_play_labels/4)))",
            "ability_episodes": "max(1,min(8,floor(candidate_sides/10)))",
            "ability_labels": (
                "max(1,min(32,floor(candidate_event_upper_bound/10)))"
            ),
        },
        "card_tokens": {
            token: {
                "full_success_episodes": _adaptive_requirement(
                    _integer(_mapping(cards[token], token).get("deck_sides"), "deck_sides"),
                    divisor=4,
                    cap=16,
                ),
                "deploy_labels": _adaptive_requirement(
                    _integer(_mapping(cards[token], token).get("play_labels"), "play_labels"),
                    divisor=4,
                    cap=64,
                ),
            }
            for token in observed_cards
        },
        "form_tokens": {
            token: {
                "resolved_form_episodes": _adaptive_requirement(
                    _integer(_mapping(forms[token], token).get("deck_sides"), "deck_sides"),
                    divisor=4,
                    cap=8,
                ),
                "resolved_form_labels": _adaptive_requirement(
                    _integer(_mapping(forms[token], token).get("play_labels"), "play_labels"),
                    divisor=4,
                    cap=16,
                ),
            }
            for token in observed_forms
        },
        "ability_tokens": {
            token: {
                "resolved_ability_episodes": _adaptive_requirement(
                    _integer(
                        _mapping(abilities[token], token).get("candidate_sides"),
                        "candidate_sides",
                    ),
                    divisor=10,
                    cap=8,
                ),
                "resolved_ability_labels": _adaptive_requirement(
                    _integer(
                        _mapping(abilities[token], token).get(
                            "candidate_event_upper_bound"
                        ),
                        "candidate_event_upper_bound",
                    ),
                    divisor=10,
                    cap=32,
                ),
            }
            for token in observed_abilities
        },
    }


def _validate_observed_sets(source: Mapping[str, Any]) -> None:
    cards = _mapping(source.get("card_tokens"), "source card token stats")
    forms = _mapping(source.get("form_tokens"), "source form token stats")
    abilities = _mapping(source.get("ability_tokens"), "source ability token stats")
    observed_cards = _unique_strings(
        source.get("observed_card_tokens"), "observed_card_tokens"
    )
    observed_forms = _unique_strings(
        source.get("observed_form_tokens"), "observed_form_tokens"
    )
    observed_abilities = _unique_strings(
        source.get("observed_ability_tokens"), "observed_ability_tokens"
    )
    expected_cards = tuple(
        token
        for token, raw in cards.items()
        if _integer(_mapping(raw, token).get("deck_sides"), "deck_sides") > 0
    )
    expected_forms = tuple(
        token
        for token, raw in forms.items()
        if _integer(_mapping(raw, token).get("deck_sides"), "deck_sides") > 0
    )
    expected_abilities = tuple(
        token
        for token, raw in abilities.items()
        if _integer(
            _mapping(raw, token).get("candidate_sides"), "candidate_sides"
        ) > 0
    )
    if (
        set(observed_cards) != set(expected_cards)
        or set(observed_forms) != set(expected_forms)
        or set(observed_abilities) != set(expected_abilities)
    ):
        raise TokenCoverageError(
            "observed token sets do not match positive source opportunities"
        )


def _validate_success_within_source(
    source: Mapping[str, Any], success: Mapping[str, Any]
) -> None:
    limits = {
        "card_tokens": (
            "deck_sides",
            "play_labels",
            "full_success_episodes",
            "deploy_labels",
        ),
        "form_tokens": (
            "deck_sides",
            "play_labels",
            "resolved_form_episodes",
            "resolved_form_labels",
        ),
        "ability_tokens": (
            "candidate_sides",
            "candidate_event_upper_bound",
            "resolved_ability_episodes",
            "resolved_ability_labels",
        ),
    }
    for group, (
        source_episode_metric,
        source_label_metric,
        success_episode_metric,
        success_label_metric,
    ) in limits.items():
        source_rows = _mapping(source.get(group), f"source {group}")
        success_rows = _mapping(success.get(group), f"success {group}")
        for token, raw_success in success_rows.items():
            if token not in source_rows:
                raise TokenCoverageError(
                    f"success token is absent from source coverage: {token}"
                )
            source_row = _mapping(source_rows[token], f"source stats for {token}")
            success_row = _mapping(raw_success, f"success stats for {token}")
            source_episodes = _integer(
                source_row.get(source_episode_metric),
                f"source {token} {source_episode_metric}",
            )
            source_labels = _integer(
                source_row.get(source_label_metric),
                f"source {token} {source_label_metric}",
            )
            successful_episodes = _integer(
                success_row.get(success_episode_metric),
                f"success {token} {success_episode_metric}",
            )
            successful_labels = _integer(
                success_row.get(success_label_metric),
                f"success {token} {success_label_metric}",
            )
            if (
                successful_episodes > source_episodes
                or successful_labels > source_labels
            ):
                raise TokenCoverageError(
                    f"success coverage exceeds source opportunities: {token}"
                )


def _deficits(
    *,
    tokens: Sequence[str],
    success: Mapping[str, Any],
    required: Mapping[str, Any] | None,
    metrics: Sequence[str],
    hard_floor: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for token in tokens:
        actual_row = _mapping(success.get(token), f"success stats for {token}")
        required_row = None if required is None else _mapping(
            required.get(token), f"quota for {token}"
        )
        for metric in metrics:
            actual = _integer(actual_row.get(metric), f"{token} {metric}")
            need = 1 if hard_floor else _integer(
                required_row.get(metric) if required_row is not None else None,
                f"{token} required {metric}",
                minimum=1,
            )
            if actual < need:
                result.append({
                    "token": token,
                    "metric": metric,
                    "actual": actual,
                    "required": need,
                    "missing": need - actual,
                })
    return result


def evaluate_token_coverage(
    source: Mapping[str, Any],
    success: Mapping[str, Any],
    quotas: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return immutable hard-floor and adaptive-quota deficits."""

    source = _mapping(source, "source coverage")
    success = _mapping(success, "success coverage")
    _validate_observed_sets(source)
    _validate_success_within_source(source, success)
    recomputed_quotas = build_adaptive_token_quotas(source)
    if quotas is None:
        quotas = recomputed_quotas
    else:
        quotas = _mapping(quotas, "token quotas")
        if canonical_json_bytes(quotas) != canonical_json_bytes(recomputed_quotas):
            raise TokenCoverageError("adaptive token quotas were changed")
    expected_sha = str(source.get("contract_sha256") or "")
    if (
        source.get("kind") != SOURCE_KIND
        or success.get("kind") != SUCCESS_KIND
        or quotas.get("kind") != QUOTA_KIND
        or not _SHA256_RE.fullmatch(expected_sha)
        or success.get("contract_sha256") != expected_sha
        or quotas.get("contract_sha256") != expected_sha
    ):
        raise TokenCoverageError("source/success/quota contract binding changed")

    token_groups = {
        "card_tokens": (
            _unique_strings(source.get("observed_card_tokens"), "observed cards"),
            ("full_success_episodes", "deploy_labels"),
        ),
        "form_tokens": (
            _unique_strings(source.get("observed_form_tokens"), "observed forms"),
            ("resolved_form_episodes", "resolved_form_labels"),
        ),
        "ability_tokens": (
            _unique_strings(
                source.get("observed_ability_tokens"), "observed abilities"
            ),
            ("resolved_ability_episodes", "resolved_ability_labels"),
        ),
    }
    hard: dict[str, list[dict[str, Any]]] = {}
    quota: dict[str, list[dict[str, Any]]] = {}
    for group, (tokens, metrics) in token_groups.items():
        success_rows = _mapping(success.get(group), f"success {group}")
        quota_rows = _mapping(quotas.get(group), f"quota {group}")
        hard[group] = _deficits(
            tokens=tokens,
            success=success_rows,
            required=None,
            metrics=metrics,
            hard_floor=True,
        )
        quota[group] = _deficits(
            tokens=tokens,
            success=success_rows,
            required=quota_rows,
            metrics=metrics,
            hard_floor=False,
        )
    hard_passed = not any(hard.values())
    quota_passed = not any(quota.values())
    return {
        "hard_floor_deficits": hard,
        "adaptive_quota_deficits": quota,
        "gate": {
            "all_observed_tokens_have_successful_training_sample": hard_passed,
            "adaptive_quotas_met": quota_passed,
            "admitted": bool(hard_passed and quota_passed),
            "waiver_supported": False,
        },
    }


def build_token_coverage_receipt(
    source: Mapping[str, Any],
    success: Mapping[str, Any],
    quotas: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonicalizable coverage receipt (without a self-hash)."""

    recomputed_quotas = build_adaptive_token_quotas(source)
    actual_quotas = recomputed_quotas if quotas is None else dict(quotas)
    evaluation = evaluate_token_coverage(source, success, actual_quotas)
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "contract_sha256": str(source.get("contract_sha256") or ""),
        "source": dict(source),
        "success": dict(success),
        "quotas": actual_quotas,
        "evaluation": evaluation,
    }


def canonical_coverage_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Serialize a receipt deterministically and reject NaN/Infinity."""

    receipt = _mapping(receipt, "token coverage receipt")
    if receipt.get("kind") != RECEIPT_KIND or receipt.get("schema_version") != 1:
        raise TokenCoverageError("token coverage receipt kind/schema is invalid")
    return canonical_json_bytes(receipt)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Canonical JSON for authenticated pure-function intermediates."""

    value = _mapping(value, "canonical JSON value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TokenCoverageError("token coverage receipt is not canonical JSON") from error


def coverage_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_coverage_receipt_bytes(receipt)).hexdigest()


def verify_coverage_receipt_sha256(
    receipt: Mapping[str, Any], expected_sha256: str
) -> bool:
    if not _SHA256_RE.fullmatch(str(expected_sha256 or "")):
        return False
    return hmac.compare_digest(
        coverage_receipt_sha256(receipt), str(expected_sha256)
    )


__all__ = [
    "COVERAGE_SCHEMA_VERSION",
    "NATIVE_ABILITY_IDENTITY_PROVENANCE",
    "QUOTA_KIND",
    "RECEIPT_KIND",
    "SOURCE_KIND",
    "SUCCESS_KIND",
    "TokenCoverageError",
    "build_adaptive_token_quotas",
    "build_token_coverage_receipt",
    "canonical_coverage_receipt_bytes",
    "canonical_json_bytes",
    "coverage_receipt_sha256",
    "evaluate_token_coverage",
    "freeze_source_token_coverage",
    "summarize_success_token_coverage",
    "verify_coverage_receipt_sha256",
]
