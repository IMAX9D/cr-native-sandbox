"""Build arbitrary base-form decks for the original native replay loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .card_catalog import catalog, replay_spells


def resolve_card(value: int | str) -> int:
    if isinstance(value, int) or str(value).strip().isdigit():
        card_id = int(value)
        if card_id not in catalog():
            raise KeyError(f"unknown live card id: {card_id}")
        return card_id
    normalized = "".join(character for character in str(value).lower()
                         if character.isalnum())
    matches = [
        card_id for card_id, row in catalog().items()
        if "".join(character for character in str(row["internal_name"]).lower()
                   if character.isalnum()) == normalized
    ]
    if len(matches) != 1:
        raise KeyError(f"card name is unknown or ambiguous: {value!r}")
    return matches[0]


def normalize_deck(
    values: Iterable[int | str | Mapping[str, Any]], *, level: int = 11
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            card_id = resolve_card(value["card_id"])
            card_level = int(value.get("level", level))
            form = str(value.get("form", "base"))
        else:
            card_id = resolve_card(value)
            card_level = level
            form = "base"
        result.append({
            "card_id": card_id, "level": card_level, "form": form
        })
    return result


def build_replay(
    template: Mapping[str, Any],
    deck0: Iterable[int | str | Mapping[str, Any]],
    deck1: Iterable[int | str | Mapping[str, Any]],
    *,
    seed: int = 1,
    level: int = 11,
) -> dict[str, Any]:
    replay = deepcopy(dict(template))
    replay["rndSeed"] = int(seed)
    battle = replay["battle"]
    battle["deck0"]["sp"] = replay_spells(normalize_deck(deck0, level=level))
    battle["deck1"]["sp"] = replay_spells(normalize_deck(deck1, level=level))
    return replay


def parse_deck_text(value: str) -> list[int | str | Mapping[str, Any]]:
    result: list[int | str | Mapping[str, Any]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "@" in item:
            card, form = item.rsplit("@", 1)
            result.append({"card_id": resolve_card(card), "form": form})
        else:
            result.append(item)
    return result
