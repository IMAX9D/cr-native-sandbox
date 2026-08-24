"""Versioned, full-card metadata for the original 15.535.29 ``libg``."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CATALOG_PATH = Path(__file__).resolve().parent / "data" / "live_card_catalog.json"


@lru_cache(maxsize=1)
def catalog() -> dict[int, dict[str, Any]]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if raw.get("kind") != "libg_live_card_catalog_v1":
        raise RuntimeError("live card catalog contract mismatch")
    return {int(item["card_id"]): dict(item) for item in raw["cards"]}


def metadata(card_id: int) -> dict[str, Any]:
    try:
        return catalog()[int(card_id)]
    except KeyError as error:
        raise KeyError(f"card id is absent from the live libg catalog: {card_id}") from error


@lru_cache(maxsize=1)
def form_index() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for base_card_id, row in catalog().items():
        for form, key in (
            ("evolution", "evolution_form_id"),
            ("hero", "hero_form_id"),
        ):
            form_id = row.get(key)
            if form_id is not None:
                result[int(form_id)] = {
                    "base_card_id": base_card_id,
                    "card_form": form,
                    "form_name": row[f"{form}_form"],
                }
    return result


def observed_card(card_id: int) -> dict[str, Any]:
    card_id = int(card_id)
    if card_id in catalog():
        return {
            "base_card_id": card_id,
            "card_form": "base",
            "form_name": catalog()[card_id]["internal_name"],
        }
    if card_id in form_index():
        return dict(form_index()[card_id])
    return {
        "base_card_id": card_id,
        "card_form": "unknown",
        "form_name": str(card_id),
    }


def card_name(card_id: int) -> str:
    return str(metadata(card_id)["display_name"])


def card_cost(card_id: int) -> int:
    value = metadata(card_id).get("elixir")
    if value is None:
        raise ValueError(f"card has no elixir cost: {card_id}")
    return int(value)


def standard_card_ids() -> tuple[int, ...]:
    return tuple(
        card_id for card_id, value in catalog().items()
        if bool(value.get("standard_1v1"))
    )


FORM_FLAGS = {"base": 0, "evolution": 1, "hero": 2, "both": 3}


def validate_deck(cards: Iterable[int | Mapping[str, Any]]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for value in cards:
        if isinstance(value, Mapping):
            card_id = int(value["card_id"])
            level = int(value.get("level", 11))
            form = str(value.get("form", "base")).lower()
        else:
            card_id = int(value)
            level = 11
            form = "base"
        row = metadata(card_id)
        if form not in FORM_FLAGS:
            raise ValueError(f"unknown card form: {form}")
        form_flags = FORM_FLAGS[form]
        if form_flags & 1 and not row.get("evolution_form"):
            raise ValueError(f"card has no native evolution form: {card_id}")
        if form_flags & 2 and not row.get("hero_form"):
            raise ValueError(f"card has no native hero form: {card_id}")
        if level < 1 or level > 16:
            raise ValueError(f"card level is outside 1..16: {level}")
        result.append({
            "card_id": card_id, "level": level, "form_flags": form_flags
        })
    if len(result) != 8:
        raise ValueError("a Clash Royale deck must contain exactly eight cards")
    if len({item["card_id"] for item in result}) != 8:
        raise ValueError("a standard deck cannot contain duplicate card ids")
    return result


def replay_spells(cards: Iterable[int | Mapping[str, Any]]) -> list[dict[str, int]]:
    """Encode card IDs, levels and native Evo/Hero form flags."""
    result: list[dict[str, int]] = []
    for item in validate_deck(cards):
        encoded = {"d": item["card_id"], "l": item["level"] - 1}
        if item["form_flags"]:
            # libg's compact LogicSpell JSON key ``el`` is a form bitmask:
            # bit 0 enables EvoForm and bit 1 enables HeroForm.
            encoded["el"] = item["form_flags"]
        result.append(encoded)
    return result
