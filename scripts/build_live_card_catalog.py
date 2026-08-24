"""Build the checked-in card catalog from the decoded 15.535.29 data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = Path(r"D:\Deepseek\cr_re\config\live_card_map.json")
DEFAULT_DATA = Path(r"D:\Deepseek\cr_re\evidence\sandbox_data.json")
DEFAULT_EXTRA = Path(r"D:\Deepseek\cr_re\evidence\sandbox_data_extra.json")
DEFAULT_HERO_DIR = Path(
    r"D:\Deepseek\cr_re\decoded_csv\characters\hero_form"
)
DEFAULT_CHARACTERS_DIR = Path(r"D:\Deepseek\cr_re\decoded_csv\characters")
DEFAULT_OUTPUT = PROJECT_ROOT / "native_core" / "data" / "live_card_catalog.json"
VISIBLE_RARITIES = {"Common", "Rare", "Epic", "Legendary", "Champion"}


def _truthy(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in {
        "1", "true", "yes"
    }


def _integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _hero_metadata(hero_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in hero_dir.glob("*_hero.toml"):
        parsed = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        owner = None
        for table in ("EXT", "CHARACTER", "BUILDING"):
            for name, row in parsed.get(table, {}).items():
                if isinstance(row, dict) and row.get("Ability"):
                    owner = (str(name), dict(row))
                    break
            if owner is not None:
                break
        abilities = parsed.get("ABILITY", {})
        ability_name = (
            str(owner[1]["Ability"]) if owner is not None
            else str(next(iter(abilities), ""))
        )
        ability = abilities.get(ability_name, {})
        normalized = "".join(
            character
            for character in path.stem.removesuffix("_hero").lower()
            if character.isalnum()
        )
        result[normalized] = {
            "character": owner[0] if owner is not None else None,
            "active_ability": ability_name or None,
            "mana_cost": _integer(ability.get("ManaCost")),
            "max_charges": _integer(ability.get("MaxCharges")),
            "cast_time_ms": _integer(ability.get("CastTime")),
            "trigger_delay_ms": _integer(ability.get("TriggerDelay")),
            "cooldown_ms": _integer(ability.get("Cooldown")),
        }
    return result


def _ability_metadata(characters_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in characters_dir.rglob("*.toml"):
        try:
            parsed = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for name, value in parsed.get("ABILITY", {}).items():
            if not isinstance(value, dict):
                continue
            result[str(name)] = {
                "mana_cost": _integer(value.get("ManaCost")),
                "max_charges": _integer(value.get("MaxCharges")),
                "cast_time_ms": _integer(value.get("CastTime")),
                "trigger_delay_ms": _integer(value.get("TriggerDelay")),
                "cooldown_ms": _integer(value.get("Cooldown")),
            }
    return result


def build_catalog(card_map: dict[str, Any], data: dict[str, Any],
                  extra: dict[str, Any],
                  hero_metadata: dict[str, dict[str, Any]],
                  ability_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cards = data["cards"]
    units = data["units"]
    evolved = extra["evolved_spells"]
    hero_forms = extra["hero_spells"]
    evolved_ids = {
        name: 13_000_000 + index for index, name in enumerate(evolved)
    }
    hero_ids = {
        name: 203_000_000 + index for index, name in enumerate(hero_forms)
    }
    rows: list[dict[str, Any]] = []
    for raw_id, internal_name in sorted(
        card_map.items(), key=lambda item: int(item[0])
    ):
        card_id = int(raw_id)
        base = cards[str(internal_name)]
        hero_name = f"{internal_name}_hero"
        hero = hero_forms.get(hero_name)
        normalized_name = "".join(
            character for character in str(internal_name).lower()
            if character.isalnum()
        )
        hero_runtime = hero_metadata.get(normalized_name)
        has_hero_form = hero_runtime is not None
        evolution_name = None
        if isinstance(hero, dict):
            evolution_name = hero.get("EvolvedSpells") or None
        if evolution_name is None:
            evolution_name = base.get("EvolvedSpells") or None
        if evolution_name is None and f"{internal_name}_EV1" in evolved:
            evolution_name = f"{internal_name}_EV1"
        evolution = evolved.get(evolution_name) if evolution_name else None
        if isinstance(evolution, dict) and (
            _truthy(evolution.get("NotInUse"))
            or _truthy(evolution.get("NotVisible"))
        ):
            evolution_name = None
            evolution = None
        summon = base.get("SummonCharacter")
        unit = units.get(summon) if summon else None
        ability = unit.get("Ability") if isinstance(unit, dict) else None
        base_ability = ability_metadata.get(str(ability), {})
        if card_id < 27_000_000:
            card_type = "troop"
        elif card_id < 28_000_000:
            card_type = "building"
        else:
            card_type = "spell"
        standard = (
            not _truthy(base.get("NotInUse"))
            and not _truthy(base.get("NotVisible"))
            and base.get("Rarity") in VISIBLE_RARITIES
        )
        rows.append({
            "card_id": card_id,
            "internal_name": str(internal_name),
            "display_name": str(internal_name),
            "type": card_type,
            "elixir": _integer(base.get("ManaCost")),
            "rarity": base.get("Rarity"),
            "unlock_arena": base.get("UnlockArena"),
            "standard_1v1": standard,
            "not_in_use": _truthy(base.get("NotInUse")),
            "not_visible": _truthy(base.get("NotVisible")),
            "summon_character": summon,
            "active_ability": ability,
            "active_ability_mana_cost": base_ability.get("mana_cost"),
            "active_ability_max_charges": base_ability.get("max_charges"),
            "active_ability_cast_time_ms": base_ability.get("cast_time_ms"),
            "active_ability_trigger_delay_ms": base_ability.get(
                "trigger_delay_ms"
            ),
            "active_ability_cooldown_ms": base_ability.get("cooldown_ms"),
            "hero_form": (
                hero_name if isinstance(hero, dict) and has_hero_form else None
            ),
            "hero_form_id": (
                hero_ids[hero_name]
                if isinstance(hero, dict) and has_hero_form else None
            ),
            "hero_character": (
                hero_runtime.get("character") if hero_runtime else None
            ),
            "hero_active_ability": (
                hero_runtime.get("active_ability") if hero_runtime else None
            ),
            "hero_ability_mana_cost": (
                hero_runtime.get("mana_cost") if hero_runtime else None
            ),
            "hero_ability_max_charges": (
                hero_runtime.get("max_charges") if hero_runtime else None
            ),
            "hero_ability_cast_time_ms": (
                hero_runtime.get("cast_time_ms") if hero_runtime else None
            ),
            "hero_ability_trigger_delay_ms": (
                hero_runtime.get("trigger_delay_ms") if hero_runtime else None
            ),
            "hero_ability_cooldown_ms": (
                hero_runtime.get("cooldown_ms") if hero_runtime else None
            ),
            "evolution_form": evolution_name,
            "evolution_form_id": (
                evolved_ids[evolution_name] if evolution_name else None
            ),
            "evolution_cycles": (
                _integer(evolution.get("PrestigeCount"))
                if isinstance(evolution, dict) else None
            ),
        })
    return {
        "schema_version": 1,
        "kind": "libg_live_card_catalog_v1",
        "game_version": "15.535.29",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "mapped": len(rows),
            "standard_1v1": sum(bool(row["standard_1v1"]) for row in rows),
            "evolutions": sum(row["evolution_form"] is not None for row in rows),
            "hero_forms": sum(row["hero_form"] is not None for row in rows),
            "base_active_abilities": sum(
                row["active_ability"] is not None for row in rows
            ),
            "hero_active_abilities": sum(
                row["hero_active_ability"] is not None for row in rows
            ),
            "active_ability_forms": sum(
                row["active_ability"] is not None for row in rows
            ) + sum(row["hero_active_ability"] is not None for row in rows),
        },
        "cards": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--sandbox-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--extra-data", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--hero-dir", type=Path, default=DEFAULT_HERO_DIR)
    parser.add_argument(
        "--characters-dir", type=Path, default=DEFAULT_CHARACTERS_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build_catalog(
        _load(args.card_map), _load(args.sandbox_data), _load(args.extra_data),
        _hero_metadata(args.hero_dir),
        _ability_metadata(args.characters_dir),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), **value["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
