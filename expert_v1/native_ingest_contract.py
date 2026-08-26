"""Frozen, machine-readable native-ingest capability contract.

The crawler must not maintain a second card/form/tower table.  This module
derives one fail-closed contract from the checked-in live libg catalog, the
RoyaleAPI alias table, the native tower/ability mappings and the frozen
runtime binding.  The emitted JSON is deliberately consumable with only the
Python standard library (or by a non-Python downloader).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from native_core.card_catalog import CATALOG_PATH, catalog

from .native_capabilities import TOWER_TROOPS
from .native_replay_plan import ROYALEAPI_CARD_ALIASES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA_VERSION = 3
CONTRACT_KIND = "cr_native_authoritative_contract_v3"
LEGACY_CONTRACT_IDENTITIES = {
    (2, "cr_native_authoritative_contract_v2"),
}
RUNTIME_VERSION = "150535029"
GAME_VERSION = "15.535.29"
SOURCE_NUMERIC_GAME_MODE_IDS = (72_000_006, 72_000_450, 72_000_464)
STANDARD_1V1_NATIVE_EXECUTION_GAME_MODE_ID = 72_000_006
NATIVE_EXECUTION_MODE_BY_SOURCE = {
    source_mode: STANDARD_1V1_NATIVE_EXECUTION_GAME_MODE_ID
    for source_mode in SOURCE_NUMERIC_GAME_MODE_IDS
}
NATIVE_EXECUTION_GAME_MODE_PROVENANCE = (
    "frozen_native_ingest_contract_mode_map_v1"
)
# Exact Ladder-mode opening probes against frozen libg 15.535.29.  These are
# integer native results, not a floating-point 1.1**level approximation.
KING_TOWER_MAX_HP_BY_LEVEL = {
    1: 2_400,
    2: 2_568,
    3: 2_736,
    4: 2_904,
    5: 3_096,
    6: 3_312,
    7: 3_528,
    8: 3_768,
    9: 4_008,
    10: 4_392,
    11: 4_824,
    12: 5_304,
    13: 5_832,
    14: 6_408,
    15: 7_032,
    16: 7_728,
}
KING_TOWER_LEVEL = 16
KING_TOWER_LEVEL_PROVENANCE_TOWER_TROOP = (
    "ranked_template_cap16_and_tower_troop_level16_v1"
)
KING_TOWER_LEVEL_PROVENANCE_FULL_HP = (
    "ranked_template_cap16_and_full_king_hp_v1"
)
KING_TOWER_LEVEL_PROVENANCES = (
    KING_TOWER_LEVEL_PROVENANCE_TOWER_TROOP,
    KING_TOWER_LEVEL_PROVENANCE_FULL_HP,
)
# This is a side-local proof, not a heuristic.  In the frozen Ranked template
# both King and Tower Troop levels are capped at 16, while the live game rule
# forbids a Tower Troop from exceeding its player's King Tower.  Therefore a
# source Tower Troop level of 16 proves King level >= 16, and the template cap
# proves King level <= 16.  The older full-HP probe remains an independent
# fallback for sides whose Tower Troop is below 16.
KING_TOWER_LEVEL_EVIDENCE: dict[str, Any] = {
    "schema_version": 1,
    "scope": "side_local_ranked_template",
    "ranked_template_level_cap": KING_TOWER_LEVEL,
    "resolved_king_tower_level": KING_TOWER_LEVEL,
    "precedence": ["tower_troop_level", "final_king_hp"],
    "accepted_provenances": list(KING_TOWER_LEVEL_PROVENANCES),
    "tower_troop_level": {
        "required_value": KING_TOWER_LEVEL,
        "inference": (
            "tower_troop_level<=king_tower_level and ranked_template_cap=16"
        ),
        "provenance": KING_TOWER_LEVEL_PROVENANCE_TOWER_TROOP,
        "official_sources": [
            "https://support.supercell.com/clash-royale/en/articles/king-tower-level.html",
            "https://support.supercell.com/clash-royale/en/articles/tower-troops-4.html",
        ],
    },
    "final_king_hp": {
        "required_value": KING_TOWER_MAX_HP_BY_LEVEL[KING_TOWER_LEVEL],
        "provenance": KING_TOWER_LEVEL_PROVENANCE_FULL_HP,
    },
    "forbidden_inference_fields": ["card_levels", "deck_cards.level"],
}
DEFAULT_BINDING_PATH = (
    PROJECT_ROOT / "bindings" / "runtime-150535029-x86_64.json"
)
DEFAULT_CONTRACT_PATH = Path(
    r"D:\AI_data\cr-native-core\expert-v1\contracts"
) / "native-ingest-v150535029.json"

_FORM_SUFFIX = re.compile(r"-(ev1|hero)$")

# This compact schema is also embedded in the artifact.  Its digest makes a
# semantics change visible even when all live tables happen to stay equal.
INGEST_SCHEMA: dict[str, Any] = {
    "card_token": {
        "base": "exact lowercase RoyaleAPI slug",
        "evolution_suffix": "-ev1",
        "hero_suffix": "-hero",
        "form_flags": {"base": 0, "evolution": 1, "hero": 2, "both": 3},
        "unknown_slug_or_form": "reject",
    },
    "tower_troop": "exact lowercase source slug; unknown values reject",
    "ability": (
        "an observed ability event requires at least one deck token listed "
        "in ability_source_tokens; live entity identity remains tick-resolved"
    ),
    "numeric_game_mode": {
        "source": "exact source allowlist membership; missing/unknown rejects",
        "native_execution": (
            "exact lookup in native_execution_mode_by_source; the source mode "
            "is retained as provenance and is never written to libg implicitly"
        ),
        "provenance": NATIVE_EXECUTION_GAME_MODE_PROVENANCE,
    },
    "king_tower_level": KING_TOWER_LEVEL_EVIDENCE,
}


class NativeIngestContractError(ValueError):
    """The frozen contract is absent, corrupt or internally inconsistent."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def contract_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash the entire top-level contract except its self-hash field."""
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return _sha256_bytes(_canonical_json(payload))


def _source_slug(internal_name: str) -> str:
    # Split ordinary CamelCase while keeping all-capital tokens such as PEKKA
    # together.  Public exceptions belong in ROYALEAPI_CARD_ALIASES.
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", str(internal_name))
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-").lower()


def _component_files(binding_path: Path) -> dict[str, Path]:
    return {
        "binding": binding_path,
        "live_card_catalog": CATALOG_PATH,
        "royaleapi_aliases": PROJECT_ROOT / "expert_v1" / "native_replay_plan.py",
        "native_capabilities": PROJECT_ROOT / "expert_v1" / "native_capabilities.py",
        "contract_generator": Path(__file__).resolve(),
    }


def _allowed_flags(row: Mapping[str, Any]) -> list[int]:
    evolution = row.get("evolution_form_id") is not None
    hero = row.get("hero_form_id") is not None
    result = [0]
    if evolution:
        result.append(1)
    if hero:
        result.append(2)
    if evolution and hero:
        result.append(3)
    return result


def build_native_ingest_contract(
    *, binding_path: str | Path = DEFAULT_BINDING_PATH,
) -> dict[str, Any]:
    """Build a deterministic contract from the current frozen components."""
    binding_source = Path(binding_path).resolve(strict=True)
    binding = json.loads(binding_source.read_text(encoding="utf-8"))
    if binding.get("runtime_version") != RUNTIME_VERSION:
        raise NativeIngestContractError("runtime binding version mismatch")
    if binding.get("libg_sha256") in (None, ""):
        raise NativeIngestContractError("runtime binding has no libg SHA-256")

    aliases_by_id: dict[int, list[str]] = {}
    for raw_slug, raw_card_id in ROYALEAPI_CARD_ALIASES.items():
        slug = str(raw_slug).strip().lower()
        card_id = int(raw_card_id)
        if not slug or _FORM_SUFFIX.search(slug):
            raise NativeIngestContractError(f"invalid base alias: {raw_slug!r}")
        aliases_by_id.setdefault(card_id, []).append(slug)

    rows: list[dict[str, Any]] = []
    allowed_tokens: set[str] = set()
    ability_tokens: set[str] = set()
    ability_card_ids: set[int] = set()
    ability_sources: list[dict[str, Any]] = []
    seen_slugs: dict[str, int] = {}
    for card_id, row in sorted(catalog().items()):
        if not bool(row.get("standard_1v1")):
            continue
        source_slugs = sorted(set(
            aliases_by_id.get(card_id) or [_source_slug(str(row["internal_name"]))]
        ))
        for slug in source_slugs:
            previous = seen_slugs.setdefault(slug, card_id)
            if previous != card_id:
                raise NativeIngestContractError(
                    f"source slug {slug!r} maps to both {previous} and {card_id}"
                )

        flags = _allowed_flags(row)
        tokens: list[str] = []
        for slug in source_slugs:
            tokens.append(slug)
            if 1 in flags:
                tokens.append(f"{slug}-ev1")
            if 2 in flags:
                tokens.append(f"{slug}-hero")
        allowed_tokens.update(tokens)

        source_forms: list[int] = []
        if row.get("active_ability"):
            source_forms.append(0)
            for slug in source_slugs:
                token = slug
                ability_tokens.add(token)
                ability_sources.append({
                    "token": token,
                    "base_card_id": card_id,
                    "native_form_id": card_id,
                    "form_flags": 0,
                    "ability_name": str(row["active_ability"]),
                })
        if row.get("hero_active_ability") and row.get("hero_form_id") is not None:
            source_forms.append(2)
            for slug in source_slugs:
                token = f"{slug}-hero"
                ability_tokens.add(token)
                ability_sources.append({
                    "token": token,
                    "base_card_id": card_id,
                    "native_form_id": int(row["hero_form_id"]),
                    "form_flags": 2,
                    "ability_name": str(row["hero_active_ability"]),
                })
        if source_forms:
            ability_card_ids.add(card_id)

        rows.append({
            "base_slugs": source_slugs,
            "card_id": card_id,
            "internal_name": str(row["internal_name"]),
            "allowed_form_flags": flags,
            "allowed_tokens": sorted(tokens),
            "evolution": {
                "supported": row.get("evolution_form_id") is not None,
                "native_form_id": row.get("evolution_form_id"),
                "cycles": row.get("evolution_cycles"),
            },
            "hero": {
                "supported": row.get("hero_form_id") is not None,
                "native_form_id": row.get("hero_form_id"),
            },
            "ability_source_form_flags": source_forms,
        })

    tower_rows = [
        {
            "slug": slug,
            "support_card_id": int(spec.support_card_id),
            "spawn_group": str(spec.spawn_group),
            "runtime_probed": bool(spec.runtime_probed),
        }
        for slug, spec in sorted(TOWER_TROOPS.items())
    ]
    if not rows or not tower_rows or not ability_sources:
        raise NativeIngestContractError("generated capability set is unexpectedly empty")

    components = {
        name: {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _file_sha256(path),
        }
        for name, path in _component_files(binding_source).items()
    }
    component_sha = _sha256_bytes(_canonical_json(components))
    schema_sha = _sha256_bytes(_canonical_json(INGEST_SCHEMA))
    catalog_raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "kind": CONTRACT_KIND,
        "game_version": GAME_VERSION,
        # These flat arrays/maps are the stable cross-project reader surface.
        # Detailed rows below are explanatory and independently useful, but
        # crawlers need not reimplement their derivation.
        "allowed_card_tokens": sorted(allowed_tokens),
        "allowed_tower_troops": [item["slug"] for item in tower_rows],
        "ability_source_tokens": sorted(ability_tokens),
        "source_numeric_game_mode_ids": list(SOURCE_NUMERIC_GAME_MODE_IDS),
        "native_execution_mode_by_source": {
            str(source_mode): execution_mode
            for source_mode, execution_mode in sorted(
                NATIVE_EXECUTION_MODE_BY_SOURCE.items()
            )
        },
        "king_tower_max_hp_by_level": {
            str(level): hp
            for level, hp in sorted(KING_TOWER_MAX_HP_BY_LEVEL.items())
        },
        "king_tower_level_evidence": KING_TOWER_LEVEL_EVIDENCE,
        "runtime": {
            "runtime_version": RUNTIME_VERSION,
            "game_version": GAME_VERSION,
            "abi": str(binding.get("abi")),
            "libg_sha256": str(binding["libg_sha256"]),
            "binding_schema_version": binding.get("schema_version"),
        },
        "cards": rows,
        "tower_troops": tower_rows,
        "ability_source_card_ids": sorted(ability_card_ids),
        "ability_sources": sorted(
            ability_sources, key=lambda item: (item["token"], item["native_form_id"])
        ),
        "counts": {
            "supported_base_cards": len(rows),
            "allowed_card_tokens": len(allowed_tokens),
            "evolution_base_cards": sum(1 in row["allowed_form_flags"] for row in rows),
            "hero_base_cards": sum(2 in row["allowed_form_flags"] for row in rows),
            "tower_troops": len(tower_rows),
            "ability_source_base_cards": len(ability_card_ids),
            "ability_source_tokens": len(ability_tokens),
        },
        "ingest_schema": INGEST_SCHEMA,
        "ingest_schema_sha256": schema_sha,
        "components": components,
        "component_sha256": component_sha,
        "source_catalog_generated_utc": catalog_raw.get("generated_utc"),
    }
    result["contract_sha256"] = contract_payload_sha256(result)
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def write_native_ingest_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
    *, binding_path: str | Path = DEFAULT_BINDING_PATH,
) -> dict[str, Any]:
    """Atomically publish the JSON artifact and its raw-file SHA sidecar."""
    destination = Path(path).resolve()
    value = build_native_ingest_contract(binding_path=binding_path)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    file_sha = _sha256_bytes(payload)
    _atomic_write(destination, payload)
    _atomic_write(
        destination.with_suffix(destination.suffix + ".sha256"),
        f"{file_sha}  {destination.name}\n".encode("ascii"),
    )
    return {
        "path": str(destination),
        "sidecar": str(destination.with_suffix(destination.suffix + ".sha256")),
        "file_sha256": file_sha,
        "contract_sha256": value["contract_sha256"],
        "counts": value["counts"],
    }


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    code: str
    value: Any


@dataclass(frozen=True)
class NativeIngestContract:
    source_path: Path
    file_sha256: str
    value: Mapping[str, Any]
    card_token_rows: Mapping[str, Mapping[str, Any]]
    base_slug_rows: Mapping[str, Mapping[str, Any]]
    tower_slugs: frozenset[str]
    ability_tokens: frozenset[str]
    source_numeric_game_mode_ids: frozenset[int]
    native_execution_mode_by_source: Mapping[int, int]
    king_tower_max_hp_by_level: Mapping[int, int]

    def validate_king_tower_level_evidence(
        self,
        *,
        king_tower_level: Any,
        provenance: Any,
        tower_troop_level: Any,
        final_king_hp: Any,
    ) -> tuple[ValidationIssue, ...]:
        """Validate one side's exact Ranked-template King-level proof.

        Tower Troop level 16 has precedence over the full-HP fallback so a
        producer cannot silently change which source field supplied truth.
        Card/deck levels are intentionally absent from this API.
        """
        if king_tower_level != KING_TOWER_LEVEL:
            return (ValidationIssue(
                "king_tower_level", "king_tower_level_missing", king_tower_level,
            ),)
        evidence_version = int(self.value.get("schema_version") or 0)
        if evidence_version >= CONTRACT_SCHEMA_VERSION and (
            tower_troop_level == KING_TOWER_LEVEL
        ):
            expected = KING_TOWER_LEVEL_PROVENANCE_TOWER_TROOP
        elif final_king_hp == self.king_tower_max_hp_by_level[KING_TOWER_LEVEL]:
            expected = KING_TOWER_LEVEL_PROVENANCE_FULL_HP
        else:
            return (ValidationIssue(
                "king_tower_level",
                "king_tower_level_exact_evidence_missing",
                {
                    "tower_troop_level": tower_troop_level,
                    "final_king_hp": final_king_hp,
                },
            ),)
        if provenance != expected:
            return (ValidationIssue(
                "king_tower_level_provenance",
                "king_tower_level_provenance_invalid",
                {"expected": expected, "actual": provenance},
            ),)
        return ()

    def validate_card_token(self, token: str) -> tuple[ValidationIssue, ...]:
        normalized = str(token).strip().lower()
        if normalized in self.card_token_rows:
            return ()
        base, form = normalized, 0
        match = _FORM_SUFFIX.search(base)
        if match:
            base = base[:match.start()]
            form = 1 if match.group(1) == "ev1" else 2
        row = self.base_slug_rows.get(base)
        if row is None:
            return (ValidationIssue("card", "native_card_mapping_missing", token),)
        return (ValidationIssue(
            "card", "native_form_mapping_missing",
            {"token": token, "card_id": row["card_id"], "form_flags": form},
        ),)

    def validate_tower_troop(self, slug: str) -> tuple[ValidationIssue, ...]:
        normalized = str(slug).strip().lower().replace("_", "-")
        if normalized in self.tower_slugs:
            return ()
        return (ValidationIssue("tower_troop", "native_tower_troop_mapping_missing", slug),)

    def validate_game_mode(self, value: Any) -> tuple[ValidationIssue, ...]:
        """Validate the source mode ID (kept under this compatibility name)."""
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in self.source_numeric_game_mode_ids
        ):
            return ()
        return (ValidationIssue("numeric_game_mode_id", "numeric_game_mode_not_allowed", value),)

    def execution_game_mode_for_source(self, value: Any) -> int:
        issues = self.validate_game_mode(value)
        if issues:
            raise NativeIngestContractError(
                f"source numeric game mode is not allowed: {value!r}"
            )
        return int(self.native_execution_mode_by_source[int(value)])

    def validate_execution_game_mode(
        self,
        source_value: Any,
        execution_value: Any,
        provenance: Any,
    ) -> tuple[ValidationIssue, ...]:
        source_issues = self.validate_game_mode(source_value)
        if source_issues:
            return source_issues
        expected = self.execution_game_mode_for_source(source_value)
        issues: list[ValidationIssue] = []
        if (
            not isinstance(execution_value, int)
            or isinstance(execution_value, bool)
            or int(execution_value) != expected
        ):
            issues.append(ValidationIssue(
                "native_execution_game_mode_id",
                "native_execution_game_mode_mismatch",
                {"source": source_value, "expected": expected, "actual": execution_value},
            ))
        if provenance != NATIVE_EXECUTION_GAME_MODE_PROVENANCE:
            issues.append(ValidationIssue(
                "native_execution_game_mode_provenance",
                "native_execution_game_mode_provenance_invalid",
                provenance,
            ))
        return tuple(issues)

    def validate_ability_source(
        self, deck_tokens: Sequence[str], *, observed_ability_events: int,
    ) -> tuple[ValidationIssue, ...]:
        if int(observed_ability_events) <= 0:
            return ()
        normalized = {str(item).strip().lower() for item in deck_tokens}
        if normalized & self.ability_tokens:
            return ()
        return (ValidationIssue(
            "ability_plays", "native_ability_source_mapping_missing",
            {"observed_ability_events": int(observed_ability_events)},
        ),)


def load_native_ingest_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
    *, verify_sidecar: bool = True,
) -> NativeIngestContract:
    """Load and fully authenticate a published contract without live probes."""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    file_sha = _sha256_bytes(raw)
    if verify_sidecar:
        sidecar = source.with_suffix(source.suffix + ".sha256")
        try:
            claimed_file_sha = sidecar.read_text(encoding="ascii").split()[0]
        except (OSError, IndexError) as error:
            raise NativeIngestContractError("contract SHA-256 sidecar missing") from error
        if claimed_file_sha != file_sha:
            raise NativeIngestContractError("contract file SHA-256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise NativeIngestContractError("contract root must be an object")
    identity = (value.get("schema_version"), value.get("kind"))
    if identity not in {
        (CONTRACT_SCHEMA_VERSION, CONTRACT_KIND),
        *LEGACY_CONTRACT_IDENTITIES,
    }:
        raise NativeIngestContractError("unsupported contract schema/kind")
    if value.get("game_version") != GAME_VERSION:
        raise NativeIngestContractError("contract game version mismatch")
    claimed = str(value.get("contract_sha256") or "")
    if claimed != contract_payload_sha256(value):
        raise NativeIngestContractError("contract canonical SHA-256 mismatch")
    if value.get("ingest_schema_sha256") != _sha256_bytes(_canonical_json(value.get("ingest_schema"))):
        raise NativeIngestContractError("ingest schema SHA-256 mismatch")

    allowed = value.get("allowed_card_tokens")
    towers = value.get("allowed_tower_troops")
    abilities = value.get("ability_source_tokens")
    source_modes = value.get("source_numeric_game_mode_ids")
    execution_modes = value.get("native_execution_mode_by_source")
    king_tower_hp = value.get("king_tower_max_hp_by_level")
    king_tower_evidence = value.get("king_tower_level_evidence")
    if not all(
        isinstance(item, list) and item
        for item in (allowed, towers, abilities, source_modes)
    ):
        raise NativeIngestContractError("contract allowlist is missing or empty")
    if len(allowed) != len(set(allowed)) or len(towers) != len(set(towers)):
        raise NativeIngestContractError("contract allowlist contains duplicates")
    if not set(abilities).issubset(set(allowed)):
        raise NativeIngestContractError("ability sources are not allowed card tokens")
    if (
        any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in source_modes
        )
        or len(source_modes) != len(set(source_modes))
    ):
        raise NativeIngestContractError("source game-mode allowlist is invalid")
    if not isinstance(execution_modes, Mapping) or not execution_modes:
        raise NativeIngestContractError("native execution game-mode map is missing")
    expected_mode_keys = {str(item) for item in source_modes}
    if set(execution_modes) != expected_mode_keys or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in execution_modes.values()
    ):
        raise NativeIngestContractError(
            "native execution game-mode map does not cover the source allowlist"
        )
    if not isinstance(king_tower_hp, Mapping) or not king_tower_hp:
        raise NativeIngestContractError("King Tower max-HP table is missing")
    if any(
        not isinstance(level, str)
        or not level.isdigit()
        or not isinstance(hp, int)
        or isinstance(hp, bool)
        for level, hp in king_tower_hp.items()
    ):
        raise NativeIngestContractError("King Tower max-HP table is invalid")
    if (
        identity == (CONTRACT_SCHEMA_VERSION, CONTRACT_KIND)
        and king_tower_evidence != KING_TOWER_LEVEL_EVIDENCE
    ):
        raise NativeIngestContractError(
            "King Tower level evidence contract is missing or changed"
        )
    if identity in LEGACY_CONTRACT_IDENTITIES and king_tower_evidence is not None:
        raise NativeIngestContractError(
            "legacy contract unexpectedly declares King Tower evidence v3"
        )
    try:
        normalized_king_tower_hp = {
            int(level): int(hp) for level, hp in king_tower_hp.items()
        }
    except (TypeError, ValueError) as error:
        raise NativeIngestContractError("King Tower max-HP table is invalid") from error
    if (
        any(str(level) not in king_tower_hp for level in normalized_king_tower_hp)
        or any(level <= 0 or hp <= 0 for level, hp in normalized_king_tower_hp.items())
        or any(
            normalized_king_tower_hp.get(level) != hp
            for level, hp in KING_TOWER_MAX_HP_BY_LEVEL.items()
        )
    ):
        raise NativeIngestContractError("King Tower max-HP table is invalid")

    token_rows: dict[str, Mapping[str, Any]] = {}
    base_rows: dict[str, Mapping[str, Any]] = {}
    for row in value.get("cards", []):
        if not isinstance(row, Mapping):
            raise NativeIngestContractError("invalid card detail row")
        for slug in row.get("base_slugs", []):
            base_rows[str(slug)] = row
        for token in row.get("allowed_tokens", []):
            token_rows[str(token)] = row
    if set(token_rows) != set(allowed):
        raise NativeIngestContractError("flat and detailed card allowlists differ")
    return NativeIngestContract(
        source_path=source,
        file_sha256=file_sha,
        value=value,
        card_token_rows=token_rows,
        base_slug_rows=base_rows,
        tower_slugs=frozenset(str(item) for item in towers),
        ability_tokens=frozenset(str(item) for item in abilities),
        source_numeric_game_mode_ids=frozenset(int(item) for item in source_modes),
        native_execution_mode_by_source={
            int(source): int(execution)
            for source, execution in execution_modes.items()
        },
        king_tower_max_hp_by_level=normalized_king_tower_hp,
    )


def validate_ingest_metadata(
    contract: NativeIngestContract,
    *, deck_tokens: Iterable[str], tower_troop: str,
    numeric_game_mode_id: Any,
    native_execution_game_mode_id: Any,
    native_execution_game_mode_provenance: Any,
    observed_ability_events: int = 0,
) -> tuple[ValidationIssue, ...]:
    """Pure fail-closed gate suitable for a downloader before persistence."""
    deck = tuple(deck_tokens)
    issues: list[ValidationIssue] = []
    for token in deck:
        issues.extend(contract.validate_card_token(token))
    issues.extend(contract.validate_tower_troop(tower_troop))
    issues.extend(contract.validate_execution_game_mode(
        numeric_game_mode_id,
        native_execution_game_mode_id,
        native_execution_game_mode_provenance,
    ))
    issues.extend(contract.validate_ability_source(
        deck, observed_ability_events=observed_ability_events
    ))
    return tuple(issues)
