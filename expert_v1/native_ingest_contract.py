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
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "cr_native_authoritative_contract_v1"
RUNTIME_VERSION = "150535029"
GAME_VERSION = "15.535.29"
STANDARD_1V1_NUMERIC_GAME_MODE_IDS = (72_000_006,)
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
    "numeric_game_mode": "exact allowlist membership; missing/unknown rejects",
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
        # The following four flat arrays are the stable cross-project reader
        # surface.  Detailed rows below are explanatory and independently
        # useful, but crawlers need not reimplement their derivation.
        "allowed_card_tokens": sorted(allowed_tokens),
        "allowed_tower_troops": [item["slug"] for item in tower_rows],
        "ability_source_tokens": sorted(ability_tokens),
        "numeric_game_mode_ids": list(STANDARD_1V1_NUMERIC_GAME_MODE_IDS),
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
    numeric_game_mode_ids: frozenset[int]

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
        if isinstance(value, int) and not isinstance(value, bool) and value in self.numeric_game_mode_ids:
            return ()
        return (ValidationIssue("numeric_game_mode_id", "numeric_game_mode_not_allowed", value),)

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
    if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise NativeIngestContractError("unsupported contract schema_version")
    if value.get("kind") != CONTRACT_KIND:
        raise NativeIngestContractError("unexpected contract kind")
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
    modes = value.get("numeric_game_mode_ids")
    if not all(isinstance(item, list) and item for item in (allowed, towers, abilities, modes)):
        raise NativeIngestContractError("contract allowlist is missing or empty")
    if len(allowed) != len(set(allowed)) or len(towers) != len(set(towers)):
        raise NativeIngestContractError("contract allowlist contains duplicates")
    if not set(abilities).issubset(set(allowed)):
        raise NativeIngestContractError("ability sources are not allowed card tokens")

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
        numeric_game_mode_ids=frozenset(int(item) for item in modes),
    )


def validate_ingest_metadata(
    contract: NativeIngestContract,
    *, deck_tokens: Iterable[str], tower_troop: str,
    numeric_game_mode_id: Any, observed_ability_events: int = 0,
) -> tuple[ValidationIssue, ...]:
    """Pure fail-closed gate suitable for a downloader before persistence."""
    deck = tuple(deck_tokens)
    issues: list[ValidationIssue] = []
    for token in deck:
        issues.extend(contract.validate_card_token(token))
    issues.extend(contract.validate_tower_troop(tower_troop))
    issues.extend(contract.validate_game_mode(numeric_game_mode_id))
    issues.extend(contract.validate_ability_source(
        deck, observed_ability_events=observed_ability_events
    ))
    return tuple(issues)

