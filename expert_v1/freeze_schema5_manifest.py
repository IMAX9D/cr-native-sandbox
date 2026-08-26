"""Freeze crawler Schema5 v2 output into a content-addressed expert manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from .native_ingest_contract import contract_payload_sha256


CONTRACT_GATE = "native_static_v2"
SOURCE_SCHEMA_VERSION = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
    temporary.replace(path)


def _contract_binding(path: Path) -> tuple[Path, str, str]:
    """Authenticate both semantic (canonical) and byte-exact contract IDs."""

    source = path.resolve(strict=True)
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("native contract root is not an object")
    canonical = str(value.get("contract_sha256") or "")
    if canonical != contract_payload_sha256(value):
        raise ValueError("native contract canonical SHA-256 mismatch")
    file_sha = hashlib.sha256(raw).hexdigest()
    sidecar = source.with_suffix(source.suffix + ".sha256")
    try:
        sidecar_sha = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as error:
        raise ValueError("native contract file SHA-256 sidecar is missing") from error
    if sidecar_sha != file_sha:
        raise ValueError("native contract file SHA-256 mismatch")
    return source, canonical, file_sha


def _index_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.resolve(strict=True).open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if (
                not isinstance(value, dict)
                or value.get("kind") != "authoritative_battle"
                or int(value.get("schema_version", -1)) != SOURCE_SCHEMA_VERSION
            ):
                raise ValueError(f"invalid authoritative index row {line_number}")
            tag = str(value.get("battle_tag") or "")
            if not tag or tag in rows:
                raise ValueError(f"duplicate/missing battle tag at index row {line_number}")
            rows[tag] = value
    return rows


def _player_tags(value: dict[str, Any]) -> tuple[list[str], list[str]]:
    team = [str(tag) for tag in value.get("team_tags") or [] if str(tag)]
    opponent = [str(tag) for tag in value.get("opponent_tags") or [] if str(tag)]
    if len(team) != 1 or len(opponent) != 1 or team[0] == opponent[0]:
        raise ValueError("Schema5 battle does not contain two distinct player tags")
    return team, opponent


def _validate_source(
    path: Path,
    *,
    battle_tag: str,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("battle_tag") != battle_tag:
        raise ValueError(f"source battle tag mismatch: {path}")
    stamp = value.get("authoritative_native_contract")
    eligibility = value.get("authoritative_eligibility")
    if (
        value.get("schema_version") != SOURCE_SCHEMA_VERSION
        or not isinstance(stamp, dict)
        or stamp.get("contract_sha256") != contract_sha256
        or not isinstance(eligibility, dict)
        or eligibility.get("status") != "accepted"
        or eligibility.get("gate") != CONTRACT_GATE
    ):
        raise ValueError(f"source Schema5/contract eligibility mismatch: {path}")
    if (
        len(value.get("team_deck") or []) != 8
        or len(value.get("opponent_deck") or []) != 8
        or value.get("normal_1v1") is not True
        or value.get("native_execution_game_mode_id") != 72_000_006
    ):
        raise ValueError(f"source 1v1/deck/execution contract mismatch: {path}")
    rounds = value.get("rounds") or []
    if len(rounds) != 1:
        raise ValueError(f"source has no single round: {path}")
    for side in ("team", "opponent"):
        players = rounds[0].get(side) or []
        if (
            len(players) != 1
            or players[0].get("king_tower_level") != 16
            or not players[0].get("king_tower_level_provenance")
        ):
            raise ValueError(f"source King Tower evidence is incomplete: {path}")
    team_tags, opponent_tags = _player_tags(value)
    source_url = str((value.get("deck_metadata") or {}).get("source_list_url") or "")
    if not source_url.startswith("http"):
        raise ValueError(f"source list provenance is missing: {path}")
    row = {
        "battle_tag": battle_tag,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "contract_file_sha256": str(stamp.get("contract_file_sha256") or ""),
        "source_numeric_game_mode_id": int(value["numeric_game_mode_id"]),
        "native_execution_game_mode_id": int(
            value["native_execution_game_mode_id"]
        ),
        "version_timestamp": int(value["version_timestamp"]),
        "battle_time_utc": str(value["battle_time_utc"]),
        "team_crowns": int(value["team_crowns"]),
        "opponent_crowns": int(value["opponent_crowns"]),
        "team_tags": team_tags,
        "opponent_tags": opponent_tags,
        "player_tags": sorted(team_tags + opponent_tags),
        "source_list_url": source_url,
    }
    row["source_group"] = hashlib.sha256(_canonical_line({
        "player_tags": row["player_tags"],
        "source_list_url": source_url,
    })).hexdigest()
    return row, value


def freeze(
    *,
    db_path: Path,
    authoritative_root: Path,
    output: Path,
    target: int,
    allow_incomplete: bool,
    native_contract_path: Path | None = None,
) -> dict[str, Any]:
    root = authoritative_root.resolve(strict=True)
    index_path = root / "index.jsonl"
    contract_path: Path | None = None
    expected_contract_sha: str | None = None
    expected_contract_file_sha: str | None = None
    if native_contract_path is not None:
        (
            contract_path,
            expected_contract_sha,
            expected_contract_file_sha,
        ) = _contract_binding(native_contract_path)
    uri = "file:" + db_path.resolve(strict=True).as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"authoritative SQLite quick_check: {quick_check}")
        rows = connection.execute(
            "SELECT battle_tag,saved_path,contract_sha256,tier "
            "FROM authoritative_results WHERE status='accepted' "
            "ORDER BY battle_tag"
        ).fetchall()
    finally:
        connection.close()
    # Publication order is file -> durable index -> accepted DB transaction.
    # Reading DB first therefore makes a live incomplete snapshot safe: the
    # subsequently read index must already contain every selected DB row and
    # may only contain additional rows accepted after this read.
    index = _index_rows(index_path)
    if not allow_incomplete and len(rows) != target:
        raise RuntimeError(f"authoritative corpus is incomplete: {len(rows)}/{target}")
    if len(rows) > target:
        raise RuntimeError(f"authoritative corpus exceeds frozen target: {len(rows)}/{target}")
    # A transactionally cap-rejected concurrent completion may have already
    # durably appended an unaccepted recovery index row.  DB membership is the
    # authority; require it to be a subset rather than mistaking an orphan for
    # an accepted battle.
    if len(index) < len(rows):
        raise RuntimeError(f"authoritative index/DB count mismatch: {len(index)}/{len(rows)}")

    manifest_rows: list[dict[str, Any]] = []
    contract_set: set[str] = set()
    file_shas: set[str] = set()
    player_tags: set[str] = set()
    for db_row in rows:
        tag = str(db_row["battle_tag"])
        if db_row["tier"] != CONTRACT_GATE or tag not in index:
            raise RuntimeError(f"accepted DB/index tier mismatch: {tag}")
        source = Path(str(db_row["saved_path"])).resolve(strict=True)
        if not source.is_relative_to(root):
            raise RuntimeError(f"accepted source escaped authoritative root: {source}")
        if Path(str(index[tag].get("saved_path") or "")).resolve() != source:
            raise RuntimeError(f"accepted index path mismatch: {tag}")
        contract_sha = str(db_row["contract_sha256"] or "")
        if expected_contract_sha is not None and contract_sha != expected_contract_sha:
            raise RuntimeError(f"accepted DB contract mismatch: {tag}")
        row, _ = _validate_source(
            source, battle_tag=tag, contract_sha256=contract_sha
        )
        if (
            expected_contract_file_sha is not None
            and row["contract_file_sha256"] != expected_contract_file_sha
        ):
            raise RuntimeError(f"accepted source contract file SHA mismatch: {tag}")
        manifest_rows.append(row)
        contract_set.add(contract_sha)
        file_shas.add(row["contract_file_sha256"])
        player_tags.update(row["player_tags"])
    if len(contract_set) != 1 or len(file_shas) != 1:
        raise RuntimeError("accepted corpus mixes native contracts")

    payload = b"".join(_canonical_line(row) for row in manifest_rows)
    _atomic_bytes(output, payload)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    metadata = {
        "schema_version": 1,
        "kind": "cr_schema5_authoritative_manifest_v1",
        "production_ready": len(manifest_rows) == target,
        "accepted_battles": len(manifest_rows),
        "target_battles": target,
        "source_database": str(db_path.resolve()),
        "source_database_quick_check": quick_check,
        "authoritative_root": str(root),
        "authoritative_index": str(index_path),
        "authoritative_index_sha256": _sha256(index_path),
        "native_contract_sha256": next(iter(contract_set), None),
        "native_contract_file_sha256": next(iter(file_shas), None),
        "native_contract_path": (
            None if contract_path is None else str(contract_path)
        ),
        "unique_players": len(player_tags),
        "manifest_path": str(output.resolve()),
        "manifest_sha256": manifest_sha,
    }
    metadata_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_bytes(metadata_path, json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n")
    return {**metadata, "metadata_path": str(metadata_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--native-contract", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = freeze(
        db_path=args.db,
        authoritative_root=args.authoritative_root,
        output=args.output,
        target=args.target,
        allow_incomplete=args.allow_incomplete,
        native_contract_path=args.native_contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
