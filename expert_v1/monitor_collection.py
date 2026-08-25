"""Continuously validate schema-v3 downloads and stop at a clean target."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

from .upgrade_base_cycles import solve_cycle


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def dump_line(value: Any) -> bytes:
    encoded = orjson.dumps(value) if orjson is not None else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return encoded + b"\n"


def line_tags(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open("rb") as handle:
        for raw in handle:
            tag = str(loads(raw).get("battle_tag") or "")
            if tag:
                result.add(tag)
    return result


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def production_pid(crawler_root: Path) -> int:
    try:
        value = loads((crawler_root / "logs" / "production.lock").read_bytes())
        return int(value.get("pid") or 0)
    except Exception:
        return 0


def db_done(crawler_root: Path) -> int:
    db = crawler_root / "data" / "progress.sqlite3"
    connection = sqlite3.connect(db, timeout=10)
    try:
        return int(connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='done' AND kind='detail'"
        ).fetchone()[0])
    finally:
        connection.close()


def validate(value: dict[str, Any], min_timestamp: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(value.get("schema_version") or 0) < 3:
        reasons.append("not_schema_v3")
    timestamp = value.get("timestamp")
    try:
        timestamp = int(timestamp)
        if timestamp > 10_000_000_000:
            timestamp //= 1000
    except (TypeError, ValueError):
        reasons.append("timestamp_missing")
        timestamp = 0
    if timestamp < min_timestamp:
        reasons.append("before_version_window")
    if value.get("draft") is not False:
        reasons.append("draft_or_unknown")
    rounds = value.get("rounds") or []
    if not (
        len(rounds) == 1
        and len(rounds[0].get("team") or []) == 1
        and len(rounds[0].get("opponent") or []) == 1
    ):
        reasons.append("not_standard_1v1")
    plays = value.get("card_plays") or []
    counts = value.get("card_counts") or {}
    for side in ("team", "opponent"):
        deck = value.get(f"{side}_deck") or []
        player = (rounds[0].get(side) or [{}])[0] if rounds else {}
        levels = player.get("card_levels") or {}
        if (
            len(deck) != 8 or len(set(deck)) != 8
            or any(not card or card == "_invalid" for card in deck)
            or len(levels) != 8
        ):
            reasons.append(f"{side}_deck_incomplete")
        base_deck = list((counts.get(side) or {}).keys())
        if len(base_deck) != 8:
            reasons.append(f"{side}_observed_cycle_incomplete")
            continue
        index = {card: position for position, card in enumerate(base_deck)}
        sequence = [
            index[str(play.get("card"))]
            for play in plays if play.get("side") == side and str(play.get("card")) in index
        ]
        solved = solve_cycle(sequence)
        if not solved["cycle_valid"]:
            reasons.append(f"{side}_cycle_invalid")
    abilities = value.get("ability_plays")
    if not isinstance(abilities, list):
        reasons.append("ability_events_missing")
    else:
        elixir = value.get("elixir_stats") or {}
        for side in ("team", "opponent"):
            expected = int(((((elixir.get(side) or {}).get("Ability") or {}).get("count")) or 0))
            observed = sum(1 for item in abilities if item.get("side") == side)
            if expected != observed:
                reasons.append(f"{side}_ability_count_mismatch")
    for key in ("team_crowns", "opponent_crowns"):
        if not isinstance(value.get(key), int):
            reasons.append(f"{key}_missing")
    return not reasons, sorted(set(reasons))


def run(args: argparse.Namespace) -> int:
    crawler = args.crawler_root.resolve(strict=True)
    index = (crawler / "data" / "index.jsonl").resolve(strict=True)
    base_manifest = args.base_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "accepted-live.jsonl"
    rejected_path = output / "rejected-live.jsonl"
    state_path = output / "state.json"
    base_tags = line_tags(base_manifest)
    accepted_tags = line_tags(accepted_path)
    rejected_tags = line_tags(rejected_path)
    processed = accepted_tags | rejected_tags
    offset = 0
    if state_path.exists():
        state = loads(state_path.read_bytes())
        offset = int(state.get("index_offset") or 0)
        if offset > index.stat().st_size:
            offset = 0

    system_python = args.crawler_python.resolve(strict=True)
    while True:
        new_rows = accepted_new = rejected_new = 0
        with index.open("rb") as handle:
            handle.seek(offset)
            with accepted_path.open("ab") as good, rejected_path.open("ab") as bad:
                for raw in handle:
                    new_rows += 1
                    item = loads(raw)
                    if item.get("kind") != "battle":
                        continue
                    saved = Path(str(item.get("saved_path") or ""))
                    source = saved if saved.is_absolute() else crawler / saved
                    try:
                        value = loads(source.resolve(strict=True).read_bytes())
                    except Exception as error:
                        bad.write(dump_line({
                            "battle_tag": "", "source_path": str(source),
                            "reasons": ["source_read_error"], "detail": str(error),
                        }))
                        rejected_new += 1
                        continue
                    tag = str(value.get("battle_tag") or "")
                    if not tag or tag in base_tags or tag in processed:
                        continue
                    valid, reasons = validate(value, args.min_timestamp)
                    record = {
                        "battle_tag": tag,
                        "source_path": str(source.resolve()),
                        "batch": "live-schema3",
                        "schema_version": int(value.get("schema_version") or 0),
                        "version_timestamp": int(value.get("timestamp") or 0),
                        "version_timestamp_quality": "exact",
                        "team_tags": value.get("team_tags", []),
                        "opponent_tags": value.get("opponent_tags", []),
                        "team_crowns": value.get("team_crowns"),
                        "opponent_crowns": value.get("opponent_crowns"),
                        "terminal_provenance": "schema_v3_list_metadata",
                    }
                    if valid:
                        good.write(dump_line(record))
                        accepted_tags.add(tag)
                        accepted_new += 1
                    else:
                        bad.write(dump_line({**record, "reasons": reasons}))
                        rejected_tags.add(tag)
                        rejected_new += 1
                    processed.add(tag)
            offset = handle.tell()

        total = len(base_tags) + len(accepted_tags)
        pid = production_pid(crawler)
        active = pid_alive(pid)
        state = {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "index_offset": offset,
            "base_accepted": len(base_tags),
            "live_accepted": len(accepted_tags),
            "live_rejected": len(rejected_tags),
            "total_accepted": total,
            "target": args.target,
            "remaining": max(0, args.target - total),
            "crawler_active": active,
            "crawler_done": db_done(crawler),
            "last_scan_rows": new_rows,
            "last_scan_accepted": accepted_new,
            "last_scan_rejected": rejected_new,
        }
        temp = state_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(state_path)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if total >= args.target:
            if active:
                subprocess.run(
                    [str(system_python), "-m", "crawler.production", "stop"],
                    cwd=str(crawler), check=False,
                )
            (output / "TARGET_REACHED").write_text(
                state["updated_utc"] + "\n", encoding="utf-8"
            )
            return 0
        if not active:
            subprocess.run(
                [str(system_python), "-m", "crawler.production", "start"],
                cwd=str(crawler), check=False,
            )
        time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawler-root", type=Path, required=True)
    parser.add_argument("--crawler-python", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-timestamp", type=int, required=True)
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--interval", type=float, default=30.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
