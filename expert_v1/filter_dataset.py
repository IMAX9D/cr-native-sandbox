"""Classify collected replay JSON into confirmed, uncertain and rejected 1v1 sets."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

try:
    import orjson
except ImportError:  # pragma: no cover - slower portable fallback
    orjson = None


BUTTON_RE = re.compile(
    r"<button[^>]*\b(?:matchup_button|replay_button)\b[^>]*>", re.S
)
ATTR_RE = re.compile(r'([:\w-]+)\s*=\s*"([^"]*)"')
FORM_SUFFIX_RE = re.compile(r"-(?:ev\d+|hero)$")


def _loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def _dumps(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def base_slug(value: str) -> str:
    return FORM_SUFFIX_RE.sub("", str(value).strip().lower())


def _attrs(tag: str) -> dict[str, str]:
    return {
        key: html.unescape(value)
        for key, value in ATTR_RE.findall(tag)
    }


def load_list_metadata(list_root: Path) -> dict[str, dict[str, Any]]:
    """Recover exact decks/forms and 1v1/draft metadata from saved list HTML."""
    result: dict[str, dict[str, Any]] = {}
    for path in list_root.glob("*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        matchup: dict[str, dict[str, str]] = {}
        replay: dict[str, dict[str, str]] = {}
        for match in BUTTON_RE.finditer(text):
            tag = match.group(0)
            attrs = _attrs(tag)
            index = attrs.get("data-index", "")
            if not index:
                continue
            classes = attrs.get("class", "")
            if "matchup_button" in classes:
                matchup[index] = attrs
            elif "replay_button" in classes:
                replay[index] = attrs
        for index, replay_attrs in replay.items():
            battle_tag = replay_attrs.get("data-replay", "").strip()
            match_attrs = matchup.get(index, {})
            if not battle_tag or not match_attrs:
                continue
            team_deck = [
                item.strip() for item in match_attrs.get("data-team-deck", "").split(",")
                if item.strip()
            ]
            opponent_deck = [
                item.strip() for item in match_attrs.get("data-opponent-deck", "").split(",")
                if item.strip()
            ]
            candidate = {
                "source_list": str(path),
                "players": match_attrs.get("data-players"),
                "draft": replay_attrs.get("data-draft"),
                "game_mode_id": match_attrs.get("data-game-mode-id"),
                "team_deck": team_deck,
                "opponent_deck": opponent_deck,
            }
            current = result.get(battle_tag)
            if current is None or (
                len(team_deck) == 8 and len(opponent_deck) == 8
                and not (
                    len(current.get("team_deck", [])) == 8
                    and len(current.get("opponent_deck", [])) == 8
                )
            ):
                result[battle_tag] = candidate
    return result


def load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with index_path.open("rb") as handle:
        for raw in handle:
            if b'"kind": "battle"' not in raw and b'"kind":"battle"' not in raw:
                continue
            try:
                item = _loads(raw)
                parsed = urlparse(str(item.get("url", "")))
                query = parse_qs(parsed.query)
                tag = (query.get("tag") or [""])[0]
                if not tag:
                    continue
                result[tag] = {
                    "saved_path": item.get("saved_path"),
                    "team_tags": (query.get("team_tags") or [""])[0].split(","),
                    "opponent_tags": (query.get("opponent_tags") or [""])[0].split(","),
                    "team_crowns": int((query.get("team_crowns") or [-1])[0]),
                    "opponent_crowns": int((query.get("opponent_crowns") or [-1])[0]),
                    "fetched_at": item.get("fetched_at"),
                }
            except Exception:
                continue
    return result


def _validate_battle(
    path: Path,
    root: Path,
    index: dict[str, dict[str, Any]],
    list_metadata: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    reasons: list[str] = []
    try:
        value = _loads(path.read_bytes())
    except Exception as error:
        return "rejected", {
            "path": str(path.relative_to(root)),
            "reasons": ["json_error"],
            "detail": str(error),
        }
    battle_tag = str(value.get("battle_tag") or "").strip()
    plays = value.get("card_plays")
    duration = value.get("duration_seconds")
    if not battle_tag:
        reasons.append("missing_battle_tag")
    if not isinstance(plays, list) or not plays:
        reasons.append("missing_card_plays")
        plays = []
    # RoyaleAPI replay duration can include the final tiebreak/settlement seconds,
    # so an ordinary five-minute 1v1 may be reported slightly above 300s.
    if not isinstance(duration, int) or isinstance(duration, bool) or not 0 < duration <= 360:
        reasons.append("invalid_duration")

    cards = {"team": set(), "opponent": set()}
    previous_tick = -1
    same_tick = False
    for play in plays:
        if not isinstance(play, dict):
            reasons.append("invalid_play_object")
            continue
        side = play.get("side")
        if side not in cards:
            reasons.append("invalid_side")
            continue
        card = str(play.get("card") or "").strip().lower()
        if not card or card == "_invalid":
            reasons.append("invalid_card")
        else:
            cards[side].add(base_slug(card))
        tick = play.get("time_raw")
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            reasons.append("invalid_tick")
        elif tick < previous_tick:
            reasons.append("nonmonotonic_tick")
        else:
            same_tick = same_tick or tick == previous_tick
            previous_tick = tick
        x, y = play.get("x"), play.get("y")
        if not isinstance(x, (int, float)) or not 0 <= x <= 18000:
            reasons.append("invalid_x")
        if not isinstance(y, (int, float)) or not 0 <= y <= 32000:
            reasons.append("invalid_y")
    if len(cards["team"]) > 8:
        reasons.append("team_more_than_8_cards")
    if len(cards["opponent"]) > 8:
        reasons.append("opponent_more_than_8_cards")

    indexed = index.get(battle_tag)
    if indexed is None:
        reasons.append("missing_index_metadata")
        indexed = {}
    team_tags = [tag for tag in indexed.get("team_tags", []) if tag]
    opponent_tags = [tag for tag in indexed.get("opponent_tags", []) if tag]
    if len(team_tags) != 1:
        reasons.append(f"team_size_{len(team_tags)}")
    if len(opponent_tags) != 1:
        reasons.append(f"opponent_size_{len(opponent_tags)}")
    for key in ("team_crowns", "opponent_crowns"):
        value_crowns = indexed.get(key, -1)
        if not isinstance(value_crowns, int) or not 0 <= value_crowns <= 3:
            reasons.append(f"invalid_{key}")

    metadata = list_metadata.get(battle_tag)
    metadata_reasons: list[str] = []
    full_decks = False
    if metadata is not None:
        if metadata.get("players") != "1v1":
            metadata_reasons.append("list_not_1v1")
        if str(metadata.get("draft")) not in ("0", "False", "false"):
            metadata_reasons.append("draft_or_special_selection")
        team_deck = metadata.get("team_deck", [])
        opponent_deck = metadata.get("opponent_deck", [])
        full_decks = len(team_deck) == 8 and len(opponent_deck) == 8
        if not full_decks:
            metadata_reasons.append("incomplete_list_deck")
        else:
            team_base = {base_slug(card) for card in team_deck}
            opponent_base = {base_slug(card) for card in opponent_deck}
            if not cards["team"].issubset(team_base):
                metadata_reasons.append("team_play_not_in_deck")
            if not cards["opponent"].issubset(opponent_base):
                metadata_reasons.append("opponent_play_not_in_deck")
    reasons.extend(metadata_reasons)
    reasons = sorted(set(reasons))
    hard_reasons = [
        reason for reason in reasons
        if reason not in {"missing_index_metadata"}
    ]
    record = {
        "schema_version": 1,
        "battle_tag": battle_tag,
        "path": str(path.relative_to(root)),
        "duration_seconds": duration,
        "event_count": len(plays),
        "team_unique_played_cards": len(cards["team"]),
        "opponent_unique_played_cards": len(cards["opponent"]),
        "same_tick_events": same_tick,
        "index": indexed,
        "list_metadata": metadata,
        "full_decks": full_decks,
        "reasons": reasons,
    }
    if hard_reasons:
        return "rejected", record
    if metadata is not None and full_decks:
        return "confirmed_1v1", record
    return "uncertain_1v1", record


def _process_shard(
    directory: Path,
    battle_root: Path,
    index: dict[str, dict[str, Any]],
    list_metadata: dict[str, dict[str, Any]],
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    files = list(directory.glob("*.json"))
    if limit > 0:
        files = files[:limit]
    return [
        _validate_battle(path, battle_root, index, list_metadata)
        for path in files
    ]


def quarantine_group(reasons: Iterable[str]) -> str:
    reason_set = set(reasons)
    if {"team_size_2", "opponent_size_2"} & reason_set:
        return "definite_2v2"
    if {"missing_battle_tag", "missing_card_plays"} & reason_set:
        return "corrupt_or_empty"
    if {"list_not_1v1", "draft_or_special_selection"} & reason_set:
        return "confirmed_special_mode"
    if {
        "invalid_duration", "invalid_team_crowns", "invalid_opponent_crowns"
    } & reason_set:
        return "invalid_result_metadata"
    if {"team_more_than_8_cards", "opponent_more_than_8_cards"} & reason_set:
        return "multi_deck_or_parser_anomaly"
    return "other"


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    battle_root = dataset_root / "data" / "raw" / "battles"
    list_root = dataset_root / "data" / "raw" / "lists"
    index_path = dataset_root / "data" / "index.jsonl"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index = load_index(index_path)
    list_metadata = load_list_metadata(list_root)
    outputs = {
        name: (output_root / f"{name}.jsonl").open("wb")
        for name in ("confirmed_1v1", "uncertain_1v1", "rejected")
    }
    counters: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    quarantine: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    started = time.perf_counter()
    shards = sorted(path for path in battle_root.iterdir() if path.is_dir())
    if args.limit > 0:
        shards = shards[: max(1, (args.limit + 999) // 1000)]
    remaining = args.limit
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for shard in shards:
                shard_limit = 0 if remaining <= 0 else min(remaining, 1000)
                futures.append(executor.submit(
                    _process_shard, shard, battle_root, index, list_metadata,
                    shard_limit,
                ))
                if remaining > 0:
                    remaining -= shard_limit
                    if remaining <= 0:
                        break
            for future in as_completed(futures):
                for classification, record in future.result():
                    outputs[classification].write(_dumps(record))
                    counters[classification] += 1
                    counters["events"] += int(record.get("event_count", 0))
                    counters["both_observed_8"] += int(
                        record.get("team_unique_played_cards") == 8
                        and record.get("opponent_unique_played_cards") == 8
                    )
                    for reason in record.get("reasons", []):
                        reasons[str(reason)] += 1
                    if classification == "rejected":
                        quarantine[quarantine_group(record.get("reasons", []))] += 1
                    metadata = record.get("list_metadata") or {}
                    if metadata.get("game_mode_id"):
                        modes[str(metadata["game_mode_id"])] += 1
    finally:
        for handle in outputs.values():
            handle.close()
    total = sum(counters[name] for name in (
        "confirmed_1v1", "uncertain_1v1", "rejected"
    ))
    summary = {
        "schema_version": 1,
        "kind": "expert_dataset_filter_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "records": total,
        "events": counters["events"],
        "confirmed_1v1": counters["confirmed_1v1"],
        "uncertain_1v1": counters["uncertain_1v1"],
        "rejected": counters["rejected"],
        "both_sides_observed_8_cards": counters["both_observed_8"],
        "index_battles": len(index),
        "saved_list_decks": len(list_metadata),
        "rejection_reasons": dict(reasons.most_common()),
        "quarantine_groups": dict(quarantine.most_common()),
        "confirmed_game_mode_ids": dict(modes.most_common()),
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_root / "summary.json").write_bytes(
        json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(r"D:\皇室战争数据集"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\expert-v1\dataset-audit"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 64:
        raise ValueError("workers must be in 1..64")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
