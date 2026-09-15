"""Static yield survey: how many sources are eligible for the production contract
at all, before any native capture is spent on them.

Runs the authoritative processing contract (prepare_for_native + compile) on each
source and records the outcome and reason. No worker, no GPU: this is the ceiling
the capture stage can never exceed, and the cheapest number to measure over a
large archive.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .prepare_600k import atomic_json


def _count_reasons(reasons: list, counter) -> None:
    """Reason lists are comma-joined, and a card-mapping reason carries the card
    names after its own colon, so a bare token continues the previous reason."""
    previous = None
    for item in reasons:
        if ':' in item:
            previous = item.split(':', 1)[0]
            counter[previous] += 1
        elif previous is not None:
            counter[previous] += 1
        else:
            counter[item] += 1


def survey_value(value: dict, downloader_root: Path) -> tuple[bool, str | None, dict]:
    """Static eligibility and plan readiness for one already-loaded source."""
    from .processing_contract import prepare_for_native, compile_processing_battle
    started = time.perf_counter()
    try:
        prepared = prepare_for_native(value, downloader_root)
    except Exception as error:
        detail = dict(stage='prepare')
        message = str(error)
        if message.startswith('native static eligibility: '):
            detail['reasons'] = message.split(': ', 1)[1].split(',')
        return False, f'prepare:{type(error).__name__}:{message[:300]}', detail
    try:
        plan = compile_processing_battle(prepared)
    except Exception as error:
        return False, f'compile:{type(error).__name__}:{str(error)[:300]}', {}
    detail = dict(stage='plan', native_replay_ready=bool(getattr(plan, 'native_replay_ready', False)),
                  replay_tier=str(getattr(plan, 'replay_tier', '')),
                  gamemode=getattr(plan, 'native_execution_game_mode_id', None),
                  actions=len(getattr(plan, 'actions', ()) or ()),
                  abilities=len(getattr(plan, 'ability_events', ()) or ()),
                  duration_ticks=getattr(plan, 'duration_ticks', None),
                  seconds=round(time.perf_counter() - started, 3))
    return bool(detail['native_replay_ready']), (None if detail['native_replay_ready'] else 'plan_not_ready'), detail


def survey_source(source_path: Path, downloader_root: Path) -> tuple[bool, str | None, dict]:
    started = time.perf_counter()
    try:
        value = json.loads(Path(source_path).read_text(encoding='utf-8'))
    except Exception as error:
        return False, f'read:{type(error).__name__}:{error}', {}
    return survey_value(value, downloader_root)


def survey_normalized(shards: list, downloader_root: Path, output: Path, limit: int = 0) -> dict:
    """Survey sources straight out of the normalized shards (no extraction)."""
    import io
    import zstandard
    from collections import Counter
    results = []
    reasons = Counter()
    schemas = Counter()
    started = time.perf_counter()
    for shard in shards:
        with Path(shard).open('rb') as handle:
            with zstandard.ZstdDecompressor().stream_reader(handle) as raw:
                with io.TextIOWrapper(raw, encoding='utf-8') as text:
                    for line in text:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        value = record.get('source_payload') or {}
                        tag = (record.get('source') or {}).get('battle_tag', '?')
                        schemas[value.get('schema_version')] += 1
                        eligible, reason, detail = survey_value(value, downloader_root)
                        if not eligible:
                            _count_reasons(detail.get('reasons') or [reason or 'unknown'], reasons)
                        results.append(dict(battle_tag=tag, eligible=eligible, reason=reason,
                                            schema_version=value.get('schema_version'), **detail))
                        if limit and len(results) >= limit:
                            break
        if limit and len(results) >= limit:
            break
    eligible = [row for row in results if row['eligible']]
    by_schema = {}
    for row in results:
        bucket = by_schema.setdefault(str(row.get('schema_version')), [0, 0])
        bucket[0] += 1
        bucket[1] += int(row['eligible'])
    summary = dict(schema='r0-static-plan-yield.v1', source='normalized', surveyed=len(results),
                   eligible=len(eligible), ineligible=len(results) - len(eligible),
                   yield_rate=round(len(eligible) / len(results), 4) if results else None,
                   source_schemas=dict(schemas), by_schema={k: dict(surveyed=v[0], eligible=v[1],
                                                                    rate=round(v[1] / v[0], 4))
                                                            for k, v in sorted(by_schema.items())},
                   reason_counts=dict(reasons.most_common()),
                   gamemodes=dict(Counter(row.get('gamemode') for row in eligible).most_common()),
                   seconds_total=round(time.perf_counter() - started, 1),
                   seconds_per_source=round((time.perf_counter() - started) / len(results), 4) if results else None,
                   battles=results[:200])
    atomic_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--normalized', type=Path, nargs='*')
    parser.add_argument('--downloader-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.normalized:
        summary = survey_normalized(args.normalized, args.downloader_root, args.output, limit=args.limit)
        print(json.dumps({k: v for k, v in summary.items() if k != 'battles'}, ensure_ascii=False, indent=2))
        return
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    records = selection['records']
    if args.limit:
        records = records[:args.limit]
    results = []
    reasons = Counter()
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        eligible, reason, detail = survey_source(Path(record['source_file']), args.downloader_root)
        if not eligible:
            _count_reasons(detail.get('reasons') or [reason or 'unknown'], reasons)
        results.append(dict(battle_tag=record['battle_tag'], eligible=eligible, reason=reason, **detail))
        if index % 10 == 0 or index == len(records):
            print(f'  surveyed {index}/{len(records)} eligible={sum(1 for r in results if r["eligible"])}', flush=True)
    eligible = [row for row in results if row['eligible']]
    summary = dict(schema='r0-static-plan-yield.v1', surveyed=len(results), eligible=len(eligible),
                   ineligible=len(results) - len(eligible),
                   yield_rate=round(len(eligible) / len(results), 4) if results else None,
                   reason_counts=dict(reasons.most_common()),
                   gamemodes=dict(Counter(row.get('gamemode') for row in eligible).most_common()),
                   seconds_total=round(time.perf_counter() - started, 1),
                   seconds_per_source=round((time.perf_counter() - started) / len(results), 3) if results else None,
                   battles=results)
    atomic_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'battles'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
