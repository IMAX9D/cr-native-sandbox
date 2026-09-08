"""Read-only pointer-chain capture. Never loads a model or starts another AVD."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time

from .mumu_live_protocol import (BattleClockGuard, DEFAULT_ADB, DEFAULT_READER,
    DEFAULT_SERIAL, install_reader, start_reader, stop_owned_reader, verify_runtime)


def summarize_changes(frames: list[dict]) -> dict:
    valid = [row for row in frames if row.get('battle_active') and row.get('coherent')]
    previous = None
    moved, changed_hands, elixir_drops, hp_changes = set(), {0: 0, 1: 0}, {0: 0, 1: 0}, 0
    elixir_ranges = {0: [], 1: []}
    for row in valid:
        players = {p['side']: p for p in row['players']}
        entities = {e['address']: e for e in row.get('entities', [])}
        for side, player in players.items():
            elixir_ranges[side].append(player['elixir_raw'])
        if previous is not None and previous['chain']['battle'] == row['chain']['battle'] and row['game_tick'] >= previous['game_tick']:
            old_players = {p['side']: p for p in previous['players']}
            old_entities = {e['address']: e for e in previous.get('entities', [])}
            for side, player in players.items():
                old = old_players.get(side)
                if old:
                    changed_hands[side] += int(old['hand_deck_indices'] != player['hand_deck_indices'])
                    elixir_drops[side] += int(player['elixir_raw'] < old['elixir_raw'])
            for address, entity in entities.items():
                old = old_entities.get(address)
                if old and old['category'] == entity['category']:
                    if (old['x'], old['y']) != (entity['x'], entity['y']):
                        moved.add((address, entity['category']))
                    hp_changes += int(old['hp'] != entity['hp'])
        previous = row
    return {'moving_entities_observed': len(moved), 'hand_transitions': changed_hands,
        'elixir_decreases': elixir_drops, 'entity_hp_changes': hp_changes,
        'elixir_raw_range': {side: [min(v), max(v)] if v else None for side, v in elixir_ranges.items()},
        'direct_player_path_frames': sum(row['chain']['player_state_path'] == [0xA8] for row in valid),
        'fallback_discovery_frames': sum(row.get('discovery_nodes', 0) > 0 for row in frames),
        'max_filtered_object_count': max((row.get('filtered_object_count', 0) for row in frames), default=0)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', type=Path, default=DEFAULT_ADB)
    parser.add_argument('--serial', default=DEFAULT_SERIAL)
    parser.add_argument('--reader', type=Path, default=DEFAULT_READER)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--interval-ms', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('artifacts/mumu-live-probes'))
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 3600 or not 20 <= args.interval_ms <= 5000:
        parser.error('seconds must be 0..3600; interval-ms must be 20..5000')
    runtime = verify_runtime(args.adb, args.serial)
    runtime['reader_sha256'] = install_reader(args.adb, args.serial, args.reader)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output / stamp
    output.mkdir(parents=True, exist_ok=False)
    count = max(2, int(args.seconds * 1000 / args.interval_ms))
    process = start_reader(args.adb, args.serial, runtime['pid'], interval_ms=args.interval_ms, max_frames=count)
    guard = BattleClockGuard()
    started = time.monotonic()
    frames, states, reader_pid = [], {}, None
    try:
        # communicate(timeout) drains both pipes and cannot hang indefinitely.
        raw, error = process.communicate(timeout=args.seconds + 15)
        if process.returncode:
            raise RuntimeError(f'reader exited {process.returncode}: {error[-500:]}')
        with (output / 'frames.jsonl').open('w', encoding='utf-8') as handle:
            for line in raw.splitlines():
                frame = json.loads(line)
                if frame.get('event') != 'mumu_live_frame':
                    continue
                reader_pid = frame.get('reader_pid')
                health = guard.observe(frame, now=frame['sample_monotonic_us'] / 1_000_000)
                states[health['status']] = states.get(health['status'], 0) + 1
                handle.write(json.dumps({'frame': frame, 'health': health}, ensure_ascii=False) + '\n')
                frames.append(frame)
        valid = [frame for frame in frames if frame.get('battle_active') and frame.get('coherent')]
        tick_values = [frame['game_tick'] for frame in valid]
        summary = {'runtime': runtime, 'output': str(output), 'frames': len(frames),
            'coherent_frames': len(valid), 'tick_first': tick_values[0] if tick_values else None,
            'tick_last': tick_values[-1] if tick_values else None,
            'distinct_ticks': len(set(tick_values)), 'health_counts': states,
            'read_us_median': statistics.median([frame['read_us'] for frame in valid]) if valid else None,
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'first_chain': valid[0]['chain'] if valid else None,
            'last_entity_count': valid[-1]['decoded_entity_count'] if valid else None,
            'touches_sent': 0, 'model_loaded': False}
        summary['observed_changes'] = summarize_changes(frames)
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if valid else 2
    finally:
        if process.poll() is None:
            stop_owned_reader(args.adb, args.serial, reader_pid, runtime['pid'])
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == '__main__':
    raise SystemExit(main())
