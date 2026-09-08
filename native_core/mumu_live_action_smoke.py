"""Bounded opt-in deployment smoke test; does not load an AI model or start games."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import re
import threading
import time

from .mumu_live_actions import ScreenLayout, UI_READY_MIN_TICK, card_receipt, own_player, send_card_taps
from .mumu_live_protocol import (BattleClockGuard, DEFAULT_ADB, DEFAULT_READER,
    DEFAULT_SERIAL, adb_run, install_reader, start_reader, stop_owned_reader, verify_runtime)

# Deliberately small, explicit test-card set. Unknown cards/skills are not guessed.
TEST_CARDS = {27000000: 3, 26000021: 4, 26000030: 1, 26000010: 1,
              26000014: 4, 26000038: 2, 28000000: 4, 28000011: 2}


def choose_test_action(frame: dict, side: int, number: int) -> dict | None:
    player = own_player(frame, side)
    if not player or len(player.get('deck_card_ids', [])) != 8:
        return None
    choices = []
    for slot, index in enumerate(player['hand_deck_indices']):
        if index not in range(8):
            continue
        card = player['deck_card_ids'][index]
        cost = TEST_CARDS.get(card)
        if cost is not None and player['elixir_raw'] >= cost * 10000:
            # First choice is stationary Cannon for coordinate read-back.
            priority = 0 if card == 27000000 else (1 if card == 26000021 else 2)
            choices.append((priority, cost, slot, card))
    if not choices:
        return None
    _, cost, slot, card = min(choices)
    row, col = (9, 8 if number % 2 == 0 else 11) if card == 27000000 else ((14, 4 if number % 2 == 0 else 13) if card == 26000021 else (8, 6 if number % 2 == 0 else 11))
    return {'slot': slot, 'card_id': card, 'cost': cost, 'cell': row * 18 + col}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adb', type=Path, default=DEFAULT_ADB)
    p.add_argument('--serial', default=DEFAULT_SERIAL)
    p.add_argument('--reader', type=Path, default=DEFAULT_READER)
    p.add_argument('--seconds', type=int, default=240)
    p.add_argument('--max-actions', type=int, default=6)
    p.add_argument('--execute', action='store_true', help='explicitly enable the bounded ordinary taps')
    p.add_argument('--output', type=Path, default=Path('artifacts/mumu-action-smoke'))
    args = p.parse_args(argv)
    if not 1 <= args.seconds <= 360 or not 1 <= args.max_actions <= 12:
        p.error('seconds 1..360; max-actions 1..12')
    runtime = verify_runtime(args.adb, args.serial)
    runtime['reader_sha256'] = install_reader(args.adb, args.serial, args.reader)
    sizes = re.findall(r'(\d+)x(\d+)', adb_run(args.adb, args.serial, 'shell', 'wm size'))
    if not sizes:
        raise RuntimeError('Cannot read Android screen size')
    layout = ScreenLayout.from_size(*map(int, sizes[-1]))
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    proc = start_reader(args.adb, args.serial, runtime['pid'], interval_ms=50, max_frames=(args.seconds + 5) * 20)
    inbox = queue.Queue(maxsize=32)
    errors = deque(maxlen=10)
    def pump():
        for line in proc.stdout:
            try:
                item = (json.loads(line), time.monotonic())
            except ValueError:
                continue
            try:
                inbox.put_nowait(item)
            except queue.Full:
                try: inbox.get_nowait()
                except queue.Empty: pass
                inbox.put_nowait(item)
    def drain_errors():
        for line in proc.stderr:
            errors.append(line.strip())
    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=drain_errors, daemon=True).start()
    guard = BattleClockGuard()
    started = time.monotonic()
    pending, battle_identity, reader_pid = None, None, None
    inactive_since = None
    confirmed = sent = frames = 0
    last_action = last_status = 0.0
    reason = 'deadline'
    events = []
    def emit(event, **fields):
        value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
        events.append(value)
        log.write(json.dumps(value, ensure_ascii=False) + '\n')
        log.flush()
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        with (output / 'events.jsonl').open('w', encoding='utf-8') as log:
            emit('armed_waiting', execute=args.execute, max_actions=args.max_actions,
                 max_battles=1, runtime=runtime, output=str(output), model_loaded=False)
            while time.monotonic() - started < args.seconds:
                try:
                    frame, received = inbox.get(timeout=.5)
                except queue.Empty:
                    if proc.poll() is not None:
                        raise RuntimeError('reader exited: ' + ' | '.join(errors))
                    continue
                try:
                    while True:
                        frame, received = inbox.get_nowait()
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now - received > .5:
                    continue
                reader_pid = frame.get('reader_pid')
                frames += 1
                health = guard.observe(frame, now)
                if now - last_status > 1:
                    status = {'event': 'progress', 'frames': frames, 'tick': frame.get('game_tick'),
                        'health': health, 'sent': sent, 'confirmed': confirmed, 'execute': args.execute}
                    (output / 'progress.json').write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
                    last_status = now
                if battle_identity is not None and health['epoch'] != battle_identity:
                    reason = 'battle_identity_changed'; break
                if health.get('can_control') and frame['game_tick'] >= UI_READY_MIN_TICK:
                    inactive_since = None
                    if battle_identity is None:
                        battle_identity = health['epoch']
                        emit('battle_bound', tick=frame['game_tick'], side=health['local_side'], chain=frame['chain'],
                             player=own_player(frame, health['local_side']))
                elif battle_identity is not None:
                    inactive_since = inactive_since or now
                    if now - inactive_since > 2:
                        reason = 'battle_ended_or_inactive'; break
                    continue
                else:
                    continue
                side = health['local_side']
                if pending:
                    receipt = card_receipt(pending['before'], frame, side=side,
                        slot=pending['action']['slot'], card_id=pending['action']['card_id'])
                    if receipt['accepted']:
                        confirmed += 1
                        emit('deployment_confirmed', action=pending['action'], receipt=receipt,
                             latency_ms=round((now - pending['at']) * 1000, 1))
                        pending = None
                    elif now - pending['at'] > 2:
                        emit('deployment_unconfirmed', action=pending['action'], receipt=receipt)
                        reason = 'unconfirmed_stop_no_retry'; break
                    continue
                if sent >= args.max_actions or now - last_action < 2:
                    continue
                action = choose_test_action(frame, side, sent)
                if action is None:
                    continue
                if not args.execute:
                    emit('dry_run_candidate', action=action, tick=frame['game_tick'])
                    reason = 'dry_run_no_touches'; break
                before = frame
                at = time.monotonic()
                coordinates = send_card_taps(args.adb, args.serial, layout, action['slot'], action['cell'], side=side)
                sent += 1
                pending = {'before': before, 'at': at, 'action': action}
                last_action = at
                emit('deployment_sent', action=action, tick=before['game_tick'], side=side, **coordinates)
            summary = {'runtime': runtime, 'output': str(output), 'execute': args.execute,
                'sent': sent, 'confirmed': confirmed, 'frames': frames, 'stop_reason': reason,
                'max_battles': 1, 'model_loaded': False, 'game_memory_written': False,
                'passed': args.execute and confirmed >= 3 and sent == confirmed and reason in ('battle_ended_or_inactive', 'battle_identity_changed')}
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            emit('smoke_finished', **summary)
            return 0 if summary['passed'] or not args.execute else 2
    finally:
        stop_owned_reader(args.adb, args.serial, reader_pid, runtime['pid'])
        if proc.poll() is None:
            proc.terminate()
        try: proc.wait(timeout=3)
        except Exception: proc.kill()


if __name__ == '__main__':
    raise SystemExit(main())
