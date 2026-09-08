from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from native_core.mumu_live_protocol import BattleClockGuard, stop_owned_reader, verify_runtime, visible_sides


def frame(tick=100, both=False, battle='0x1000'):
    return {'schema_version': 2, 'pid': 123, 'battle_active': True, 'coherent': True,
        'game_tick': tick, 'sequence': tick, 'chain': {'root': '0x2000', 'context': '0x3000',
        'battle': battle, 'player_state': '0x4000', 'player_state_path': [0xA8]},
        'decoded_entity_count': 2, 'entities': [{'side': side, 'card_id': -1,
            'x': 9000, 'hp': 4000, 'max_hp': 4000} for side in (0, 1)],
        'players': [{'side': 0, 'hand_deck_indices': [0, 1, 2, 3], 'next_deck_index': 4},
                    {'side': 1, 'hand_deck_indices': [0, 1, 2, 3] if both else [-1]*4,
                     'next_deck_index': 4 if both else -1}]}


class ClockGuardTests(unittest.TestCase):
    def test_repeated_paused_frame_never_admits(self):
        guard = BattleClockGuard()
        for i in range(30):
            state = guard.observe(frame(), i / 10)
            self.assertFalse(state['can_control'])
        self.assertEqual(state['status'], 'paused_or_stalled')

    def test_requires_real_advancing_ticks_then_stops_when_stale(self):
        guard = BattleClockGuard()
        self.assertFalse(guard.observe(frame(100), 0)['can_control'])
        self.assertFalse(guard.observe(frame(101), 0.05)['can_control'])
        self.assertTrue(guard.observe(frame(102), 0.1)['can_control'])
        self.assertFalse(guard.observe(frame(102), 1.2)['can_control'])

    def test_both_hands_visible_never_enables_control_even_while_playing(self):
        guard = BattleClockGuard()
        for i in range(8):
            state = guard.observe(frame(100 + i, both=True), i / 10)
            self.assertFalse(state['can_control'])
        self.assertEqual(state['status'], 'both_hands_visible_readonly')

    def test_seek_or_battle_identity_change_resets_epoch_and_warmup(self):
        guard = BattleClockGuard()
        for i in range(3):
            last = guard.observe(frame(100 + i), i / 10)
        old_epoch = last['epoch']
        after = guard.observe(frame(10), 0.4)
        self.assertFalse(after['can_control'])
        self.assertGreater(after['epoch'], old_epoch)
        changed = guard.observe(frame(11, battle='0x5000'), 0.5)
        self.assertGreater(changed['epoch'], after['epoch'])

    def test_invalid_incoherent_or_unverified_chain_is_rejected(self):
        for change in ({'coherent': False}, {'battle_active': False}, {'game_tick': True},
                       {'schema_version': 1}, {'chain': {'battle': '0x1', 'player_state_path': [16, 152]}}):
            self.assertFalse(BattleClockGuard().observe({**frame(), **change}, 0)['can_control'])

    def test_invalid_duplicate_hand_does_not_identify_self(self):
        bad = frame()
        bad['players'][0]['hand_deck_indices'] = [0, 0, 0, 0]
        self.assertEqual(visible_sides(bad), [])

    def test_empty_scene_or_dead_king_never_admits(self):
        for bad in ({**frame(), 'entities': [], 'decoded_entity_count': 0},
                    {**frame(), 'entities': [{'side': 0, 'card_id': -1, 'x': 9000, 'hp': 0, 'max_hp': 4000}]}):
            guard = BattleClockGuard()
            for i in range(5):
                self.assertFalse(guard.observe({**bad, 'game_tick': 100 + i}, i / 10)['can_control'])

    def test_stop_verifies_exact_reader_command_not_game_or_another_process(self):
        with patch('native_core.mumu_live_protocol.root_run', return_value='something_else\x00123\x00') as run:
            stop_owned_reader(Path('adb'), 'serial', 555, 123)
            self.assertEqual(run.call_count, 1)
        with patch('native_core.mumu_live_protocol.root_run') as run:
            stop_owned_reader(Path('adb'), 'serial', 123, 123)
            run.assert_not_called()
        with patch('native_core.mumu_live_protocol.root_run', side_effect=['/data/local/tmp/mumu-live-reader-v2\x00123\x00100\x00', '']) as run:
            stop_owned_reader(Path('adb'), 'serial', 555, 123)
            self.assertEqual(run.call_args.args[2], 'kill -TERM 555')

    def test_runtime_rejects_changed_libg_hash(self):
        with patch('native_core.mumu_live_protocol.adb_run', side_effect=['device',
                  'versionCode=160402002 primaryCpuAbi=arm64-v8a', '123']), \
             patch('native_core.mumu_live_protocol.root_run', side_effect=[
                  '1000-2000 r-xp 00000000 00:00 0 /data/app/a/lib/arm64/libg.so',
                  '0'*64 + '  /data/app/a/lib/arm64/libg.so']):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                verify_runtime(Path('adb'), 'serial')


if __name__ == '__main__':
    unittest.main()
