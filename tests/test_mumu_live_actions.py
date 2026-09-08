from __future__ import annotations
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from native_core.mumu_live_actions import ScreenLayout, card_receipt, send_card_taps
from native_core.mumu_live_action_smoke import choose_test_action

def before():
    return {'pid': 1, 'game_tick': 100, 'coherent': True,
        'chain': {'battle': '0xa', 'player_state': '0xb'}, 'entities': [],
        'players': [{'side': 0, 'elixir_raw': 100000, 'hand_deck_indices': [0,1,2,3],
        'deck_card_ids': [27000000,26000021,26000030,26000010,28000000,28000011,26000014,26000038]}]}

class ActionTests(unittest.TestCase):
    def test_selected_slot_and_elixir_both_required(self):
        initial = before(); after = copy.deepcopy(initial); after['game_tick'] = 105
        after['players'][0]['hand_deck_indices'][0] = 4
        self.assertFalse(card_receipt(initial, after, side=0, slot=0, card_id=27000000)['accepted'])
        after['players'][0]['elixir_raw'] = 71000
        self.assertTrue(card_receipt(initial, after, side=0, slot=0, card_id=27000000)['accepted'])
        self.assertFalse(card_receipt(initial, after, side=0, slot=1, card_id=26000021)['accepted'])

    def test_other_battle_cannot_confirm(self):
        initial = before(); after = copy.deepcopy(initial)
        after['chain']['battle'] = '0xdead'
        self.assertEqual(card_receipt(initial, after, side=0, slot=0, card_id=27000000)['reason'], 'battle_changed')

    def test_taps_use_correct_hotbar_and_guest_50ms_wait(self):
        with patch('native_core.mumu_live_actions.adb_run') as call:
            result = send_card_taps(Path('adb'), 'serial', ScreenLayout.from_size(1080,1920), 3, 170)
            self.assertEqual(result['hand_screen'], (950,1709))
            self.assertIn('; sleep 0.05; input tap ', call.call_args.args[3])
            self.assertEqual(call.call_count, 1)

    def test_invalid_touch_is_rejected_before_adb(self):
        with patch('native_core.mumu_live_actions.adb_run') as call:
            with self.assertRaises(ValueError):
                send_card_taps(Path('adb'), 'serial', ScreenLayout.from_size(1080,1920), 4, 0)
            with self.assertRaises(ValueError):
                send_card_taps(Path('adb'), 'serial', ScreenLayout.from_size(1080,1920), 0, 576)
            call.assert_not_called()

    def test_side_one_horizontal_mapping_matches_actor_canonicalization(self):
        layout = ScreenLayout.from_size(1080,1920)
        left = layout.deployment_point(9*18+4, side=0)
        right = layout.deployment_point(9*18+4, side=1)
        self.assertEqual(left[1],right[1])
        self.assertEqual(left[0]+right[0],1080)
        self.assertEqual(layout.deployment_point(170, side=1),(567,1085))

    def test_test_policy_prefers_stationary_cannon_and_respects_elixir(self):
        initial = before()
        self.assertEqual(choose_test_action(initial,0,0)['card_id'],27000000)
        initial['players'][0]['elixir_raw'] = 9000
        self.assertIsNone(choose_test_action(initial,0,0))
