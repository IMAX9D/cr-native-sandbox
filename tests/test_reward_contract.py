from __future__ import annotations

from copy import deepcopy
import unittest

from training.schema import DefensiveTowerReward, PotentialReward


def _state(side0_damage: int = 0, side1_damage: int = 0) -> dict:
    towers = []
    maxima = (4824, 3052, 3052)
    for side, damage in ((0, side0_damage), (1, side1_damage)):
        remaining_damage = damage
        for index, maximum in enumerate(maxima):
            applied = min(maximum, remaining_damage)
            remaining_damage -= applied
            towers.append({
                "side": side,
                "type": "king" if index == 0 else "princess",
                "hp": maximum - applied,
                "max_hp": maximum,
            })
    return {
        "episode": {"crown_towers": towers},
        "players": [{"side": 0, "elixir": 10}, {"side": 1, "elixir": 0}],
        "entities": [{"side": 0, "card_id": 26000003, "hp": 9999, "max_hp": 9999}],
    }


class RewardContractTests(unittest.TestCase):
    def test_defensive_dense_reward_matches_reference_weights(self):
        reward = DefensiveTowerReward()
        previous = _state(side0_damage=0, side1_damage=0)
        current = _state(side0_damage=1000, side1_damage=2000)
        value = reward.transition(previous, current)
        self.assertAlmostEqual(value[0], 2.0 - 1.2)
        self.assertAlmostEqual(value[1], 1.0 - 2.4)

    def test_defensive_dense_reward_adds_towers_and_terminal_outcome(self):
        reward = DefensiveTowerReward()
        previous = _state()
        current = _state(side1_damage=4824)
        value = reward.transition(previous, current)
        self.assertAlmostEqual(value[0], 4.824 + 5.0)
        self.assertAlmostEqual(value[1], -5.7888 - 5.0)
        terminal = reward.transition(
            current,
            None,
            terminal_rewards={0: 1.0, 1: -1.0},
            done=True,
        )
        self.assertEqual(terminal, {0: 10.0, 1: -10.0})

    def test_potential_is_only_normalized_total_tower_hp_difference(self):
        reward = PotentialReward(gamma=0.99995, shaping_scale=0.20)
        state = _state(side0_damage=1200, side1_damage=2800)
        expected = (1.0 - 1200 / 10928) - (1.0 - 2800 / 10928)
        self.assertAlmostEqual(reward.potential(state, 0), expected)
        self.assertAlmostEqual(reward.potential(state, 1), -expected)

        changed = deepcopy(state)
        changed["players"][0]["elixir"] = 0
        changed["players"][1]["elixir"] = 10
        changed["entities"] = [{
            "side": 1, "card_id": 26000003, "hp": 1, "max_hp": 999999
        }]
        self.assertEqual(reward.potential(changed, 0), reward.potential(state, 0))

    def test_transition_matches_frozen_formula_and_is_zero_sum(self):
        gamma = 0.99995
        reward = PotentialReward(gamma=gamma, shaping_scale=0.20)
        previous = _state(side0_damage=1000, side1_damage=2000)
        current = _state(side0_damage=1100, side1_damage=2400)
        value = reward.transition(previous, current)
        expected = 0.20 * (
            gamma * reward.potential(current, 0)
            - reward.potential(previous, 0)
        )
        self.assertAlmostEqual(value[0], expected)
        self.assertAlmostEqual(value[1], -expected)

        terminal = reward.transition(
            current,
            None,
            terminal_rewards={0: 1.0, 1: -1.0},
            done=True,
        )
        self.assertAlmostEqual(
            terminal[0], 1.0 - 0.20 * reward.potential(current, 0)
        )
        self.assertAlmostEqual(terminal[1], -terminal[0])

    def test_discounted_shaping_telescopes_to_terminal_zero_potential(self):
        gamma = 0.99995
        scale = 0.20
        reward = PotentialReward(gamma=gamma, shaping_scale=scale)
        states = [
            _state(side0_damage=800, side1_damage=100),
            _state(side0_damage=900, side1_damage=500),
            _state(side0_damage=1200, side1_damage=1700),
        ]
        first = reward.transition(states[0], states[1])[0]
        second = reward.transition(states[1], states[2])[0]
        terminal = reward.transition(states[2], None, done=True)[0]
        discounted = first + gamma * second + gamma * gamma * terminal
        self.assertAlmostEqual(
            discounted, -scale * reward.potential(states[0], 0), places=7
        )


if __name__ == "__main__":
    unittest.main()
