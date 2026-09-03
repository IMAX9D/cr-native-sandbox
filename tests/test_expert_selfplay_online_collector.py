from __future__ import annotations

from copy import deepcopy
import threading
import time
import unittest

import torch
from torch import Tensor, nn

from expert_selfplay_v1.batched_policy import BatchedPolicyService, SampledPolicyAction
from expert_selfplay_v1.decks import DeckFixture
from expert_selfplay_v1.native_observation import NativeObservationEncoder
from expert_selfplay_v1.online_collector import (
    OnlineEpisodeSpec,
    OnlineSelfPlayCollector,
)
from expert_selfplay_v1.rollout import EpisodeHeader
from expert_selfplay_v1.rollout_storage import LearnerEpisodeChunker
from expert_v1.training_v1.model import ExpertPolicyConfig, ExpertPolicyOutput


CARDS = (
    26_000_000, 26_000_001, 26_000_003, 26_000_010,
    26_000_014, 26_000_021, 27_000_000, 28_000_001,
)
CARD_MAP = {card_id: index + 1 for index, card_id in enumerate(CARDS)}


def _towers(*, enemy_damage: int = 0) -> list[dict]:
    answer = []
    for side in (0, 1):
        for index, (x, y, maximum) in enumerate((
            (9000, 3000 if side == 0 else 29000, 4000),
            (3500, 6500 if side == 0 else 25500, 3000),
            (14500, 6500 if side == 0 else 25500, 3000),
        )):
            damage = enemy_damage if side == 1 and index == 1 else 0
            row = {
                "side": side,
                "type": "king" if index == 0 else "princess",
                "x": x, "y": y, "hp": maximum - damage, "max_hp": maximum,
            }
            if index:
                row["lane"] = "left" if index == 1 else "right"
            answer.append(row)
    return answer


def _state(tick: int, *, enemy_damage: int = 0, entities: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "kind": "libg_native_train_state_v1",
        "coherent": True,
        "tick": tick,
        "entity_count": len(entities or []),
        "players": [
            {"side": 0, "elixir_raw": 100_000, "hand_deck_indices": [0, 1, 2, 3],
             "next_deck_index": 4, "refill_timer": 0},
            {"side": 1, "elixir_raw": 100_000, "hand_deck_indices": [0, 1, 2, 3],
             "next_deck_index": 4, "refill_timer": 0},
        ],
        "entities": deepcopy(entities or []),
        "episode": {
            "terminated": False, "truncated": False, "outcome": "ongoing",
            "crowns": [0, 0], "commands_allowed": True,
            "crown_towers": _towers(enemy_damage=enemy_damage),
        },
    }


def _replay(seed: int, *, hero: bool = False) -> dict:
    spells = [
        {"d": card, "l": 10, **({"el": 2} if hero and index == 0 else {})}
        for index, card in enumerate(CARDS)
    ]
    return {
        "rndSeed": seed,
        "battle": {
            "deck0": {"sp": deepcopy(spells)}, "deck1": {"sp": deepcopy(spells)},
            "avatar0": {"accountID.hi": 1, "accountID.lo": 2},
            "avatar1": {"accountID.hi": 3, "accountID.lo": 4},
        },
    }


def _fixture(seed: int, learner_side: int = 0, *, hero: bool = False) -> DeckFixture:
    return DeckFixture(
        episode_index=seed,
        learner_side=learner_side,
        learner_deck_sha256="c" * 64,
        opponent_deck_sha256="d" * 64,
        opponent_preset="fake.json",
        replay=_replay(seed, hero=hero),
    )


def _header(seed: int, learner_side: int, learner_hash: str, opponent_hash: str) -> EpisodeHeader:
    return EpisodeHeader(
        episode_id=f"episode-{seed}", batch_id="batch-1", seed=seed,
        learner_side=learner_side, behavior_policy_version=1,
        behavior_actor_sha256=learner_hash, opponent_policy_id="opponent",
        opponent_actor_sha256=opponent_hash,
        learner_deck_sha256="c" * 64, opponent_deck_sha256="d" * 64,
        curriculum_stage="stage1_critic", initial_hidden_sha256="e" * 64,
    )


class FakeNativeEnv:
    def __init__(self, *, terminal_after: int = 2, late_terminal: bool = False,
                 pending_advance: bool = False,
                 terminal_advance: int | None = None,
                 entities: list[dict] | None = None) -> None:
        self.terminal_after = terminal_after
        self.late_terminal = late_terminal
        self.pending_advance = pending_advance
        self.terminal_advance = terminal_advance
        self.entities = deepcopy(entities or [])
        self.calls = 0
        self.transitions: list[list[dict]] = []
        self.probes: list[tuple[int, int]] = []
        self.decks: list[list[dict]] = []
        self._state = _state(100, entities=self.entities)

    def reset_rpc_profile(self) -> None:
        pass

    def reset(self, replay, *, warmup_steps: int = 100):
        self.calls = 0
        self._state = _state(warmup_steps, entities=self.entities)
        self.decks = [[
            {"card_id": int(card["d"]), "level": int(card["l"]) + 1,
             "form_flags": int(card.get("el", 0))}
            for card in replay["battle"][f"deck{side}"]["sp"]
        ] for side in (0, 1)]
        return deepcopy(self._state)

    def probe_grid(self, *, side: int, deck_index: int):
        self.probes.append((side, deck_index))
        return {"rows": ["1" * 18 for _ in range(32)]}

    def joint_training_transition(self, actions, *, steps: int = 1):
        assert 1 <= steps <= 16
        self.calls += 1
        copied = [dict(row) for row in actions]
        self.transitions.append(copied)
        pending_late = self.late_terminal and self.calls == self.terminal_after
        terminal = (
            self.calls >= self.terminal_after + 1
            if self.late_terminal
            else self.calls >= self.terminal_after
        )
        pending_advance = self.pending_advance and self.calls == 1
        terminal = terminal and not self.pending_advance
        if self.pending_advance and self.calls >= 3:
            terminal = True
        zero_delta = pending_late or pending_advance or (terminal and self.late_terminal)
        receipts = []
        for action in copied:
            accepted = not pending_late
            receipts.append({
                "side": action["side"],
                "result": {"accepted": accepted, "result_code": 0 if accepted else 3},
            })
        if not zero_delta or (self.pending_advance and self.calls == 2):
            advance = (
                self.terminal_advance
                if terminal and self.terminal_advance is not None
                else steps
            )
            self._state = _state(
                int(self._state["tick"]) + int(advance),
                enemy_damage=100 if self.calls >= 1 else 0,
                entities=self.entities,
            )
        episode = deepcopy(self._state["episode"])
        if terminal:
            episode.update({
                "terminated": True, "truncated": False,
                "outcome": "side_0_win", "terminal_tick": int(self._state["tick"]),
                "crowns": [1, 0], "rewards": [1.0, -1.0],
                "rewards_by_side": {0: 1.0, 1: -1.0},
            })
        result = {"joint_action": {"actions": receipts}, "step": {"episode": episode}}
        if not terminal:
            result["state"] = deepcopy(self._state)
        return result


class ConstantActor(nn.Module):
    def __init__(self, *, position: int = 0) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = ExpertPolicyConfig(
            grid_channels=8, public_scalar_size=16, card_vocab_size=9,
            ability_vocab_size=1, max_ability_slots=2,
            hidden_size=8, card_embedding_size=4, spatial_size=4,
            lambda_max=20.0, native_tick_seconds=0.05,
        )
        self.position = position
        self.calls = 0

    def initial_hidden(self, batch_size: int, *, device):
        return (torch.zeros(1, batch_size, 8, device=device),
                torch.zeros(1, batch_size, 8, device=device))

    def forward_sequence(self, *, public_scalars: Tensor, hidden, **_kwargs):
        self.calls += 1
        batch, steps = public_scalars.shape[:2]
        device = public_scalars.device
        positions = torch.zeros(batch, steps, 4, 576, device=device)
        positions[..., 0, self.position] = 10
        return ExpertPolicyOutput(
            torch.full((batch, steps), 20.0, device=device),
            torch.tensor([1.0, -1.0], device=device).expand(batch, steps, -1),
            torch.tensor([5.0, 0.0, 0.0, 0.0], device=device).expand(batch, steps, -1),
            positions,
            torch.zeros(batch, steps, 2, device=device),
            torch.zeros(batch, steps, 2, 576, device=device),
            (hidden[0] + 1, hidden[1] + 1),
        )


class FixedAbilityService:
    def __init__(self) -> None:
        self.calls = 0
        self.hidden = {}

    def reset_episode(self, _worker_id):
        return 0

    def act(self, requests):
        self.calls += 1
        answer = []
        for request in requests:
            self.hidden[(request.actor_sha256, request.worker_id, request.side)] = (
                torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)
            )
            ability = request.side == 0
            answer.append(SampledPolicyAction(
                worker_id=request.worker_id, side=request.side,
                actor_sha256=request.actor_sha256, delta_ticks=1,
                event_happened=ability, action_kind=1 if ability else 0,
                card_slot=0, position=0, ability_slot=0, ability_position=0,
                ability_requires_target=False, lambda_per_second=1.0,
                event_probability=0.1, logp_total=-0.1, logp_timing=-0.1,
                logp_action_type=0.0, logp_slot=0.0, logp_position=0.0,
                logp_mark=0.0,
            ))
        return answer

    def last_pre_action_hidden(self, *, actor_sha256, worker_id, side):
        return self.hidden[(actor_sha256, worker_id, side)]


def _encoder(*, hero: bool = False) -> NativeObservationEncoder:
    mapping = dict(CARD_MAP)
    abilities = {}
    if hero:
        mapping.pop(26_000_000)
        mapping[203_000_000] = 1
        abilities[203_000_000] = 1
    return NativeObservationEncoder(
        card_id_to_token=mapping, ability_id_to_token=abilities,
        max_ability_slots=2, card_vocab_size=9,
        ability_vocab_size=2 if hero else 1,
    )


class OnlineCollectorTests(unittest.TestCase):
    def test_batch_encodes_all_sides_once_and_groups_same_actor_hash(self):
        digest = "a" * 64
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor = ConstantActor(position=0)
        service.register_actor(actor, actor_sha256=digest)
        collector = OnlineSelfPlayCollector(_encoder(), service)
        env0, env1 = FakeNativeEnv(), FakeNativeEnv()
        specs = []
        for index, (env, learner) in enumerate(((env0, 0), (env1, 1)), 1):
            fixture = _fixture(index, learner)
            specs.append(OnlineEpisodeSpec(
                index, env, fixture, _header(index, learner, digest, digest),
                {0: digest, 1: digest},
            ))
        results = collector.collect_batch(specs)
        self.assertEqual(actor.calls, 2)  # one forward for four sides per native Tick
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual([len(row.episode.decisions) for row in results], [2, 2])
        self.assertTrue(all(
            decision.side == result.episode.header.learner_side
            for result in results for decision in result.episode.decisions
        ))
        # Side 1 canonical cell zero is absolute bottom-right.
        side1_action = results[1].step_payloads[0]["native_action"]
        self.assertEqual((side1_action["x"], side1_action["y"]), (17_500, 31_500))
        self.assertEqual(results[0].step_payloads[0]["actor_inputs"]["grid"].shape,
                         (1, 8, 32, 18))
        self.assertTrue(torch.equal(
            results[0].step_payloads[0]["actor_inputs"]["hidden"][0],
            torch.zeros(1, 1, 8),
        ))
        self.assertTrue(torch.equal(
            results[0].step_payloads[1]["actor_inputs"]["hidden"][0],
            torch.ones(1, 1, 8),
        ))
        self.assertEqual(len(results[0].step_payloads), len(results[0].episode.decisions))

    def test_different_actor_hashes_are_dispatched_in_the_same_service_call(self):
        hashes = {0: "a" * 64, 1: "b" * 64}
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actors = {side: ConstantActor() for side in (0, 1)}
        for side in (0, 1):
            service.register_actor(actors[side], actor_sha256=hashes[side])
        fixture = _fixture(7, 0)
        result = OnlineSelfPlayCollector(_encoder(), service).collect_episode(
            env=FakeNativeEnv(terminal_after=1), fixture=fixture,
            header=_header(7, 0, hashes[0], hashes[1]), actor_hashes=hashes,
        )
        self.assertEqual([actors[side].calls for side in (0, 1)], [1, 1])
        self.assertEqual(service.forward_calls, 2)  # one hash group each, one act() call
        self.assertTrue(result.episode.decisions[-1].terminated)

    def test_zero_delta_late_terminal_merges_into_previous_tick_and_reward_is_auditable(self):
        digest = "a" * 64
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor = ConstantActor()
        service.register_actor(actor, actor_sha256=digest)
        fixture = _fixture(9, 0)
        env = FakeNativeEnv(terminal_after=2, late_terminal=True)
        result = OnlineSelfPlayCollector(_encoder(), service).collect_episode(
            env=env, fixture=fixture,
            header=_header(9, 0, digest, digest), actor_hashes={0: digest, 1: digest},
        )
        self.assertEqual([row.tick for row in result.episode.decisions], [100])
        decision = result.episode.decisions[0]
        self.assertTrue(decision.terminated)
        self.assertAlmostEqual(decision.reward_damage_dealt, 0.1)
        self.assertAlmostEqual(decision.reward_terminal, 10.0)
        self.assertAlmostEqual(decision.reward_total, 10.1)
        self.assertTrue(result.step_payloads[0]["terminal_merged_from_zero_delta"])
        self.assertEqual(result.native_ticks_advanced, 1)
        self.assertEqual(actor.calls, 2)
        self.assertEqual(env.calls, 3)
        self.assertEqual(env.transitions[-1], [])
        self.assertIn("late_terminal_first_joint_action", result.step_payloads[0])

    def test_zero_delta_ongoing_then_tick_advance_reuses_the_same_policy_decision(self):
        digest = "a" * 64
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor = ConstantActor()
        service.register_actor(actor, actor_sha256=digest)
        fixture = _fixture(10, 0)
        env = FakeNativeEnv(pending_advance=True)
        result = OnlineSelfPlayCollector(_encoder(), service).collect_episode(
            env=env, fixture=fixture,
            header=_header(10, 0, digest, digest), actor_hashes={0: digest, 1: digest},
        )
        # Three native RPCs: pending zero, empty resolution to Tick+1, then
        # terminal Tick+1.  Only two policy decisions/records are produced.
        self.assertEqual(env.calls, 3)
        self.assertEqual(actor.calls, 2)
        self.assertEqual(env.transitions[1], [])
        self.assertEqual([row.tick for row in result.episode.decisions], [100, 101])
        self.assertEqual(result.native_ticks_advanced, 2)
        self.assertEqual(result.step_payloads[0]["native_advanced_ticks"], 1)

    def test_step_ticks_four_records_full_native_advances(self):
        digest = "a" * 64
        service = BatchedPolicyService(device="cpu", deterministic=True)
        service.register_actor(ConstantActor(), actor_sha256=digest)
        fixture = _fixture(12, 0)
        result = OnlineSelfPlayCollector(
            _encoder(), service, step_ticks=4
        ).collect_episode(
            env=FakeNativeEnv(terminal_after=2), fixture=fixture,
            header=_header(12, 0, digest, digest), actor_hashes={0: digest, 1: digest},
        )
        self.assertEqual([row.tick for row in result.episode.decisions], [100, 104])
        self.assertEqual([row.delta_ticks for row in result.episode.decisions], [4, 4])
        self.assertEqual(result.native_ticks_advanced, 8)
        self.assertTrue(all(
            float(payload["actor_inputs"]["delta_ticks"][0]) == 4.0
            for payload in result.step_payloads
        ))

    def test_step_ticks_four_early_terminal_uses_actual_delta_in_gae(self):
        digest = "a" * 64
        service = BatchedPolicyService(device="cpu", deterministic=True)
        service.register_actor(ConstantActor(), actor_sha256=digest)
        fixture = _fixture(13, 0)
        result = OnlineSelfPlayCollector(
            _encoder(), service, step_ticks=4
        ).collect_episode(
            env=FakeNativeEnv(terminal_after=2, terminal_advance=2), fixture=fixture,
            header=_header(13, 0, digest, digest), actor_hashes={0: digest, 1: digest},
        )
        self.assertEqual([row.tick for row in result.episode.decisions], [100, 104])
        self.assertEqual([row.delta_ticks for row in result.episode.decisions], [4, 2])
        self.assertEqual(result.native_ticks_advanced, 6)
        chunks = LearnerEpisodeChunker().chunk(
            result.episode, step_payloads=result.step_payloads
        )
        self.assertEqual(
            [row["delta_ticks"] for row in chunks[0]["decisions"]], [4, 2]
        )

    def test_available_native_ability_maps_to_entity_command(self):
        hero = {
            "category": 5_000_007, "side": 0, "x": 4000, "y": 8000,
            "card_id": 203_000_000, "level": 11, "hp": 1000, "max_hp": 2000,
            "behavior_state": 1, "ability_slot": 1, "ability_state_code": 0,
            "ability_available": True, "ability_cooldown_remaining_ms": 0,
            "ability_charges_remaining": 1, "ability_pending_ms": 0,
            "ability_mana_cost": 2,
        }
        digest = "f" * 64
        service = FixedAbilityService()
        fixture = _fixture(11, 0, hero=True)
        env = FakeNativeEnv(terminal_after=1, entities=[hero])
        result = OnlineSelfPlayCollector(_encoder(hero=True), service).collect_episode(
            env=env, fixture=fixture, header=_header(11, 0, digest, digest),
            actor_hashes={0: digest, 1: digest},
        )
        self.assertEqual(service.calls, 1)
        self.assertEqual(env.transitions[0][0], {
            "type": "ability", "side": 0, "entity_id": 5_000_007,
        })
        self.assertEqual(result.episode.decisions[0].action_kind, 1)


if __name__ == "__main__":
    unittest.main()
