from __future__ import annotations

from pathlib import Path
import gc
import tempfile
import unittest

import numpy as np
import torch

from expert_selfplay_v1.actor_adapter import actor_state_digest, assert_actor_equivalence
from expert_selfplay_v1.contracts import (
    BatchManifest,
    EntityInputContractError,
    EntityInputGuard,
)
from expert_selfplay_v1.gae import discount_interval_rewards, variable_time_gae
from expert_selfplay_v1.hazard import (
    marked_hazard_entropy,
    marked_hazard_kl,
    marked_hazard_log_prob,
)
from expert_selfplay_v1.league import (
    LeagueState,
    MatchupStats,
    OpponentScheduler,
    exact_quotas,
)
from expert_selfplay_v1.decks import DeckScheduler
from expert_selfplay_v1.stages import build_optimizer, configure_stage
from expert_selfplay_v1.actions import (
    ExpertActionMasks,
    RecordedExpertAction,
    evaluate_expert_action,
    expert_policy_kl,
)
from expert_selfplay_v1.promotion import (
    PromotionCriteria,
    apply_promotion,
    decide_promotion,
    fitted_candidate_elo,
)
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.losses import CriticTargets, critic_loss, explained_variance
from expert_selfplay_v1.rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from expert_selfplay_v1.ppo import recurrent_ppo_loss
from expert_selfplay_v1.update_guard import evaluate_update
from expert_selfplay_v1.critic import (
    ExpertActorCritic,
    PrivilegedCritic,
    PrivilegedCriticConfig,
)
from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    ExpertPolicyOutput,
    RecurrentExpertPolicy,
)
from expert_v1.training_v1.schema import read_manifest
from expert_v1.training_v1.smoke_data import create_smoke_dataset


class ExpertSelfPlayContractTests(unittest.TestCase):
    def test_marked_hazard_log_prob_is_stable_and_joint(self):
        rate = torch.tensor([2.0, 2.0], requires_grad=True)
        ticks = torch.tensor([1, 1])
        event = torch.tensor([False, True])
        legal = torch.tensor([True, True])
        mark = torch.tensor([0.0, -0.7])
        value = marked_hazard_log_prob(
            lambda_per_second=rate,
            delta_ticks=ticks,
            event_happened=event,
            can_act=legal,
            mark_log_prob=mark,
        )
        exposure = 0.1
        self.assertAlmostEqual(float(value.timing[0].detach()), -exposure, places=6)
        self.assertAlmostEqual(
            float(value.timing[1].detach()), np.log1p(-np.exp(-exposure)), places=6
        )
        self.assertAlmostEqual(
            float(value.total[1].detach()),
            float(value.timing[1].detach()) - 0.7,
            places=6,
        )
        (-value.total.mean()).backward()
        self.assertTrue(torch.isfinite(rate.grad).all())

    def test_expert_hierarchy_builds_one_joint_log_probability(self):
        config = ExpertPolicyConfig(
            grid_channels=8, public_scalar_size=16, card_vocab_size=12,
            ability_vocab_size=4, max_ability_slots=2,
            hidden_size=8, card_embedding_size=8, spatial_size=8,
        )
        prefix = (1, 2)
        output = ExpertPolicyOutput(
            rate_logits=torch.zeros(prefix),
            action_kind_logits=torch.zeros(*prefix, 2),
            card_logits=torch.zeros(*prefix, 4),
            position_logits=torch.zeros(*prefix, 4, 576),
            ability_logits=torch.zeros(*prefix, 2),
            ability_position_logits=torch.zeros(*prefix, 2, 576),
            hidden=(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8)),
        )
        kind = torch.zeros(*prefix, 2, dtype=torch.bool)
        kind[..., 0] = True
        cards = torch.zeros(*prefix, 4, dtype=torch.bool)
        cards[..., 0] = True
        positions = torch.zeros(*prefix, 4, 576, dtype=torch.bool)
        positions[..., 0, 10:12] = True
        abilities = torch.zeros(*prefix, 2, dtype=torch.bool)
        ability_positions = torch.zeros(*prefix, 2, 576, dtype=torch.bool)
        masks = ExpertActionMasks(
            action_kind=kind,
            cards=cards,
            positions=positions,
            abilities=abilities,
            ability_positions=ability_positions,
            ability_requires_target=torch.zeros(*prefix, 2, dtype=torch.bool),
        )
        action = RecordedExpertAction(
            event_happened=torch.tensor([[True, False]]),
            action_kind=torch.zeros(prefix, dtype=torch.long),
            card_slot=torch.tensor([[0, 3]]),
            position=torch.tensor([[10, 575]]),
            ability_slot=torch.tensor([[0, 1]]),
            ability_position=torch.tensor([[0, 575]]),
            ability_requires_target=torch.zeros(prefix, dtype=torch.bool),
        )
        evaluated = evaluate_expert_action(
            output=output, config=config, masks=masks, action=action,
            delta_ticks=torch.ones(prefix),
        )
        self.assertAlmostEqual(float(evaluated.mark_log_prob[0, 0]), -np.log(2), places=6)
        self.assertEqual(float(evaluated.mark_log_prob[0, 1]), 0.0)
        self.assertTrue(torch.isfinite(evaluated.log_prob.total).all())
        identical_kl = expert_policy_kl(
            source=output, target=output, config=config, masks=masks,
            delta_ticks=torch.ones(prefix),
        )
        self.assertTrue(torch.allclose(identical_kl, torch.zeros_like(identical_kl)))
        shifted = output._replace(
            rate_logits=output.rate_logits + 0.2,
            position_logits=output.position_logits + torch.linspace(
                0.0, 0.1, 576
            ).view(1, 1, 1, 576),
        )
        shifted_kl = expert_policy_kl(
            source=output, target=shifted, config=config, masks=masks,
            delta_ticks=torch.ones(prefix),
        )
        self.assertTrue(torch.isfinite(shifted_kl).all())
        self.assertTrue(bool((shifted_kl > 0).all()))
        invalid = RecordedExpertAction(
            **{**action.__dict__, "position": torch.tensor([[12, 575]])}
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            evaluate_expert_action(
                output=output, config=config, masks=masks, action=invalid,
                delta_ticks=torch.ones(prefix),
            )

    def test_forced_wait_has_zero_log_prob_gradient_and_illegal_event_fails(self):
        rate = torch.tensor([3.0], requires_grad=True)
        value = marked_hazard_log_prob(
            lambda_per_second=rate,
            delta_ticks=torch.tensor([4]),
            event_happened=torch.tensor([False]),
            can_act=torch.tensor([False]),
            mark_log_prob=torch.tensor([0.0]),
        )
        self.assertEqual(float(value.total[0].detach()), 0.0)
        value.total.sum().backward()
        self.assertEqual(float(rate.grad[0]), 0.0)
        with self.assertRaisesRegex(ValueError, "cannot occur"):
            marked_hazard_log_prob(
                lambda_per_second=torch.tensor([1.0]),
                delta_ticks=torch.tensor([1]),
                event_happened=torch.tensor([True]),
                can_act=torch.tensor([False]),
                mark_log_prob=torch.tensor([0.0]),
            )

    def test_joint_entropy_and_kl_include_marks_only_on_event_mass(self):
        rate = torch.tensor([2.0])
        ticks = torch.tensor([1])
        legal = torch.tensor([True])
        mark_entropy = torch.tensor([0.8])
        entropy = marked_hazard_entropy(
            lambda_per_second=rate,
            delta_ticks=ticks,
            can_act=legal,
            mark_entropy=mark_entropy,
        )
        p = 1.0 - np.exp(-0.1)
        expected = -(p * np.log(p) + (1 - p) * np.log(1 - p)) + p * 0.8
        self.assertAlmostEqual(float(entropy[0]), expected, places=6)
        kl = marked_hazard_kl(
            source_lambda=rate,
            target_lambda=rate,
            delta_ticks=ticks,
            can_act=legal,
            mark_kl=torch.tensor([0.0]),
        )
        self.assertAlmostEqual(float(kl[0]), 0.0, places=7)

    def test_variable_time_gae_uses_per_tick_powers_and_truncation_bootstrap(self):
        rewards = np.asarray([1.0, 2.0], dtype=np.float32)
        values = np.asarray([0.5, 0.75], dtype=np.float32)
        terminated = np.asarray([False, False])
        ticks = np.asarray([2, 3])
        gamma, trace = 0.9, 0.8
        advantages, returns = variable_time_gae(
            rewards,
            values,
            terminated,
            ticks,
            bootstrap_value=1.25,
            gamma_per_tick=gamma,
            gae_lambda_per_tick=trace,
        )
        delta1 = 2.0 + gamma**3 * 1.25 - 0.75
        expected1 = delta1
        delta0 = 1.0 + gamma**2 * 0.75 - 0.5
        expected0 = delta0 + gamma**2 * trace**2 * expected1
        np.testing.assert_allclose(advantages, [expected0, expected1], rtol=1e-6)
        np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)
        self.assertAlmostEqual(
            discount_interval_rewards(np.asarray([1.0, 2.0]), gamma_per_tick=0.5),
            2.0,
        )

    def test_entity_input_guard_fails_closed_on_native_nonempty_encoded_empty(self):
        guard = EntityInputGuard()
        guard.observe(
            native_eligible_entities=2,
            entity_tokens=[4, 9, 0],
            entity_positions=[12, 513, 0],
            entity_mask=[True, True, False],
        )
        self.assertEqual(guard.summary()["encoded_entities"], 2)
        with self.assertRaisesRegex(EntityInputContractError, "encoded.*empty"):
            guard.observe(
                native_eligible_entities=1,
                entity_tokens=[0],
                entity_positions=[0],
                entity_mask=[False],
            )

    def test_batch_manifest_is_content_addressed_and_version_locked(self):
        value = BatchManifest(
            run_id="run",
            batch_id="B0001",
            policy_version=7,
            behavior_actor_sha256="a" * 64,
            encoder_schema_sha256="b" * 64,
            action_schema_sha256="c" * 64,
            reward_schema_sha256="d" * 64,
            native_lib_sha256="e" * 64,
            episode_count=16,
        )
        self.assertEqual(value.digest(), value.digest())
        with self.assertRaisesRegex(ValueError, "behavior_actor_sha256"):
            BatchManifest(**{**value.__dict__, "behavior_actor_sha256": "bad"}).validate()

    def test_actor_feature_adapter_is_policy_and_state_dict_equivalent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_smoke_dataset(Path(temporary) / "dataset")
            manifest = read_manifest(root)
            dataset = NativeExpertSequenceDataset(
                root, split="train", sequence_length=8, burn_in=2
            )
            batch = collate_sequences([dataset[0]])
            dimensions = manifest["dimensions"]
            actor = RecurrentExpertPolicy(ExpertPolicyConfig(
                grid_channels=dimensions["grid_channels"],
                public_scalar_size=dimensions["public_scalar_size"],
                card_vocab_size=dimensions["card_vocab_size"],
                ability_vocab_size=dimensions["ability_vocab_size"],
                max_ability_slots=dimensions["max_ability_slots"],
                hidden_size=32,
                card_embedding_size=16,
                spatial_size=16,
            ))
            inputs = {
                "grid": batch["grid"],
                "public_scalars": batch["public_scalars"],
                "own_deck_tokens": batch["own_deck_tokens"],
                "hand_tokens": batch["hand_tokens"],
                "next_card_token": batch["next_card_token"],
                "revealed_enemy_tokens": batch["revealed_enemy_tokens"],
                "ability_tokens": batch["ability_tokens"],
                "delta_ticks": batch["delta_ticks"],
                "entity_tokens": batch["entity_tokens"],
                "entity_positions": batch["entity_positions"],
                "entity_relations": batch["entity_relations"],
                "entity_numeric": batch["entity_numeric"],
                "entity_mask": batch["entity_mask"],
            }
            report = assert_actor_equivalence(actor, inputs)
            self.assertEqual(report["latent_shape"], [1, 8, 32])
            del inputs, batch, actor
            dataset.close()
            del dataset
            gc.collect()

    def test_critic_is_independent_and_cannot_backpropagate_into_actor(self):
        config = ExpertPolicyConfig(
            grid_channels=8,
            public_scalar_size=16,
            card_vocab_size=24,
            ability_vocab_size=8,
            max_ability_slots=4,
            hidden_size=32,
            card_embedding_size=16,
            spatial_size=16,
        )
        actor = RecurrentExpertPolicy(config)
        critic = PrivilegedCritic(PrivilegedCriticConfig(
            actor_latent_size=32,
            card_vocab_size=24,
            scalar_size=20,
        ))
        wrapper = ExpertActorCritic(actor, critic)
        batch, steps, entities, private = 2, 3, 4, 12
        actor_inputs = {
            "grid": torch.randn(batch, steps, 8, 32, 18),
            "public_scalars": torch.randn(batch, steps, 16),
            "own_deck_tokens": torch.randint(1, 24, (batch, steps, 8)),
            "hand_tokens": torch.randint(1, 24, (batch, steps, 4)),
            "next_card_token": torch.randint(1, 24, (batch, steps)),
            "revealed_enemy_tokens": torch.zeros(batch, steps, 8, dtype=torch.long),
            "ability_tokens": torch.zeros(batch, steps, 4, dtype=torch.long),
            "delta_ticks": torch.ones(batch, steps),
            "entity_tokens": torch.randint(1, 24, (batch, steps, entities)),
            "entity_positions": torch.randint(0, 576, (batch, steps, entities)),
            "entity_relations": torch.randint(0, 2, (batch, steps, entities)),
            "entity_numeric": torch.rand(batch, steps, entities, 3),
            "entity_mask": torch.ones(batch, steps, entities, dtype=torch.bool),
        }
        critic_inputs = {
            "grid": torch.randn(batch, steps, 8, 32, 18),
            "entity_tokens": torch.randint(1, 24, (batch, steps, entities)),
            "entity_positions": torch.randint(0, 576, (batch, steps, entities)),
            "entity_relations": torch.randint(0, 2, (batch, steps, entities)),
            "entity_numeric": torch.rand(batch, steps, entities, 3),
            "entity_mask": torch.ones(batch, steps, entities, dtype=torch.bool),
            "private_card_tokens": torch.randint(1, 24, (batch, steps, private)),
            "private_card_owners": torch.randint(0, 2, (batch, steps, private)),
            "private_card_slots": torch.arange(private).view(1, 1, -1).expand(batch, steps, -1),
            "private_card_mask": torch.ones(batch, steps, private, dtype=torch.bool),
            "scalars": torch.randn(batch, steps, 20),
        }
        before = actor_state_digest(actor)
        _actor_output, critic_output = wrapper(
            actor_inputs=actor_inputs, critic_inputs=critic_inputs
        )
        critic_output.values.mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in critic.parameters()))
        self.assertEqual(before, actor_state_digest(actor))
        self.assertTrue(torch.isfinite(critic_output.values).all())

    def test_production_critic_stays_in_requested_parameter_budget(self):
        critic = PrivilegedCritic(PrivilegedCriticConfig(
            actor_latent_size=256,
            card_vocab_size=181,
            scalar_size=32,
        ))
        parameters = sum(value.numel() for value in critic.parameters())
        self.assertGreaterEqual(parameters, 9_000_000)
        self.assertLessEqual(parameters, 15_000_000)

    def test_opponent_scheduler_enforces_40_40_20_and_side_balance(self):
        history = [f"H{index}" for index in range(8)]
        league = LeagueState(
            base_policy_id="BASE",
            champion_policy_id="CHAMPION",
            active_history_ids=history,
        )
        for index, opponent in enumerate(history):
            league.training_matchups[
                league.matchup_key("CANDIDATE", opponent)
            ] = MatchupStats(wins=index, losses=7 - index)
        rows = OpponentScheduler(league).build_batch(
            episode_count=100, candidate_id="CANDIDATE", seed=7
        )
        categories = {name: sum(row.category == name for row in rows)
                      for name in ("champion", "history", "base")}
        self.assertEqual(categories, {"champion": 40, "history": 40, "base": 20})
        self.assertEqual(sum(row.learner_side == 0 for row in rows), 50)
        modes = {name: sum(row.history_mode == name for row in rows)
                 for name in ("close", "hard", "uniform")}
        self.assertEqual(modes, {"close": 20, "hard": 10, "uniform": 10})
        counts = {
            opponent: sum(row.policy_id == opponent for row in rows)
            for opponent in history
        }
        self.assertLessEqual(max(counts.values()), 10)

    def test_empty_history_bootstrap_is_40_champion_60_base(self):
        league = LeagueState(base_policy_id="BASE", champion_policy_id="BASE")
        rows = OpponentScheduler(league).build_batch(
            episode_count=20, candidate_id="CANDIDATE", seed=3
        )
        self.assertEqual(sum(row.category == "champion" for row in rows), 8)
        self.assertEqual(sum(row.category == "base" for row in rows), 12)
        self.assertFalse(any(row.category == "history" for row in rows))
        self.assertEqual(exact_quotas(7, {"a": 0.4, "b": 0.4, "c": 0.2}),
                         {"a": 3, "b": 3, "c": 1})

    def test_deck_scheduler_keeps_fixed_learner_and_balances_sides(self):
        root = Path(__file__).parents[1]
        opponent_root = Path(
            r"D:\AI_data\cr-native-core\expert-v1\audits\top-deck-presets-v1"
        )
        scheduler = DeckScheduler(
            learner_preset=root / "examples" / "user-selected-heavy-control.json",
            opponent_presets=sorted(opponent_root.glob("deck-*.json")),
        )
        rows = scheduler.build_batch(episode_count=20, seed=11)
        self.assertEqual(sum(row.learner_side == 0 for row in rows), 10)
        self.assertEqual(sum(row.learner_side == 1 for row in rows), 10)
        self.assertEqual(len({row.opponent_preset for row in rows}), 10)
        for row in rows:
            learner = row.replay["battle"][f"deck{row.learner_side}"]["sp"]
            opponent = row.replay["battle"][f"deck{1-row.learner_side}"]["sp"]
            self.assertEqual(len(learner), 8)
            self.assertEqual(len(opponent), 8)

    @staticmethod
    def _small_actor_critic() -> ExpertActorCritic:
        actor = RecurrentExpertPolicy(ExpertPolicyConfig(
            grid_channels=8,
            public_scalar_size=16,
            card_vocab_size=24,
            ability_vocab_size=8,
            max_ability_slots=4,
            hidden_size=32,
            card_embedding_size=16,
            spatial_size=16,
        ))
        critic = PrivilegedCritic(PrivilegedCriticConfig(
            actor_latent_size=32, card_vocab_size=24, scalar_size=20
        ))
        return ExpertActorCritic(actor, critic)

    def test_stage1_trains_only_critic_and_stage2_has_named_groups(self):
        model = self._small_actor_critic()
        stage1 = configure_stage(model, "stage1_critic")
        self.assertTrue(stage1["trainable_names"])
        self.assertTrue(all(
            name.startswith("critic.") for name in stage1["trainable_names"]
        ))
        self.assertTrue(all(not value.requires_grad for value in model.actor.parameters()))
        optimizer, mapping = build_optimizer(model, "stage2_reaction")
        group_names = {group["group_name"] for group in optimizer.param_groups}
        self.assertEqual(group_names, {
            "critic", "actor_entity", "actor_spatial", "actor_recurrent", "actor_timing"
        })
        self.assertIn("actor_adapter.actor.rate_head.weight", mapping)
        self.assertFalse(model.actor.card_query.weight.requires_grad)
        self.assertTrue(model.actor.rate_head.weight.requires_grad)

    def test_stage1_optimizer_step_preserves_actor_hash(self):
        model = self._small_actor_critic()
        optimizer, _mapping = build_optimizer(model, "stage1_critic")
        before = actor_state_digest(model.actor)
        loss = sum(parameter.square().mean() for parameter in model.critic.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(before, actor_state_digest(model.actor))

    def test_critic_warmup_loss_is_finite_and_masked(self):
        output = type("Output", (), {
            "values": torch.tensor([[0.0, 0.2, -0.1]], requires_grad=True),
            "wdl_logits": torch.zeros(1, 3, 3, requires_grad=True),
            "crown_difference": torch.zeros(1, 3, requires_grad=True),
            "tower_hp_difference": torch.zeros(1, 3, requires_grad=True),
            "future_damage": torch.zeros(1, 3, 2, requires_grad=True),
        })()
        targets = CriticTargets(
            returns=torch.tensor([[0.5, 0.1, -0.4]]),
            wdl_class=torch.tensor([[2, 1, 0]]),
            crown_difference=torch.tensor([[1.0, 0.0, -1.0]]),
            tower_hp_difference=torch.tensor([[0.2, 0.0, -0.3]]),
            future_damage=torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]]),
            loss_mask=torch.tensor([[True, False, True]]),
        )
        loss = critic_loss(output, targets)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        ev = explained_variance(output.values.detach(), targets.returns, targets.loss_mask)
        self.assertTrue(torch.isfinite(ev))

    def test_ppo_uses_one_joint_ratio_and_bc_anchor(self):
        old = torch.tensor([[0.0, -0.2]])
        new = old + torch.tensor([[np.log(1.05), np.log(1.30)]])
        loss = recurrent_ppo_loss(
            new_log_prob=new,
            old_log_prob=old,
            advantages=torch.tensor([[1.0, 1.0]]),
            values=torch.tensor([[0.0, 0.0]]),
            returns=torch.tensor([[0.5, -0.5]]),
            joint_entropy=torch.tensor([[0.4, 0.6]]),
            bc_kl=torch.tensor([[0.01, 0.03]]),
            loss_mask=torch.tensor([[True, True]]),
            clip_epsilon=0.10,
            bc_kl_coefficient=2.0,
        )
        # Ratios are clipped once after all timing/mark components are summed.
        self.assertAlmostEqual(float(loss.policy), -(1.05 + 1.10) / 2.0, places=6)
        self.assertAlmostEqual(float(loss.bc_kl), 0.02, places=6)
        self.assertAlmostEqual(float(loss.clip_fraction), 0.5, places=6)
        self.assertTrue(torch.isfinite(loss.total))

    def test_bad_update_retries_once_then_halts(self):
        metrics = {
            "loss": 1.0,
            "approx_update_kl": 0.02,
            "clip_fraction": 0.45,
            "rate_mean_before": 0.5,
            "rate_mean_after": 1.5,
        }
        first = evaluate_update(metrics, retry_attempt=0)
        self.assertEqual(first.action, "retry")
        self.assertEqual(first.actor_lr_multiplier, 0.5)
        self.assertEqual(first.ppo_epochs, 1)
        self.assertEqual(first.bc_kl_multiplier, 2.0)
        second = evaluate_update(metrics, retry_attempt=1)
        self.assertEqual(second.action, "halt")
        finite = evaluate_update({**metrics, "loss": float("nan")}, retry_attempt=0)
        self.assertEqual(finite.action, "halt")

    def test_safe_update_is_admitted(self):
        decision = evaluate_update({
            "loss": 0.8,
            "approx_update_kl": 0.004,
            "clip_fraction": 0.12,
            "rate_mean_before": 0.5,
            "rate_mean_after": 0.55,
        }, retry_attempt=0)
        self.assertEqual(decision.action, "accept")

    def test_candidate_promotion_uses_fixed_winrate_panel_and_elo(self):
        league = LeagueState(
            base_policy_id="BASE",
            champion_policy_id="CHAMPION",
            active_history_ids=["HISTORY"],
            eval_elo={"BASE": 1000.0, "CHAMPION": 1050.0, "HISTORY": 1025.0},
        )
        panel = {
            "CHAMPION": MatchupStats(wins=320, draws=0, losses=192),
            "BASE": MatchupStats(wins=180, draws=0, losses=76),
            "HISTORY": MatchupStats(wins=80, draws=0, losses=48),
        }
        rating = fitted_candidate_elo(panel, league.eval_elo)
        self.assertGreater(rating, league.eval_elo["CHAMPION"])
        decision = decide_promotion(
            candidate_id="CANDIDATE", league=league, panel=panel
        )
        self.assertTrue(decision.promoted, decision.reasons)
        apply_promotion(league, candidate_id="CANDIDATE", decision=decision)
        self.assertEqual(league.champion_policy_id, "CANDIDATE")
        self.assertIn("CHAMPION", league.active_history_ids)

    def test_failed_candidate_never_enters_history(self):
        league = LeagueState(
            base_policy_id="BASE",
            champion_policy_id="CHAMPION",
            eval_elo={"BASE": 1000.0, "CHAMPION": 1100.0},
        )
        panel = {
            "CHAMPION": MatchupStats(wins=100, losses=412),
            "BASE": MatchupStats(wins=100, losses=156),
        }
        decision = decide_promotion(
            candidate_id="BAD", league=league, panel=panel,
            criteria=PromotionCriteria(),
        )
        self.assertFalse(decision.promoted)
        apply_promotion(league, candidate_id="BAD", decision=decision)
        self.assertIn("BAD", league.failed_candidate_ids)
        self.assertNotIn("BAD", league.active_history_ids)
        self.assertEqual(league.champion_policy_id, "CHAMPION")

    def test_rollout_ledger_is_idempotent_and_commits_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = RolloutLedger(Path(temporary) / "ledger.sqlite")
            ledger.open_batch("B1", policy_version=3, actor_sha256="a" * 64)
            ledger.transition("B1", "COLLECTING")
            self.assertTrue(ledger.record_shard(
                "B1", shard_uuid="S1", content_sha256="b" * 64
            ))
            self.assertFalse(ledger.record_shard(
                "B1", shard_uuid="S1", content_sha256="b" * 64
            ))
            with self.assertRaisesRegex(RuntimeError, "conflicting"):
                ledger.record_shard("B1", shard_uuid="S1", content_sha256="c" * 64)
            self.assertEqual(ledger.close_collection("B1"), ["S1"])
            ledger.transition("B1", "UPDATING")
            ledger.transition("B1", "VALIDATING")
            ledger.commit("B1")
            self.assertEqual(ledger.state("B1"), "COMMITTED")
            self.assertEqual(ledger.shards("B1"), [("S1", "b" * 64, True)])
            with self.assertRaisesRegex(RuntimeError, "only a validated"):
                ledger.commit("B1")
            ledger.close()

    def test_failed_update_returns_to_closed_without_consuming_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = RolloutLedger(Path(temporary) / "ledger.sqlite")
            ledger.open_batch("B2", policy_version=4, actor_sha256="d" * 64)
            ledger.transition("B2", "COLLECTING")
            ledger.record_shard("B2", shard_uuid="S2", content_sha256="e" * 64)
            ledger.close_collection("B2")
            ledger.transition("B2", "UPDATING")
            ledger.transition("B2", "CLOSED")
            self.assertEqual(ledger.shards("B2"), [("S2", "e" * 64, False)])
            ledger.close()

    @staticmethod
    def _decision(**overrides):
        value = dict(
            tick=1, delta_ticks=1, side=0, event_happened=False,
            action_kind=0, card_slot=0, position=0, ability_slot=0,
            ability_position=0, old_logp_total=0.0, old_logp_timing=0.0,
            old_logp_action_type=0.0, old_logp_slot=0.0,
            old_logp_position=0.0, reward_damage_dealt=0.0,
            reward_damage_received=0.0, reward_towers_dealt=0.0,
            reward_towers_received=0.0, reward_terminal=0.0,
            reward_total=0.0, value=0.0, terminated=True, truncated=False,
            native_entity_count=0, encoded_entity_count=0,
        )
        value.update(overrides)
        return DecisionRecord(**value)

    def test_episode_buffer_accepts_only_learner_side_and_complete_episode(self):
        header = EpisodeHeader(
            episode_id="E1", batch_id="B1", seed=1, learner_side=0,
            behavior_policy_version=2, behavior_actor_sha256="a" * 64,
            opponent_policy_id="OPP", opponent_actor_sha256="b" * 64,
            learner_deck_sha256="c" * 64, opponent_deck_sha256="d" * 64,
            curriculum_stage="stage1_critic", initial_hidden_sha256="e" * 64,
        )
        buffer = LearnerEpisodeBuffer(header)
        with self.assertRaisesRegex(ValueError, "opponent trajectory"):
            buffer.append(self._decision(side=1))
        buffer.append(self._decision())
        frozen = buffer.freeze()
        self.assertEqual(len(frozen["decisions"]), 1)
        self.assertEqual(len(frozen["content_sha256"]), 64)

    def test_episode_buffer_rejects_empty_encoded_native_scene_and_bad_reward_sum(self):
        header = EpisodeHeader(
            episode_id="E2", batch_id="B1", seed=2, learner_side=0,
            behavior_policy_version=2, behavior_actor_sha256="a" * 64,
            opponent_policy_id="OPP", opponent_actor_sha256="b" * 64,
            learner_deck_sha256="c" * 64, opponent_deck_sha256="d" * 64,
            curriculum_stage="stage1_critic", initial_hidden_sha256="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "encoded as empty"):
            LearnerEpisodeBuffer(header).append(
                self._decision(native_entity_count=1, encoded_entity_count=0)
            )
        with self.assertRaisesRegex(ValueError, "reward components"):
            LearnerEpisodeBuffer(header).append(
                self._decision(reward_total=1.0)
            )


if __name__ == "__main__":
    unittest.main()
