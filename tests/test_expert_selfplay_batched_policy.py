from __future__ import annotations

import math
import unittest

import torch
from torch import Tensor, nn

from expert_selfplay_v1.actions import ExpertActionMasks
from expert_selfplay_v1.batched_policy import (
    BatchedPolicyService,
    PolicyRequest,
)
from expert_v1.training_v1.model import ExpertPolicyConfig, ExpertPolicyOutput


class CountingActor(nn.Module):
    def __init__(self, *, rate_logit: float = 20.0, nan: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = ExpertPolicyConfig(
            grid_channels=1,
            public_scalar_size=1,
            card_vocab_size=8,
            ability_vocab_size=3,
            max_ability_slots=2,
            hidden_size=4,
            card_embedding_size=4,
            spatial_size=4,
            lambda_max=20.0,
            native_tick_seconds=0.05,
        )
        self.rate_logit = rate_logit
        self.nan = nan
        self.calls = 0
        self.hidden_seen: list[Tensor] = []
        self.entity_shape: tuple[int, ...] | None = None

    def initial_hidden(self, batch_size: int, *, device: torch.device | str):
        return (
            torch.zeros(1, batch_size, 4, device=device),
            torch.zeros(1, batch_size, 4, device=device),
        )

    def forward_sequence(
        self,
        *,
        public_scalars: Tensor,
        delta_ticks: Tensor,
        hidden: tuple[Tensor, Tensor],
        entity_tokens: Tensor | None = None,
        **_unused,
    ) -> ExpertPolicyOutput:
        self.calls += 1
        self.hidden_seen.append(hidden[0].detach().cpu().clone())
        if entity_tokens is not None:
            self.entity_shape = tuple(entity_tokens.shape)
        batch, steps = public_scalars.shape[:2]
        device = public_scalars.device
        rate = torch.full((batch, steps), self.rate_logit, device=device)
        if self.nan:
            rate[0, 0] = float("nan")
        # A negative scalar prefers an ability, a nonnegative one a card.
        kind = torch.stack(
            (public_scalars[..., 0], -public_scalars[..., 0]), dim=-1
        )
        cards = torch.tensor([0.0, 1.0, 5.0, 2.0], device=device).expand(
            batch, steps, -1
        )
        positions = torch.zeros(batch, steps, 4, 576, device=device)
        positions[..., 2, 17] = 9.0
        abilities = torch.tensor([0.0, 4.0], device=device).expand(batch, steps, -1)
        ability_positions = torch.zeros(batch, steps, 2, 576, device=device)
        ability_positions[..., 1, 23] = 8.0
        return ExpertPolicyOutput(
            rate,
            kind,
            cards,
            positions,
            abilities,
            ability_positions,
            (hidden[0] + 1, hidden[1] + 1),
        )


def masks(*, card: bool = True, ability: bool = True) -> ExpertActionMasks:
    kinds = torch.tensor([card, ability])
    cards = torch.tensor([False, True, True, False]) if card else torch.zeros(4, dtype=torch.bool)
    positions = torch.zeros(4, 576, dtype=torch.bool)
    if card:
        positions[1, 9] = True
        positions[2, 17] = True
    abilities = torch.tensor([False, True]) if ability else torch.zeros(2, dtype=torch.bool)
    ability_positions = torch.zeros(2, 576, dtype=torch.bool)
    if ability:
        ability_positions[1, 23] = True
    return ExpertActionMasks(
        action_kind=kinds,
        cards=cards,
        positions=positions,
        abilities=abilities,
        ability_positions=ability_positions,
        ability_requires_target=torch.tensor([False, ability]),
    )


def request(
    worker: int,
    digest: str,
    *,
    scalar: float = 1.0,
    action_masks: ExpertActionMasks | None = None,
    delta_ticks: int = 10,
    reset: bool = False,
    extra_inputs: dict[str, Tensor] | None = None,
) -> PolicyRequest:
    inputs: dict[str, Tensor] = {
        "public_scalars": torch.tensor([scalar]),
        "delta_ticks": torch.tensor(float(delta_ticks)),
    }
    inputs.update(extra_inputs or {})
    return PolicyRequest(
        worker_id=worker,
        side=worker % 2,
        actor_sha256=digest,
        actor_inputs=inputs,
        masks=action_masks or masks(),
        delta_ticks=delta_ticks,
        reset_hidden=reset,
    )


class BatchedPolicyServiceTests(unittest.TestCase):
    def test_groups_by_content_hash_and_never_forwards_per_worker(self):
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor_a = CountingActor()
        actor_b = CountingActor()
        hash_a, hash_b = "a" * 64, "b" * 64
        service.register_actor(actor_a, actor_sha256=hash_a)
        service.register_actor(actor_b, actor_sha256=hash_b)
        answers = service.act([
            request(0, hash_a),
            request(1, hash_b),
            request(2, hash_a),
        ])
        self.assertEqual(actor_a.calls, 1)
        self.assertEqual(actor_b.calls, 1)
        self.assertEqual(service.forward_calls, 2)
        self.assertEqual([row.worker_id for row in answers], [0, 1, 2])
        self.assertEqual([row.actor_sha256 for row in answers], [hash_a, hash_b, hash_a])

    def test_recurrent_hidden_is_scoped_and_reset_at_episode_boundary(self):
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor = CountingActor()
        digest = "c" * 64
        service.register_actor(actor, actor_sha256=digest)
        service.act([request(10, digest), request(11, digest)])
        first_pre = service.last_pre_action_hidden(
            actor_sha256=digest, worker_id=10, side=0
        )
        service.act([request(10, digest)])
        second_pre = service.last_pre_action_hidden(
            actor_sha256=digest, worker_id=10, side=0
        )
        service.act([request(10, digest, reset=True)])
        self.assertTrue(torch.equal(actor.hidden_seen[0], torch.zeros(1, 2, 4)))
        self.assertTrue(torch.equal(actor.hidden_seen[1], torch.ones(1, 1, 4)))
        self.assertTrue(torch.equal(actor.hidden_seen[2], torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(first_pre[0], torch.zeros(1, 1, 4)))
        self.assertTrue(torch.equal(second_pre[0], torch.ones(1, 1, 4)))
        self.assertEqual(service.reset_episode(10), 1)

    def test_deterministic_sampling_obeys_hierarchy_and_joint_logp(self):
        service = BatchedPolicyService(device="cpu", deterministic=True)
        actor = CountingActor()
        digest = "d" * 64
        service.register_actor(actor, actor_sha256=digest)
        card, ability, wait = service.act([
            request(0, digest, scalar=1.0, action_masks=masks(card=True, ability=False)),
            request(1, digest, scalar=-1.0, action_masks=masks(card=False, ability=True)),
            request(2, digest, action_masks=masks(card=False, ability=False)),
        ])
        self.assertTrue(card.event_happened)
        self.assertEqual((card.action_kind, card.card_slot, card.position), (0, 2, 17))
        self.assertTrue(ability.event_happened)
        self.assertEqual(
            (ability.action_kind, ability.ability_slot, ability.ability_position),
            (1, 1, 23),
        )
        self.assertTrue(ability.ability_requires_target)
        self.assertFalse(wait.event_happened)
        self.assertEqual(wait.logp_total, 0.0)
        for answer in (card, ability, wait):
            self.assertEqual(answer.delta_ticks, 10)
            self.assertTrue(all(math.isfinite(value) for value in (
                answer.lambda_per_second,
                answer.event_probability,
                answer.logp_total,
                answer.logp_timing,
                answer.logp_action_type,
                answer.logp_slot,
                answer.logp_position,
                answer.logp_mark,
            )))
            self.assertAlmostEqual(
                answer.logp_total,
                answer.logp_timing + answer.logp_action_type
                + answer.logp_slot + answer.logp_position,
                places=6,
            )

    def test_seeded_sampling_is_reproducible_and_entity_rows_are_padded(self):
        digest = "e" * 64
        services = [
            BatchedPolicyService(device="cpu", deterministic=False, seed=91),
            BatchedPolicyService(device="cpu", deterministic=False, seed=91),
        ]
        actors = [CountingActor(rate_logit=0.0), CountingActor(rate_logit=0.0)]
        batches = []
        for service, actor in zip(services, actors, strict=True):
            service.register_actor(actor, actor_sha256=digest)
            batches.append(service.act([
                request(0, digest, extra_inputs={
                    "entity_tokens": torch.tensor([1, 2]),
                    "entity_positions": torch.tensor([4, 5]),
                    "entity_relations": torch.tensor([0, 1]),
                    "entity_numeric": torch.ones(2, 3),
                    "entity_mask": torch.tensor([True, True]),
                }),
                request(1, digest, extra_inputs={
                    "entity_tokens": torch.tensor([3, 4, 5, 6]),
                    "entity_positions": torch.tensor([6, 7, 8, 9]),
                    "entity_relations": torch.tensor([1, 1, 0, 0]),
                    "entity_numeric": torch.ones(4, 3),
                    "entity_mask": torch.tensor([True, True, True, True]),
                }),
            ]))
        signature = lambda values: [
            (row.event_happened, row.action_kind, row.card_slot, row.position,
             row.ability_slot, row.ability_position)
            for row in values
        ]
        self.assertEqual(signature(batches[0]), signature(batches[1]))
        self.assertEqual(actors[0].entity_shape, (2, 1, 4))

    def test_nonfinite_actor_output_and_illegal_hierarchy_fail_closed(self):
        service = BatchedPolicyService(device="cpu", deterministic=True)
        digest = "f" * 64
        service.register_actor(CountingActor(nan=True), actor_sha256=digest)
        with self.assertRaises(FloatingPointError):
            service.act([request(0, digest)])

        service = BatchedPolicyService(device="cpu", deterministic=True)
        digest = "1" * 64
        service.register_actor(CountingActor(), actor_sha256=digest)
        broken = masks(card=True, ability=False)
        broken.positions.zero_()
        with self.assertRaisesRegex(ValueError, "placement"):
            service.act([request(0, digest, action_masks=broken)])


if __name__ == "__main__":
    unittest.main()
