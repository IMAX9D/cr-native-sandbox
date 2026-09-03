import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

from scripts.experiment_expert_position_stability import (
    transform_scores, position_summary, configure_threads, forward_position_variant,
)
from expert_v1.training_v1.model import ExpertPolicyOutput


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_query = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.SiLU(), torch.nn.Linear(4, 4))
        self.cell_features = torch.nn.Conv2d(3, 4, 1)

    def forward_batch(self, batch):
        query = self.position_query(batch['query_context'])
        cells = self.cell_features(batch['spatial_context']).flatten(2).transpose(1, 2).reshape(2, 3, 6, 4)
        positions = torch.einsum('btse,btpe->btsp', query, cells) / 2
        z = positions.new_zeros(2, 3)
        return ExpertPolicyOutput(z, z[..., None].expand(2,3,2), z[...,None].expand(2,3,4),
                                  positions, z[...,None], positions[:,:,:1], (z,z))


def toy_batch():
    mask = torch.tensor([[False, True, False], [False, False, True]])
    return {'query_context':torch.randn(2,3,4,6), 'spatial_context':torch.randn(6,3,2,3),
            'loss_mask':torch.ones(2,3,dtype=torch.bool), 'position_label_mask':mask,
            'card_slot':torch.tensor([[0,2,0],[0,0,1]]), 'position':torch.tensor([[0,4,0],[0,0,1]]),
            'position_mask':torch.ones(2,3,6,dtype=torch.bool), 'sample_weight':torch.ones(2,3)}


class PositionExperimentTests(unittest.TestCase):
    def test_thread_configuration_is_idempotent(self):
        with patch('torch.get_num_interop_threads', return_value=1), patch('torch.set_num_interop_threads') as interop, patch('torch.set_num_threads'):
            configure_threads(6)
            configure_threads(6)
            interop.assert_not_called()

    def test_sparse_override_matches_dense_loss_and_gradients(self):
        torch.manual_seed(7)
        dense = ToyPolicy()
        sparse = deepcopy(dense)
        batch = toy_batch()
        b,t = batch['position_label_mask'].nonzero(as_tuple=True)
        c = batch['card_slot'][b,t]
        labels = batch['position'][b,t]
        first = transform_scores(dense.forward_batch(batch).position_logits[b,t,c], 'fp32_softcap20')
        second = forward_position_variant(sparse, batch, 'fp32_softcap20').position_logits[b,t,c]
        a = torch.nn.functional.cross_entropy(first, labels)
        z = torch.nn.functional.cross_entropy(second, labels)
        a.backward(); z.backward()
        self.assertTrue(torch.allclose(a,z,atol=1e-6))
        for p,q in zip(dense.parameters(),sparse.parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-6))

    def test_empty_positions_preserve_zero_gradients(self):
        torch.manual_seed(8)
        model = ToyPolicy()
        batch = toy_batch()
        batch['position_label_mask'].zero_()
        output = forward_position_variant(model,batch,'fp32_softcap10')
        (output.position_logits.sum()*0).backward()
        self.assertTrue(all(p.grad is not None and not bool(p.grad.any()) for p in model.parameters()))

    def test_centered_softcap_is_shift_invariant_and_bounded(self):
        x = torch.tensor([[2., -3., 20., 5.], [-2., 1., 4., 3.]])
        a = transform_scores(x, "fp32_softcap10")
        b = transform_scores(x + 10000, "fp32_softcap10")
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(bool((a.abs() <= 10).all()))
        self.assertTrue(torch.equal(x.argmax(-1), a.argmax(-1)))

    def test_weighted_position_loss_and_legal_mask(self):
        x = torch.tensor([[1., 3., 20.], [2., 1., 0.]], requires_grad=True)
        labels = torch.tensor([1, 0])
        legal = torch.tensor([[True, True, False], [True, True, True]])
        weights = torch.tensor([1., 3.])
        metrics, loss = position_summary(x, labels, legal, weights)
        expected = (torch.nn.functional.cross_entropy(x[:1, :2], labels[:1]) +
                    3 * torch.nn.functional.cross_entropy(x[1:], labels[1:])) / 4
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=6)
        self.assertEqual(metrics["legal_target_failures"], 0)
        loss.backward()
        self.assertEqual(float(x.grad[0, 2]), 0.)

    def test_softcap_extreme_scores_has_finite_gradients(self):
        x = torch.tensor([[-16000., -15600., -15800.]], requires_grad=True)
        y = transform_scores(x, "fp32_softcap20")
        metrics, loss = position_summary(y, torch.tensor([0]), torch.ones_like(x, dtype=torch.bool), torch.ones(1))
        loss.backward()
        self.assertTrue(metrics["finite"])
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertLess(float(loss.detach()), 42.)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            transform_scores(torch.zeros(1, 2), "unknown")


if __name__ == "__main__":
    unittest.main()
