import unittest
from expert_v1.training_v1.losses import MetricAccumulator


class MetricAccumulatorTests(unittest.TestCase):
    def test_empty_ability_batches_do_not_dilute_accuracy(self):
        m = MetricAccumulator()
        for _ in range(100):
            m.add({'ability_count': 0, 'ability_top1': 0, 'timing_count': 100})
        m.add({'ability_count': 5, 'ability_top1': .6, 'timing_count': 100})
        self.assertEqual(m.result()['ability_count'], 5)
        self.assertAlmostEqual(m.result()['ability_top1'], .6)

    def test_all_supervised_families_use_actual_label_counts(self):
        for name, metric in [('ability', 'top1'), ('card', 'top1'),
                             ('position', 'mean_cell_error'), ('kind', 'top1')]:
            with self.subTest(name=name):
                m = MetricAccumulator()
                m.add({f'{name}_count': 0, f'{name}_{metric}': float('nan')})
                m.add({f'{name}_count': 2, f'{name}_{metric}': .5})
                m.add({f'{name}_count': 8, f'{name}_{metric}': 1.})
                self.assertAlmostEqual(m.result()[f'{name}_{metric}'], .9)
                self.assertEqual(m.result()[f'{name}_count'], 10)

    def test_no_labels_keeps_zero_count_not_fake_sample(self):
        m = MetricAccumulator()
        m.add({'ability_count': 0, 'ability_top1': 0})
        self.assertEqual(m.result(), {'ability_count': 0., 'ability_top1': 0.})
        self.assertEqual(m.weights['ability_top1'], 0.)

    def test_loss_aggregation_contract_is_unchanged(self):
        m = MetricAccumulator()
        m.add({'timing_count': 100, 'ability_count': 0, 'loss': 6., 'loss_ability': 0.})
        m.add({'timing_count': 20, 'ability_count': 5, 'loss': 3., 'loss_ability': 2.})
        self.assertAlmostEqual(m.result()['loss'], 5.5)
        self.assertAlmostEqual(m.result()['loss_ability'], 1/3)


if __name__ == '__main__':
    unittest.main()
