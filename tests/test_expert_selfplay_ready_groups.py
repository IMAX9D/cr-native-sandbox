from types import SimpleNamespace as NS
import threading
import unittest

from expert_selfplay_v1.ready_groups import ReadyGroupCollector


class ReadyGroupTests(unittest.TestCase):
    def specs(self, count=3):
        return [NS(worker_id=i, env=object(), header=NS(episode_id=str(i))) for i in range(count)]

    def test_independent_groups_complete_out_of_order_but_return_scheduled_order(self):
        second_done = threading.Event()
        closed = []

        def factory(index, cancel):
            def collect(rows, **kwargs):
                if index == 0:
                    self.assertTrue(second_done.wait(2), 'second group blocked by first')
                else:
                    second_done.set()
                return [NS(episode_id=row.header.episode_id) for row in reversed(rows)]
            return NS(collect_batch=collect, last_profile={'total_seconds': 1, 'actor_rows': 2},
                      policy_service=NS(close=lambda: closed.append(index)))

        collector = ReadyGroupCollector(factory, group_size=1)
        result = collector.collect_batch(self.specs())
        self.assertEqual([r.episode_id for r in result], ['0', '1', '2'])
        self.assertEqual(sorted(closed), [0, 1, 2])
        self.assertEqual(collector.last_profile['actor_rows'], 6)
        self.assertEqual(collector.last_profile['ready_group_count'], 3)

    def test_native_worker_cannot_be_owned_by_two_groups(self):
        rows = self.specs(2)
        rows[1].env = rows[0].env
        with self.assertRaisesRegex(ValueError, 'multiple worker'):
            ReadyGroupCollector(lambda *_: self.fail(), group_size=1).collect_batch(rows)

    def test_group_limit_rejected_before_start(self):
        with self.assertRaisesRegex(ValueError, 'concurrency limit'):
            ReadyGroupCollector(lambda *_: self.fail(), group_size=1,
                                max_groups=2).collect_batch(self.specs())

    def test_failure_cancels_other_group_and_no_partial_result_is_returned(self):
        waiting = threading.Event()
        closed = []

        def factory(index, cancel):
            def collect(rows, **kwargs):
                if index == 0:
                    waiting.set()
                    self.assertTrue(cancel.wait(2))
                    raise RuntimeError('cancelled')
                self.assertTrue(waiting.wait(2))
                raise ValueError('native failed')
            return NS(collect_batch=collect, last_profile={},
                      policy_service=NS(close=lambda: closed.append(index)))

        with self.assertRaisesRegex(ValueError, 'native failed'):
            ReadyGroupCollector(factory, group_size=1).collect_batch(self.specs(2))
        self.assertEqual(sorted(closed), [0, 1])

    def test_rolling_same_worker_stays_in_same_group(self):
        rows = self.specs(2)
        rows.append(NS(worker_id=0, env=rows[0].env, header=NS(episode_id='2')))
        assignments = []

        def factory(index, cancel):
            def collect(group, **kwargs):
                self.assertTrue(kwargs['rolling_replacement'])
                assignments.append([r.header.episode_id for r in group])
                return [NS(episode_id=r.header.episode_id) for r in group]
            return NS(collect_batch=collect, last_profile={}, policy_service=NS(close=lambda: None))

        result = ReadyGroupCollector(factory, group_size=1).collect_batch(rows, rolling_replacement=True)
        self.assertEqual([r.episode_id for r in result], ['0', '1', '2'])
        self.assertIn(['0', '2'], assignments)


if __name__ == '__main__':
    unittest.main()
