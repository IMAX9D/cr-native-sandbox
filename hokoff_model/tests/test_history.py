import contextlib
import io
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from policy_v1.smoke import create_fixture
from policy_v1.data import open_arrays, close_arrays
from policy_v1.train import load_checkpoint
from hokoff_model.history import HistoryIndex, HISTORY_FIELDS
from hokoff_model.decision_data import prepare, DecisionWindows, collate_decisions
from hokoff_model.train_fixed import FixedConfig, FixedPolicy, parser, run
from hokoff_model.metrics import bc_loss


class HistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_fixture(self.root/'data', steps=81)
        a = open_arrays(self.root/'data/shards/validation-00000')
        self.a = {k: v.copy() for k,v in a.items()}
        close_arrays(a)

    def tearDown(self): self.tmp.cleanup()

    def test_both_sides_clock_rotation_unknown_and_strict_past(self):
        a = self.a
        # Side 1 normal play at source tick 103, in its own coordinate system.
        row = 81+3
        a['action_kind'][row] = 0
        a['card_label_mask'][row] = a['position_label_mask'][row] = 1
        a['card_slot'][row] = 1
        a['position'][row] = 100
        history = HistoryIndex(a, a['sequence_offsets'])
        b = history.query(np.array([2,3,4]), np.array([102,103,104]), 4)
        self.assertFalse(b['history_mask'][0].any())
        self.assertEqual(int(b['history_card'][1,0,0]),1)
        self.assertFalse(b['history_mask'][1,1].any())
        self.assertEqual(int(b['history_card'][2,1,0]),2)
        self.assertEqual(int(b['history_position'][2,1,0]),475)
        self.assertAlmostEqual(float(b['history_age'][2,1,0]),.05)
        opposite = history.query(np.array([85]), np.array([104]), 4)
        self.assertEqual(int(opposite['history_position'][0,0,0]),100)
        self.assertEqual(int(opposite['history_position'][0,1,0]),375)
        a['card_label_mask'][2] = 0
        unknown = HistoryIndex(a,a['sequence_offsets']).query(np.array([3]),np.array([103]),4)
        self.assertTrue(unknown['history_mask'][0,0,0])
        self.assertFalse(unknown['history_known'][0,0,0])
        self.assertEqual(int(unknown['history_card'][0,0,0]),0)
        # Future labels cannot change an earlier history.
        a['play_now'][10:] = 0
        earlier = HistoryIndex(a,a['sequence_offsets']).query(np.array([3]),np.array([103]),4)
        for key in HISTORY_FIELDS: torch.testing.assert_close(unknown[key],earlier[key])

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.root/'data',self.root/'cache',sampling='fixed',allow_smoke=True,auxiliary_frame_window=3)
        return DecisionWindows(self.root/'data',self.root/'cache','validation',sampling='fixed',
            decision_period=4,targets=4,frame_window=3,history_length=4)

    def test_fixed_labels_do_not_enter_history_early_and_backward(self):
        ds = self.prepare()
        b = ds[0]
        self.assertTrue(b['play_now'][0])  # tick 102 label is aligned to tick 100.
        self.assertFalse(b['history_mask'][0].any())
        self.assertTrue(b['history_mask'][1,0,0])  # tick 104 sees the actual past play.
        batch = collate_decisions([ds[0],ds[1]])
        model = FixedPolicy(FixedConfig(12,4,width=16,hidden_size=32,frame_window=3,
                                       history_length=4,spatial_type_dim=8))
        out = model(batch)
        loss,_ = bc_loss(out,batch)
        loss.backward()
        self.assertGreater(float(model.history_cards.weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(model.history_summary[0].weight.grad).all())
        changed = {k:v.clone() for k,v in batch.items()}
        for key in ('play_now','card_slot','position'): changed[key].zero_()
        torch.testing.assert_close(model(changed)['timing'],out['timing'])
        changed['history_card'][:,-1] = 3
        torch.testing.assert_close(model(changed)['timing'][:,:-1],out['timing'][:,:-1])
        with self.assertRaisesRegex(NotImplementedError,'BC-only'): model.forward_stream(batch)
        ds.close()

    def test_train_resume_and_evaluate(self):
        ds = self.prepare(); ds.close()
        def train(name,steps,resume=False):
            argv=['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
                  '--run',str(self.root/name),'--allow-smoke','--device','cpu','--workers','0',
                  '--cpu-threads','1','--width','16','--hidden-size','32','--frame-window','3',
                  '--targets','4','--batch-size','2','--history-length','4','--max-steps',str(steps),
                  '--eval-batches','1']
            if resume: argv += ['--resume',str(self.root/name/'last.pt')]
            with contextlib.redirect_stdout(io.StringIO()):run(parser().parse_args(argv))
            return load_checkpoint(self.root/name/'last.pt')
        full = train('full',4); train('resume',2); resumed = train('resume',4,True)
        for key in full['model']: torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
        from hokoff_model.evaluate_fixed import main
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(['--data',str(self.root/'data'),'--cache',str(self.root/'cache'),
                '--checkpoint',str(self.root/'full/last.pt'),'--output',str(self.root/'eval'),
                '--device','cpu','--workers','0','--batches','1','--batch-size','2'])
        self.assertIn('timing_ranking',result)

    def test_vector_queries_match_simple_reference(self):
        h = HistoryIndex(self.a, self.a['sequence_offsets'])
        # Deliberately include an unknown event and multiple plays at one tick.
        h.events[0] = np.array([[102,1,200,1],[106,0,0,0],[106,2,300,1],[150,3,400,1]])
        h.events[1] = np.array([[103,2,100,1],[140,4,500,1]])
        for owner in (0,1):
            rows = owner*81 + np.arange(81)
            ticks = 100 + np.arange(81)
            for length in (1,4,16):
                actual = h.query(rows,ticks,length)
                expected = {k:torch.zeros_like(v) for k,v in actual.items()}
                for relation,seq in enumerate((owner,owner^1)):
                    for t,tick in enumerate(ticks):
                        events = h.events[seq]
                        past = events[events[:,0]<tick][-length:][::-1]
                        for j,(at,card,pos,known) in enumerate(past):
                            expected['history_card'][t,relation,j] = int(card)
                            expected['history_position'][t,relation,j] = int(575-pos if relation and known else pos)
                            expected['history_age'][t,relation,j] = float((tick-at)/20)
                            expected['history_mask'][t,relation,j] = True
                            expected['history_known'][t,relation,j] = bool(known)
                for key in HISTORY_FIELDS:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)

    def test_cache_reuses_index_after_source_eviction(self):
        from unittest.mock import patch
        ds = self.prepare()
        first = ds[0]
        files = list((self.root/'cache/public-history-v1').glob('*.npz'))
        self.assertEqual(len(files),1)
        ds.close()
        with patch.object(HistoryIndex,'__init__',side_effect=AssertionError('unexpected source rescan')):
            second = ds[0]
        for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        ds.close()
        # Metadata identity changes select a new sidecar, never a stale index.
        ds.records[0]['metadata_sha256'] = 'different-content-key'
        ds[0]
        self.assertEqual(len(list((self.root/'cache/public-history-v1').glob('*.npz'))),2)
        ds.close()

    def test_cache_atomic_concurrent_publish(self):
        from concurrent.futures import ThreadPoolExecutor
        h = HistoryIndex(self.a,self.a['sequence_offsets'])
        path = self.root/'history/index.npz'
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _:h.save(path),range(8)))
        loaded = HistoryIndex.load(path)
        np.testing.assert_array_equal(h.offsets,loaded.offsets)
        for a,b in zip(h.events,loaded.events):np.testing.assert_array_equal(a,b)
        self.assertFalse(list(path.parent.glob('*.partial')))


if __name__ == '__main__': unittest.main()
