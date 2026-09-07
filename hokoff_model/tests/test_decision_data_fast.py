from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from policy_v1.data import collate
from policy_v1.smoke import create_fixture
from hokoff_model.decision_data import DecisionWindows, prepare, ragged_indices, collate_decisions


class FastDataTests(unittest.TestCase):
    def test_ragged_gather_handles_empty_and_repeated_rows(self):
        offsets = np.array([0, 2, 2, 5])
        rows = np.array([2, 1, 0, 2])
        q, c, src = ragged_indices(offsets, rows)
        np.testing.assert_array_equal(q, [0,0,0,2,2,3,3,3])
        np.testing.assert_array_equal(c, [0,1,2,0,1,0,1,2])
        np.testing.assert_array_equal(src, [2,3,4,0,1,2,3,4])
        q,c,src = ragged_indices(offsets, np.array([1,1]))
        self.assertEqual(len(src),0)

    def test_reopen_headers_and_collate_match_original_without_aliasing(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root/'data', steps=40)
            # Exercise Fortran-order numeric storage in the header cache.
            path=root/'data/shards/train-00000/public_scalars.npy'
            np.save(path, np.asfortranarray(np.load(path)))
            prepare(root/'data',root/'cache',allow_smoke=True)
            ds=DecisionWindows(root/'data',root/'cache','train',targets=4,frame_window=3)
            first=ds[0]
            ds.close()  # next access reopens memmaps using saved dtype/shape/order/offset
            second=ds[0]
            for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
            second['public_scalars'].zero_(); second['frame_ticks'].zero_()
            again=ds[0]
            torch.testing.assert_close(first['public_scalars'],again['public_scalars'])
            torch.testing.assert_close(first['frame_ticks'],again['frame_ticks'])
            short=ds[0]; long=ds[1]
            for k in ('entity_tokens','entity_positions','entity_relations','entity_numeric','entity_mask'):
                short[k]=short[k][:,:0]
            reference=collate([short,long]); actual=collate_decisions([short,long])
            for k in reference:torch.testing.assert_close(reference[k],actual[k],rtol=0,atol=0)
            ds.close()


if __name__=='__main__':unittest.main()
