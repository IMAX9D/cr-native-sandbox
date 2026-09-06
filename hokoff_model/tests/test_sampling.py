import unittest
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from policy_v1.sampling import ResumableBatchSampler


class CountingData(Dataset):
    def __init__(self):
        self.reads = []

    def __len__(self):
        return 11

    def __getitem__(self, index):
        self.reads.append(index)
        return index


class SamplingTests(unittest.TestCase):
    def test_resume_never_reads_skipped_rows_and_keeps_tail(self):
        data = CountingData()
        sampler = ResumableBatchSampler(SequentialSampler(data), 3)
        sampler.start_batch = 2
        loader = DataLoader(data, batch_sampler=sampler, num_workers=0)
        self.assertEqual([b.tolist() for b in loader], [[6,7,8],[9,10]])
        self.assertEqual(data.reads, [6,7,8,9,10])
        self.assertEqual(len(loader), 2)
        self.assertEqual(sampler.total_batches, 4)
        sampler.start_batch = 4
        self.assertEqual(list(loader), [])
        sampler.start_batch = 0
        self.assertEqual(len(loader), 4)
