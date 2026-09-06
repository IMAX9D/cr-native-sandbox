"""Resume a batch stream by skipping indices before any dataset I/O."""
from itertools import islice
from torch.utils.data import BatchSampler


class ResumableBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)
        self.start_batch = 0

    @property
    def total_batches(self):
        return super().__len__()

    def __len__(self):
        return max(0, self.total_batches - self.start_batch)

    def __iter__(self):
        return islice(super().__iter__(), self.start_batch, None)
