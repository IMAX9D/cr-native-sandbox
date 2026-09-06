"""Inclusive [t, t+H] timing supervision; future observations never enter the model."""
from bisect import bisect_right

import numpy as np
import torch
from policy_v1.data import Windows


TARGET_CONTRACT = 'action_in_inclusive_t_to_t_plus_h_v1'


def future_targets(actions, valid, horizon, mask_horizon=None):
    """Conservatively censor incomplete/gapped future, even if a positive is seen."""
    mask_horizon = horizon if mask_horizon is None else mask_horizon
    if horizon < 0 or mask_horizon < horizon:
        raise ValueError('mask horizon must cover a nonnegative target horizon')
    actions = np.asarray(actions, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if actions.ndim != 1 or actions.shape != valid.shape:
        raise ValueError('matching 1D arrays required')
    n = len(actions)
    rows = np.arange(n)
    event_sum = np.r_[0, np.cumsum(actions & valid)]
    bad_sum = np.r_[0, np.cumsum(~valid)]
    target_end = np.minimum(rows+horizon+1, n)
    mask_end = np.minimum(rows+mask_horizon+1, n)
    known = (rows+mask_horizon < n) & (bad_sum[mask_end] == bad_sum[rows])
    target = (event_sum[target_end]-event_sum[rows] > 0) & known
    return target, known


class HorizonWindows(Windows):
    def __init__(self, *args, timing_horizon_ticks=0, timing_mask_horizon_ticks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_horizon_ticks = timing_horizon_ticks
        self.timing_mask_horizon_ticks = (timing_horizon_ticks if timing_mask_horizon_ticks is None
                                          else timing_mask_horizon_ticks)
        if self.timing_horizon_ticks < 0 or self.timing_mask_horizon_ticks < self.timing_horizon_ticks:
            raise ValueError('invalid horizons')

    def __getitem__(self, index):
        b = super().__getitem__(index)
        sh = bisect_right(self.prefix, index)-1
        local = index-self.prefix[sh]
        seq = int(np.searchsorted(self.sequence_prefix[sh], local, side='right')-1)
        begin,end = self.records[sh]['offsets'][seq:seq+2]
        target = begin+(local-int(self.sequence_prefix[sh][seq]))*self.targets
        start = max(begin,target-self.frame_window+1)
        stop = start+len(b['frame_ticks'])
        arrays,_,_ = self._open(sh)
        future_stop = min(end,stop+self.timing_mask_horizon_ticks)
        # Only label arrays extend beyond the normal input window. Do not read
        # future hand, scalar, grid, entity or event observations here.
        y,known = future_targets(arrays['play_now'][start:future_stop],
                                 arrays['timing_label_mask'][start:future_stop],
                                 self.timing_horizon_ticks,self.timing_mask_horizon_ticks)
        b['timing_target'] = torch.from_numpy(y[:stop-start].copy())
        b['timing_target_mask'] = torch.from_numpy(known[:stop-start].copy())
        return b
