"""BC-only public normal-play history, derived from original source ticks."""
import numpy as np
import torch

HISTORY_FIELDS = ('history_card', 'history_position', 'history_age', 'history_mask', 'history_known')
CONTRACT = 'paired_source_tick_normal_plays_strict_past_unknown_slots_v1'


class HistoryIndex:
    def __init__(self, arrays, offsets):
        self.offsets = np.asarray(offsets, dtype=np.int64).copy()
        self.events = []
        self.starts = []
        if (len(offsets)-1) % 2:
            raise ValueError('history requires paired actors')
        for seq, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            lo, hi = int(lo), int(hi)
            start = int(np.rint(arrays['public_scalars'][lo, 0]*6000))
            self.starts.append(start)
            ticks = start + np.arange(hi-lo)
            if not np.array_equal(np.rint(arrays['public_scalars'][lo:hi, 0]*6000), np.minimum(ticks, 6000)):
                raise ValueError('history requires contiguous source clock')
            rows = lo + np.flatnonzero(arrays['play_now'][lo:hi])
            events = []
            for row in rows:
                kind_known = bool(arrays['kind_label_mask'][row])
                if kind_known and int(arrays['action_kind'][row]) == 1:
                    continue  # First version deliberately excludes identified skills.
                known = (kind_known and int(arrays['action_kind'][row]) == 0
                         and bool(arrays['timing_label_mask'][row])
                         and bool(arrays['card_label_mask'][row])
                         and bool(arrays['position_label_mask'][row]))
                token, position = 0, 0
                if known:
                    slot = int(arrays['card_slot'][row])
                    if not 0 <= slot < 4: raise ValueError('invalid history hand slot')
                    token = int(arrays['hand_tokens'][row, slot])
                    position = int(arrays['position'][row])
                    if token <= 0 or not 0 <= position < 576:
                        raise ValueError('invalid history card/position')
                # Unknown executions occupy a slot instead of fabricating no play.
                events.append((start+int(row)-lo, token, position, int(known)))
            self.events.append(np.asarray(events, dtype=np.int64).reshape(-1, 4))
            if seq % 2 and (start != self.starts[-2] or hi-lo != offsets[seq]-offsets[seq-1]):
                raise ValueError('paired history clocks differ')

    def query(self, rows, ticks, length):
        owners = np.searchsorted(self.offsets, rows, side='right')-1
        if not np.all(owners == owners[0]): raise ValueError('history window crosses actors')
        owner = int(owners[0])
        result = {k: np.zeros((len(rows), 2, length), dtype=(bool if k in
                  ('history_mask', 'history_known') else np.float32 if k == 'history_age' else np.int64))
                  for k in HISTORY_FIELDS}
        for relation, seq in enumerate((owner, owner ^ 1)):
            events = self.events[seq]
            for t, tick in enumerate(ticks):
                stop = np.searchsorted(events[:, 0], tick, side='left')
                chosen = events[max(0, stop-length):stop][::-1]
                n = len(chosen)
                if not n: continue
                pos = chosen[:, 2].copy()
                if relation: pos = np.where(chosen[:, 3].astype(bool), 575-pos, 0)
                result['history_card'][t, relation, :n] = chosen[:, 1]
                result['history_position'][t, relation, :n] = pos
                result['history_age'][t, relation, :n] = (tick-chosen[:, 0])/20.0
                result['history_mask'][t, relation, :n] = True
                result['history_known'][t, relation, :n] = chosen[:, 3].astype(bool)
        return {k: torch.from_numpy(v) for k, v in result.items()}
