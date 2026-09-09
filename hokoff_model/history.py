"""BC-only public normal-play history, derived from original source ticks."""
import os
import tempfile
from pathlib import Path
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
            kind_known = arrays['kind_label_mask'][rows].astype(bool)
            # Keep unknown executions as unknown slots; exclude identified skills.
            rows = rows[~(kind_known & (arrays['action_kind'][rows] == 1))]
            known = (arrays['kind_label_mask'][rows].astype(bool)
                     & (arrays['action_kind'][rows] == 0)
                     & arrays['timing_label_mask'][rows].astype(bool)
                     & arrays['card_label_mask'][rows].astype(bool)
                     & arrays['position_label_mask'][rows].astype(bool))
            events = np.zeros((len(rows), 4), dtype=np.int64)
            events[:, 0] = start + rows - lo
            events[:, 3] = known
            valid_rows = rows[known]
            slots = arrays['card_slot'][valid_rows]
            if np.any((slots < 0) | (slots >= 4)):
                raise ValueError('invalid history hand slot')
            tokens = arrays['hand_tokens'][valid_rows, slots]
            positions = arrays['position'][valid_rows]
            if np.any((tokens <= 0) | (positions < 0) | (positions >= 576)):
                raise ValueError('invalid history card/position')
            events[known, 1] = tokens
            events[known, 2] = positions
            self.events.append(events)
            if seq % 2 and (start != self.starts[-2] or hi-lo != offsets[seq]-offsets[seq-1]):
                raise ValueError('paired history clocks differ')

    def save(self, path):
        """Atomic, compact sidecar; competing loader workers may build it safely."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(e) for e in self.events])]
        events = np.concatenate(self.events, axis=0)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as f:
                temporary = f.name
                np.savez(f, source_offsets=self.offsets, event_offsets=offsets, events=events,
                         starts=np.asarray(self.starts, dtype=np.int64), contract=np.asarray(CONTRACT))
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary): os.unlink(temporary)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if str(z['contract']) != CONTRACT: raise ValueError('history cache contract differs')
            result = cls.__new__(cls)
            result.offsets = z['source_offsets']
            result.starts = z['starts'].tolist()
            offsets, events = z['event_offsets'], z['events']
        if (len(offsets) != len(result.offsets) or len(offsets) < 3
                or (len(offsets)-1) % 2 or offsets[0] != 0
                or offsets[-1] != len(events) or np.any(np.diff(offsets) < 0)
                or events.ndim != 2 or events.shape[1] != 4):
            raise ValueError('invalid history cache layout')
        result.events = [events[lo:hi] for lo,hi in zip(offsets[:-1],offsets[1:])]
        return result

    def query(self, rows, ticks, length):
        owners = np.searchsorted(self.offsets, rows, side='right')-1
        if not np.all(owners == owners[0]): raise ValueError('history window crosses actors')
        owner = int(owners[0])
        result = {k: np.zeros((len(rows), 2, length), dtype=(bool if k in
                  ('history_mask', 'history_known') else np.float32 if k == 'history_age' else np.int64))
                  for k in HISTORY_FIELDS}
        for relation, seq in enumerate((owner, owner ^ 1)):
            events = self.events[seq]
            if not len(events): continue
            stop = np.searchsorted(events[:, 0], ticks, side='left')
            indices = stop[:, None] - 1 - np.arange(length)[None, :]
            valid = indices >= 0
            chosen = events[np.maximum(indices, 0)]
            pos = chosen[..., 2]
            if relation: pos = np.where(chosen[..., 3].astype(bool), 575-pos, 0)
            result['history_card'][:, relation] = np.where(valid, chosen[..., 1], 0)
            result['history_position'][:, relation] = np.where(valid, pos, 0)
            result['history_age'][:, relation] = np.where(valid, (np.asarray(ticks)[:, None]-chosen[..., 0])/20.0, 0)
            result['history_mask'][:, relation] = valid
            result['history_known'][:, relation] = valid & chosen[..., 3].astype(bool)
        return {k: torch.from_numpy(v) for k, v in result.items()}
