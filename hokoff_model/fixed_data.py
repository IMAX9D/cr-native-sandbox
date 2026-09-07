"""Fixed observations with legal, at-most-(period-1)-tick advanced action targets."""
from collections import Counter
import numpy as np

CONTRACT = 'fixed_period_legal_action_windows_v1'
SUPPORTED_COMPONENTS = {
    'compiler_sha256': '55e8380d8fddbf8c191e338c7f6dcd1604caa01fc874b0c1c0d7235125e0aab7',
    'deployment_masks_sha256': '294f5df6f1f8d762e2a1ad9304c275da94ba147347b8a2b54bc1f54bee252ecf',
}


def check_mask_proof(manifest, allow_smoke=False):
    if allow_smoke and manifest.get('smoke_only'):
        return
    compiler = manifest.get('compiler', {})
    if any(compiler.get('components', {}).get(k) != v for k, v in SUPPORTED_COMPONENTS.items()):
        raise ValueError('fixed-period mask transfer requires the audited compiler/mask versions')


def action_rejection(a, row, event):
    """None means all supervised action components are legal at the CURRENT row.

    The audited compiler masks dynamic-choice cards at non-action rows. Static
    deployment masks depend only on the selected card and crown towers. Comparing
    raw tower HP and tower grid entries proves the future selected mask unchanged;
    troop movement never enters that mask derivation. No future features are fed
    to the policy. Ability entity identity is unavailable, so never advance it.
    """
    kind = int(a['action_kind'][event])
    if not a['kind_label_mask'][event] or kind not in (0, 1):
        return 'unknown_action'
    if not a['action_kind_mask'][row, kind] or a['public_scalars'][row, 2] <= 0:
        return 'current_kind_unavailable'
    if kind == 1:
        return 'advanced_ability' if row != event else None
    slot, position = int(a['card_slot'][event]), int(a['position'][event])
    if not (a['card_label_mask'][event] and a['position_label_mask'][event]
            and 0 <= slot < 4 and 0 <= position < 576):
        return 'unknown_deploy'
    if not np.array_equal(a['hand_tokens'][row], a['hand_tokens'][event]):
        return 'hand_changed'
    if not a['card_mask'][row, slot]:
        return 'current_card_unavailable'
    stored = a['selected_position_mask_rows']
    i = int(np.searchsorted(stored, event))
    if (i == len(stored) or stored[i] != event or
            not (int(a['selected_position_mask_packed'][i, position // 8]) >> (position % 8)) & 1):
        return 'missing_position_mask'
    if row != event:
        if not np.array_equal(a['public_scalars'][row, 6:12], a['public_scalars'][event, 6:12]):
            return 'tower_changed'
        def towers(r):
            lo, hi = a['grid_offsets'][r:r+2]
            idx = a['grid_indices'][lo:hi]
            keep = idx < 4*576
            return idx[keep], a['grid_values'][lo:hi][keep]
        current, future = towers(row), towers(event)
        if any(not np.array_equal(x, y) for x, y in zip(current, future)):
            return 'tower_changed'
    return None


def fixed_indices(a, period=4, *, auxiliary=True, frame_window=17):
    if not 1 <= period <= 32767 or frame_window < 1:
        raise ValueError('invalid fixed period/history')
    offsets = a['sequence_offsets']; n = len(a['play_now'])
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
        raise ValueError('invalid sequence offsets')
    if not np.all(a['delta_ticks'] == 1) or not np.all(a['timing_exposure_ticks'] == 1):
        raise ValueError('requires contiguous one-tick source')
    fields = {k: [] for k in ('rows', 'ticks', 'elapsed', 'delay', 'delay_mask',
                              'supervision', 'label_rows', 'timing_mask')}
    packed_offsets = [0]; owners = []; roles = []; audit = Counter()

    def append(rows, initial, lo, elapsed, labels, timing, supervision, owner, role):
        values = (rows, rows-lo+initial, elapsed, np.zeros(len(rows), dtype=np.int64),
                  np.zeros(len(rows), dtype=bool), supervision, labels, timing)
        for key, value in zip(fields, values): fields[key].append(value)
        packed_offsets.append(packed_offsets[-1]+len(rows)); owners.append(owner); roles.append(role)

    for owner, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        lo, hi = int(lo), int(hi)
        recorded = np.rint(a['public_scalars'][lo:hi, 0]*6000).astype(np.int64)
        initial = int(recorded[0]); clock = initial+np.arange(hi-lo)
        if not 0 <= initial < 6000 or not np.array_equal(recorded, np.minimum(clock, 6000)):
            raise ValueError('source clock is not unambiguously contiguous')
        valid = np.asarray(a['timing_label_mask'][lo:hi], dtype=bool)
        starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])+lo
        ends = np.flatnonzero(valid & ~np.r_[valid[1:], False])+lo+1
        for start, end in zip(starts, ends):
            rows = np.arange(start, end, period)
            elapsed = np.r_[0, np.diff(rows)]
            events = np.flatnonzero(a['play_now'][start:end])+start
            first = np.searchsorted(events, rows)
            counts = np.searchsorted(events, np.minimum(rows+period, end))-first
            full = rows+period <= end
            known = full & (counts == 0)
            labels = np.full(len(rows), -1, dtype=np.int64)
            audit.update(primary_rows=len(rows), source_actions=len(events),
                         noop_periods=int(known.sum()), incomplete_periods=int((~full).sum()),
                         multiple_action_periods=int((full & (counts > 1)).sum()))
            for j in np.flatnonzero(full & (counts == 1)):
                event = int(events[first[j]])
                reason = action_rejection(a, int(rows[j]), event)
                if reason is None:
                    labels[j] = event; known[j] = True
                    audit['positive_periods'] += 1
                    audit['advance_%d_ticks' % (event-rows[j])] += 1
                else:
                    audit['excluded_'+reason] += 1
            audit['unknown_periods'] += int((~known).sum())
            append(rows, initial, lo, elapsed, labels, known, np.ones(len(rows), dtype=np.uint8), owner, 0)
            missed = np.setdiff1d(events, labels[labels >= 0], assume_unique=True)
            audit['unrepresented_primary_actions'] += len(missed)
            if auxiliary:
                for event in missed:
                    pos = int(np.searchsorted(rows, event)); begin = max(0, pos-frame_window+1)
                    history = np.r_[rows[begin:pos], event]
                    dt = np.r_[elapsed[begin:pos], event-rows[pos-1] if pos else 0]
                    supervision = np.zeros(len(history), dtype=np.uint8); supervision[-1] = 2
                    targets = np.full(len(history), -1, dtype=np.int64); targets[-1] = event
                    append(history, initial, lo, dt, targets, np.zeros(len(history), dtype=bool), supervision, owner, 1)
                    audit['auxiliary_actions'] += 1
    out = {k: np.concatenate(v) if v else np.empty(0, dtype=np.int64) for k, v in fields.items()}
    for k in ('rows', 'ticks', 'elapsed', 'delay', 'label_rows'): out[k] = out[k].astype(np.int64)
    for k in ('delay_mask', 'timing_mask'): out[k] = out[k].astype(bool)
    out['supervision'] = out['supervision'].astype(np.uint8)
    out.update(offsets=np.array(packed_offsets, dtype=np.int64), owners=np.array(owners, dtype=np.int64),
               roles=np.array(roles, dtype=np.uint8))
    return out, dict(audit)
