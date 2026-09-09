"""Read-only ordered labels at original tick offsets; no future state inputs."""
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import numpy as np
from policy_v1.data import digest, within
from .fixed_data import CONTRACT, check_mask_proof
from .ordered_model import STOP, DEPLOY, ABILITY


def locate(dataset, index):
    sh = bisect_right(dataset.prefix, index)-1
    local = index-dataset.prefix[sh]
    p = dataset.sequence_prefix[sh]
    seq = int(np.searchsorted(p, local, side='right')-1)
    if dataset.records[sh]['segment_roles'][seq] != 0:
        return None
    start = dataset.records[sh]['offsets'][seq]
    return sh, seq, start+local-int(p[seq])  # targets=1 only


def labels_at(a, row, actor_end, period=4):
    """Expert sequence forecast, NOT legality of executing everything at row.

    No future masks are returned as network inputs. They authenticate labels at
    their own recorded execution time. >2 events, partial labels, ambiguous hero
    types and censored/incomplete periods are rejected, never relabeled STOP.
    """
    if row+period > actor_end or not np.all(a['timing_label_mask'][row:row+period]):
        return None, 'incomplete_or_censored'
    events = np.flatnonzero(a['play_now'][row:row+period])+row
    if len(events) > 2:
        return None, 'more_than_two'
    result = dict(type=np.zeros(2, dtype=np.int64), token=np.zeros(2, dtype=np.int64),
                  position=np.full(2, 576, dtype=np.int64), tick=np.zeros(2, dtype=np.int64),
                  mask=np.array([True, len(events)>0]))
    for step, event in enumerate(events):
        kind = int(a['action_kind'][event])
        if not a['kind_label_mask'][event] or kind not in (0, 1):
            return None, 'unknown_kind'
        if not a['action_kind_mask'][event, kind]:
            return None, 'illegal_recorded_kind'
        result['type'][step] = DEPLOY if kind == 0 else ABILITY
        result['tick'][step] = event-row
        if kind == 0:
            slot, position = int(a['card_slot'][event]), int(a['position'][event])
            if not (a['card_label_mask'][event] and a['position_label_mask'][event]
                    and 0 <= slot < 4 and 0 <= position < 576 and a['card_mask'][event, slot]):
                return None, 'unknown_or_illegal_card'
            stored = a['selected_position_mask_rows']; i = int(np.searchsorted(stored, event))
            if (i == len(stored) or stored[i] != event or
                    not (int(a['selected_position_mask_packed'][i, position//8]) >> (position%8)) & 1):
                return None, 'unknown_or_illegal_position'
            result['token'][step] = a['hand_tokens'][event, slot]
            result['position'][step] = position
        else:
            slot = int(a['ability_slot'][event])
            if not (a['ability_label_mask'][event] and 0 <= slot < a['ability_tokens'].shape[1]
                    and a['ability_mask'][event, slot]):
                return None, 'unknown_or_illegal_ability'
            token = int(a['ability_tokens'][event, slot])
            if token == 0 or np.count_nonzero(a['ability_tokens'][event] == token) != 1:
                return None, 'ambiguous_hero_type'
            result['token'][step] = token
        if result['token'][step] <= 0:
            return None, 'missing_token'
    return result, None


def audit_pairs(root, cache):
    root, cache = Path(root), Path(cache)
    manifest = json.loads((root/'manifest.json').read_text())
    index = json.loads((cache/'index.json').read_text())
    check_mask_proof(manifest)
    if index['decision_contract'] != CONTRACT or index['decision_period'] != 4:
        raise ValueError('this pilot requires the audited fixed4 cache')
    if index['manifest_sha256'] != digest(root/'manifest.json'):
        raise ValueError('source manifest changed')
    report = dict(manifest_sha256=index['manifest_sha256'], cache_sha256=digest(cache/'index.json'),
                  period=4, same_tick_collisions='rejected_by_source_compiler',
                  exact_costs_available=False, hero_instance_ids_available=False, splits={})
    for split in ('validation', 'train'):
        counts = Counter(); pairs = []
        for ri, record in enumerate(index['splits'][split]):
            counts.update({k: record['label_audit'].get(k, 0) for k in
                           ('primary_rows', 'positive_periods', 'multiple_action_periods')})
            if not record['label_audit'].get('multiple_action_periods'):
                continue
            shard = within(root, record['path']); path = within(cache, record['indices'])
            if digest(shard/'shard.json') != record['metadata_sha256'] or digest(path) != record['indices_sha256']:
                raise ValueError('shard metadata or cache changed')
            a = {k: np.load(shard/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                 for k in ('play_now','timing_label_mask','action_kind','kind_label_mask','sequence_offsets')}
            with np.load(path, allow_pickle=False) as z:
                rows, offsets, roles, owners = (z[k] for k in ('rows','offsets','roles','owners'))
            for seq in np.flatnonzero(roles == 0):
                lo, hi = offsets[seq:seq+2]; rr = rows[lo:hi]
                end = int(a['sequence_offsets'][owners[seq]+1])
                events = np.flatnonzero(a['play_now'][rr[0]:min(end, rr[-1]+4)])+rr[0]
                first = np.searchsorted(events, rr)
                n = np.searchsorted(events, rr+4)-first
                for j in np.flatnonzero(n > 1):
                    row = int(rr[j])
                    if row+4 > end or not a['timing_label_mask'][row:row+4].all(): continue
                    ee = events[first[j]:first[j]+n[j]]
                    kinds = [int(a['action_kind'][e]) if a['kind_label_mask'][e] else -1 for e in ee]
                    counts['pattern_'+'_'.join(map(str,kinds))] += 1
                    counts['first_offset_'+str(ee[0]-row)] += 1
                    if len(ee) == 2: counts['gap_'+str(ee[1]-ee[0])] += 1
                    pairs.append(dict(record=ri, seq=int(seq), row=row, events=ee.tolist(), kinds=kinds))
            for v in a.values(): v._mmap.close()
        if len(pairs) != counts['multiple_action_periods']:
            raise ValueError('pair audit disagrees with fixed cache')
        report['splits'][split] = dict(counts=dict(counts), pairs=pairs)
        print('pair_audit', split, dict(counts), flush=True)
    return report


def target_for_index(dataset, index):
    located = locate(dataset, index)
    if located is None: return None, 'auxiliary_window'
    sh, seq, pos = located
    a, d = dataset._open(sh)
    # Owners are retained in each decision NPZ, including auxiliary segments.
    offsets = np.load(dataset.shard_paths[sh]/'sequence_offsets.npy', mmap_mode='r', allow_pickle=False)
    end = int(offsets[int(d['owners'][seq])+1]); offsets._mmap.close()
    return labels_at(a, int(d['rows'][pos]), end)


def sample_plan(dataset, pairs, *, examples_per_class=678, natural_count=0, seed=42):
    if dataset.targets != 1: raise ValueError('ordered pilot uses one target per history window')
    rng = np.random.default_rng(seed)
    chosen = {}; rejected = Counter()
    for pair in pairs:
        sh, seq = pair['record'], pair['seq']
        _, d = dataset._open(sh)
        begin, end = dataset.records[sh]['offsets'][seq:seq+2]
        pos = int(np.searchsorted(d['rows'][begin:end], pair['row']))
        index = dataset.prefix[sh]+int(dataset.sequence_prefix[sh][seq])+pos
        target, reason = target_for_index(dataset, index)
        if target is None: rejected[reason] += 1
        else: chosen[index] = target
    wanted = {0: examples_per_class, 1: examples_per_class}
    counts = Counter(int((t['type'] != STOP).sum()) for t in chosen.values())
    attempts = 0
    while any(counts[k] < n for k,n in wanted.items()):
        attempts += 1
        if attempts > 2000*max(1, examples_per_class):
            raise ValueError('insufficient usable examples for stratified pilot')
        index = int(rng.integers(len(dataset)))
        if index in chosen: continue
        located = locate(dataset, index)
        if located is None: continue
        sh, _, pos = located
        _, d = dataset._open(sh)
        # Cheap screening; avoid gathering sparse observations/masks for WAIT candidates.
        if not d['timing_mask'][pos]: continue
        n = int(d['label_rows'][pos] >= 0)
        if counts[n] >= wanted[n]: continue
        target, reason = target_for_index(dataset, index)
        if target is None: rejected[reason] += 1; continue
        actual_n = int((target['type'] != STOP).sum())
        if actual_n != n: raise ValueError('source targets disagree with fixed single/WAIT label')
        chosen[index] = target; counts[n] += 1
    natural = {}; tries = 0
    while len(natural) < natural_count:
        tries += 1
        if tries > max(1, natural_count)*100: raise ValueError('insufficient natural windows')
        index = int(rng.integers(len(dataset)))
        if index in natural or locate(dataset,index) is None: continue
        target, reason = target_for_index(dataset, index)
        if target is None: continue
        natural[index] = target
    def pack(samples):
        keys = sorted(samples)
        return dict(indices=keys, targets={k:np.stack([samples[i][k] for i in keys])
                                         for k in ('type','token','position','tick','mask')})
    return pack(chosen), pack(natural) if natural else None, dict(counts=dict(counts), rejected=dict(rejected))
