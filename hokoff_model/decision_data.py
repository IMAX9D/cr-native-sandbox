"""Bounded, event-preserving decision indices over immutable native tick shards.

Preparation reads labels; the policy only receives selected current observations.
Unknown timing regions split recurrent sequences and never become WAIT examples.
"""
import argparse
import hashlib
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from policy_v1.data import (SCALARS, ROW_TOKENS, LABELS, MASKS, ENTITY_FIELDS,
                            EVENT_FIELDS, FIELDS, digest, within, open_arrays, close_arrays)

CONTRACT = 'bounded_observe_after_action_censored_v1'
from .independent_data import CONTRACT as INDEPENDENT_CONTRACT, independent_indices
from .fixed_data import CONTRACT as FIXED_CONTRACT, fixed_indices, check_mask_proof
CONTRACTS = dict(legacy=CONTRACT, independent=INDEPENDENT_CONTRACT, fixed=FIXED_CONTRACT)
READER_FIELDS = tuple(k for k in FIELDS if k not in (
    'sequence_offsets', 'delta_ticks', 'timing_exposure_ticks', 'replay_extent'))


def decision_indices(offsets, scalars, delta, exposure, actions, valid, max_delay):
    """Return source-row indices; one output segment per contiguous valid region."""
    if not 1 <= max_delay <= 32767:
        raise ValueError('max_delay must be in 1..32767')
    n = len(actions)
    if (len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n
            or np.any(np.diff(offsets) <= 0) or len(valid) != n):
        raise ValueError('invalid source sequence offsets')
    if not np.all(delta == 1) or not np.all(exposure == 1):
        raise ValueError('preparation requires contiguous one-tick source data')
    chunks = {k: [] for k in ('rows', 'ticks', 'elapsed', 'delay', 'delay_mask')}
    out_offsets, owners = [0], []
    for owner, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        lo, hi = int(lo), int(hi)
        recorded = np.rint(scalars[lo:hi, 0] * 6000).astype(np.int64)
        initial = int(recorded[0])
        clock = initial + np.arange(hi-lo)
        if not 0 <= initial < 6000 or not np.array_equal(recorded, np.minimum(clock, 6000)):
            raise ValueError('source clock is not unambiguously contiguous')
        v = np.asarray(valid[lo:hi], dtype=bool)
        events = np.flatnonzero(np.asarray(actions[lo:hi], dtype=bool) & v)
        starts = np.flatnonzero(v & ~np.r_[False, v[:-1]])
        ends = np.flatnonzero(v & ~np.r_[v[1:], False]) + 1
        for start, end in zip(starts, ends):
            anchors = np.r_[start, events[(events > start) & (events < end)]]
            stops = np.r_[anchors[1:], end]
            rows = np.concatenate([np.arange(s, e, max_delay) for s, e in zip(anchors, stops)])
            elapsed = np.r_[0, np.diff(rows)]
            delay = np.r_[np.diff(rows), 0]
            known = np.ones(len(rows), dtype=bool)
            known[-1] = False  # no invented exact duration at any unknown/terminal boundary
            for name, value in [('rows', rows+lo), ('ticks', rows+initial),
                                ('elapsed', elapsed), ('delay', delay), ('delay_mask', known)]:
                chunks[name].append(value)
            out_offsets.append(out_offsets[-1]+len(rows))
            owners.append(owner)
    result = {k: np.concatenate(v) if v else np.empty(0, dtype=bool if k == 'delay_mask' else np.int64)
              for k, v in chunks.items()}
    result['offsets'] = np.asarray(out_offsets, dtype=np.int64)
    result['owners'] = np.asarray(owners, dtype=np.int64)
    result['delay_mask'] = result['delay_mask'].astype(bool)
    return result


def prepare(root, cache, *, max_delay=8, splits=None, max_shards_per_split=None,
            verify_hashes=False, allow_smoke=False, sampling='legacy', sampling_seed=42,
            auxiliary_split='validation', auxiliary_frame_window=17, decision_period=4):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    if root == cache or root in cache.parents:
        raise ValueError('decision cache must be outside source data')
    if (cache/'index.json').exists():
        raise FileExistsError('use a new cache directory')
    if not 1 <= max_delay <= 32767 or (max_shards_per_split is not None and max_shards_per_split < 1):
        raise ValueError('invalid delay/shard limit')
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('kind') != 'cr_native_expert_bc_dataset_v3'
            or manifest.get('observation_mode') != 'native_state_v1'
            or manifest.get('actor_information') != 'public_only_v1'):
        raise ValueError('requires public-only native_state_v1 source')
    if not manifest.get('production_ready') and not (allow_smoke and manifest.get('smoke_only')):
        raise ValueError('non-production source; synthetic smoke needs explicit opt-in')
    if (manifest['feature_schema']['public_scalars'] != SCALARS
            or manifest['feature_schema']['entity_numeric'] != ['level_ratio', 'hp_ratio', 'log_max_hp']
            or manifest['dimensions']['grid_channels'] != 8):
        raise ValueError('unsupported observation schema')
    if sampling == 'fixed':
        check_mask_proof(manifest, allow_smoke)
    cache.mkdir(parents=True, exist_ok=True)
    if sampling not in CONTRACTS:
        raise ValueError('unknown observation sampling')
    index = dict(version=1, decision_contract=CONTRACTS[sampling], max_delay=max_delay,
                 manifest_sha256=digest(root/'manifest.json'), dimensions=manifest['dimensions'],
                 smoke_only=bool(manifest.get('smoke_only')), verified_source_hashes=verify_hashes,
                 max_shards_per_split=max_shards_per_split, event_contract='no_event_inputs_v1', splits={})
    if sampling in ('independent', 'fixed'):
        index.update(sampling=sampling,sampling_seed=sampling_seed,auxiliary_split=auxiliary_split,
                     auxiliary_frame_window=auxiliary_frame_window)
    if sampling == 'fixed':
        index.update(decision_period=decision_period, target_window='[t,t+period)',
                     mask_proof='audited_static_card_and_equal_crown_towers_v1')
    seen = set()
    for split in (splits or manifest['splits']):
        records = []
        for relative in manifest['splits'][split][:max_shards_per_split]:
            shard = within(root, relative)
            meta = json.loads((shard/'shard.json').read_text())
            metadata_sha = digest(shard/'shard.json')
            expected_meta = manifest.get('shard_metadata_sha256', {}).get(relative)
            if expected_meta and expected_meta != metadata_sha:
                raise ValueError('source metadata checksum mismatch')
            arrays = open_arrays(shard)
            try:
                if verify_hashes:
                    for name in FIELDS:
                        if digest(shard/(name+'.npy')) != meta['file_sha256'][name+'.npy']:
                            raise ValueError('source checksum mismatch: '+name)
                identities = meta['sequence_identity']
                offsets = arrays['sequence_offsets']
                if len(identities)+1 != len(offsets) or len(identities) % 2:
                    raise ValueError('invalid paired actor identities')
                tags = []
                for j in range(0, len(identities), 2):
                    left, right = identities[j:j+2]
                    tag = left['battle_tag']
                    if left['actor_side'] != 0 or right['actor_side'] != 1 or tag != right['battle_tag'] or tag in seen:
                        raise ValueError('duplicate battle or invalid actor pairing')
                    seen.add(tag); tags.append(tag)
                inputs=(offsets, arrays['public_scalars'], arrays['delta_ticks'],
                        arrays['timing_exposure_ticks'], arrays['play_now'], arrays['timing_label_mask'], max_delay)
                audit = None
                if sampling == 'fixed':
                    packed, audit = fixed_indices(arrays, decision_period, auxiliary=split==auxiliary_split,
                                                  frame_window=auxiliary_frame_window)
                elif sampling=='independent':
                    seed=int.from_bytes(hashlib.sha256((str(sampling_seed)+':'+relative).encode()).digest()[:8],'little')
                    packed=independent_indices(*inputs,seed=seed,auxiliary=split==auxiliary_split,
                                               frame_window=auxiliary_frame_window)
                else:
                    packed=decision_indices(*inputs)
                if np.any(arrays['kind_label_mask'] & ~arrays['play_now']):
                    raise ValueError('conditional kind label without action')
                name = '%s-%05d-decisions.npz' % (split, len(records))
                with (cache/(name+'.partial')).open('wb') as f:
                    np.savez_compressed(f, **packed)
                (cache/(name+'.partial')).replace(cache/name)
                records.append(dict(path=relative, metadata_sha256=metadata_sha, indices=name,
                                    indices_sha256=digest(cache/name), offsets=packed['offsets'].tolist(),
                                    identities=[identities[int(j)] for j in (packed['owners'] if sampling=='legacy'
                                        else packed['owners'][packed['roles']==0])], battle_tags=tags,
                                    source_rows=len(arrays['play_now']), valid_rows=int(arrays['timing_label_mask'].sum()),
                                    decision_rows=len(packed['rows']),
                                    actions=int((arrays['play_now'].astype(bool) & arrays['timing_label_mask'].astype(bool)).sum())))
                if audit is not None:
                    records[-1]['label_audit'] = audit
                if sampling in ('independent', 'fixed'):
                    records[-1].update(segment_roles=packed['roles'].tolist(),
                        identity_layout='primary_segments_only',
                        primary_rows=int((packed['supervision']==1).sum()),
                        auxiliary_actions=int((packed['supervision']==2).sum()))
            finally:
                close_arrays(arrays)
            if len(records) % 100 == 0:
                print('prepared decisions', split, len(records), flush=True)
        index['splits'][split] = records
    (cache/'index.json.partial').write_text(json.dumps(index, ensure_ascii=False))
    (cache/'index.json.partial').replace(cache/'index.json')
    return index


def selected_masks(a, prefix, rows):
    """Gather packed supervised masks without unpacking intervening tick rows."""
    out = np.zeros((len(rows), 576), dtype=bool)
    stored = a[prefix+'_rows']
    pos = np.searchsorted(stored, rows)
    hit = np.flatnonzero(pos < len(stored))
    hit = hit[np.asarray(stored[pos[hit]]) == rows[hit]]
    if len(hit):
        out[hit] = np.unpackbits(a[prefix+'_packed'][pos[hit]], axis=-1, bitorder='little')[:, :576]
    return out


def ragged_indices(offsets, rows):
    """Vectorized CSR gather locations, including empty rows and repeated queries."""
    starts = np.asarray(offsets[rows])
    counts = np.asarray(offsets[rows+1])-starts
    query = np.repeat(np.arange(len(rows)), counts)
    before = np.r_[0, np.cumsum(counts[:-1])]
    columns = np.arange(int(counts.sum()))-np.repeat(before, counts)
    return query, columns, np.repeat(starts, counts)+columns


def collate_decisions(items):
    """Allocate each padded batch tensor once instead of padding then stacking windows."""
    B = len(items)
    T = max(len(b['frame_ticks']) for b in items)
    N = max(b['entity_tokens'].shape[1] for b in items)
    E = max(1, max(len(b['event_ticks']) for b in items))
    output = {}
    for k, first in items[0].items():
        length = E if k in EVENT_FIELDS else T
        shape = ((B, length, N)+tuple(first.shape[2:]) if k in ENTITY_FIELDS
                 else (B, length)+tuple(first.shape[1:]))
        batch = torch.full(shape, -100 if k in LABELS else 0, dtype=first.dtype)
        for i, b in enumerate(items):
            v = b[k]
            if k in ENTITY_FIELDS:
                batch[i, :len(v), :v.shape[1]] = v
            else:
                batch[i, :len(v)] = v
        output[k] = batch
    return output


class DecisionWindows(Dataset):
    def __init__(self, root, cache, split, *, targets=32, frame_window=17,
                 event_window=1, max_open=2, max_delay=8, sampling=None, decision_period=None, history_length=0):
        self.history_length = history_length
        if history_length < 0: raise ValueError("negative history length")
        self.root, self.cache = Path(root).resolve(), Path(cache).resolve()
        self.index = json.loads((self.cache/'index.json').read_text())
        if (self.index.get('version') != 1 or self.index.get('decision_contract') not in CONTRACTS.values()
                or self.index.get('max_delay') != max_delay):
            raise ValueError('decision cache contract/max delay differs')
        if sampling is not None and sampling not in CONTRACTS:
            raise ValueError('unknown observation sampling')
        if sampling is not None and self.index.get('decision_contract') != CONTRACTS[sampling]:
            raise ValueError('requested observation sampling differs from cache')
        if decision_period is not None and self.index.get('decision_period') != decision_period:
            raise ValueError('fixed decision period differs from cache')
        if (self.index.get('decision_contract') in (INDEPENDENT_CONTRACT, FIXED_CONTRACT) and split==self.index['auxiliary_split']
                and frame_window>self.index['auxiliary_frame_window']):
            raise ValueError('auxiliary cache history is shorter than model history')
        if digest(self.root/'manifest.json') != self.index['manifest_sha256']:
            raise ValueError('source manifest changed')
        if min(targets, frame_window, max_open) < 1:
            raise ValueError('positive window sizes required')
        self.targets, self.frame_window, self.max_open = targets, frame_window, max_open
        self.records = self.index['splits'][split]
        self.opened = OrderedDict()
        self.layouts = {}  # small immutable NPY headers, not cached observations
        self.shard_paths, self.index_paths = [], []
        self.prefix, self.sequence_prefix = [0], []
        for r in self.records:
            shard_path = within(self.root, r['path'])
            index_path = within(self.cache, r['indices'])
            self.shard_paths.append(shard_path)
            self.index_paths.append(index_path)
            if digest(shard_path/'shard.json') != r['metadata_sha256']:
                raise ValueError('source shard metadata changed')
            if digest(index_path) != r['indices_sha256']:
                raise ValueError('decision indices changed')
            lengths = np.diff(r['offsets'])
            counts=(lengths+targets-1)//targets
            if 'segment_roles' in r:
                counts=np.where(np.asarray(r['segment_roles'])==1,1,counts)
            p = np.r_[0, np.cumsum(counts)]
            self.sequence_prefix.append(p); self.prefix.append(self.prefix[-1]+int(p[-1]))

    def __len__(self):
        return self.prefix[-1]

    def _open(self, i):
        if i not in self.opened:
            record = self.records[i]
            with np.load(self.index_paths[i], allow_pickle=False) as z:
                indices = {k: z[k].copy() for k in z.files}
            source = self._open_source(i)
            if self.history_length:
                from .history import HistoryIndex, CONTRACT as HISTORY_CONTRACT
                # Content-keyed sidecar survives the tiny source-shard LRU. Random
                # sampling otherwise rebuilds a whole shard for almost every window.
                identity = json.dumps([HISTORY_CONTRACT, self.index['manifest_sha256'],
                                       record['path'], record['metadata_sha256']])
                key = hashlib.sha256(identity.encode()).hexdigest()
                path = self.cache/'public-history-v1'/(key+'.npz')
                try:
                    if path.is_file():
                        source['_history'] = HistoryIndex.load(path)
                    else:
                        offsets = np.load(self.shard_paths[i]/'sequence_offsets.npy', allow_pickle=False)
                        source['_history'] = HistoryIndex(source, offsets)
                        source['_history'].save(path)
                except Exception:
                    close_arrays(source)
                    raise
            self.opened[i] = (source, indices)
            while len(self.opened) > self.max_open:
                _, (old, _) = self.opened.popitem(last=False)
                close_arrays(old)
        self.opened.move_to_end(i)
        return self.opened[i]

    def _open_source(self, i):
        # Random window sampling revisits shards after LRU eviction. Reuse their
        # checked NPY layouts rather than reparsing every array header on each visit.
        layout = self.layouts.get(i)
        arrays = {}
        try:
            for name in READER_FIELDS:
                path = self.shard_paths[i]/(name+'.npy')
                if layout is None:
                    arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
                else:
                    dtype, shape, offset, order = layout[name]
                    arrays[name] = np.memmap(path, mode='r', dtype=dtype, shape=shape,
                                             offset=offset, order=order)
            if layout is None:
                self.layouts[i] = {name: (a.dtype, a.shape, a.offset,
                    'F' if a.flags.f_contiguous else 'C') for name, a in arrays.items()}
            return arrays
        except Exception:
            close_arrays(arrays)
            raise

    def __getstate__(self):
        return dict(self.__dict__, opened=OrderedDict(), layouts={})

    def close(self):
        while self.opened:
            _, (a, _) = self.opened.popitem(); close_arrays(a)

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        sh = bisect_right(self.prefix, i)-1
        local = i-self.prefix[sh]
        p = self.sequence_prefix[sh]
        seq = int(np.searchsorted(p, local, side='right')-1)
        begin, end = self.records[sh]['offsets'][seq:seq+2]
        target = begin+(local-int(p[seq]))*self.targets
        start, stop = max(begin, target-self.frame_window+1), min(end, target+self.targets)
        roles=self.records[sh].get('segment_roles')
        if roles is not None and roles[seq]==1:
            target=end-1;start=max(begin,end-self.frame_window);stop=end
        a, d = self._open(sh)
        rows = d['rows'][start:stop]
        T = len(rows)
        # Fancy indexing owns its output; dtype conversion in NumPy avoids hundreds
        # of tiny torch.tensor copies per window.
        b = {k: torch.from_numpy(np.asarray(a[k][rows], dtype=np.int64)) for k in ROW_TOKENS+LABELS}
        b.update({k: torch.from_numpy(np.asarray(a[k][rows], dtype=bool)) for k in MASKS})
        for k in ('public_scalars', 'sample_weight'):
            b[k] = torch.from_numpy(np.asarray(a[k][rows], dtype=np.float32))
        b['frame_ticks'] = torch.from_numpy(d['ticks'][start:stop].astype(np.int64, copy=True))
        b['prev_elapsed_ticks'] = torch.from_numpy(d['elapsed'][start:stop].astype(np.float32, copy=True))
        b['delay_target_ticks'] = torch.from_numpy(d['delay'][start:stop].astype(np.int64, copy=True))
        b['delay_label_mask'] = torch.from_numpy(d['delay_mask'][start:stop].astype(bool, copy=True))
        b['frame_mask'] = torch.ones(T, dtype=torch.bool)
        b['loss_mask'] = torch.arange(T) >= target-start
        label_rows = rows
        if 'label_rows' in d:
            label_rows = d['label_rows'][start:stop]
            positive = label_rows >= 0
            safe_rows = np.maximum(label_rows, 0)
            for k in LABELS:
                b[k] = torch.from_numpy(np.where(positive, a[k][safe_rows], -100).astype(np.int64))
            for k in ('kind_label_mask', 'card_label_mask', 'position_label_mask',
                      'ability_label_mask', 'ability_position_label_mask'):
                b[k] = torch.from_numpy(np.asarray(a[k][safe_rows], dtype=bool) & positive)
            b['play_now'] = torch.from_numpy(positive)
            b['timing_label_mask'] = torch.from_numpy(d['timing_mask'][start:stop].copy())
        if 'supervision' in d:
            supervision=torch.from_numpy(d['supervision'][start:stop].copy())
            b['loss_mask'] &= supervision>0
            b['timing_label_mask'] &= supervision==1
        b['position_mask'] = torch.from_numpy(selected_masks(a, 'selected_position_mask', label_rows))
        b['ability_position_mask'] = torch.from_numpy(selected_masks(a, 'ability_position_mask', label_rows))
        grid = np.zeros((T, 8*576), dtype=np.float32)
        grid_rows, _, grid_indices = ragged_indices(a['grid_offsets'], rows)
        grid[grid_rows, a['grid_indices'][grid_indices]] = a['grid_values'][grid_indices]/255.0
        b['grid'] = torch.from_numpy(grid.reshape(T, 8, 32, 18))
        entity_rows, columns, source = ragged_indices(a['entity_offsets'], rows)
        N = max(1, int(columns.max())+1) if len(columns) else 1
        for k in ENTITY_FIELDS[:-1]:
            shape = (T, N, a[k].shape[-1]) if k == 'entity_numeric' else (T, N)
            values = np.zeros(shape, dtype=np.float32 if k == 'entity_numeric' else np.int64)
            values[entity_rows, columns] = a[k][source]
            b[k] = torch.from_numpy(values)
        entity_mask = np.zeros((T, N), dtype=bool)
        entity_mask[entity_rows, columns] = True
        b['entity_mask'] = torch.from_numpy(entity_mask)
        if self.history_length:
            b.update(a['_history'].query(rows, b['frame_ticks'].numpy(), self.history_length))
        # Shared collate expects event fields. HoKoff never consumes event history.
        for k in EVENT_FIELDS:
            b[k] = torch.empty(0, dtype=torch.bool if k == 'event_mask' else torch.long)
        return b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--max-delay', type=int, default=8)
    p.add_argument('--sampling', choices=['legacy','independent','fixed'], default='legacy')
    p.add_argument('--decision-period', type=int, default=4)
    p.add_argument('--sampling-seed', type=int, default=42)
    p.add_argument('--auxiliary-split', default='validation')
    p.add_argument('--auxiliary-frame-window', type=int, default=17)
    p.add_argument('--splits', nargs='+', default=['validation', 'train'])
    p.add_argument('--max-shards-per-split', type=int)
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    index = prepare(args.data, args.cache, max_delay=args.max_delay, splits=args.splits,
                    max_shards_per_split=args.max_shards_per_split,
                    verify_hashes=args.verify_hashes, allow_smoke=args.allow_smoke,
                    sampling=args.sampling,sampling_seed=args.sampling_seed,
                    auxiliary_split=args.auxiliary_split,auxiliary_frame_window=args.auxiliary_frame_window,
                    decision_period=args.decision_period)
    print(json.dumps({s: {k: sum(r[k] for r in rr) for k in ('source_rows', 'valid_rows', 'decision_rows', 'actions')}
                      for s, rr in index['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
