"""Read only the small compiled-label columns; no replay or libg regeneration."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np


def inspect_shard(path):
    labels = np.load(path/'ability_label_mask.npy', mmap_mode='r', allow_pickle=False)
    indices = np.flatnonzero(labels)
    result = Counter(labels=len(indices))
    if not len(indices): return result
    legal = np.load(path/'ability_mask.npy', mmap_mode='r', allow_pickle=False)[indices].astype(bool)
    target = np.load(path/'ability_slot.npy', mmap_mode='r', allow_pickle=False)[indices].astype(int)
    weight = np.load(path/'sample_weight.npy', mmap_mode='r', allow_pickle=False)[indices]
    counts = legal.sum(axis=1)
    result['positive_weight_labels'] = int((weight > 0).sum())
    result['zero_weight_labels'] = int((weight == 0).sum())
    result['nonfinite_weights'] = int((~np.isfinite(weight)).sum())
    result['no_legal_candidate'] = int((counts == 0).sum())
    result['single_legal_candidate'] = int((counts == 1).sum())
    result['multiple_legal_candidates'] = int((counts > 1).sum())
    valid = (target >= 0) & (target < legal.shape[1])
    target_legal = np.zeros(len(target), dtype=bool)
    target_legal[valid] = legal[np.flatnonzero(valid), target[valid]]
    result['illegal_or_missing_target'] = int((~target_legal).sum())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    manifest = json.loads((a.dataset_root/'manifest.json').read_text())
    shards = [a.dataset_root/x for x in manifest['splits'][a.split]]
    total = Counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(inspect_shard, shards): total.update(result)
    value = {'split': a.split, 'shards': len(shards), 'counts': dict(total),
             'scope': 'compiled raw supervised ability rows; no model inference or training mutation'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
