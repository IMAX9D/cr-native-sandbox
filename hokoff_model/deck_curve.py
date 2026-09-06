"""Fixed-pool learning curves for mixed versus exact own-deck BC."""
from collections import Counter
from dataclasses import asdict
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from policy_v1.data import Windows, collate
from policy_v1.train import seed_all, optimizer_update
from .experiments import parser as base_parser, initialize
from .model import Policy, config_from_args
from .metrics import bc_loss, summarize
from .evaluate_timing import probability_report


def catalog(dataset, limit, seed):
    """Scan first-row deck tokens; at most one selected actor per battle later."""
    rng = random.Random(seed)
    shards = list(range(len(dataset.records)))
    rng.shuffle(shards)
    rows = []
    scanned = 0
    for sh in shards:
        record = dataset.records[sh]
        arrays, _, _ = dataset._open(sh)
        battles = list(range(len(record['battle_tags'])))
        rng.shuffle(battles)
        for battle in battles:
            scanned += 1
            for seq in (2 * battle, 2 * battle + 1):
                start, stop = record['offsets'][seq:seq + 2]
                if stop <= start:
                    continue
                deck = tuple(sorted(int(x) for x in arrays['own_deck_tokens'][start]))
                # Unknown or incomplete decks are not an exact-deck experiment.
                if len(deck) != 8 or min(deck) <= 0 or len(set(deck)) != 8:
                    continue
                rows.append(dict(shard=sh, sequence=seq, battle=str(record['battle_tags'][battle]),
                                 deck=deck, frames=int(stop-start)))
            if scanned >= limit:
                return rows
        print(json.dumps({'phase': 'deck_scan', 'shards_scanned_through': sh,
                          'actors': len(rows)}), flush=True)
    return rows


def unique_battles(rows, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    seen = set()
    result = []
    for row in shuffled:
        if row['battle'] not in seen:
            result.append(row)
            seen.add(row['battle'])
    return result


def select_pools(train, validation, ntrain, nval, seed):
    if {r['battle'] for r in train} & {r['battle'] for r in validation}:
        raise ValueError('battle overlap across source splits')
    counts = Counter(deck for deck, battle in {(r['deck'], r['battle']) for r in train})
    # Choose deck ONLY from training frequency; validation cannot choose a winner.
    deck = min(counts, key=lambda d: (-counts[d], d)) if counts else None
    fixed_train = unique_battles([r for r in train if r['deck'] == deck], seed)
    fixed_val = unique_battles([r for r in validation if r['deck'] == deck], seed+1)
    if len(fixed_train) < ntrain or len(fixed_val) < nval:
        raise ValueError('most frequent training deck has only %d train / %d validation battles; '
                         'increase --scan-battles or reduce --train-battles / --val-battles; see selection-audit.json'
                         % (len(fixed_train), len(fixed_val)))
    mixed_train = unique_battles(train, seed)
    mixed_val = unique_battles(validation, seed+1)
    return deck, {'mixed': (mixed_train[:ntrain], mixed_val[:nval]),
                  'fixed_deck': (fixed_train[:ntrain], fixed_val[:nval])}


def indices_for(dataset, rows):
    indices = []
    for row in rows:
        sh, seq = row['shard'], row['sequence']
        start = dataset.prefix[sh] + int(dataset.sequence_prefix[sh][seq])
        stop = dataset.prefix[sh] + int(dataset.sequence_prefix[sh][seq+1])
        indices.extend(range(start, stop))
    return indices


def measure(model, dataset, indices, args, device):
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        num_workers=0, collate_fn=collate)
    probabilities, labels, totals = [], [], {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(b)
            _, stats = bc_loss(out, b, timing_positive_weight=args.positive_weight)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            valid = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
            probabilities.append(out['timing'][valid].sigmoid().cpu().numpy())
            labels.append(b['play_now'][valid].cpu().numpy())
    report = probability_report(np.concatenate(probabilities), np.concatenate(labels))
    keys = ('valid_frames', 'actual_actions', 'actual_action_rate', 'average_precision',
            'constant_score_ap_baseline', 'ap_lift_over_prevalence', 'roc_auc')
    return {**summarize(totals), **{k: report[k] for k in keys}}


def parser():
    p = base_parser('compare')
    p.description = __doc__
    p.set_defaults(steps=5000)
    p.add_argument('--train-battles', type=int, default=32)
    p.add_argument('--val-battles', type=int, default=16)
    p.add_argument('--scan-battles', type=int, default=8192)
    p.add_argument('--eval-windows', type=int, default=2048)
    p.add_argument('--eval-every', type=int, default=500)
    return p


def run(args):
    if min(args.train_battles, args.val_battles, args.scan_battles, args.eval_windows,
           args.eval_every, args.batch_size) < 1 or args.workers < 0:
        raise ValueError('invalid experiment size')
    root = initialize(args, 'deck-curve')
    datasets = [Windows(args.data, args.cache, split, targets=args.targets,
                        frame_window=args.frame_window, event_window=1)
                for split in (args.train_split, args.val_split)]
    if datasets[0].index['smoke_only'] and not args.allow_smoke:
        raise ValueError('synthetic requires --allow-smoke')
    catalogs = [catalog(ds, args.scan_battles, args.seed+i) for i, ds in enumerate(datasets)]
    audit = {'manifest_sha256': datasets[0].index['manifest_sha256'],
             'train': catalogs[0], 'validation': catalogs[1]}
    (root/'selection-audit.json').write_text(json.dumps(audit, indent=2))
    deck, pools = select_pools(*catalogs, args.train_battles, args.val_battles, args.seed)
    (root/'selected-pools.json').write_text(json.dumps({'deck_tokens': deck, 'pools': pools}, indent=2))
    fixed_val = indices_for(datasets[1], pools['fixed_deck'][1])
    rng = random.Random(123)
    fixed_eval = sorted(rng.sample(fixed_val, min(len(fixed_val), args.eval_windows)))
    device = torch.device(args.device)
    summaries = []
    for arm, (training, validation) in pools.items():
        seed_all(args.seed)
        train_indices = indices_for(datasets[0], training)
        val_indices = indices_for(datasets[1], validation)
        rng = random.Random(123)
        eval_train = sorted(rng.sample(train_indices, min(len(train_indices), args.eval_windows)))
        eval_val = sorted(rng.sample(val_indices, min(len(val_indices), args.eval_windows)))
        arm_dir = root/arm
        arm_dir.mkdir()
        (arm_dir/'evaluation-windows.json').write_text(json.dumps(
            {'training': eval_train, 'validation': eval_val, 'common_fixed_validation': fixed_eval}))
        config = config_from_args(args, datasets[0].index['dimensions'])
        model = Policy(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        loader = DataLoader(Subset(datasets[0], train_indices), batch_size=args.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(args.seed),
                            num_workers=args.workers, collate_fn=collate,
                            persistent_workers=args.workers > 0)
        iterator = iter(loader)
        print(json.dumps({'phase': 'deck_curve_start', 'arm': arm, 'deck_tokens': deck,
                          'training_windows': len(train_indices), 'validation_windows': len(val_indices),
                          'training_decks': len({r['deck'] for r in training}),
                          'validation_decks': len({r['deck'] for r in validation}),
                          'precision': 'fp32', 'positive_weight': args.positive_weight,
                          'run_dir': str(arm_dir)}), flush=True)
        for step in range(args.steps+1):
            if step:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                model.train()
                b = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, _ = bc_loss(model(b), b, timing_positive_weight=args.positive_weight)
                loss.backward()
                optimizer_update(model, optimizer, scaler, 1.0)
            if step % args.eval_every == 0 or step == args.steps:
                result = {'phase': 'deck_learning_curve', 'arm': arm, 'step': step,
                          'train': measure(model, datasets[0], eval_train, args, device),
                          'validation': measure(model, datasets[1], eval_val, args, device),
                          'common_fixed_validation': measure(model, datasets[1], fixed_eval, args, device)}
                print(json.dumps(result), flush=True)
                with (arm_dir/'curve.jsonl').open('a') as f:
                    f.write(json.dumps(result)+'\n')
                torch.save({'model': model.state_dict(), 'config': asdict(config), 'step': step,
                            'purpose': 'fixed-pool diagnostic; not resumable training checkpoint'},
                           arm_dir/'diagnostic.pt')
            elif step % 100 == 0:
                print(json.dumps({'phase': 'deck_curve_progress', 'arm': arm, 'step': step}), flush=True)
        summaries.append(result)
        del iterator, loader, optimizer, model
    result = {'phase': 'deck_curve_summary', 'run_dir': str(root), 'arms': summaries,
              'note': 'Own deck only; opponents and card levels remain diverse. Equal battles and updates, '
                      'not necessarily equal frames. Compare common_fixed_validation across arms; '
                      'curves diagnose memorization/generalization, not causal proof of dirty data.'}
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
