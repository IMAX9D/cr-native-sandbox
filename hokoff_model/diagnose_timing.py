"""Read-only availability strata and one-to-one timing event diagnostics."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from policy_v1.data import Windows, collate
from policy_v1.train import load_checkpoint, move
from .model import Config, Policy
from .evaluate_timing import probability_report, THRESHOLDS


TOLERANCES = (("exact", 0, 0), ("plus_minus_250ms", 5, 5),
              ("plus_minus_500ms", 10, 10),
              ("early_only_250ms", 5, 0), ("early_only_500ms", 10, 0))
MASK_NOTE = (
    "Positive masks confirm at least one available action. All-zero masks do NOT prove forced WAIT: "
    "dynamic-choice exact masks may be unavailable outside supervised deployment ticks. "
    "Mask-conditioned results may have selection bias; do not use them alone to change training."
)


def match_events(predicted, actual, early, late):
    """Maximum-count ordered matching for equal tolerance intervals, one-to-one.

    Offset = prediction tick - expert tick. Not a minimum-distance assignment.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    i = j = 0
    offsets = []
    while i < len(predicted) and j < len(actual):
        if predicted[i] < actual[j] - early:
            i += 1
        elif predicted[i] > actual[j] + late:
            j += 1
        else:
            offsets.append(int(predicted[i] - actual[j]))
            i += 1
            j += 1
    return len(offsets), len(predicted)-len(offsets), len(actual)-len(offsets), offsets


def segments(record):
    """Keep actor boundaries and masked/gapped intervals separate."""
    ticks = np.asarray(record['ticks'])
    valid = np.asarray(record['valid'], dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(ticks[indices]) != 1)) + 1
    for ix in np.split(indices, cuts):
        yield ticks[ix], np.asarray(record['probabilities'])[ix], np.asarray(record['labels'], dtype=bool)[ix]


def event_metrics(records, threshold, early, late, detector):
    tp = fp = fn = boundary_actions = total_segments = 0
    offsets = []
    for record in records:
        for ticks, probabilities, labels in segments(record):
            total_segments += 1
            above = probabilities > threshold
            if detector == 'rising_edge':
                # Causal: one alarm when crossing threshold. A plateau is one
                # alarm, never a post-hoc maximum chosen with future scores.
                alarms = above & np.r_[True, ~above[:-1]]
            elif detector == 'every_frame':
                alarms = above
            else:
                raise ValueError('unknown detector')
            actual = ticks[labels]
            a, b, c, delta = match_events(ticks[alarms], actual, early, late)
            tp += a; fp += b; fn += c; offsets.extend(delta)
            boundary_actions += int(((actual-early < ticks[0]) | (actual+late > ticks[-1])).sum())
    predicted = tp + fp
    actual = tp + fn
    return {
        'threshold': float(threshold), 'detector': detector,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp/predicted if predicted else None,
        'recall': tp/actual if actual else None,
        'f1': 2*tp/(2*tp+fp+fn) if actual else None,
        'predicted_events': predicted, 'actual_events': actual,
        'boundary_limited_events': boundary_actions, 'valid_segments': total_segments,
        'median_offset_ms': float(np.median(offsets)*50) if offsets else None,
        'matched_after_expert': int(np.count_nonzero(np.asarray(offsets)>0)),
    }


def strata_report(records):
    p = np.concatenate([r['probabilities'] for r in records])
    y = np.concatenate([r['labels'] for r in records]).astype(bool)
    valid = np.concatenate([r['valid'] for r in records]).astype(bool)
    available = np.concatenate([r['available'] for r in records]).astype(bool)
    cards = np.concatenate([r['card_available'] for r in records]).astype(bool)
    elixir = np.concatenate([r['elixir'] for r in records])
    selectors = {
        'all_valid': valid,
        'mask_confirms_any_action': valid & available,
        'no_action_confirmed_by_mask': valid & ~available,
        'mask_confirms_card_play': valid & cards,
    }
    out = {name: probability_report(p[mask],y[mask]) if mask.any() else None
           for name,mask in selectors.items()}
    # This second stratification uses a public scalar, not action-label-dependent
    # mask coverage. Elixir ratio is NOT the exact affordability of a given card.
    out['elixir_strata'] = {}
    for low,high in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.00001)):
        mask = valid & (elixir>=low) & (elixir<high)
        out['elixir_strata']['%.1f_to_%.1f' % (low,min(high,1.))] = (
            probability_report(p[mask],y[mask]) if mask.any() else None)
    out['action_frames_without_confirmed_mask'] = int((valid & y & ~available).sum())
    out['mask_caveat'] = MASK_NOTE
    return out


def shifted_control(records, seed=9187):
    """Preserve score distribution and most temporal structure, break alignment.

    A single circular shift per valid segment is a diagnostic null, not a
    significance test. Very short segments cannot be shifted beyond tolerance.
    """
    rng = random.Random(seed)
    result = []
    for record in records:
        control = dict(record)
        scores = np.asarray(record['probabilities']).copy()
        ticks = np.asarray(record['ticks'])
        indices = np.flatnonzero(record['valid'])
        if len(indices):
            cuts = np.flatnonzero((np.diff(indices)!=1) | (np.diff(ticks[indices])!=1))+1
            for ix in np.split(indices,cuts):
                n = len(ix)
                shift = rng.randrange(11,n-10) if n>21 else n//2
                scores[ix] = np.roll(scores[ix],shift)
        control['probabilities'] = scores
        result.append(control)
    return result


def analyze(records):
    if not records:
        raise ValueError('no sequences')
    strata = strata_report(records)
    if strata['all_valid'] is None:
        raise ValueError('no valid timing frames')
    probabilities = np.concatenate([r['probabilities'][r['valid']] for r in records])
    thresholds = sorted(set(THRESHOLDS) | set(np.quantile(
        probabilities,[.1,.25,.5,.75,.9,.95,.99,.994,.995,.999]).tolist()))
    tables = {}; best = {}
    control_records = shifted_control(records)
    for name,early,late in TOLERANCES:
        rows = [event_metrics(records,t,early,late,detector)
                for detector in ('every_frame','rising_edge') for t in thresholds]
        tables[name] = rows
        best[name] = {}
        for detector in ('every_frame','rising_edge'):
            candidates = [row for row in rows if row['detector']==detector]
            best[name][detector] = dict(max(candidates,key=lambda row: (
                row['f1'] if row['f1'] is not None else -1,
                row['precision'] if row['precision'] is not None else -1)))
            controls = [event_metrics(control_records,t,early,late,detector) for t in thresholds]
            control_best = max(controls,key=lambda row: row['f1'] if row['f1'] is not None else -1)
            best[name][detector]['shift_control_best_f1'] = control_best['f1']
            best[name][detector]['shift_control_best_threshold'] = control_best['threshold']
    return {'strata':strata, 'event_tables':tables, 'best_event_f1_on_this_validation_sample':best,
            'matching':'ordered one-to-one within each contiguous valid actor segment',
            'interpretation':'Best thresholds are descriptive validation selections, not deployment settings. '
             'Symmetric tolerance credits late reactions; early-only excludes them. '
             'Edges have less observed context and are counted in boundary_limited_events. '
             'Shift-control scores are circularly shifted within valid segments (seed 9187); '
             'its F1 is maximized on the same threshold grid. This single control is not a significance test.'}


def sequence_plan(dataset, count, seed):
    pairs = [(sh,seq) for sh,r in enumerate(dataset.records) for seq in range(len(r['offsets'])-1)]
    chosen = random.Random(seed).sample(pairs,min(count,len(pairs)))
    indices, owners = [], []
    for owner,(sh,seq) in enumerate(chosen):
        start = dataset.prefix[sh] + int(dataset.sequence_prefix[sh][seq])
        end = dataset.prefix[sh] + int(dataset.sequence_prefix[sh][seq+1])
        indices.extend(range(start,end)); owners.extend([owner]*(end-start))
    return chosen,indices,owners


def collect(model,dataset,chosen,indices,owners,args,device):
    records = [{k:[] for k in ('ticks','probabilities','labels','valid','available','card_available','elixir')}
               for _ in chosen]
    loader = DataLoader(dataset,sampler=indices,batch_size=args.batch_size,num_workers=args.workers,
                        collate_fn=collate,pin_memory=device.type=='cuda',
                        generator=torch.Generator().manual_seed(args.seed))
    cursor=0
    with torch.no_grad():
        for bi,batch in enumerate(loader,1):
            b=move(batch,device)
            logits=model(b)['timing'].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite timing logits')
            p=logits.sigmoid().cpu().numpy()
            for j in range(len(p)):
                target=(batch['frame_mask'][j] & batch['loss_mask'][j]).numpy()
                record=records[owners[cursor]]
                values={
                    'ticks':batch['frame_ticks'][j].numpy(), 'probabilities':p[j],
                    'labels':batch['play_now'][j].numpy(),
                    'valid':batch['timing_label_mask'][j].numpy(),
                    'available':(batch['card_mask'][j].any(-1) | batch['ability_mask'][j].any(-1)).numpy(),
                    'card_available':batch['card_mask'][j].any(-1).numpy(),
                    'elixir':batch['public_scalars'][j,:,1].numpy(),
                }
                for key,value in values.items():record[key].append(value[target])
                cursor+=1
            if bi%25==0:
                print(json.dumps({'phase':'timing_context_progress','batches':bi,
                                  'total_windows':len(indices)}),flush=True)
    if cursor != len(indices):raise ValueError('missing windows')
    for record,(sh,seq) in zip(records,chosen):
        for key in record:record[key]=np.concatenate(record[key])
        expected=dataset.records[sh]['offsets'][seq+1]-dataset.records[sh]['offsets'][seq]
        if len(record['ticks'])!=expected or np.any(np.diff(record['ticks'])!=1):
            raise ValueError('duplicate, missing or reordered sequence frames')
    return records


def compact_stratum(value):
    if value is None:return None
    return {k:value[k] for k in ('valid_frames','actual_actions','actual_action_rate',
                               'average_precision','ap_lift_over_prevalence','roc_auc')}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dir',type=Path)
    p.add_argument('--checkpoint',type=Path,help='optional single-model diagnostic')
    p.add_argument('--runs',type=Path,default=Path('/root/autodl-tmp/runs'))
    p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/expert-dataset/native-bc-v1'))
    p.add_argument('--cache',type=Path,default=Path('/root/autodl-tmp/policy-v1-cache'))
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--sequences',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--output',type=Path)
    p.add_argument('--allow-smoke',action='store_true')
    return p


def run(args):
    if min(args.sequences,args.batch_size,args.cpu_threads)<1 or args.workers<0:
        raise ValueError('invalid counts')
    if args.checkpoint and args.comparison_dir:raise ValueError('choose checkpoint OR comparison directory')
    if args.checkpoint:
        checkpoints=[('single',args.checkpoint)]
        root=args.checkpoint.parent
    else:
        root=args.comparison_dir
        if root is None:
            dirs=[d for d in args.runs.glob('hokoff-weight-compare-*')
                  if (d/'baseline/last.pt').is_file() and (d/'weighted/last.pt').is_file()]
            if not dirs:raise FileNotFoundError('No completed weight comparison found; pass --comparison-dir')
            root=max(dirs,key=lambda d:(d/'weighted/last.pt').stat().st_mtime_ns)
        checkpoints=[(arm,root/arm/'last.pt') for arm in ('baseline','weighted')]
    output=args.output or root/('timing-context-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
    if output.exists():raise FileExistsError(output)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.cpu_threads)
    arms=[]; reports={}; reference=None
    for arm,path in checkpoints:
        saved=load_checkpoint(path)
        if saved['config'].get('architecture')!='hokoff_cr_lstm_v1':raise ValueError('wrong architecture')
        contract=saved['contract']; split=contract['val_split']
        if split==contract['train_split'] or split=='test':raise ValueError('requires non-test held-out split')
        signature=(saved['config'],{k:v for k,v in contract.items() if k!='timing_positive_weight'},saved['step'])
        if reference is not None and signature!=reference:raise ValueError('comparison contracts/steps differ')
        reference=signature
        config=Config(**saved['config'])
        dataset=Windows(args.data,args.cache,split,targets=contract['targets'],
                        frame_window=config.frame_window,event_window=1)
        if dataset.index['manifest_sha256']!=contract['manifest_sha256']:raise ValueError('manifest differs')
        if dataset.index['smoke_only'] and not args.allow_smoke:raise ValueError('synthetic requires --allow-smoke')
        chosen,indices,owners=sequence_plan(dataset,args.sequences,args.seed)
        if not chosen:raise ValueError('empty split')
        print(json.dumps({'phase':'timing_context_start','arm':arm,'checkpoint':str(path),
                          'step':saved['step'],'split':split,'sequences':len(chosen),
                          'windows':len(indices),'precision':'fp32'}),flush=True)
        model=Policy(config).to(device).eval();model.load_state_dict(saved['model'])
        records=collect(model,dataset,chosen,indices,owners,args,device)
        report=analyze(records)
        report.update(checkpoint=str(path),checkpoint_step=saved['step'],split=split,
                      seed=args.seed,sequences=len(chosen),precision='fp32',
                      selection=[{'shard':dataset.records[sh]['path'],'sequence_index':seq} for sh,seq in chosen],
                      context='same finite-history model.forward as training; unique target frames stitched by actor')
        reports[arm]=report
        strata=report['strata']
        summary={'arm':arm,'sequences':len(chosen),
                 'all_valid':compact_stratum(strata['all_valid']),
                 'mask_confirms_any_action':compact_stratum(strata['mask_confirms_any_action']),
                 'no_action_confirmed_by_mask':compact_stratum(strata['no_action_confirmed_by_mask']),
                 'action_frames_without_confirmed_mask':strata['action_frames_without_confirmed_mask'],
                 'best_event_f1_on_validation':report['best_event_f1_on_this_validation_sample']}
        arms.append(summary)
        print(json.dumps({'phase':'timing_context_arm',**summary}),flush=True)
        del model,records,dataset,saved
    result={'phase':'timing_context_summary','report_path':str(output),'arms':arms,'mask_caveat':MASK_NOTE}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump({'summary':result,'reports':reports},f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
    return result


def main():
    run(parser().parse_args())


if __name__=='__main__':main()
