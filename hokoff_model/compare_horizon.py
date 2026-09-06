"""Matched point-vs-500ms forecast experiment, with shared held-out targets."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from policy_v1.data import Windows
from policy_v1.train import load_checkpoint
from .model import Config, Policy
from .train import parser as train_parser, run as train
from .experiments import parser as experiment_parser
from .horizon import future_targets
from .evaluate_timing import probability_report, THRESHOLDS
from .diagnose_timing import collect, sequence_plan, segments, shifted_control, compact_stratum


def lead_events(records, threshold, lead, common_horizon, detector):
    """Censor incomplete future; use edge references as ignored matching context.

    Predictions need fully observed [t,t+common_horizon]. Scored references need
    common_horizon context on both sides. Alarms matched to unscored edge
    references are ignored, not incorrectly declared false positives.
    """
    tp=fp=fn=ignored=eligible_frames=0
    for record in records:
        for ticks,p,y in segments(record):
            if len(ticks) <= 2*common_horizon:
                continue
            actual=ticks[y]
            core=(actual>=ticks[0]+common_horizon) & (actual<=ticks[-1]-common_horizon)
            available=ticks<=ticks[-1]-common_horizon
            above=p>threshold
            if detector=='rising_edge':
                alarms=above & np.r_[True,~above[:-1]]
            elif detector=='every_frame':
                alarms=above
            else:
                raise ValueError('unknown detector')
            predictions=ticks[alarms & available]
            eligible_frames+=int(available.sum())
            i=j=matched=halo=0
            while i<len(predictions) and j<len(actual):
                if predictions[i]<actual[j]-lead:
                    i+=1
                elif predictions[i]>actual[j]:
                    j+=1
                else:
                    if core[j]:matched+=1
                    else:halo+=1
                    i+=1;j+=1
            tp+=matched;fp+=len(predictions)-matched-halo;fn+=int(core.sum())-matched;ignored+=halo
    return {'threshold':float(threshold),'detector':detector,'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if tp+fn else None,
            'scored_actual_events':tp+fn,'ignored_edge_matches':ignored,
            'eligible_prediction_frames':eligible_frames}


def analyze(records, horizon):
    ps=[]; now=[]; soon=[]; excluded=valid_total=0
    for record in records:
        target,known=future_targets(record['labels'],record['valid'],horizon)
        # collect() guarantees consecutive physical ticks; invalid spans are
        # rejected by future_targets, not bridged by filtering rows first.
        ps.append(record['probabilities'][known]);now.append(record['labels'][known]);soon.append(target[known])
        valid_total+=int(record['valid'].sum());excluded+=int((record['valid'] & ~known).sum())
    scores=np.concatenate(ps)
    if not len(scores):raise ValueError('no complete future intervals')
    point=probability_report(scores,np.concatenate(now))
    forecast=probability_report(scores,np.concatenate(soon))
    thresholds=sorted(set(THRESHOLDS) | set(np.quantile(scores,[.25,.5,.75,.9,.95,.99,.994,.999]).tolist()))
    controls=shifted_control(records)
    events={};tables={}
    for name,lead in [('exact',0),('early_only_500ms',horizon)]:
        events[name]={};tables[name]={}
        for detector in ('every_frame','rising_edge'):
            rows=[lead_events(records,t,lead,horizon,detector) for t in thresholds]
            shifted=[lead_events(controls,t,lead,horizon,detector) for t in thresholds]
            best=dict(max(rows,key=lambda r:r['f1'] if r['f1'] is not None else -1))
            null=max(shifted,key=lambda r:r['f1'] if r['f1'] is not None else -1)
            best['shift_control_best_f1']=null['f1']
            events[name][detector]=best;tables[name][detector]=rows
    return {'point_target':point,'within_500ms_target':forecast,'best_event_f1_on_validation':events,
            'event_tables':tables,'valid_source_frames':valid_total,'excluded_incomplete_future_frames':excluded,
            'notes':'Both models scored on the same frames and both label definitions. '
             'Do not compare AP across label definitions. Event matching is one-to-one and early-only, '
             'with common boundary censoring and ignored edge-reference matches. '
             'Best thresholds and a single time-shift control are diagnostics, not deployment settings or significance tests.'}


def evaluate(checkpoint,args,root,arm):
    saved=load_checkpoint(checkpoint);contract=saved['contract'];config=Config(**saved['config'])
    if config.architecture!='hokoff_cr_lstm_v1':raise ValueError('wrong architecture')
    split=contract['val_split']
    if split==contract['train_split'] or split=='test':raise ValueError('requires held-out non-test split')
    dataset=Windows(args.data,args.cache,split,targets=contract['targets'],frame_window=config.frame_window,event_window=1)
    if dataset.index['manifest_sha256']!=contract['manifest_sha256']:raise ValueError('manifest differs')
    if dataset.index['smoke_only'] and not args.allow_smoke:raise ValueError('synthetic requires --allow-smoke')
    chosen,indices,owners=sequence_plan(dataset,args.sequences,123)
    if not chosen:raise ValueError('empty split')
    device=torch.device(args.device)
    model=Policy(config).to(device).eval();model.load_state_dict(saved['model'])
    print(json.dumps({'phase':'horizon_eval_start','arm':arm,'sequences':len(chosen),'windows':len(indices)}),flush=True)
    records=collect(model,dataset,chosen,indices,owners,args,device)
    report=analyze(records,10)
    report.update(checkpoint=str(checkpoint),checkpoint_step=saved['step'],timing_horizon_ticks=contract.get('timing_horizon_ticks',0),
                  split=split,seed=123,sequences=len(chosen),
                  selection=[{'shard':dataset.records[sh]['path'],'sequence_index':seq} for sh,seq in chosen])
    with (root/arm/'horizon-eval.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report


def parser():
    p=experiment_parser('compare')
    p.description=__doc__
    # The comparison changes horizon only. Weighting stays at one in both arms.
    p.set_defaults(positive_weight=1.0)
    p.add_argument('--sequences',type=int,default=128)
    return p


def run(args):
    if args.positive_weight != 1.0:
        raise ValueError('this experiment fixes positive weight at 1 in both arms')
    if min(args.steps,args.batch_size,args.sequences,args.width,args.hidden_size,args.cpu_threads,args.targets,args.frame_window)<1 or args.workers<0:
        raise ValueError('invalid arguments')
    if args.train_split==args.val_split or 'test' in (args.train_split,args.val_split):
        raise ValueError('use distinct non-test training/validation splits')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.cpu_threads)
    root=args.run_dir or Path('/root/autodl-tmp/runs')/('hokoff-horizon-compare-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    root.mkdir(parents=True,exist_ok=False)
    (root/'experiment.json').write_text(json.dumps(vars(args),default=str,indent=2))
    results=[];reference=None
    for arm,horizon in [('point',0),('forecast_500ms',10)]:
        options={'data':args.data,'cache':args.cache,'run-dir':root/arm,'device':args.device,'precision':'fp32',
                 'train-split':args.train_split,'val-split':args.val_split,'width':args.width,'hidden-size':args.hidden_size,
                 'frame-window':args.frame_window,'targets':args.targets,'batch-size':args.batch_size,
                 'workers':args.workers,'cpu-threads':args.cpu_threads,'seed':args.seed,'max-steps':args.steps,
                 'epochs':args.steps,'log-every':100,'save-every':500,'eval-batches':min(args.eval_batches,100),
                 'timing-positive-weight':1,'timing-horizon-ticks':horizon,'timing-mask-horizon-ticks':10}
        argv=[part for k,v in options.items() for part in ('--'+k,str(v))]
        if args.allow_smoke:argv.append('--allow-smoke')
        print(json.dumps({'phase':'horizon_arm_start','arm':arm,'horizon_ticks':horizon,'common_mask_horizon':10,
                          'positive_weight':1,'steps':args.steps,'run_dir':str(root/arm)}),flush=True)
        train(train_parser().parse_args(argv))
        report=evaluate(root/arm/'last.pt',args,root,arm)
        signature=(report['checkpoint_step'],report['selection'],report['point_target']['valid_frames'],
                   report['point_target']['actual_actions'],report['within_500ms_target']['actual_actions'])
        if reference is not None and signature!=reference:raise ValueError('comparison sample/step mismatch')
        reference=signature
        summary={'arm':arm,'checkpoint_step':report['checkpoint_step'],'point_target':compact_stratum(report['point_target']),
                 'within_500ms_target':compact_stratum(report['within_500ms_target']),
                 'excluded_incomplete_future_frames':report['excluded_incomplete_future_frames'],
                 'best_event_f1_on_validation':report['best_event_f1_on_validation']}
        results.append(summary)
        print(json.dumps({'phase':'horizon_arm_summary',**summary}),flush=True)
    result={'phase':'horizon_comparison_summary','run_dir':str(root),'arms':results,
            'note':'Compare arms under the SAME target. A higher forecast positive rate alone is not improvement. '
             'Forecast output means an action within [now, now+0.5s], not execute immediately.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


def main():
    run(parser().parse_args())


if __name__=='__main__':main()
