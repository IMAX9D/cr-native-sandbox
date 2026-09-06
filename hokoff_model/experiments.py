"""Small diagnostic experiments; executed on the user's training machine."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from policy_v1.data import Windows, collate
from policy_v1.train import seed_all, optimizer_update
from .model import Policy, config_from_args
from .metrics import bc_loss, summarize
from .evaluate_timing import probability_report, run as evaluate, parser as eval_parser
from .train import run as train, parser as train_parser


DATA = "/root/autodl-tmp/expert-dataset/native-bc-v1"
CACHE = "/root/autodl-tmp/policy-v1-cache"
RUNS = Path("/root/autodl-tmp/runs")


def parser(mode):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(DATA))
    p.add_argument('--cache', type=Path, default=Path(CACHE))
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--train-split', default='validation')
    p.add_argument('--val-split', default='train')
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--hidden-size', type=int, default=512)
    p.add_argument('--frame-window', type=int, default=128)
    p.add_argument('--targets', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=1000 if mode=='overfit' else 2000)
    p.add_argument('--positive-weight', type=float, default=32.0)
    p.add_argument('--allow-smoke', action='store_true')
    if mode == 'overfit':
        p.add_argument('--positive-windows', type=int, default=8)
        p.add_argument('--wait-windows', type=int, default=8)
        p.add_argument('--scan-windows', type=int, default=4096)
        p.add_argument('--log-every', type=int, default=100)
    else:
        p.add_argument('--batch-size', type=int, default=32)
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--eval-batches', type=int, default=200)
    return p


def initialize(args, mode):
    if min(args.steps,args.width,args.hidden_size,args.frame_window,args.targets,args.cpu_threads) < 1:
        raise ValueError('positive dimensions/steps required')
    if not math.isfinite(args.positive_weight) or args.positive_weight <= 1:
        raise ValueError('comparison positive weight must exceed 1 and be finite')
    if args.train_split == 'test' or args.val_split == 'test' or args.train_split == args.val_split:
        raise ValueError('use distinct training/validation splits; never test')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only for local checks')
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    root = args.run_dir or RUNS / ('hokoff-' + mode + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    root.mkdir(parents=True, exist_ok=False)
    (root/'experiment.json').write_text(json.dumps(vars(args),default=str,indent=2))
    return root


def select_windows(dataset, positive_windows, wait_windows, scan_windows, seed):
    if min(positive_windows,wait_windows,scan_windows) < 1:
        raise ValueError('positive and wait window counts and scan limit must be positive')
    indices = random.Random(seed).sample(range(len(dataset)),min(scan_windows,len(dataset)))
    positives, waits = [], []
    for scanned,index in enumerate(indices,1):
        b = dataset[index]
        valid = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
        if not valid.any():
            continue
        has_action = bool((valid & b['play_now']).any())
        if has_action and len(positives)<positive_windows:
            positives.append((index,b))
        elif not has_action and len(waits)<wait_windows:
            waits.append((index,b))
        if len(positives)==positive_windows and len(waits)==wait_windows:
            return positives+waits
        if scanned%100==0:
            print(json.dumps({'phase':'overfit_selection','scanned':scanned,
                              'positive_windows':len(positives),'wait_windows':len(waits)}),flush=True)
    raise ValueError('not enough action/wait windows; increase --scan-windows or reduce requested counts')


def overfit(args):
    if args.log_every < 1:
        raise ValueError('positive log interval required')
    root = initialize(args,'overfit')
    dataset = Windows(args.data,args.cache,args.train_split,targets=args.targets,
                      frame_window=args.frame_window,event_window=1)
    if dataset.index['smoke_only'] and not args.allow_smoke:
        raise ValueError('synthetic data requires --allow-smoke')
    selected = select_windows(dataset,args.positive_windows,args.wait_windows,args.scan_windows,args.seed)
    indices, windows = zip(*selected)
    batch = collate(windows)
    (root/'selected-windows.json').write_text(json.dumps({
        'split':args.train_split,'manifest_sha256':dataset.index['manifest_sha256'],
        'indices':indices,'selection':'action-window-enriched training subset; NOT held-out evaluation',
        'frame_window':args.frame_window,'targets':args.targets},indent=2))
    device = torch.device(args.device)
    b = {k:v.to(device) for k,v in batch.items()}
    valid = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
    truth = b['play_now'][valid].cpu().numpy()
    config = config_from_args(args,dataset.index['dimensions'])
    model = Policy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    scaler = (torch.amp.GradScaler('cuda',enabled=False) if hasattr(torch.amp,'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=False))
    print(json.dumps({'phase':'overfit_start','run_dir':str(root),'windows':len(windows),
                      'valid_frames':int(valid.sum()),'actual_actions':int(truth.sum()),
                      'positive_weight':args.positive_weight,'precision':'fp32',
                      'parameters':sum(p.numel() for p in model.parameters()),
                      'scope':'memorization diagnostic, not generalization'}),flush=True)

    def measure(step):
        model.eval()
        with torch.no_grad():
            output=model(b)
            _,stats=bc_loss(output,b,timing_positive_weight=args.positive_weight)
            report=probability_report(output['timing'][valid].sigmoid().cpu().numpy(),truth)
        result={'phase':'overfit','step':step,'scope':'same fixed training windows',
                **summarize(stats),**report}
        print(json.dumps(result),flush=True)
        with (root/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        return result

    initial=measure(0)
    for step in range(1,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,_=bc_loss(model(b),b,timing_positive_weight=args.positive_weight)
        loss.backward()
        optimizer_update(model,optimizer,scaler,1.0)
        if step%args.log_every==0 or step==args.steps:
            final=measure(step)
    # Diagnostic checkpoint is intentionally not a general-trainer checkpoint.
    torch.save({'config':asdict(config),'model':model.state_dict(),'step':args.steps,
                'purpose':'overfit diagnostic only','selected_indices':indices},root/'diagnostic.pt')
    summary={'phase':'overfit_summary','run_dir':str(root),
             'initial_ap':initial['average_precision'],'final_ap':final['average_precision'],
             'baseline_ap':final['constant_score_ap_baseline'],
             'final_action_precision':final['action_precision'],
             'final_action_recall':final['action_recall'],
             'initial_timing_unweighted_loss':initial['timing_unweighted_loss'],
             'final_timing_unweighted_loss':final['timing_unweighted_loss'],
             'actual_actions':int(truth.sum()),
             'note':'training-set memorization only; failure is not proof that data must be recollected'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary


def compare(args):
    if min(args.batch_size,args.eval_batches) < 1 or args.workers < 0:
        raise ValueError('invalid batch/worker arguments')
    root=initialize(args,'weight-compare')
    reports=[]
    for name,weight in [('baseline',1.0),('weighted',args.positive_weight)]:
        run_dir=root/name
        options={'data':args.data,'cache':args.cache,'run-dir':run_dir,
                 'train-split':args.train_split,'val-split':args.val_split,'device':args.device,
                 'precision':'fp32','width':args.width,'hidden-size':args.hidden_size,
                 'frame-window':args.frame_window,'targets':args.targets,
                 'batch-size':args.batch_size,'workers':args.workers,'cpu-threads':args.cpu_threads,
                 'seed':args.seed,'max-steps':args.steps,'epochs':args.steps,
                 'log-every':100,'save-every':500,'eval-batches':min(args.eval_batches,100),
                 'timing-positive-weight':weight}
        argv=[part for k,v in options.items() for part in ('--'+k,str(v))]
        if args.allow_smoke:argv.append('--allow-smoke')
        print(json.dumps({'phase':'comparison_arm','arm':name,'positive_weight':weight,
                          'run_dir':str(run_dir),'steps':args.steps,'precision':'fp32'}),flush=True)
        train(train_parser().parse_args(argv))
        opts={'checkpoint':run_dir/'last.pt','data':args.data,'cache':args.cache,
              'device':args.device,'batch-size':args.batch_size,'workers':args.workers,
              'cpu-threads':args.cpu_threads,'batches':args.eval_batches,'seed':123,
              'output':run_dir/'timing-eval.json'}
        argv=[part for k,v in opts.items() for part in ('--'+k,str(v))]
        if args.allow_smoke:argv.append('--allow-smoke')
        report=evaluate(eval_parser().parse_args(argv))
        reports.append({'arm':name,'positive_weight':weight,'run_dir':str(run_dir),
                        **{key:report[key] for key in ('checkpoint_step','valid_frames','actual_actions',
                          'actual_action_rate','average_precision','constant_score_ap_baseline',
                          'ap_lift_over_prevalence','roc_auc','mean_predicted_probability')}})
    if reports[0]['checkpoint_step']!=reports[1]['checkpoint_step']:
        raise RuntimeError('unequal completed update counts')
    result={'phase':'comparison_summary','run_dir':str(root),'arms':reports,
            'note':'same initialization seed, sample order and FP32 update count; compare AP, not weighted loss or recall alone'}
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result
