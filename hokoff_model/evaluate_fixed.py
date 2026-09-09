"""Evaluate fixed-period action imitation on held-out recorded observations."""
import argparse
from collections import defaultdict, Counter
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from policy_v1.data import digest
from policy_v1.train import load_checkpoint, move
from .decision_data import DecisionWindows, collate_decisions
from .capacity_model import policy_from_config
from .metrics import bc_loss, summarize
from .diagnostic_metrics import timing_diagnostics
from .console_log import format_console


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batches', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = p.parse_args(argv)
    if args.output.exists(): raise FileExistsError('use a new evaluation output directory')
    if args.batches < 1 or args.batch_size < 1: raise ValueError('positive evaluation size required')
    if args.workers < 0 or args.cpu_threads < 1: raise ValueError('invalid worker/thread count')
    torch.set_num_threads(args.cpu_threads)
    checkpoint_sha = digest(args.checkpoint)
    saved = load_checkpoint(args.checkpoint); config, model = policy_from_config(saved['config'])
    if checkpoint_sha != digest(args.checkpoint): raise ValueError('checkpoint changed while loading; use a frozen copy')
    evaluation_contract=saved['contract']
    if saved.get('kind')=='hokoff_fixed_ppo_checkpoint_v1':
        source_path=args.checkpoint.parent/'il_source.pt'
        if digest(source_path)!=saved['run_contract']['source_sha256']:
            raise ValueError('PPO frozen IL source changed')
        source=load_checkpoint(source_path)
        if digest(source_path)!=saved['run_contract']['source_sha256']:
            raise ValueError('PPO frozen IL source changed while loading')
        evaluation_contract=source['contract']
    if evaluation_contract['decision_cache_sha256'] != digest(args.cache/'index.json'):
        raise ValueError('checkpoint cache differs')
    device = torch.device(args.device)
    model = model.to(device); model.load_state_dict(saved['model']); model.eval()
    ds = DecisionWindows(args.data,args.cache,evaluation_contract['val_split'],
        targets=evaluation_contract['targets'],frame_window=config.frame_window,
        max_delay=config.max_delay,sampling='fixed',decision_period=config.decision_period)
    if any(any(r.get('segment_roles',[])) for r in ds.records):
        raise ValueError('evaluation requires primary-only held-out sequences')
    ids = list(range(len(ds))); random.Random(123).shuffle(ids)
    ids = ids[:args.batches*args.batch_size]
    loader = DataLoader(ds,batch_size=args.batch_size,sampler=ids,num_workers=args.workers,
                       collate_fn=collate_decisions,pin_memory=device.type=='cuda')
    stats=defaultdict(float); probability=[]; actual=[]; joint=Counter(); started=time.monotonic()
    try:
        for bi,b in enumerate(loader):
            b=move(b,device)
            with torch.inference_mode():
                with torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext():
                    out=model(b)
                _,part=bc_loss(out,b)
                for k,v in part.items(): stats[k]+=v
                valid=b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
                probability.append(out['timing'][valid].float().sigmoid().cpu().numpy())
                actual.append(b['play_now'][valid].cpu().numpy())
                deploy=valid & b['card_label_mask'] & b['position_label_mask']
                kind=out['kind'].float().masked_fill(~b['action_kind_mask'],-1e9).argmax(-1)
                card=out['card'].float().masked_fill(~b['card_mask'],-1e9).argmax(-1)
                # Joint success requires the correct card, so its expert native
                # mask is sufficient for scoring these successful predictions.
                slot=b['card_slot'].clamp_min(0)
                pos=out['position'].gather(2,slot[...,None,None].expand(*slot.shape,1,576)).squeeze(2)
                position=pos.float().masked_fill(~b['position_mask'],-1e9).argmax(-1)
                correct=(kind==0) & (card==b['card_slot']) & (position==b['position'])
                joint['deploy_count']+=int(deploy.sum())
                joint['deploy_joint_correct']+=int((deploy & correct).sum())
                joint['deploy_with_timing_correct']+=int((deploy & correct & (out['timing']>0)).sum())
            if (bi+1)%25==0: print('fixed evaluation batches',bi+1,flush=True)
        prob=np.concatenate(probability); truth=np.concatenate(actual)
        if not len(prob): raise ValueError('no scored fixed periods')
        audit=Counter()
        for record in ds.records: audit.update(record['label_audit'])
        ranking, curve = timing_diagnostics(prob, truth)
        result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_sha,
            checkpoint_step=saved['step'],cache_sha256=digest(args.cache/'index.json'),
            sampling='fixed_shuffled_windows_seed_123',windows=len(ids),period_ticks=config.decision_period,
            observations_per_6000_ticks=(6000+config.decision_period-1)//config.decision_period,
            metrics=summarize(stats),metrics_timing_positive_weight=1.0,timing_ranking=ranking,
            deploy_joint_accuracy=joint['deploy_joint_correct']/max(joint['deploy_count'],1),
            deploy_with_timing_accuracy=joint['deploy_with_timing_correct']/max(joint['deploy_count'],1),
            joint_counts=dict(joint),validation_label_audit=dict(audit),elapsed_seconds=time.monotonic()-started,
            limitations=['fixed-window legal imitation targets may advance expert actions up to period-1 ticks',
                'unknown/illegal/multiple-action periods excluded from scoring; counts reported',
                'conditional position accuracy uses the expert card and native mask',
                'recorded expert states; no counterfactual game actions executed; not win rate'])
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output/'timing_predictions.npz',probability=prob,actual=truth,windows=ids)
        np.savez_compressed(args.output/'pr_curve.npz', **curve)
        (args.output/'results.json').write_text(json.dumps(result,indent=2))
        print(format_console(dict(phase='timing_diagnostic',step=saved['step'],
            average_precision=ranking['average_precision'],positive_rate=ranking['positive_rate'],
            ap_over_positive_rate=ranking['ap_over_positive_rate'],
            **{k:result['metrics'][k] for k in ('action_precision','action_recall','card_accuracy','position_accuracy')},
            actual_rate_budget=ranking['action_budgets']['actual_rate'],results=str(args.output/'results.json'))),flush=True)
    finally: ds.close()
    return result


if __name__ == '__main__': main()
