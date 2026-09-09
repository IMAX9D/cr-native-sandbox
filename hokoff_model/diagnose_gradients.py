"""Measure fixed-policy task gradients without changing model or optimizer state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from policy_v1.data import digest
from policy_v1.loss import bc_loss
from policy_v1.train import load_checkpoint, move
from .decision_data import DecisionWindows, collate_decisions
from .capacity_model import policy_from_config
from .diagnostic_metrics import gradient_geometry, summarize_gradient_batches
from .console_log import format_console

TASK_MASKS = dict(timing='timing_label_mask', kind='kind_label_mask', card='card_label_mask',
    position='position_label_mask', ability='ability_label_mask', ability_position='ability_position_label_mask')
ENCODER_MODULES = ('cards', 'positions', 'sides', 'entity', 'grid', 'scene', 'time_projection')


def shared_parameters(model):
    groups = dict(encoder=[], lstm=[], context=[])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        root = name.split('.')[0]
        if root in ENCODER_MODULES: groups['encoder'].append((name,parameter))
        elif root in groups: groups[root].append((name,parameter))
    if not all(groups.values()): raise ValueError('unrecognized shared model structure')
    return groups


def isolated_task_losses(output, batch, timing_positive_weight):
    # Reuse exactly the training loss's legal masks, sample weights and per-task
    # denominators. Only label masks for OTHER heads are disabled.
    result = {}; counts = {}
    base = batch['frame_mask'] & batch['loss_mask']
    for task, keep in TASK_MASKS.items():
        counts[task] = int((base & batch[keep]).sum())
        if not counts[task]:
            result[task] = None
            continue
        masked = dict(batch)
        for label_mask in TASK_MASKS.values():
            if label_mask != keep: masked[label_mask] = torch.zeros_like(batch[label_mask])
        result[task], _ = bc_loss(output, masked, timing_positive_weight=timing_positive_weight)
    return result, counts


def measure_batch(model, batch, timing_positive_weight):
    groups = shared_parameters(model)
    named = [entry for entries in groups.values() for entry in entries]
    parameters = [p for _,p in named]
    output = model(batch)
    losses, counts = isolated_task_losses(output, batch, timing_positive_weight)
    gradients = []
    for task in TASK_MASKS:
        loss = losses[task]
        gradients.append(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                         if loss is not None else (None,)*len(parameters))
    gram_total = np.zeros((len(TASK_MASKS), len(TASK_MASKS)), dtype=np.float64)
    geometry = {}; offset = 0
    for group, entries in groups.items():
        vectors = []
        for gradient in gradients:
            vectors.append(torch.cat([g.detach().reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g,(_,p) in zip(gradient[offset:offset+len(entries)], entries)]))
        matrix = torch.stack(vectors)
        gram = (matrix @ matrix.T).detach().double().cpu().numpy()
        geometry[group] = gradient_geometry(gram, list(TASK_MASKS))
        gram_total += gram; offset += len(entries)
    geometry['all_shared'] = gradient_geometry(gram_total, list(TASK_MASKS))
    valid = batch['frame_mask'] & batch['loss_mask']
    return dict(losses={k:float(v.detach()) if v is not None else None for k,v in losses.items()},
        label_counts=counts, primary_targets=int((valid & batch['timing_label_mask']).sum()),
        auxiliary_action_targets=int((valid & batch['kind_label_mask'] & ~batch['timing_label_mask']).sum()),
        groups=geometry)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','cache','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('training','validation'),default='training',help='role, not the historical source split name')
    p.add_argument('--batches',type=int,default=50)
    p.add_argument('--batch-size',type=int,help='defaults to the checkpoint training batch size')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args(argv)
    if args.output.exists(): raise FileExistsError('use a new diagnostic output directory')
    if args.batches < 1 or args.workers < 0 or args.cpu_threads < 1: raise ValueError('invalid batch/worker/thread count')
    torch.set_num_threads(args.cpu_threads)
    # FP32 and no TF32/AMP/scaler, so the measured vectors are unscaled gradients.
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    checkpoint_sha=digest(args.checkpoint)
    saved=load_checkpoint(args.checkpoint)
    if checkpoint_sha!=digest(args.checkpoint): raise ValueError('checkpoint changed while loading; use a frozen copy')
    config,model=policy_from_config(saved['config']); contract=saved['contract']
    if contract['decision_cache_sha256']!=digest(args.cache/'index.json'): raise ValueError('checkpoint cache differs')
    batch_size=contract['batch_size_per_rank'] if args.batch_size is None else args.batch_size
    if batch_size < 1: raise ValueError('positive batch size required')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device)
    model=model.to(device);model.load_state_dict(saved['model'])
    # cuDNN LSTM backward needs training mode. This architecture has no dropout or
    # batch-normalization, so this preserves its evaluation-time computation.
    model.train()
    split=contract['train_split' if args.split=='training' else 'val_split']
    ds=DecisionWindows(args.data,args.cache,split,targets=contract['targets'],frame_window=config.frame_window,
                       max_delay=config.max_delay,sampling='fixed',decision_period=config.decision_period)
    ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(args.seed))[:args.batches*batch_size].tolist()
    if not ids: ds.close(); raise ValueError('empty diagnostic dataset')
    loader=DataLoader(ds,batch_size=batch_size,sampler=ids,num_workers=args.workers,
                      collate_fn=collate_decisions,pin_memory=device.type=='cuda')
    records=[];started=time.monotonic()
    try:
        for i,b in enumerate(loader):
            measured=measure_batch(model,move(b,device),contract['timing_positive_weight'])
            measured.update(batch=i,window_ids=ids[i*batch_size:(i+1)*batch_size]);records.append(measured)
            if (i+1)%10==0: print('gradient diagnostic batches',i+1,flush=True)
        if any(p.grad is not None for p in model.parameters()): raise AssertionError('diagnostic populated parameter .grad')
        labels=Counter()
        for record in records: labels.update(record['label_counts'])
        groups=shared_parameters(model)
        result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_step=saved['step'],
            cache_sha256=digest(args.cache/'index.json'),split_role=args.split,source_split=split,
            seed=args.seed,sampling='uniform_shuffled_windows',batch_size=batch_size,
            training_batch_size=contract['batch_size_per_rank'],training_world_size=contract['world_size'],
            batches=len(records),windows=len(ids),precision='fp32',timing_positive_weight=contract['timing_positive_weight'],
            label_counts=dict(labels),auxiliary_action_targets=sum(r['auxiliary_action_targets'] for r in records),
            shared_parameter_groups={k:[n for n,_ in v] for k,v in groups.items()},
            groups=summarize_gradient_batches(records,[*groups,'all_shared'],list(TASK_MASKS)),
            elapsed_seconds=time.monotonic()-started,
            limitations=['raw per-task gradients before clipping and Adam; no optimizer steps',
                'pair cosines exclude absent/zero gradients; a negative cosine alone does not prove harmful training interference',
                'norm_fraction is a fraction of gradient norms, not a fraction of the optimizer update',
                'sampled minibatches; different batch sizes change gradient noise; FP32 may differ slightly from FP16 training',
                'single-process diagnostic; does not reproduce cross-rank DDP gradient averaging'])
        args.output.mkdir(parents=True)
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        with (args.output/'batches.jsonl').open('w') as f:
            for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
        focus=result['groups']['all_shared']
        console=dict(phase='gradient_diagnostic',step=saved['step'],batches=len(records),
            timing_positive_weight=contract['timing_positive_weight'])
        for task in ('timing','card','position'):
            console[task+'_gradient_norm']=focus['tasks'][task]['norm']['mean']
        for pair in ('timing__card','timing__position','card__position'):
            console[pair+'_cosine']=focus['pairs'][pair]['mean']
            console[pair+'_negative_rate']=focus['pairs'][pair]['negative_cosine_rate']
        console.update(results=str(args.output/'results.json'))
        print(format_console(console),flush=True)
    finally: ds.close()
    return result


if __name__=='__main__': main()
