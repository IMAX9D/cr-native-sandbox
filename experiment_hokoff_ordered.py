#!/usr/bin/env python3
"""Bounded ordered-decoder probe over a frozen fixed4 encoder; no gameplay changes."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from policy_v1.data import digest
from policy_v1.train import load_checkpoint
from hokoff_model.train_fixed import FixedConfig, FixedPolicy
from hokoff_model.decision_data import DecisionWindows, collate_decisions
from hokoff_model.ordered_data import audit_pairs, sample_plan
from hokoff_model.ordered_model import OrderedDecoder, ordered_loss, sequence_metrics, continuation_metrics


class ContextPolicy(FixedPolicy):
    def heads(self, recurrent, batch):
        return {'context': self.context(recurrent)}


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def extract_contexts(model, dataset, plan, device, batch_size):
    contexts = []
    started = time.monotonic()
    for start in range(0, len(plan['indices']), batch_size):
        items = [dataset[i] for i in plan['indices'][start:start+batch_size]]
        batch = {k:v.to(device) for k,v in collate_decisions(items).items()}
        with torch.inference_mode():
            out = model(batch)['context']
            last = batch['frame_mask'].sum(-1)-1
            contexts.append(out[torch.arange(len(items), device=device), last].cpu())
        if start % (batch_size*20) == 0:
            print(f'encode: {start+len(items)}/{len(plan["indices"])} elapsed_seconds: {time.monotonic()-started:.1f}', flush=True)
    return dict(context=torch.cat(contexts), targets={k:torch.from_numpy(v) for k,v in plan['targets'].items()},
                indices=plan['indices'])


def evaluate(model, data, device, *, oracle_prefix=False):
    model.eval(); predictions = []
    with torch.inference_mode():
        for start in range(0, len(data['context']), 256):
            prefix = None if not oracle_prefix else {k:v[start:start+256,0].to(device)
                       for k,v in data['targets'].items() if k != 'mask'}
            _, p = model(data['context'][start:start+256].to(device),first_action=prefix)
            predictions.append({k:v.cpu() for k,v in p.items()})
    prediction = {k:torch.cat([p[k] for p in predictions]) for k in predictions[0]}
    target = data['targets']
    if oracle_prefix:
        # Exclude trivial oracle STOP windows. First action is gold; second is free-running.
        active = target['type'][:,0] != 0
        prediction = {k:v[active] for k,v in prediction.items()}
        target = {k:v[active] for k,v in target.items()}
    return continuation_metrics(prediction,target) if oracle_prefix else sequence_metrics(prediction,target)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('/home/lenovo/cr-data/expert-dataset/native-bc-v1'))
    p.add_argument('--cache',type=Path,default=Path('/home/lenovo/cr-data/hokoff-fixed-cache-p4'))
    p.add_argument('--source-checkpoint',type=Path,default=Path('/home/lenovo/cr-data/runs/hokoff-fixed-p4/last.pt'))
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--examples-per-class',type=int,default=678)
    p.add_argument('--validation-per-class',type=int,default=256)
    p.add_argument('--natural-validation',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--encode-batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    args = p.parse_args(argv)
    if min(args.steps,args.examples_per_class,args.validation_per_class,args.natural_validation,
           args.batch_size,args.encode_batch_size,args.cpu_threads) < 1 or not np.isfinite(args.lr) or args.lr<=0:
        raise ValueError('positive finite probe settings required')
    output = args.output or Path('/home/lenovo/cr-data/runs')/('hokoff-ordered-probe-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_hash = digest(args.source_checkpoint)
    saved = load_checkpoint(args.source_checkpoint)
    if digest(args.source_checkpoint) != source_hash: raise ValueError('source changed while reading')
    if saved['config']['architecture'] != FixedConfig.architecture:
        raise ValueError('use the original fixed4 checkpoint')
    config = FixedConfig(**saved['config'])
    if config.decision_period != 4: raise ValueError('requires period4')
    audit = audit_pairs(args.data,args.cache)
    if saved['contract']['decision_cache_sha256'] != audit['cache_sha256']:
        raise ValueError('checkpoint/cache mismatch')
    json_write(output/'data_audit.json',audit)
    contract = dict(experiment='ordered_forecast_frozen_encoder_probe_v1',training_only=True,
                    source_checkpoint=str(args.source_checkpoint.resolve()), source_sha256=source_hash,
                    source_step=saved['step'], source_config=asdict(config), period=4,max_actions=2,
                    stop_id=0,deploy_id=1,ability_id=2,preserve_recorded_tick_offsets=True,
                    decoder_tokens='card_and_ability_types_not_instances',executable_macro_policy=False,
                    future_observation_inputs=False,train_sampling='stratified_0_1_2_diagnostic_only',
                    cache_sha256=audit['cache_sha256'], steps=args.steps,seed=args.seed,
                    batch_size=args.batch_size,lr=args.lr,backbone_frozen=True)
    json_write(output/'contract.json',contract)
    model = ContextPolicy(config).to(args.device).eval()
    model.load_state_dict(saved['model']); model.requires_grad_(False)
    datasets = {}; features = {}; plans = {}; sampling = {}
    try:
        for name, split, n, natural_n in [('train','validation',args.examples_per_class,0),
                                         ('validation','train',args.validation_per_class,args.natural_validation)]:
            ds = DecisionWindows(args.data,args.cache,split,targets=1,frame_window=config.frame_window,
                                 max_delay=config.max_delay,sampling='fixed',decision_period=4,max_open=2)
            datasets[name] = ds
            print('sampling',name,flush=True)
            plan,natural,sample_audit = sample_plan(ds,audit['splits'][split]['pairs'],examples_per_class=n,
                                                   natural_count=natural_n,seed=args.seed+(name=='validation'))
            sampling[name] = sample_audit
            for key, selected in [(name,plan),(name+'_natural',natural)]:
                if selected is not None:
                    plans[key] = selected['indices']
                    features[key] = extract_contexts(model,ds,selected,args.device,args.encode_batch_size)
            ds.close()
    finally:
        for ds in datasets.values(): ds.close()
    json_write(output/'sampling.json',sampling)
    json_write(output/'windows.json',plans)
    torch.save(features,output/'features.pt')
    torch.manual_seed(args.seed)
    initial = OrderedDecoder(config.width,config.card_vocab_size,config.ability_vocab_size)
    # Reuse categorical/position knowledge; output token classifiers and recurrent decoder are new.
    initial.card_embedding.load_state_dict(model.cards.state_dict())
    initial.ability_embedding.load_state_dict(model.abilities.state_dict())
    initial.position_embedding.load_state_dict(model.positions.state_dict())
    initial.position.load_state_dict(model.position_head.state_dict())
    initial_state = deepcopy(initial.state_dict())
    del model, saved
    if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
    rng = np.random.default_rng(args.seed)
    order = rng.integers(len(features['train']['context']),size=(args.steps,args.batch_size))
    train_context = features['train']['context'].to(args.device)
    train_targets = {k:v.to(args.device) for k,v in features['train']['targets'].items()}
    results = dict(contract=contract,sampling=sampling,trials={})
    for variant, conditioned in [('unconditioned',False),('ordered',True)]:
        decoder = OrderedDecoder(config.width,config.card_vocab_size,config.ability_vocab_size,
                                 conditioned=conditioned).to(args.device)
        decoder.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(decoder.parameters(),lr=args.lr,weight_decay=1e-4)
        initial_metrics = evaluate(decoder,features['validation'],args.device)
        started = time.monotonic()
        with (output/(variant+'.jsonl')).open('w') as log:
            for step, ids in enumerate(order,1):
                decoder.train(); ids=torch.from_numpy(ids).to(args.device)
                targets={k:v[ids] for k,v in train_targets.items()}
                outputs,_=decoder(train_context[ids],targets)
                loss,stats=ordered_loss(outputs,targets)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
                if not torch.isfinite(norm): raise FloatingPointError('non-finite decoder gradients')
                optimizer.step()
                if step%100==0 or step==1 or step==args.steps:
                    row=dict(variant=variant,step=step,loss=float(loss.detach()),**{k+'_loss':v for k,v in stats.items()})
                    log.write(json.dumps(row)+'\n');log.flush()
                    print('#'*64+'\n'+f'{variant} | step {step}/{args.steps}',flush=True)
                    for k,v in row.items():
                        if k not in ('variant','step'): print(f'{k:>28}: {v:.4f}',flush=True)
        if not all(torch.isfinite(v).all() for v in decoder.state_dict().values()):
            raise FloatingPointError('non-finite checkpoint')
        metrics = {name:evaluate(decoder,data,args.device) for name,data in features.items()}
        oracle = {name:evaluate(decoder,data,args.device,oracle_prefix=True) for name,data in features.items()}
        results['trials'][variant] = dict(initial_validation=initial_metrics,metrics=metrics,oracle_first_action_metrics=oracle,
                                         elapsed_seconds=time.monotonic()-started,
                                         parameter_count=sum(v.numel() for v in decoder.parameters()))
        torch.save(dict(decoder=decoder.state_dict(),contract=contract,conditioned=conditioned,
                        decoder_config=dict(width=config.width,card_vocab=config.card_vocab_size,
                                            ability_vocab=config.ability_vocab_size,period=4),step=args.steps),
                   output/(variant+'.pt'))
        json_write(output/'results.json',results)
    results['source_unchanged'] = digest(args.source_checkpoint)==source_hash
    if not results['source_unchanged']: raise ValueError('source checkpoint changed during experiment')
    json_write(output/'results.json',results)
    print('output:',output,flush=True)
    return results


if __name__=='__main__':main()
