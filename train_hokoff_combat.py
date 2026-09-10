"""Migrate current fixed policy to static combat inputs, then resume for 15 hours."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from hokoff_model.train_fixed import FixedConfig,FixedPolicy
from policy_v1.train import load_checkpoint
from policy_v1.data import digest
import train_hokoff_fixed

BASE=Path.home()/'cr-data'
TABLE=Path(__file__).resolve().parent/'hokoff_model/combat_features.json'


def migrate(saved, features):
    if saved['config'].get('architecture') != FixedConfig.architecture:
        raise ValueError('requires fixed-policy BC checkpoint')
    source_config=FixedConfig(**saved['config'])
    if source_config.combat_features is not None:raise ValueError('source already has combat inputs')
    old=FixedPolicy(source_config)
    old.load_state_dict(saved['model'],strict=True)
    new_config=FixedConfig(**dict(asdict(source_config),combat_features=features))
    new=FixedPolicy(new_config)
    state=dict(saved['model'])
    name='entity.0.weight';weight=state[name]
    state[name]=torch.cat((weight,weight.new_zeros(weight.shape[0],20)),1)
    state['combat_features']=new.combat_features
    new.load_state_dict(state,strict=True)
    # Same parameter names/order: only one existing matrix gains 20 columns.
    names=[n for n,_ in old.named_parameters()]
    if names != [n for n,_ in new.named_parameters()]:raise ValueError('parameter ordering changed')
    ids=[i for group in saved['optimizer']['param_groups'] for i in group['params']]
    if len(ids)!=len(names):raise ValueError('optimizer parameter layout differs')
    pid=ids[names.index(name)]
    for key,value in saved['optimizer']['state'].get(pid,{}).items():
        if isinstance(value,torch.Tensor) and value.ndim:
            if value.shape!=weight.shape:raise ValueError('unexpected optimizer tensor')
            saved['optimizer']['state'][pid][key]=torch.cat((value,value.new_zeros(value.shape[0],20)),1)
    saved['config']=asdict(new_config);saved['model']=state
    return saved


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=BASE/'runs/hokoff-fixed-p4-history4-posskip16/last.pt')
    p.add_argument('--run-dir',type=Path,default=BASE/'runs/hokoff-fixed-p4-history4-posskip16-combat')
    p.add_argument('--data',type=Path,default=BASE/'expert-dataset/native-bc-v1')
    p.add_argument('--cache',type=Path,default=BASE/'hokoff-fixed-cache-p4')
    p.add_argument('--hours',type=float,default=15)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    path=args.run_dir/'last.pt';table=json.loads(TABLE.read_text());manifest=json.loads((args.data/'manifest.json').read_text())
    if table['card_vocabulary']!=manifest['card_vocabulary']:p.error('combat vocabulary differs from dataset')
    source=path if path.exists() else args.source
    source_sha=digest(source);saved=load_checkpoint(source);contract=saved['contract']
    if digest(source)!=source_sha:p.error('source changed while loading')
    if contract['manifest_sha256']!=digest(args.data/'manifest.json'):p.error('dataset differs from checkpoint')
    if contract['decision_cache_sha256']!=digest(args.cache/'index.json'):p.error('decision cache differs from checkpoint')
    if contract['world_size']!=1:p.error('launcher requires single-device checkpoint')
    if not args.dry_run and args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; migration/training not started')
    c=saved['config']
    command=['--data',str(args.data),'--cache',str(args.cache),'--run-dir',str(args.run_dir),
             '--resume',str(path),'--combat-features-file',str(TABLE),'--device',args.device,
             '--precision',contract['precision'],'--workers',str(args.workers),'--hours',str(args.hours),
             '--epochs','100','--eval-every','2000','--save-every','2000','--eval-batches','100']
    for flag,key in [('width','width'),('hidden-size','hidden_size'),('frame-window','frame_window'),
                     ('history-length','history_length'),('spatial-type-dim','spatial_type_dim'),
                     ('spatial-skip-channels','spatial_skip_channels'),('decision-period','decision_period'),('max-delay','max_delay')]:
        command+=['--'+flag,str(c.get(key,0))]
    for flag,key in [('batch-size','batch_size_per_rank'),('targets','targets'),('seed','seed'),('lr','lr'),
                     ('weight-decay','weight_decay'),('grad-clip','grad_clip'),('timing-positive-weight','timing_positive_weight'),
                     ('train-split','train_split'),('val-split','val_split')]:command+=['--'+flag,str(contract[key])]
    if path.exists():
        if c.get('combat_features')!=table['features']:p.error('existing run has different combat table')
    elif not args.dry_run:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):p.error('new run directory must be empty')
        if digest(source)!=source_sha:p.error('source changed during loading')
        old_step=saved['step'];migrated=migrate(saved,table['features'])
        args.run_dir.mkdir(parents=True,exist_ok=True)
        temporary=args.run_dir/'migration.pt.partial';torch.save(migrated,temporary);temporary.replace(path)
        (args.run_dir/'migration.json').write_text(json.dumps(dict(source=str(source),source_sha256=source_sha,
             step=old_step,table_sha256=digest(TABLE),optimizer_preserved=True,new_columns_zero=True),indent=2))
    print(json.dumps(dict(source=str(source),run_dir=str(args.run_dir),migration_needed=not source==path,
                         step=saved['step'],hours=args.hours,training_arguments=command),indent=2),flush=True)
    if not args.dry_run:train_hokoff_fixed.main(command)

if __name__=='__main__':main()
