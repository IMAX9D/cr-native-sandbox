"""Run paired short fine-tuning trials: original linear timing vs zero-output residual MLP."""
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import torch
import numpy as np

from policy_v1.data import digest
from policy_v1.train import load_checkpoint,move
from hokoff_model.train_fixed import FixedConfig
from hokoff_model.capacity_model import CapacityConfig,initialize_from_source
from hokoff_model.train_capacity import parser as train_parser,run as train
from hokoff_model.decision_data import DecisionWindows,collate_decisions
from hokoff_model.evaluate_fixed import main as evaluate

BASE=Path.home()/'cr-data'


def audit_initial(source,data,cache,hidden,seed,device,precision):
    saved=load_checkpoint(source);c=saved['config'];contract=saved['contract']
    configs=dict(linear=FixedConfig(**c),residual=CapacityConfig(**dict(c,
        architecture=CapacityConfig.architecture,timing_hidden_size=hidden)))
    ds=DecisionWindows(data,cache,contract['val_split'],targets=contract['targets'],frame_window=c['frame_window'],
                       max_delay=c['max_delay'],sampling='fixed',decision_period=c['decision_period'])
    try:
        b=move(collate_decisions([ds[i] for i in range(min(4,len(ds)))]),torch.device(device))
        outputs={};counts={}
        for variant,config in configs.items():
            torch.manual_seed(seed)
            model=initialize_from_source(config,checkpoint=source).to(device).eval()
            for name,value in saved['model'].items():
                torch.testing.assert_close(model.state_dict()[name].cpu(),value,rtol=0,atol=0)
            counts[variant]=sum(p.numel() for p in model.parameters())
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else contextlib.nullcontext()
            with torch.inference_mode(),amp:
                outputs[variant]={k:v.float().cpu() for k,v in model(b).items()}
            del model
        for name in outputs['linear']:
            torch.testing.assert_close(outputs['linear'][name],outputs['residual'][name],rtol=0,atol=0)
        return dict(source_step=saved['step'],parameters=counts,added_parameters=counts['residual']-counts['linear'],
                    all_initial_outputs_exactly_equal=True,all_source_parameters_preserved=True,
                    audit_precision=precision,audit_windows=min(4,len(ds)))
    finally:ds.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=BASE/'expert-dataset/native-bc-v1')
    p.add_argument('--cache',type=Path,default=BASE/'hokoff-fixed-cache-p4')
    p.add_argument('--source-checkpoint',type=Path,default=BASE/'runs/hokoff-fixed-p4/last.pt')
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--timing-hidden-size',type=int,default=256)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--eval-batches',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    args=p.parse_args(argv)
    if min(args.steps,args.timing_hidden_size,args.eval_every,args.eval_batches,args.cpu_threads)<1 or args.workers<0:
        p.error('invalid trial sizes')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    precision='bf16' if args.device=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    torch.set_num_threads(args.cpu_threads)
    output=args.output or BASE/'runs'/('hokoff-timing-capacity-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True,exist_ok=False);source=output/'source.pt'
    with args.source_checkpoint.open('rb') as src,source.open('xb') as dst:shutil.copyfileobj(src,dst)
    saved=load_checkpoint(source);contract=saved['contract'];config=saved['config']
    if config['architecture']!=FixedConfig.architecture:raise ValueError('requires original fixed-policy source')
    if contract['world_size']!=1:raise ValueError('this paired trial reproduces single-GPU training only')
    audit=audit_initial(source,args.data,args.cache,args.timing_hidden_size,args.seed,args.device,precision)
    result=dict(source_checkpoint=str(args.source_checkpoint),source_checkpoint_sha256=digest(source),
        source_step=saved['step'],initial_audit=audit,steps_per_variant=args.steps,seed=args.seed,
        precision=precision,source_precision=contract['precision'],optimizer_initialization='fresh_in_both',
        sampling='same_seed_shuffled_training_windows_from_epoch_zero',train_scope='all_original_trainable_parameters_plus_residual',
        trials={})
    (output/'experiment.json').write_text(json.dumps(result,indent=2))
    print('Capacity experiment:',output,flush=True);print('Initial audit:',json.dumps(audit),flush=True)
    del saved;gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
    for variant in ('linear','residual'):
        trial=output/variant
        argv=['--data',str(args.data),'--cache',str(args.cache),'--run',str(trial),
            '--source-checkpoint',str(source),'--variant',variant,'--timing-hidden-size',str(args.timing_hidden_size),
            '--device',args.device,'--precision',precision,'--seed',str(args.seed),'--workers',str(args.workers),
            '--cpu-threads',str(args.cpu_threads),'--max-steps',str(args.steps),'--epochs','100',
            '--eval-every',str(args.eval_every),'--save-every',str(args.eval_every),
            '--eval-batches',str(args.eval_batches),'--log-every','100']
        for key in ('width','hidden_size','frame_window','max_delay','decision_period'):
            argv+=['--'+key.replace('_','-'),str(config[key])]
        for arg,key in [('train-split','train_split'),('val-split','val_split'),('targets','targets'),
                        ('batch-size','batch_size_per_rank'),('lr','lr'),('weight-decay','weight_decay'),
                        ('grad-clip','grad_clip'),('timing-positive-weight','timing_positive_weight')]:
            argv+=['--'+arg,str(contract[key])]
        index=json.loads((args.cache/'index.json').read_text())
        if index.get('smoke_only'):argv+=['--allow-smoke']
        print('Training',variant,'for',args.steps,'updates; log:',output/(variant+'.console.log'),flush=True)
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        with (output/(variant+'.console.log')).open('w') as f,contextlib.redirect_stdout(f):
            train(train_parser().parse_args(argv))
        trained=load_checkpoint(trial/'last.pt')
        if trained['step']!=args.steps:raise AssertionError('trial ended before the requested update count')
        if any(json.loads(s).get('phase')=='amp_overflow' for s in (trial/'metrics.jsonl').read_text().splitlines()):
            raise AssertionError('skipped updates invalidate the paired sample comparison')
        for name in ('delay.weight','delay.bias'):
            torch.testing.assert_close(trained['model'][name],load_checkpoint(source)['model'][name],rtol=0,atol=0)
        cursor=dict(epoch=trained['epoch'],next_batch=trained['next_batch'])
        del trained;gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
        evaluation=evaluate(['--data',str(args.data),'--cache',str(args.cache),'--checkpoint',str(trial/'last.pt'),
            '--output',str(trial/'evaluation'),'--batches',str(args.eval_batches),'--batch-size',str(contract['batch_size_per_rank']),
            '--workers',str(args.workers),'--cpu-threads',str(args.cpu_threads),'--device',args.device])
        result['trials'][variant]=dict(checkpoint=str(trial/'last.pt'),checkpoint_sha256=evaluation['checkpoint_sha256'],
            cursor=cursor,metrics=evaluation['metrics'],timing_ranking=evaluation['timing_ranking'],
            deploy_joint_accuracy=evaluation['deploy_joint_accuracy'],deploy_with_timing_accuracy=evaluation['deploy_with_timing_accuracy'])
        (output/'experiment.json').write_text(json.dumps(result,indent=2))
    assert result['trials']['linear']['cursor']==result['trials']['residual']['cursor']
    with np.load(output/'linear/evaluation/timing_predictions.npz') as a,np.load(output/'residual/evaluation/timing_predictions.npz') as b:
        for key in ('windows','actual'):np.testing.assert_array_equal(a[key],b[key])
    result['matched_training_progress']=True;result['matched_evaluation_windows_and_labels']=True
    result['limitations']=['single paired seed and short fine-tuning; no win-rate evaluation',
        'both optimizers reset; training precision shared across variants, possibly different from source',
        'end-of-budget checkpoints compared, not separately selected best validation checkpoints']
    (output/'experiment.json').write_text(json.dumps(result,indent=2))
    print('Completed paired capacity experiment:',output/'experiment.json',flush=True)
    return result


if __name__=='__main__':main()
