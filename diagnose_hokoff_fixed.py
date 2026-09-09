"""One-command AP and shared-gradient diagnostics on one frozen fixed-policy checkpoint."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import torch

from policy_v1.train import load_checkpoint
from policy_v1.data import digest
from hokoff_model.evaluate_fixed import main as evaluate_ap
from hokoff_model.capacity_model import SUPPORTED_ARCHITECTURES
from hokoff_model.diagnose_gradients import main as evaluate_gradients

BASE=Path.home()/'cr-data'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=BASE/'expert-dataset/native-bc-v1')
    p.add_argument('--cache',type=Path,default=BASE/'hokoff-fixed-cache-p4')
    p.add_argument('--checkpoint',type=Path,default=BASE/'runs/hokoff-fixed-p4/last.pt')
    p.add_argument('--output',type=Path,help='new directory; default is timestamped beside the checkpoint')
    p.add_argument('--mode',choices=('both','ap','gradients'),default='both')
    p.add_argument('--ap-batches',type=int,default=100)
    p.add_argument('--gradient-batches',type=int,default=50)
    p.add_argument('--gradient-split',choices=('training','validation'),default='training')
    p.add_argument('--batch-size',type=int,help='default: checkpoint training batch size')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=p.parse_args(argv)
    if min(args.ap_batches,args.gradient_batches,args.cpu_threads)<1 or args.workers<0:
        p.error('batch/thread counts must be positive; workers must be nonnegative')
    if args.batch_size is not None and args.batch_size<1:p.error('batch-size must be positive')
    if not args.checkpoint.is_file():p.error('checkpoint is missing: '+str(args.checkpoint))
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    output=args.output or args.checkpoint.parent/('diagnostics-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/'model.pt'
    # Read one open inode: the trainer may atomically replace last.pt, but both
    # diagnostics must examine the exact same checkpoint bytes.
    with args.checkpoint.open('rb') as source, frozen.open('xb') as target:
        shutil.copyfileobj(source,target)
    saved=load_checkpoint(frozen)
    if saved.get('config',{}).get('architecture') not in SUPPORTED_ARCHITECTURES:
        raise ValueError('requires a fixed-policy checkpoint')
    batch_size=args.batch_size or saved['contract']['batch_size_per_rank']
    result=dict(source_checkpoint=str(args.checkpoint),frozen_checkpoint=str(frozen),
                checkpoint_sha256=digest(frozen),checkpoint_step=saved['step'],mode=args.mode)
    del saved
    common=['--data',str(args.data),'--cache',str(args.cache),'--checkpoint',str(frozen),
            '--device',args.device,'--batch-size',str(batch_size),'--workers',str(args.workers),
            '--cpu-threads',str(args.cpu_threads)]
    print('Frozen checkpoint:',frozen,'step:',result['checkpoint_step'],flush=True)
    if args.mode in ('both','ap'):
        r=evaluate_ap(common+['--output',str(output/'ap'),'--batches',str(args.ap_batches)])
        result['ap']=dict(results=str(output/'ap/results.json'),average_precision=r['timing_ranking']['average_precision'],
            positive_rate=r['timing_ranking']['positive_rate'],actual_rate_budget=r['timing_ranking']['action_budgets']['actual_rate'])
        assert r['checkpoint_sha256']==result['checkpoint_sha256']
    if args.mode in ('both','gradients'):
        r=evaluate_gradients(common+['--output',str(output/'gradients'),'--batches',str(args.gradient_batches),
                                     '--split',args.gradient_split])
        result['gradients']=dict(results=str(output/'gradients/results.json'),batches=r['batches'])
        assert r['checkpoint_sha256']==result['checkpoint_sha256']
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Diagnostic summary:',output/'summary.json',flush=True)
    return result


if __name__=='__main__':main()
