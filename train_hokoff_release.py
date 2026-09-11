"""Restart BC from the pinned release; subsequent invocations resume full state."""
import argparse
from pathlib import Path
from policy_v1.train import load_checkpoint
import train_hokoff_fixed

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=ROOT/'models/hokoff-bc-step1037042.pt')
    p.add_argument('--hours', type=float, default=8)
    p.add_argument('--steps', type=int, help='bounded short check instead of hours')
    p.add_argument('--lr', type=float, default=None, help='new run defaults to 1e-4; resume inherits')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args(argv)
    command=['--data',str(a.data),'--cache',str(a.cache),'--run-dir',str(a.run_dir),
             '--device',a.device,'--workers',str(a.workers),'--epochs','100',
             '--eval-every','2000','--save-every','1000','--eval-batches','100']
    last=a.run_dir/'last.pt'
    if last.exists():
        saved=load_checkpoint(last); c=saved['config']; contract=saved['contract']
        if not contract.get('weights_restart'):
            p.error('run is not a release restart experiment')
        for value,key in ((a.lr,'lr'),(a.batch_size,'batch_size_per_rank')):
            if value is not None and value != contract[key]:
                p.error('resume settings differ: '+key)
        command+=['--resume',str(last)]
        for key in ('width','hidden_size','frame_window','max_delay','decision_period',
                    'spatial_type_dim','history_length','spatial_skip_channels'):
            command+=['--'+key.replace('_','-'),str(c[key])]
        if c.get('combat_features') is not None:
            command+=['--combat-features-file',str(ROOT/'hokoff_model/combat_features.json')]
        for flag,key in [('lr','lr'),('batch-size','batch_size_per_rank'),('precision','precision'),
                         ('targets','targets'),('seed','seed'),('weight-decay','weight_decay'),
                         ('grad-clip','grad_clip'),('timing-positive-weight','timing_positive_weight'),
                         ('train-split','train_split'),('val-split','val_split')]:
            command+=['--'+flag,str(contract[key])]
    else:
        command+=['--init-weights',str(a.weights),'--lr',str(a.lr if a.lr is not None else 1e-4),
                  '--batch-size',str(a.batch_size if a.batch_size is not None else 32)]
    command+=['--steps',str(a.steps)] if a.steps is not None else ['--hours',str(a.hours)]
    if a.dry_run: command+=['--dry-run']
    return train_hokoff_fixed.main(command)


if __name__=='__main__': main()
