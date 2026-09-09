"""Controlled timing-capacity fine-tuning with fresh optimizers in both variants."""
from functools import partial
from pathlib import Path
from dataclasses import asdict
from policy_v1.data import digest
from policy_v1.train import parser as base_parser,run as base_run,load_checkpoint
from .train import adapt_parser
from .train_fixed import FixedConfig,FixedPolicy
from .capacity_model import CapacityConfig,CapacityPolicy,initialize_from_source
from .fixed_data import CONTRACT
from .decision_data import DecisionWindows,collate_decisions
from .metrics import bc_loss,summarize
from .console_log import format_console


def parser():
    p=adapt_parser(base_parser())
    p.description=__doc__
    p.set_defaults(frame_window=17,targets=32,train_split='validation',val_split='train',
                   precision='bf16',max_steps=5000,eval_every=1000,save_every=1000,
                   log_every=100,eval_batches=100,eval_shuffle=True)
    p.add_argument('--source-checkpoint',type=Path,required=True)
    p.add_argument('--variant',choices=('linear','residual'),required=True)
    p.add_argument('--timing-hidden-size',type=int,default=256)
    p.add_argument('--max-delay',type=int,default=8)
    p.add_argument('--decision-period',type=int,default=4)
    p.add_argument('--timing-positive-weight',type=float,default=32)
    return p


def config_from_args(args,dims):
    kwargs={k:dims[k] for k in ('card_vocab_size','ability_vocab_size','public_scalar_size','entity_numeric_size','grid_channels')}
    kwargs.update(width=args.width,hidden_size=args.hidden_size,frame_window=args.frame_window,
                  max_delay=args.max_delay,decision_period=args.decision_period)
    return FixedConfig(**kwargs) if args.variant=='linear' else CapacityConfig(**kwargs,timing_hidden_size=args.timing_hidden_size)


def run(args):
    if args.precision=='fp16':
        raise ValueError('use BF16 or FP32: AMP-skipped batches would confound equal-update comparisons')
    if args.resume is None and args.run.exists() and any(args.run.iterdir()):
        raise FileExistsError('use a new/empty trial directory')
    source_hash=digest(args.source_checkpoint);source=load_checkpoint(args.source_checkpoint)
    if source_hash!=digest(args.source_checkpoint):raise ValueError('source changed while loading')
    if source['config'].get('architecture')!=FixedConfig.architecture:raise ValueError('requires original fixed-policy source')
    cache_hash=digest(args.cache/'index.json')
    if source['contract']['decision_cache_sha256']!=cache_hash:raise ValueError('source cache differs')
    for key in ('train_split','val_split','timing_positive_weight'):
        if getattr(args,key)!=source['contract'][key]:raise ValueError('source '+key+' differs')
    factory=FixedPolicy if args.variant=='linear' else CapacityPolicy
    if args.resume is None:factory=partial(initialize_from_source,checkpoint=args.source_checkpoint)
    return base_run(args,model_factory=factory,config_factory=config_from_args,
        dataset_factory=partial(DecisionWindows,sampling='fixed',max_delay=args.max_delay,decision_period=args.decision_period),
        collate_fn=collate_decisions,bc_loss=partial(bc_loss,timing_positive_weight=args.timing_positive_weight),
        summarize=summarize,console_formatter=format_console,
        contract_extra=dict(decision_contract=CONTRACT,decision_period=args.decision_period,
            decision_cache_sha256=cache_hash,timing_positive_weight=args.timing_positive_weight,
            training_only=True,delay_enabled=False,capacity_experiment='timing_residual_v1',
            variant=args.variant,source_checkpoint_sha256=source_hash,source_step=source['step'],optimizer_initialization='fresh'))


if __name__=='__main__':run(parser().parse_args())
