"""CPU-only synthetic R0 smoke. Never starts a native worker or cloud service."""
from __future__ import annotations

import argparse
import json
import time

import torch

from .catalog import CardVocabulary
from .config import ModelConfig, TrainingConfig
from .learning import update_minibatch
from .model import R0Policy
from .observation import ObservationBuilder
from .synthetic import rollout


def run(profile: str = 'base', steps: int = 4, rich: bool = False) -> dict:
    if profile not in ('base','reference','elite','tiny') or not 1 <= steps <= 160:
        raise ValueError('invalid smoke profile/length')
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    vocabulary = CardVocabulary.from_native()
    config = ModelConfig.reference(allow_incomplete_capture=True) if profile in ('base','reference') else ModelConfig(allow_incomplete_capture=True) if profile=='elite' else ModelConfig(width=32,layers=1,heads=4,hidden=64,spatial_channels=16,native_graph_layers=1,allow_incomplete_capture=True)
    model = R0Policy(vocabulary,config).cpu()
    builder = ObservationBuilder(vocabulary,config)
    started = time.perf_counter()
    segment = rollout(model,builder,steps=steps,sides=(0,1),rich=rich)
    settings = TrainingConfig()
    optimizer = torch.optim.Adam(model.parameters(),lr=settings.learning_rate)
    metrics = update_minibatch(model,optimizer,segment,expected_behavior=segment.behavior,config=settings)
    return dict(status='synthetic_cpu_smoke_pass',profile=profile,rich_fixture=rich,parameters=sum(p.numel() for p in model.parameters()),
                vocabulary_size=vocabulary.size,observation_schema_sha256=builder.schema_hash,
                elapsed_seconds=round(time.perf_counter()-started,3),metrics=metrics,
                native_worker_started=False,native_same_offset_certified=False,
                real_dataset_read=False,gpu_used=False,formal_training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('base','reference','elite','tiny'),default='elite')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--rich',action='store_true',help='exercise nonempty effects, grouped entities, relations and commands')
    args = parser.parse_args()
    print(json.dumps(run(args.profile,args.steps,args.rich),ensure_ascii=False,indent=2))
