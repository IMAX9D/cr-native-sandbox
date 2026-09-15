"""Bounded native opening-layout probe; no source mutation or training admission."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from native_core.env import NativeRoyaleEnv
from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import load_template
from expert_v1.native_seed_search import resolve_native_seed
from .prepare_600k import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=39431);p.add_argument('--maximum-seeds',type=int,default=128)
    a=p.parse_args();value=json.loads(a.source.read_bytes());started=time.monotonic()
    if a.output.exists():raise ValueError('preserve prior probe')
    plan=compile_battle({**value,'schema_version':3})
    plan=replace(plan,source_schema_version=value['schema_version'],native_replay_ready=False,original_state_exact=False,
                 replay_tier='diagnostic_source_without_native_contract')
    template=load_template(Path(__file__).resolve().parents[1]/'examples/eight-card-bootstrap.json')
    report=dict(source=str(a.source),source_sha256=digest(a.source),maximum_seeds=a.maximum_seeds,
                training_ready=False,source_seed_recovered=False,method='bounded native seed search preserving source deck order/form flags')
    with NativeRoyaleEnv(host=a.host,port=a.port,timeout=30) as env:
        try:
            result=resolve_native_seed(env,plan,template,preferred_seed=1,maximum_seeds_to_test=a.maximum_seeds,warmup_tick=10)
            report.update(success=True,chosen_seed=result.chosen_seed,seeds_tested=result.seeds_tested,
                          native_resets=result.native_resets,observed_players=result.state['players'],both_cycles_compatible=True)
        except Exception as error:report.update(success=False,error=type(error).__name__+': '+str(error))
    report['elapsed_seconds']=time.monotonic()-started;atomic_json(a.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
