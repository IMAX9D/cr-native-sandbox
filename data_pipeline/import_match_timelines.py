"""Extract version-pinned mode/timeline facts, including array continuation rows.

The source tables, not source replay duration or final outcome, define clocks.
Reference: FirstLight CR timeline.py, Apache-2.0, commit 28d66cc0.
"""
import argparse
import csv
import io
from pathlib import Path
from training_r0.build_native_semantics import decode
from .prepare_600k import atomic_json, digest


def build(root):
    timelines = {}; current = None
    for row in csv.DictReader(io.StringIO(decode(root/'battle_timelines.csv'))):
        name = row['Name']
        if name.lower() == 'string': continue
        if name:
            current = dict(starting_elixir=int(row['StartingElixir'] or 0), sections=[], rates=[])
            timelines[name] = current
        if current is None: raise ValueError('timeline continuation before name')
        if row['SectionLength'] and row['SectionType']:
            current['sections'].append(dict(ticks=int(row['SectionLength'])*20, phase=row['SectionType'].lower()))
        if row['ElixirRateLength'] and row['ElixirFullBarMS']:
            current['rates'].append(dict(ticks=int(row['ElixirRateLength'])*20,
                full_bar_ms=int(row['ElixirFullBarMS']), visible=int(row['ElixirRateVisible']) if row['ElixirRateVisible'] else None))
    modes = {}
    for row in csv.DictReader(io.StringIO(decode(root/'game_modes.csv'))):
        name = row['Name']
        if not name or name.lower() == 'string': continue
        modes[str(72000000+len(modes))] = dict(name=name, timeline=row['BattleTimeline'])
    for mode in modes.values():
        if mode['timeline'] not in timelines: raise ValueError('mode has missing timeline')
    for timeline in timelines.values():
        if not timeline['sections'] or not timeline['rates']: raise ValueError('incomplete timeline')
        if any(r['full_bar_ms'] <= 0 or r['ticks'] < 0 for r in timeline['rates']): raise ValueError('invalid elixir schedule')
    return dict(schema='r0-match-timelines.v1', runtime_sha256='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba',
        sources={n:dict(path=str(root/n),sha256=digest(root/n)) for n in ('game_modes.csv','battle_timelines.csv')},modes=modes,timelines=timelines)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing timeline artifact')
    atomic_json(a.output,build(a.source))
