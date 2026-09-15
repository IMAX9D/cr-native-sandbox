"""Bounded, read-only source/schema/runtime audit; never guesses missing history."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sqlite3
import sys

from .prepare_600k import digest,atomic_json
from .runtime_registry import inventory
from .processing_contract import resource_mode_evidence


def source_sample(path,downloader_root):
    root=Path(downloader_root).resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from crawler.authoritative import upgrade_exact_replay_body
    path=Path(path);value=json.loads(path.read_bytes());schema=value.get('schema_version')
    upgraded,error=upgrade_exact_replay_body(value) if schema in (3,4) else (None,'not_legacy_schema')
    missing=[name for name in ('numeric_game_mode_id','numeric_game_mode_provenance','matchup_players',
        'normal_1v1','battle_index','battle_index_provenance','version_timestamp','version_timestamp_provenance',
        'battle_time_utc','battle_time_utc_provenance','final_tower_hp','deck_crosscheck_complete') if not value.get(name)]
    for side in ('team','opponent'):
        players=value.get('rounds',[{}])[0].get(side,[])
        if len(players)!=1:missing.append(side+'.single_player');continue
        for name in ('tower_troop_level','king_tower_level','king_tower_level_provenance'):
            if not players[0].get(name):missing.append(side+'.'+name)
    return dict(path=str(path.resolve()),sha256=digest(path),battle_tag=value.get('battle_tag'),schema=schema,
        original_timestamp=value.get('timestamp'),timestamp_not_client_build=True,
        exact_body_upgrade_possible=upgraded is not None,body_upgrade_error=error,
        deployment_count=len(value.get('card_plays',[])),ability_count=len(value.get('ability_plays',[])),
        missing_processing_metadata=missing,source_list_url=value.get('deck_metadata',{}).get('source_list_url'),
        full_source_retained=True,training_ready=False)


def run(catalog,archive,downloader_root,sources):
    catalog=Path(catalog).resolve()
    with sqlite3.connect(catalog.as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('select schema_version,count(*),min(timestamp),max(timestamp) from battles where archive=? group by schema_version',(archive,)).fetchall()
        total=db.execute('select count(*) from battles').fetchone()[0]
    utc=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
    registry=inventory();samples=[source_sample(p,downloader_root) for p in sources]
    supplementary=[]
    old_arm=Path(os.environ.get('CR_NATIVE_LEGACY_ARM_LIBRARY', str(Path(__file__).resolve().parents[1]/'local/legacy-arm64/libg.so')))
    if old_arm.exists():supplementary.append(dict(client_version='150535029',abi='arm64-v8a',libg=str(old_arm),libg_sha256=digest(old_arm),
        resource_pairing_certified=False,linux_r0_capture_certified=False,role='additional official ABI binary, not another historical update snapshot'))
    return dict(kind='r0-source-runtime-coverage-audit.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        archive=archive,catalog=str(catalog),catalog_total_including_other_archives=total,
        primary_schema_counts=[dict(schema=s,count=n,earliest_utc=utc(lo),latest_utc=utc(hi)) for s,n,lo,hi in rows],
        primary_total=sum(r[1] for r in rows),samples=samples,sample_scope='Four deliberately selected exact legacy bodies; not a population pass-rate estimate.',
        global_mode_distribution='Not present in catalog; deliberately not estimated without scanning JSON payloads.',
        runtime_registry=registry,supplementary_binaries=supplementary,
        distinct_libg_hashes_in_three_resource_entries=len({e['libg_sha256'] for e in registry['entries']}),
        mode_static_evidence=resource_mode_evidence(),
        implementation_actions=[
            dict(priority=1,task='Use explicit processing contract for complete known eight-card decks with unplayed cards',
                status='implemented',files=['data_pipeline/processing_contract.py','expert_v1/native_replay_plan.py'],
                remaining='Native compatible-seed verification; retain ambiguous initial hand prefix provenance.'),
            dict(priority=2,task='Add resource-supported CW_Battle_1v1 direct mode 72000268',status='implemented_static_route_only',
                remaining='Verify native startup/timeline/tower levels; do not fill missing tower troop/King levels.'),
            dict(priority=3,task='Recover schema3 list metadata',status='body_upgrade_available_but_metadata_missing_in_four_samples',
                existing_entry='crawler.authoritative.upgrade_exact_replay_body + crawler.authoritative_manifest.prepare_upgrade_groups',
                remaining='Read local original list metadata cache or re-fetch exact battle-tag list metadata; old page no longer containing tag is not proof.'),
            dict(priority=4,task='Route newer cards/mechanics to ARM64 160402002 + correct resource overlay',status='archives_present_linux_backend_unproven',
                remaining='ARM64/Bionic dependency/SCID path, function relocations, state fields and full battle validation; no silent fallback to old x86.'),
            dict(priority=5,task='Historical effective interval ledger',status='missing',
                remaining='Record verified libg+ABI+resource fingerprint+overlay dependencies+mode+effective time evidence; capture date is not patch effective date.')],
        guardrails=['Original sources are never rewritten/deleted.','Successful static compatibility is not exact historical reproduction.',
            'A body-only schema upgrade cannot manufacture mode, final HP, tower/King levels.','Three resource archives are not three distinct libg versions.'],
        training_ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--archive',default='royaleapi-600000.tar.xz')
    p.add_argument('--downloader-root',type=Path,required=True);p.add_argument('--source',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('keep earlier audit evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True);result=run(a.catalog,a.archive,a.downloader_root,a.source);atomic_json(a.output,result)
    print(json.dumps(dict(primary_total=result['primary_total'],schemas=result['primary_schema_counts'],samples=len(result['samples']),output=str(a.output)),indent=2))


if __name__=='__main__':main()
