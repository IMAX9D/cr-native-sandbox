"""Explicit processing compatibility, distinct from downloader/historical proof.

Retained complete decks may contain unplayed cards. The acquisition filter is
unchanged; a nonempty compatible-cycle set replaces only its all-eight-observed
filter. Native seed/hand checks remain mandatory.
"""
import copy
import csv
from dataclasses import asdict,replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from expert_v1.native_ingest_contract import DEFAULT_CONTRACT_PATH,load_native_ingest_contract,NATIVE_EXECUTION_GAME_MODE_PROVENANCE,ValidationIssue
from expert_v1.native_replay_plan import compatible_cycle,split_card_token
from training_r0.build_native_semantics import decode

MODE_PROVENANCE='processing_resource_mode_route_v1'
RUNTIME_SHA='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
RESOURCE_ROOT=Path(os.environ.get('CR_NATIVE_RESOURCE_ROOT', str(Path(__file__).resolve().parents[1]/'local/resources/csv_logic')))
NONCOMBAT_MODE_DIFFERENCES={'Name','Icon','DescriptionTID','ValidLadderMode','ShowTrophies','AllowBotPractice'}


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def resource_mode_evidence(root=RESOURCE_ROOT):
    path=Path(root)/'game_modes.csv';raw=path.read_bytes()
    rows=[r for r in csv.DictReader(io.StringIO(decode(path))) if r['Name'] and r['Name'].lower()!='string']
    ladder,cw=rows[6],rows[268]
    differences={k:[ladder[k],cw[k]] for k in ladder if ladder[k]!=cw[k]}
    if ladder['Name']!='Ladder' or cw['Name']!='CW_Battle_1v1' or set(differences)-NONCOMBAT_MODE_DIFFERENCES:
        raise ValueError('CW direct processing route no longer matches pinned combat resource assumptions')
    return dict(resource_path=str(path.resolve()),resource_sha256=hashlib.sha256(raw).hexdigest(),
        source_mode=72000268,execution_mode=72000268,source_name=cw['Name'],timeline=cw['BattleTimeline'],
        reference_mode=72000006,differences=differences,native_execution_verified=False,
        proof_scope='Static combat fields match; direct native mode startup, levels and timing must still be verified.')


def processing_document():
    base=load_native_ingest_contract();routes=dict(base.native_execution_mode_by_source);routes[72000268]=72000268
    value=dict(kind='cr-native-processing-contract.v1',schema_version=3,game_version='15.535.29',
        runtime_sha256=RUNTIME_SHA,base_contract_sha256=base.value['contract_sha256'],base_contract_file_sha256=base.file_sha256,
        routes={str(k):v for k,v in routes.items()},mode_evidence=resource_mode_evidence(),
        cycle_policy='complete_source_eight_card_deck_and_exact_play_sequence_with_nonempty_compatible_initial_state_set',
        native_seed_calibration_required=True,original_state_exact=False,historical_ruleset_certified=False)
    value['contract_sha256']=_hash(value)
    return value


class ProcessingContract:
    """Planner-facing explicit extension; original acquisition contract is intact."""
    def __init__(self):
        self.value=processing_document();self.file_sha256=_hash(self.value)
        self.base=load_native_ingest_contract();self.routes={int(k):v for k,v in self.value['routes'].items()}
    def __getattr__(self,name):return getattr(self.base,name)
    def validate_execution_game_mode(self,source,execution,provenance):
        if provenance!=MODE_PROVENANCE:return (ValidationIssue('native_execution_game_mode_provenance','processing_mode_provenance_invalid',provenance),)
        expanded=replace(self.base,source_numeric_game_mode_ids=frozenset(self.routes),native_execution_mode_by_source=self.routes)
        return expanded.validate_execution_game_mode(source,execution,NATIVE_EXECUTION_GAME_MODE_PROVENANCE)
    def validate_king_tower_level_evidence(self,**kwargs):
        provenance=kwargs.get('provenance','')
        if provenance.startswith('processing_template16_'):
            kwargs['provenance']=provenance.replace('processing_template16_','ranked_template_cap16_',1)
        return self.base.validate_king_tower_level_evidence(**kwargs)


def complete_deck_cycle_evidence(value):
    evidence={}
    for side in ('team','opponent'):
        tokens=value[side+'_deck'];bases=[split_card_token(t)[0] for t in tokens]
        player=value['rounds'][0][side][0]
        if len(tokens)!=8 or len(set(bases))!=8 or player['full_deck']!=tokens:
            raise ValueError('compatible-cycle processing needs independently observed complete eight-card deck')
        plays=[p for p in value['card_plays'] if p['side']==side]
        indices=[bases.index(split_card_token(p['card'])[0]) for p in plays]
        cycle=compatible_cycle(indices)
        evidence[side]=dict(**asdict(cycle),observed_distinct_cards=len(set(indices)),
            unplayed_known_cards=[tokens[i] for i in range(8) if i not in indices],
            full_deck_source=value['deck_metadata']['source'],source_initial_hand_recovered=False,
            native_seed_calibration_required=True)
    return evidence


def prepare_for_native(source,downloader_root):
    root=Path(downloader_root).resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from crawler.authoritative import load_native_contract,apply_native_contract_metadata,evaluate_native_eligibility
    base=load_native_contract(DEFAULT_CONTRACT_PATH,expected_game_version='15.535.29');processing=ProcessingContract()
    contract=replace(base,source_numeric_game_mode_ids=frozenset(processing.routes),native_execution_mode_by_source=processing.routes)
    value=copy.deepcopy(source)
    if value.get('schema_version')!=5:raise ValueError('use separately audited schema-3 upgrade and recover missing list metadata before native admission')
    metadata=value.get('deck_metadata',{})
    if metadata.get('source')!='battle_list_html' or metadata.get('complete') is not True:raise ValueError('source deck metadata incomplete')
    metadata['authoritative_complete']=True  # Private validator input, removed below.
    original_gate_value=copy.deepcopy(value)
    apply_native_contract_metadata(original_gate_value,base)
    original_eligibility=evaluate_native_eligibility(original_gate_value,native_contract=base)
    apply_native_contract_metadata(value,contract)
    eligibility=evaluate_native_eligibility(value,native_contract=contract)
    allowed_reasons={s+'_incomplete_observed_8_card_cycle' for s in ('team','opponent')}
    if not eligibility.accepted and (eligibility.tier!='events' or set(eligibility.reasons)-allowed_reasons):
        raise ValueError('native static eligibility: '+','.join(eligibility.reasons))
    cycles=complete_deck_cycle_evidence(value)
    metadata.pop('authoritative_complete',None);metadata['processing_complete']=True
    value['native_execution_game_mode_provenance']=MODE_PROVENANCE
    if value['numeric_game_mode_id']==72000268:
        for side in ('team','opponent'):
            player=value['rounds'][0][side][0]
            player['king_tower_level_provenance']=player['king_tower_level_provenance'].replace('ranked_template_cap16_','processing_template16_',1)
    value['processing_native_contract']=dict(game_version=processing.value['game_version'],contract_sha256=processing.value['contract_sha256'],contract_file_sha256=processing.file_sha256)
    value['processing_eligibility']=dict(status='accepted',gate='processing_static_v1',native_execution_verified=False,
        downloader_original_result=original_eligibility.as_dict(),processing_validator_result=eligibility.as_dict(),
        processing_exceptions=list(eligibility.reasons),cycle_evidence=cycles)
    value['processing_provenance']=dict(source_schema_unchanged=True,source_historical_runtime_not_inferred=True,
        validator_sha256=hashlib.sha256((root/'crawler/authoritative.py').read_bytes()).hexdigest(),
        processing_contract=processing.value,processing_runtime='150535029-x86_64',
        note='Processing compatibility only; native execution/seed/fields and historical rule matching are separate gates.')
    return value


def compile_processing_battle(value):
    from expert_v1.native_replay_plan import compile_battle
    return compile_battle(value,processing_contract=ProcessingContract())
