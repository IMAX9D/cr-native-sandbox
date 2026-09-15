from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from data_pipeline.processing_contract import prepare_for_native,compile_processing_battle,ProcessingContract,MODE_PROVENANCE
from expert_v1.native_ingest_contract import load_native_ingest_contract
from expert_v1.native_replay_plan import compile_battle

from data_pipeline.processing_contract import RESOURCE_ROOT
DOWNLOADER=Path(os.environ.get('ROYALEAPI_DOWNLOADER_ROOT', str(Path(__file__).resolve().parents[2]/'royaleapi-downloader')))
pytestmark=pytest.mark.skipif(not (DOWNLOADER/'crawler/authoritative.py').exists() or not (RESOURCE_ROOT/'game_modes.csv').exists(),reason='external downloader validator and matching game_modes.csv required')


def fixture():
    value=json.loads((Path(__file__).parent/'fixtures/expert_schema5_authoritative.json').read_text('utf-8'))
    for side in ('team','opponent'):
        value['elixir_stats'][side]['Total']={'count':sum(p['side']==side for p in value['card_plays'])}
    return value


def test_complete_deck_need_not_have_played_every_card():
    source=fixture();last=max(i for i,p in enumerate(source['card_plays']) if p['side']=='opponent')
    source['card_plays'].pop(last)
    for i,p in enumerate(source['card_plays']):p['marker_index']=i
    source['elixir_stats']['opponent']['Total']['count']-=1
    preserved=deepcopy(source);value=prepare_for_native(source,DOWNLOADER);plan=compile_processing_battle(value)
    assert source==preserved
    assert value['processing_eligibility']['processing_exceptions']==['opponent_incomplete_observed_8_card_cycle']
    assert value['processing_eligibility']['cycle_evidence']['opponent']['compatible_initial_state_count']>0
    assert len(plan.sides[1].deck)==8 and not plan.original_state_exact
    assert 'processing' in plan.state_provenance
    assert value.get('authoritative_native_contract')==source.get('authoritative_native_contract')
    with pytest.raises(ValueError):compile_battle(value)


def test_resource_supported_cw_is_direct_route_not_friendly_fallback():
    source=fixture();source['numeric_game_mode_id']=72000268;source['battle_type']='riverRacePvP'
    value=prepare_for_native(source,DOWNLOADER);plan=compile_processing_battle(value)
    assert plan.numeric_game_mode_id==plan.native_execution_game_mode_id==72000268
    assert plan.native_execution_game_mode_provenance==MODE_PROVENANCE
    assert value['processing_eligibility']['downloader_original_result']['accepted'] is False
    assert value['processing_eligibility']['native_execution_verified'] is False
    assert load_native_ingest_contract().validate_game_mode(72000268)  # Original strict downloader contract unchanged.


def test_unknown_mode_and_missing_levels_fail_closed():
    source=fixture();source['numeric_game_mode_id']=72000001
    with pytest.raises(ValueError,match='mode_not_allowed'):prepare_for_native(source,DOWNLOADER)
    source=fixture();source['rounds'][0]['opponent'][0]['tower_troop_level']=0
    source['final_tower_hp']['opponent']['king']=1000
    hp=source['final_tower_hp']['opponent'];hp['total']=hp['king']+hp['princess0']+hp['princess1']
    with pytest.raises(ValueError,match='king_tower_level'):prepare_for_native(source,DOWNLOADER)
    source=fixture();source['schema_version']=3
    with pytest.raises(ValueError,match='schema-3'):prepare_for_native(source,DOWNLOADER)


def test_incompatible_partial_cycle_is_not_admitted():
    source=fixture();first=source['card_plays'][0]
    own=[p for p in source['card_plays'] if p['side']==first['side']]
    for p in own[1:]:
        p['card']=first['card'];p['card_base']=first.get('card_base',first['card']);p['card_form']=first.get('card_form',first['card'])
    with pytest.raises(ValueError):prepare_for_native(source,DOWNLOADER)
