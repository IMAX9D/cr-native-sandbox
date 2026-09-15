"""Bind a compiled dataset to the current R0 contract and reject raw pilots.

Validation is deliberately distinct from data retention. Unknown or partial
native captures remain useful source artifacts, not accepted training shards.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from training_r0.config import ModelConfig,NATIVE_HZ,DECISION_TICKS,OFFSET_TICKS,OBSERVATION_SCHEMA,ACTION_SCHEMA
from training_r0.feature_contract import FEATURES
from training_r0.elite_tensorizer import REQUIRED_CAPTURE
from training_r0.catalog import CardVocabulary
from training_r0.observation import ObservationBuilder


class R0AdmissionError(ValueError):pass


def current_contract():
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/'training_r0').glob('*.py'))+sorted((root/'training_r0'/'data').glob('*.json'))
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config=ModelConfig();builder=ObservationBuilder(CardVocabulary.from_native(),config)
    contract=dict(kind='r0-training-data-contract.v1',observation=OBSERVATION_SCHEMA,actions=ACTION_SCHEMA,
        native_hz=NATIVE_HZ,decision_ticks=DECISION_TICKS,offset_ticks=list(OFFSET_TICKS),micro_actions=2,
        features={k:list(v) for k,v in FEATURES.items()},required_capture=list(REQUIRED_CAPTURE),
        model_config=asdict(config),observation_schema_hash=builder.schema_hash,source_hashes=hashes)
    canonical=json.dumps(contract,sort_keys=True,separators=(',',':')).encode()
    return dict(contract,sha256=hashlib.sha256(canonical).hexdigest())


def require_manifest(manifest,contract=None):
    contract=current_contract() if contract is None else contract
    issues=[]
    if manifest.get('kind')!='r0-compiled-sequences.v1':issues.append('not_compiled_r0_sequences')
    if manifest.get('contract_sha256')!=contract['sha256']:issues.append('r0_contract_mismatch')
    if manifest.get('allow_incomplete_capture') is not False:issues.append('incomplete_capture_not_for_production')
    if manifest.get('source_stage')!='validated_raw_capture':issues.append('raw_capture_not_validated')
    checks=manifest.get('checks',{})
    for name in ('source_hashes','native_identity','ruleset_resolved','opening_cycle_compatible','tick_zero_capture',
                 'continuous_frames','continuous_events','entity_generation_identity','public_information_only',
                 'feature_applicability_and_known_masks','legal_candidate_alignment','rejected_action_loss_masks',
                 'action_offsets','sequence_boundaries','battle_level_split','shard_readback','strict_r0_forward'):
        if checks.get(name) is not True:issues.append(name+'_not_passed')
    certificates=manifest.get('capture_certificates',{})
    for capability in contract['required_capture']:
        proof=certificates.get(capability,{})
        if proof.get('status')!='passed' or not proof.get('evidence_sha256'):issues.append('capture_'+capability+'_not_certified')
    if not manifest.get('shards'):issues.append('no_training_shards')
    if issues:raise R0AdmissionError('; '.join(issues))


def require_batch(batch,model):
    import torch
    if model.config.allow_incomplete_capture:raise R0AdmissionError('diagnostic R0 cannot certify training data')
    model._check_batch(batch)
    banks=(batch.semantic.elite.child,batch.semantic.elite.tower,batch.semantic.elite.groups,
           batch.semantic.elite.own_cards,batch.semantic.elite.enemy_cards,batch.semantic.elite.match,
           batch.semantic.elite.combat_features,batch.semantic.elite.candidate_ability_features)
    for bank in banks:
        if not torch.isfinite(bank).all():raise R0AdmissionError('nonfinite R0 feature')
        if not ((bank[...,1::2]==0)|(bank[...,1::2]==1)).all():raise R0AdmissionError('invalid feature known bits')
    if not torch.isfinite(batch.semantic.elite.lifecycle).all():raise R0AdmissionError('nonfinite lifecycle')
    # False known bits may be legitimate N/A. Dataset-level applicability
    # certificates, not anonymous padding or nonzero counts, justify that.
    return True


def require_dataset_files(root,contract=None):
    """Verify referenced shard/certificate bytes, not just declaration booleans."""
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    require_manifest(manifest,contract)
    def verify(reference):
        if not isinstance(reference,dict):raise R0AdmissionError('invalid evidence record')
        path=(root/str(reference.get('path',''))).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise R0AdmissionError('evidence path missing/outside dataset')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=reference.get('sha256'):raise R0AdmissionError('evidence/shard SHA256 mismatch')
    for shard in manifest['shards']:verify(shard)
    for certificate in manifest['capture_certificates'].values():
        verify(dict(path=certificate.get('evidence_path'),sha256=certificate.get('evidence_sha256')))
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--export-contract',type=Path);p.add_argument('--check-dataset',type=Path)
    args=p.parse_args()
    if args.export_contract:
        contract=current_contract();args.export_contract.parent.mkdir(parents=True,exist_ok=True)
        with args.export_contract.open('x',encoding='utf-8') as f:json.dump(contract,f,ensure_ascii=False,indent=2)
        print(contract['sha256'])
    if args.check_dataset:require_dataset_files(args.check_dataset);print('R0 dataset manifest and referenced bytes verified; strict batch checks remain required at load time.')
