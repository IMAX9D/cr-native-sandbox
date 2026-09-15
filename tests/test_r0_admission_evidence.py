"""Admission must reject tampered or unverifiable evidence, not just report it.

These are negative controls for the certificate pipeline: a manifest is only
usable when every referenced byte matches and every check was produced by a real
verification of the actual artifacts.
"""
import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from data_pipeline.r0_admission import R0AdmissionError, current_contract, require_dataset_files, require_manifest
from data_pipeline.r0_certify import stream_facts, verify_raw_checks


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _certified_dataset(root, contract):
    """A minimal but complete manifest: 17 checks, 12 certificates, one shard."""
    root = Path(root)
    (root / 'evidence').mkdir(parents=True, exist_ok=True)
    shard = root / 'train' / 'shard.npz'
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b'not-a-real-npz-but-hashed')
    certificates = {}
    for capability in ('entity_runtime', 'archetypes', 'tower_runtime', 'groups', 'combat_events',
                       'card_runtime', 'match_runtime', 'commands', 'effects', 'relations',
                       'geometry', 'lifecycle'):
        evidence = root / 'evidence' / f'capability-{capability}.json'
        evidence.write_text(json.dumps(dict(capability=capability, status='passed')), encoding='utf-8')
        certificates[capability] = dict(status='passed', capability=capability,
                                        evidence_path=str(evidence.relative_to(root)),
                                        evidence_sha256=_digest(evidence))
    checks = ('source_hashes', 'native_identity', 'ruleset_resolved', 'opening_cycle_compatible',
              'tick_zero_capture', 'continuous_frames', 'continuous_events', 'entity_generation_identity',
              'public_information_only', 'feature_applicability_and_known_masks', 'legal_candidate_alignment',
              'rejected_action_loss_masks', 'action_offsets', 'sequence_boundaries', 'battle_level_split',
              'shard_readback', 'strict_r0_forward')
    manifest = dict(kind='r0-compiled-sequences.v1', contract_sha256=contract['sha256'],
                    allow_incomplete_capture=False, source_stage='validated_raw_capture',
                    checks={name: True for name in checks}, capture_certificates=certificates,
                    shards=[dict(path='train/shard.npz', sha256=_digest(shard))])
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root, manifest


def test_complete_manifest_is_accepted(tmp_path):
    contract = current_contract()
    root, manifest = _certified_dataset(tmp_path / 'dataset', contract)
    require_manifest(manifest, contract)
    require_dataset_files(root, contract)


def test_tampered_shard_is_rejected(tmp_path):
    contract = current_contract()
    root, manifest = _certified_dataset(tmp_path / 'dataset', contract)
    (root / 'train' / 'shard.npz').write_bytes(b'tampered-after-certification')
    with pytest.raises(R0AdmissionError, match='SHA256 mismatch'):
        require_dataset_files(root, contract)


def test_tampered_certificate_evidence_is_rejected(tmp_path):
    contract = current_contract()
    root, manifest = _certified_dataset(tmp_path / 'dataset', contract)
    (root / 'evidence' / 'capability-lifecycle.json').write_text('{"status": "passed"}', encoding='utf-8')
    with pytest.raises(R0AdmissionError, match='SHA256 mismatch'):
        require_dataset_files(root, contract)


def test_missing_certificate_or_check_is_rejected(tmp_path):
    contract = current_contract()
    root, manifest = _certified_dataset(tmp_path / 'dataset', contract)
    for mutate, expected in (
            (lambda m: m['capture_certificates'].pop('lifecycle'), 'capture_lifecycle_not_certified'),
            (lambda m: m['checks'].pop('ruleset_resolved'), 'ruleset_resolved_not_passed'),
            (lambda m: m.update(source_stage='retained_raw_capture_pending_admission'), 'raw_capture_not_validated'),
            (lambda m: m.update(contract_sha256='0' * 64), 'r0_contract_mismatch'),
    ):
        broken = json.loads(json.dumps(manifest))
        mutate(broken)
        with pytest.raises(R0AdmissionError, match=expected):
            require_manifest(broken, contract)


def _write_stream(root, damage_sequences, ticks):
    frames = []
    frames.append(dict(kind='episode', capture_mode='production_processing_contract',
                       identity=dict(libg_sha256='a' * 64, host_sha256='b' * 64, bridge_sha256='c' * 64,
                                     elite_telemetry='x86-v1'),
                       runtime_bundle=dict(libg_sha256='a' * 64, worker_files=[
                           dict(name='libg.so', sha256='a' * 64),
                           dict(name='lifecycle-probe.jar', sha256='b' * 64),
                           dict(name='libnative_host_bridge.so', sha256='c' * 64)])))
    for tick, sequences in zip(ticks, damage_sequences):
        frames.append(dict(kind='frame', state=dict(tick=tick),
                           telemetry=dict(epoch=1, gap=False, events=[
                               dict(sequence=value, tick=tick, source_key=5000001, attacker_key=5000001,
                                    target_key=5000002, source_side=0, target_side=1, target_kind=15,
                                    effective_damage=10, lethal=False) for value in sequences],
                               lifecycle=dict(gap=False, rejected=0, null_id_removals=0, events=[]))))
    payload = ('\n'.join(json.dumps(frame) for frame in frames) + '\n').encode()
    (root / 'native-frames.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(payload))
    (root / 'report.json').write_text(json.dumps(dict(status='native_terminal')), encoding='utf-8')


def _bundle(tmp_path, name='bundle.json'):
    path = tmp_path / name
    path.write_text(json.dumps(dict(libg_sha256='a' * 64, worker_files=[
        dict(name='libg.so', sha256='a' * 64), dict(name='lifecycle-probe.jar', sha256='b' * 64),
        dict(name='libnative_host_bridge.so', sha256='c' * 64)])), encoding='utf-8')
    return path


def test_deleted_damage_receipt_breaks_event_continuity(tmp_path):
    """Dropping one receipt leaves a sequence hole that must fail the check."""
    root = tmp_path / 'capture'
    root.mkdir()
    _write_stream(root, [[1, 2, 3], [4, 5]], [0, 5])
    bundle = _bundle(tmp_path)
    facts = stream_facts(root)
    assert facts['damage_gap_count'] == 0
    checks = verify_raw_checks(facts, current_contract(),
                               _certified_dataset(tmp_path / 'dataset', current_contract())[0], bundle)
    assert checks['continuous_events']['passed'] is True

    root2 = tmp_path / 'capture-deleted'
    root2.mkdir()
    _write_stream(root2, [[1, 2, 3], [4, 6]], [0, 5])   # receipt 5 deleted
    facts2 = stream_facts(root2)
    assert facts2['damage_gap_count'] == 1
    checks2 = verify_raw_checks(facts2, current_contract(),
                                _certified_dataset(tmp_path / 'dataset2', current_contract())[0], bundle)
    assert checks2['continuous_events']['passed'] is False


def test_swapped_bundle_identity_is_rejected(tmp_path):
    """Certifying a capture against a bundle it was not taken with must fail."""
    root = tmp_path / 'capture'
    root.mkdir()
    _write_stream(root, [[1]], [0])
    swapped = tmp_path / 'swapped.json'
    swapped.write_text(json.dumps(dict(libg_sha256='d' * 64, worker_files=[
        dict(name='libg.so', sha256='d' * 64), dict(name='lifecycle-probe.jar', sha256='e' * 64),
        dict(name='libnative_host_bridge.so', sha256='f' * 64)])), encoding='utf-8')
    checks = verify_raw_checks(stream_facts(root), current_contract(),
                               _certified_dataset(tmp_path / 'dataset', current_contract())[0], swapped)
    assert checks['native_identity']['passed'] is False
    assert checks['native_identity']['observed']['libg_sha256'] != checks['native_identity']['expected']


def test_template_default_ruleset_is_rejected(tmp_path, monkeypatch):
    """A capture whose payload equals the bootstrap template defaults was not
    resolved by the processing contract, so the check must fail."""
    root = tmp_path / 'capture'
    root.mkdir()
    _write_stream(root, [[1]], [0])
    bundle = _bundle(tmp_path)
    dataset = _certified_dataset(tmp_path / 'dataset', current_contract())[0]
    # No processing input and no plan: the resolver cannot confirm the ruleset.
    checks = verify_raw_checks(stream_facts(root), current_contract(), dataset, bundle)
    assert checks['ruleset_resolved']['passed'] is False


def test_cross_side_parent_is_rejected_by_the_adapter():
    """A lifecycle parent on the other side must fail capture, not be accepted."""
    from training_r0.replay_adapter import ReplayFeatureAssembler
    replay = {'battle': {'gamemode': 72000006,
                         **{f'deck{s}': {'sp': [{'d': 26000000, 'l': 10}] * 1} for s in (0, 1)}}}
    assembler = ReplayFeatureAssembler(replay, 'synthetic')

    def frame(tick, lifecycle_events):
        entities = [dict(id=f'0x{i}', category=5000000 + i, kind=15, side=i - 1, x=5000, y=10000, card_id=26000000,
                         hp=100, max_hp=100) for i in (1, 2)]
        details = [dict(key=e['category'], native_data_id=34000000, native_name='Knight',
                        created_tick_candidate=0) for e in entities]
        return dict(state=dict(tick=tick, coherent=True, entities=entities, effects=[], projectiles=[],
                               episode=dict(crown_towers=[])),
                    details=dict(tick=tick, objects=details),
                    telemetry=dict(epoch=1, rejected=0, unattributed=0, gap=False, events=[],
                                   lifecycle=dict(gap=False, rejected=0, events=lifecycle_events)),
                    masks=[])

    for tick in range(4):
        events = []
        if tick == 1:
            events = [dict(sequence=1, operation=1, key=3000001, side=0, tick=1, parent_key=5000002,
                           command_sequence=0, birth_tick=-1, native_data_id=22000001)]
        assembler.ingest(frame(tick, events))
    assert 'native_parent_owner_mismatch' in assembler.capture_failures
    assert assembler.issues.get('native_registration_parent_owner_mismatch') == 1
    assert assembler.damage.births[3000001].parent_key is None


def test_future_lifecycle_summary_is_rejected():
    """A lifecycle summary stamped in the future must not enter an input."""
    from dataclasses import replace
    from training_r0.catalog import CardVocabulary
    from training_r0.config import ModelConfig
    from training_r0.lifecycle import DamageSummary
    from training_r0.observation import ObservationBuilder
    from training_r0.synthetic import rich_public_frame
    builder = ObservationBuilder(CardVocabulary.from_native(), ModelConfig(allow_incomplete_capture=True))
    frame = rich_public_frame(builder)
    detail = frame.semantics.entities[0]
    future = DamageSummary(frame.semantics.episode_uid, detail.key, frame.semantics.tick + 100,
                           (0., 0., 0., 0., 0., 0.), True)
    spoiled = replace(detail, damage=future)
    broken = replace(frame, semantics=replace(frame.semantics, entities=(spoiled, *frame.semantics.entities[1:])))
    with pytest.raises(ValueError, match='future/stale lifecycle summary'):
        builder.batch([broken])
