"""Start a new BC run from source-bound release weights, without resetting heads."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from policy_v1.data import digest


def prepare_restart(checkpoint, contract_path, data):
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8-sig'))
    source_hash = digest(checkpoint)
    if contract.get('kind') != 'hokoff_match_encoder_contract_v1':
        raise ValueError('unsupported release encoder contract')
    if source_hash not in contract['allowed_weights']:
        raise ValueError('release weight hash is not bound to encoder contract')
    manifest_path = Path(data)/'manifest.json'
    if digest(manifest_path) != contract['source_manifest_sha256']:
        raise ValueError('dataset manifest differs from release source; obtain the original dataset')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    for key in ('card_vocabulary', 'ability_vocabulary', 'dimensions'):
        if manifest[key] != contract['encoder'][key]:
            raise ValueError('release encoder differs from dataset: '+key)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if digest(checkpoint) != source_hash:
        raise ValueError('source weights changed during loading')
    if not isinstance(saved.get('step'), int) or saved['step'] < 0:
        raise ValueError('invalid source step')
    from .train_fixed import FixedConfig
    if saved['config'].get('architecture') != FixedConfig.architecture:
        raise ValueError('release must be a fixed-period model')
    config = FixedConfig(**saved['config'])
    for key in ('card_vocab_size', 'ability_vocab_size', 'grid_channels',
                'public_scalar_size', 'entity_numeric_size'):
        if getattr(config, key) != manifest['dimensions'][key]:
            raise ValueError('release model dimension differs: '+key)
    if not all(bool(torch.isfinite(v).all()) for v in saved['model'].values() if v.is_floating_point()):
        raise ValueError('release contains non-finite parameters')
    return saved, dict(source_step=saved['step'], source_sha256=source_hash,
                       encoder_contract_sha256=digest(contract_path),
                       source_manifest_sha256=digest(manifest_path),
                       optimizer_initialization='fresh', step_semantics='updates_since_weights_restart')


def initialize_weights(config, *, saved):
    from .train_fixed import FixedConfig, FixedPolicy
    if asdict(FixedConfig(**saved['config'])) != asdict(config):
        raise ValueError('release model configuration differs from training configuration')
    model = FixedPolicy(config)
    model.load_state_dict(saved['model'], strict=True)
    return model
