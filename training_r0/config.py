"""Versioned R0 defaults; reference-inspired, not historical checkpoint settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

NATIVE_HZ = 20
DECISION_TICKS = 5
HEIGHT, WIDTH = 32, 18
CELLS = HEIGHT * WIDTH
OFFSET_TICKS = (0, 1, 2, 3, 4)
ENTITY_FEATURE_NAMES = ('x_over_18000','y_over_32000','hp_fraction','hp_known','level_over_16','level_known','is_unit','is_king','is_princess','own_ability_available')
CANDIDATE_FEATURE_NAMES = ('cost_over_10','hand_slot_over_3','source_x_over_18000','source_y_over_32000','source_position_known','requires_grid','observed_form_known','first_only')
EVENT_FEATURE_NAMES = ('observation_age_ticks_over_1200','raw_card_known','kind_known','target_x_over_18000','target_y_over_32000','target_known','action_occurred')
PUBLIC_SCALAR_NAMES = ('own_elixir_over_10','tick_over_6000','own_crowns_over_3','enemy_crowns_over_3','terminated','commands_allowed')
ENTITY_FEATURES = len(ENTITY_FEATURE_NAMES)
CANDIDATE_FEATURES = len(CANDIDATE_FEATURE_NAMES)
EVENT_FEATURES = len(EVENT_FEATURE_NAMES)
PUBLIC_SCALARS = len(PUBLIC_SCALAR_NAMES)
OBSERVATION_SCHEMA = 'r0-public-observation.v1'
ACTION_SCHEMA = 'r0-two-micro-actions.v1'


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Temperatures:
    gate: float = 1.0
    action: float = 1.0
    continuation: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError('temperatures must be finite and positive')


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    layers: int = 4
    heads: int = 8
    hidden: int = 512
    spatial_channels: int = 64
    max_entities: int = 256  # Includes public towers; never silently truncate.
    max_candidates: int = 6
    max_history: int = 32

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('model dimensions must be positive integers')
        if self.width % self.heads or self.max_candidates < 1:
            raise ValueError('width must be divisible by attention heads')


@dataclass(frozen=True)
class TrainingConfig:
    collection_seconds: int = 40
    tbptt_steps: int = 48
    gamma: float = 0.9997
    gae_lambda: float = 0.98
    learning_rate: float = 1e-5
    clip_range: float = 0.1
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01

    def __post_init__(self):
        if type(self.collection_seconds) is not int or type(self.tbptt_steps) is not int or min(self.collection_seconds, self.tbptt_steps) <= 0:
            raise ValueError('segment and TBPTT sizes must be positive integers')
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError('discount and trace lambda must be in (0,1]')
        for name in ('learning_rate', 'clip_range', 'max_grad_norm', 'target_kl'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'invalid {name}')
        for name in ('value_coefficient', 'entropy_coefficient'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'invalid {name}')

    @property
    def collection_steps(self) -> int:
        return self.collection_seconds * NATIVE_HZ // DECISION_TICKS
