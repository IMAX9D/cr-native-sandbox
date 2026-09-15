"""Full feature names from the attributed FirstLight V4 snapshot; no truncation."""
import json
from pathlib import Path

REFERENCE_FEATURES=json.loads((Path(__file__).with_name('data')/'reference_features.json').read_text(encoding='utf-8'))
FEATURES={name:tuple(values) for name,values in REFERENCE_FEATURES['features'].items()}
CHILD_FEATURES=FEATURES['CHILD_FEATURE_NAMES']
TOWER_FEATURES=FEATURES['TOWER_FEATURE_NAMES']
GROUP_FEATURES=FEATURES['GROUP_FEATURE_NAMES']
CARD_FEATURES=FEATURES['CARD_RUNTIME_FEATURE_NAMES']
MATCH_FEATURES=FEATURES['MATCH_SCALAR_NAMES']
COMBAT_FEATURES=FEATURES['EVENT_FEATURE_NAMES']
ABILITY_FEATURES=FEATURES['ABILITY_FEATURE_NAMES']
COMBAT_KINDS=tuple(REFERENCE_FEATURES['event_kinds'])
