from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import load_template
from expert_v1.native_seed_search import clear_native_seed_cache, resolve_native_seed
from native_core.env import NativeRoyaleEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
    r"\source-round-03-selected-schema3-20260826\raw\battles"
)
REGRESSIONS = (
    (SOURCE_ROOT / "00" / "008YLPVGR8GR.json", 9),
    (SOURCE_ROOT / "02" / "022YYLPR8C0R.json", 13),
    (SOURCE_ROOT / "02" / "02PYPJJRY9VG.json", 53),
)


@unittest.skipUnless(
    os.environ.get("CR_NATIVE_INTEGRATION") == "1",
    "requires a live frozen-libg Worker and the local expert source corpus",
)
class NativeSeedSearchIntegrationTest(unittest.TestCase):
    def test_three_form_layout_oscillation_regressions(self) -> None:
        port = int(os.environ.get("CR_NATIVE_PORT", "38032"))
        template = load_template(PROJECT_ROOT / "examples" / "eight-card-bootstrap.json")
        clear_native_seed_cache()
        with NativeRoyaleEnv(port=port, timeout=60.0) as env:
            for source_path, expected_seed in REGRESSIONS:
                source = json.loads(source_path.read_text(encoding="utf-8-sig"))
                plan = compile_battle(
                    source,
                    terminal_crowns=(
                        int(source["team_crowns"]),
                        int(source["opponent_crowns"]),
                    ),
                )
                result = resolve_native_seed(
                    env,
                    plan,
                    template,
                    preferred_seed=424_242,
                    maximum_seeds_to_test=100,
                    warmup_tick=10,
                )
                self.assertEqual(result.chosen_seed, expected_seed)
                self.assertEqual(result.seeds_tested, expected_seed)
                self.assertFalse(result.source_seed_recovered)
                self.assertEqual(
                    result.mappings, (tuple(range(8)), tuple(range(8)))
                )


if __name__ == "__main__":
    unittest.main()
