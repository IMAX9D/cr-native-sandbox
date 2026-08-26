from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android_probe" / "native" / "jni_bridge.cpp"


class NativeActionGuardDiagnosticsContractTest(unittest.TestCase):
    """Pin the fail-closed native action diagnostic ABI.

    The fields are intentionally sourced from the same frozen libg structures
    used by D8D520.  These source-contract checks complement the Android bridge
    compile and keep a future cleanup from silently dropping rejection evidence.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BRIDGE.read_text(encoding="utf-8")
        cls.action = cls.source.split(
            "Java_royale_nativehost_JniHost_nativeAct", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeUseAbility", 1
        )[0]

    def test_code_4_and_13_have_evidence_backed_reasons(self) -> None:
        self.assertIn('result_code == 4', self.action)
        self.assertIn('"battle_command_gate"', self.action)
        self.assertIn('result_code == 13', self.action)
        self.assertIn('"insufficient_elixir"', self.action)

    def test_guard_snapshot_precedes_authoritative_execute(self) -> None:
        snapshot = self.action.index("const bool command_gate_before")
        execute = self.action.index("const int32_t result_code")
        self.assertLess(snapshot, execute)
        for field in (
            '"guard_before"',
            '"hard_gate"',
            '"command_gate"',
            '"logic_end_counter_198"',
            '"logic_player_count_60"',
            '"mode_counter_194"',
        ):
            self.assertIn(field.replace('"', '\\"'), self.action)

    def test_resource_snapshot_matches_native_cost_guard_inputs(self) -> None:
        self.assertIn("player_address + 0x2F8", self.action)
        self.assertIn("static_cast<uint32_t>(packed_selection) >> 28", self.action)
        for field in (
            '"resource_before"',
            '"elixir_raw"',
            '"card_cost"',
            '"deficit_raw"',
            '"refill_timer"',
            '"next_deck_index"',
        ):
            self.assertIn(field.replace('"', '\\"'), self.action)


if __name__ == "__main__":
    unittest.main()
