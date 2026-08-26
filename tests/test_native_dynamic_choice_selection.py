from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android_probe" / "native" / "jni_bridge.cpp"


class NativeDynamicChoiceSelectionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BRIDGE.read_text(encoding="utf-8")
        cls.helper = cls.source.split("int32_t build_deployment_selection", 1)[1].split(
            "const char* native_data_string_chars", 1
        )[0]
        cls.action = cls.source.split(
            "Java_royale_nativehost_JniHost_nativeAct", 1
        )[1].split("Java_royale_nativehost_JniHost_nativeUseAbility", 1)[0]
        cls.grid = cls.source.split(
            "Java_royale_nativehost_JniHost_nativeProbeGrid", 1
        )[1].split("static jstring observe_state_json", 1)[0]

    def test_choice_capability_is_native_vtable_not_card_id(self) -> None:
        self.assertIn("kDynamicChoiceSpellVtableRva", self.helper)
        self.assertIn("reinterpret_cast<uintptr_t>(entry) + 0x10", self.helper)
        self.assertNotIn("26000104", self.helper)
        self.assertNotIn("26000105", self.helper)
        self.assertNotIn("28000025", self.helper)

    def test_exactly_one_native_builder_is_selected(self) -> None:
        self.assertIn("const uintptr_t builder_rva = dynamic_choice", self.helper)
        self.assertEqual(self.helper.count("build(output, entry, player, mode)"), 1)
        self.assertIn("return dynamic_choice ? 1 : 0", self.helper)

    def test_act_and_grid_fail_closed_and_share_helper(self) -> None:
        for section in (self.action, self.grid):
            self.assertIn("build_deployment_selection(", section)
            self.assertIn("selection_strategy < 0", section)
            self.assertIn("selection_strategy == 1", section)
            self.assertIn("selection_strategy", section)
            self.assertIn("selection_builder_rva", section)
            self.assertIn("selection_form_index", section)
        self.assertIn("resolved_data_id", self.grid)
        self.assertIn("card_cost_raw", self.grid)


if __name__ == "__main__":
    unittest.main()
