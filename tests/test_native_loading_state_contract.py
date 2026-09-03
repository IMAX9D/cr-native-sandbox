from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "android_probe" / "native" / "jni_bridge.cpp"
HOST = ROOT / "android_probe" / "java" / "royale" / "nativehost" / "JniHost.java"


class NativeLoadingStateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.host = HOST.read_text(encoding="utf-8")
        cls.game_main_init = cls.bridge.split(
            "Java_royale_nativehost_JniHost_nativeInitGameMain", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeProbePrerequisites", 1
        )[0]
        cls.probe = cls.bridge.split(
            "Java_royale_nativehost_JniHost_nativeProbePrerequisites", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeInitResources", 1
        )[0]
        cls.preload = cls.bridge.split(
            "Java_royale_nativehost_JniHost_nativePreloadCoreData", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeProbePrerequisites", 1
        )[0]
        cls.init_resources = cls.bridge.split(
            "Java_royale_nativehost_JniHost_nativeInitResources", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeInitManager", 1
        )[0]
        cls.manager_pump = cls.bridge.split(
            "Java_royale_nativehost_JniHost_nativePumpManager", 1
        )[1].split(
            "Java_royale_nativehost_JniHost_nativeLoadReplay", 1
        )[0]

    def test_readiness_is_observed_from_natural_loading_state(self) -> None:
        self.assertIn("kDataTablesFactoryGlobalRva", self.probe)
        for field in (
            '"loading_phase"',
            '"data_load_resource_count"',
            '"data_tables_complete_latch"',
            '"data_tables_factory_structurally_valid"',
            '"natural_loading_phase_ready"',
            '"natural_loading_postprocess_ready"',
            '"data_load_task_structurally_ready"',
            '"data_load_task_complete"',
            '"natural_data_tables_ready"',
        ):
            self.assertIn(field.replace('"', '\\"'), self.probe)
        self.assertIn("loading_phase >= 3", self.probe)
        self.assertIn("loading_phase == 5", self.probe)
        self.assertIn("data_load_task_tables + 0x6E8", self.probe)
        self.assertNotIn(
            "reinterpret_cast<int32_t*>(loading_state + 0x0C)", self.probe
        )

    def test_startup_uses_bounded_manager_updates_not_manual_full_range(self) -> None:
        self.assertIn("CR_NATIVE_LOADING_MAX_FRAMES", self.host)
        self.assertIn("CR_NATIVE_LOADING_TIMEOUT_MS", self.host)
        self.assertIn("CR_NATIVE_LOADING_INITIAL_SETTLE_MS", self.host)
        self.assertIn("direct_loading_initial_settle", self.host)
        self.assertIn("500L", self.host)
        self.assertIn("30_000L", self.host)
        self.assertIn("factorySetupObserved", self.host)
        self.assertIn("natural_data_tables_ready", self.host)
        self.assertIn("manualDataTablesDiagnostic &&", self.host)
        self.assertIn("CR_NATIVE_UNSAFE_MANUAL_DATATABLE_PUMP", self.host)
        initial_probe = self.host.index(
            "nativeProbePrerequisites(args[0] + \"/libg.so\")"
        )
        settle = self.host.index("direct_loading_initial_settle")
        manager_loop = self.host.index("while (loadingFrames < loadingMaxFrames")
        self.assertLess(initial_probe, settle)
        self.assertLess(settle, manager_loop)

    def test_native_config_ab_switches_are_independent_and_default_off(self) -> None:
        for name in (
            "CR_BINDERLESS_NATIVE_CONFIG_REGISTRATIONS",
            "CR_BINDERLESS_NATIVE_CONFIG_GETTERS",
            "CR_BINDERLESS_NATIVE_CONFIG_POSTPROCESS",
        ):
            self.assertIn(name, self.game_main_init)
        for variable in (
            "native_config_registrations",
            "native_config_getters",
            "native_config_postprocess",
        ):
            self.assertIn(f"bool {variable} = false", self.game_main_init)
        self.assertIn("if (!native_config_registrations)", self.game_main_init)
        self.assertIn("if (native_config_getters)", self.game_main_init)
        self.assertIn("if (native_config_postprocess)", self.game_main_init)
        self.assertIn("BINDERLESS_NATIVE_CONFIG_POLICY", self.game_main_init)
        self.assertIn("config registration guard rejected", self.game_main_init)
        self.assertIn("guarded config getter rejected", self.game_main_init)
        self.assertIn("guarded config postprocess rejected", self.game_main_init)
        self.assertIn("kBinderlessConfigGetterGuardCaveRva", self.game_main_init)
        self.assertIn("BINDERLESS_CONFIG_GETTER_GUARD", self.game_main_init)
        self.assertIn(
            "binderless_config_getter_guard_cave_code", self.game_main_init
        )
        self.assertIn("call[0] = 0xE8", self.game_main_init)
        self.assertIn(
            "kBinderlessConfigPostprocessGuardCaveRva", self.game_main_init
        )
        self.assertIn(
            "BINDERLESS_CONFIG_POSTPROCESS_GUARD", self.game_main_init
        )
        self.assertIn(
            "binderless_config_postprocess_guard_cave_code",
            self.game_main_init,
        )

    def test_loading_screen_experiment_is_compiled_out(self) -> None:
        self.assertIn("kBinderlessLoadingScreenInitRva", self.bridge)
        self.assertIn("BINDERLESS_LOADING_SCREEN_BYPASS", self.manager_pump)
        self.assertIn("binderless_loading_screen_bypass_installed", self.manager_pump)

    def test_deferred_null_dependency_guard_is_opt_in_and_fail_closed(self) -> None:
        self.assertIn(
            "CR_BINDERLESS_DEFER_NULL_DEPENDENCIES", self.manager_pump
        )
        self.assertIn(
            "const bool defer_null_dependencies", self.manager_pump
        )
        self.assertIn("if (defer_null_dependencies)", self.manager_pump)
        self.assertIn(
            "kBinderlessDeferredNullDependencyGuardRva", self.manager_pump
        )
        self.assertIn(
            "kBinderlessDeferredNullDependencyCaveRva", self.manager_pump
        )
        self.assertIn(
            "deferred_null_dependency_cave_code", self.manager_pump
        )
        self.assertIn(
            "BINDERLESS_DEFER_NULL_DEPENDENCY_GUARD", self.manager_pump
        )
        self.assertIn(
            "deferred null dependency guard rejected", self.manager_pump
        )

    def test_optional_client_globals_deferral_is_narrow_and_opt_in(self) -> None:
        self.assertIn(
            "CR_BINDERLESS_DEFER_OPTIONAL_CLIENT_GLOBALS", self.manager_pump
        )
        self.assertIn(
            "const bool defer_optional_client_globals", self.manager_pump
        )
        self.assertIn(
            "if (defer_optional_client_globals)", self.manager_pump
        )
        self.assertIn(
            "kBinderlessDeferredOptionalClientGlobalsRva", self.manager_pump
        )
        self.assertIn(
            "deferred optional ClientGlobals guard rejected", self.manager_pump
        )
        self.assertIn("scope=non_battle_meta", self.manager_pump)
        self.assertIn("BOAT_PVE", self.manager_pump)
        self.assertIn("ROYAL_TOURNAMENTS", self.manager_pump)

    def test_sc3d_context_guard_is_binderless_only_and_preserves_valid_path(self) -> None:
        for symbol in (
            "kBinderlessSc3dContextGuardRva",
            "kBinderlessSc3dContextGuardCaveRva",
            "kBinderlessSc3dContextGlobalRva",
            "kBinderlessSc3dContextNullTargetRva",
            "kBinderlessSc3dContextValidResumeRva",
            "kBinderlessSc3dCapabilityGuardRva",
            "kBinderlessSc3dCapabilityGuardCaveRva",
            "kBinderlessSc3dCapabilityNullTargetRva",
            "kBinderlessSc3dCapabilityValidResumeRva",
        ):
            self.assertIn(symbol, self.manager_pump)
        self.assertIn("if (binderless_android)", self.manager_pump)
        self.assertIn(
            "binderless_sc3d_context_global_expected", self.manager_pump
        )
        self.assertIn(
            "global_target != kBinderlessSc3dContextGlobalRva",
            self.manager_pump,
        )
        self.assertIn(
            "null_target != kBinderlessSc3dContextNullTargetRva",
            self.manager_pump,
        )
        self.assertIn(
            "valid_resume != kBinderlessSc3dContextValidResumeRva",
            self.manager_pump,
        )
        self.assertIn("BINDERLESS_SC3D_CONTEXT_GUARD", self.manager_pump)
        self.assertIn(
            "binderless_sc3d_capability_global_expected", self.manager_pump
        )
        self.assertIn(
            "capability_null_target != kBinderlessSc3dCapabilityNullTargetRva",
            self.manager_pump,
        )
        self.assertIn(
            "kBinderlessSc3dCapabilityValidResumeRva", self.manager_pump
        )
        self.assertIn("null=return valid=replay_original", self.manager_pump)
        self.assertNotIn("kDataTablesLoadRangeRva = 0x72E2AC", self.bridge)

    def test_sc3d_feature_guard_uses_native_fallback_and_cleanup(self) -> None:
        for symbol in (
            "kBinderlessSc3dFeatureGuardRva",
            "kBinderlessSc3dFeatureGuardCaveRva",
            "kBinderlessSc3dFeatureNullTargetRva",
            "kBinderlessSc3dFeatureValidResumeRva",
        ):
            self.assertIn(symbol, self.manager_pump)
        self.assertIn("binderless_sc3d_feature_guard_expected", self.manager_pump)
        self.assertIn("BINDERLESS_SC3D_FEATURE_GUARD", self.manager_pump)
        self.assertIn("null=fallback_cleanup", self.manager_pump)
        self.assertIn("valid=replay_original", self.manager_pump)
        self.assertNotIn("kDataTablesLoadRangeRva = 0x725AA7", self.bridge)

    def test_ui_locale_guard_skips_only_a_missing_presentation_container(self) -> None:
        for symbol in (
            "kBinderlessUiLocaleContainerGuardRva",
            "kBinderlessUiLocaleContainerGuardCaveRva",
            "kBinderlessUiLocaleContainerNullTargetRva",
            "kBinderlessUiLocaleContainerValidResumeRva",
            "kBinderlessUiLocaleRegistrationTargetRva",
        ):
            self.assertIn(symbol, self.manager_pump)
        self.assertIn("binderless_ui_locale_container_guard_expected", self.manager_pump)
        self.assertIn("original_registration_target", self.manager_pump)
        self.assertIn("cave_registration_target", self.manager_pump)
        self.assertIn("BINDERLESS_UI_LOCALE_CONTAINER_GUARD", self.manager_pump)
        self.assertIn("null=cleanup valid=replay_original", self.manager_pump)
        self.assertNotIn("kDataTablesLoadRangeRva = 0x14ABA01", self.bridge)

    def test_core_preload_is_explicit_texts_only_and_resolver_verified(self) -> None:
        self.assertIn("CR_BINDERLESS_PRELOAD_CORE_DATA", self.host)
        self.assertIn("nativePreloadCoreData", self.host)
        self.assertIn("direct_core_data_preload", self.host)
        self.assertIn("kCoreTextsPathRva", self.preload)
        self.assertIn('"csv_client/texts.csv"', self.preload)
        self.assertIn("kSafeResourceResolveRva", self.preload)
        self.assertIn("kResourceLoadedPredicateRva", self.preload)
        self.assertIn("loaded_before", self.preload)
        self.assertIn("resolver_before", self.preload)
        self.assertIn("if (!loaded_before)", self.preload)
        self.assertIn("blocking_request(path, false)", self.preload)
        self.assertIn("loaded_after", self.preload)
        self.assertIn("resolver_after", self.preload)
        self.assertIn("const bool success = loaded_after", self.preload)
        self.assertNotIn("texts_patch", self.preload)
        self.assertNotIn("data_manifest", self.preload)
        direct_loading = self.host.index("direct_loading_state")
        preload_stage = self.host.index("direct_core_data_preload")
        loading_loop = self.host.index("while (loadingFrames < loadingMaxFrames")
        self.assertLess(direct_loading, preload_stage)
        self.assertLess(preload_stage, loading_loop)

    def test_binderless_resource_init_sets_assets_as_process_cwd(self) -> None:
        self.assertIn("if (binderless_android)", self.init_resources)
        self.assertIn("chdir(direct_asset_root.c_str())", self.init_resources)
        self.assertIn("getcwd(direct_cwd.data()", self.init_resources)
        self.assertIn("DIRECT_CWD", self.init_resources)
        self.assertIn("cannot chdir to direct asset root", self.init_resources)
        mounts = self.init_resources.index(
            'mount_path(kTempPathGetterRva, "temp:")'
        )
        chdir_call = self.init_resources.index(
            "chdir(direct_asset_root.c_str())"
        )
        self.assertLess(mounts, chdir_call)


if __name__ == "__main__":
    unittest.main()
