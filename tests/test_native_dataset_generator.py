from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from expert_v1.native_dataset_generator import (
    COORDINATE_PROVENANCE,
    _failure_class,
    _failure_domain,
    _combined_phase_metrics,
    _mask_invalid_censor_evidence,
    NativeDatasetTask,
    PreflightFullTraceDivergence,
    RecordingCountingEnv,
    StagedTickSink,
    StoredFrameRegistry,
    execute_task,
    execute_two_phase_plan,
    native_result_pipeline_contract_valid,
    atomic_json,
    prepare_run,
    reconcile_result_files,
    recover_unmanifested_final_shards,
    requeue_failed_infrastructure,
    select_tasks,
    should_retry_failure,
    summarize_results,
    verify_published_audit_prefix_store,
    verify_published_tick_store,
)
from expert_v1.native_profile import native_teacher_forced_profile
from expert_v1.tick_store_v1.schema import (
    EpisodeState,
    PlayerPrivate,
    TickState,
)
from expert_v1.tick_store_v1.deployment_masks import (
    DYNAMIC_RULE,
    EPISODE_METADATA_KEY,
    DeploymentMaskStore,
    NativeDeploymentMaskCapture,
)
from expert_v1.tick_store_v1.shard import (
    AUDIT_PREFIX_STORE_KIND,
    WorkerShardSink,
    build_store_manifest,
)
from expert_v1.tick_store_v1.work_queue import TickStoreWorkQueue


def tick_states(count: int = 4) -> list[TickState]:
    return [
        TickState(
            tick=10 + index,
            players=(
                PlayerPrivate(0, 50_000, (0, 1, 2, 3), 4),
                PlayerPrivate(1, 50_000, (4, 5, 6, 7), 0),
            ),
            towers=(),
            entities=(),
            episode=EpisodeState(1, 0, 1, 0, 0, 0, 0, 0, 0),
        )
        for index in range(count)
    ]


def candidate(tag: str, path: Path, *, abilities: int) -> dict:
    return {
        "ability_count_reported": abilities,
        "ability_events_observed": abilities,
        "ability_log_tier": (
            "observed_ticks_identity_runtime_resolved"
            if abilities else "source_reports_zero"
        ),
        "authoritative_native_full_candidate": True,
        "battle_tag": tag,
        "compiler_native_replay_ready": True,
        "coordinate_tier": "all_card_events_raw_data_i",
        "deployment_actions": 10,
        "duration_ticks": 100,
        "eligibility_tier": (
            "authoritative_native_ability_exact"
            if abilities else "authoritative_native_deployment_only"
        ),
        "source_path": str(path),
        "source_schema_version": 3,
        "source_sha256": "a" * 64,
    }


def write_candidates(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def replay_result_stub(
    *, success: bool, chosen_seed: int = 7, failure: str | None = None,
    action_sequence: tuple[dict, ...] = (),
    collected_states: tuple[TickState, ...] = (),
    with_masks: bool = False,
) -> SimpleNamespace:
    mask_metadata = None
    mask_payloads = {}
    mask_probe_count = 0
    if with_masks:
        slots = [
            {
                "side": side,
                "deck_index": deck_index,
                "card_id": 26_000_000 + deck_index,
                "level": 16,
                "form_flags": 0,
                "source_token": f"card-{deck_index}",
                "base_token": f"card-{deck_index}",
            }
            for side in (0, 1)
            for deck_index in range(8)
        ]

        class Probe:
            def probe_grid(self, *, side: int, deck_index: int) -> dict:
                rows = ["1" * 18 for _ in range(32)]
                return {
                    "width": 18, "height": 32, "cell_size": 1000,
                    "valid_cells": 576,
                    "resolved_data_id": 26_000_000 + deck_index,
                    "packed_selection": 0,
                    "card_cost": 1, "card_cost_raw": 10_000,
                    "selection_form_index": -1,
                    "selection_strategy": "canonical",
                    "selection_builder_rva": "0x1",
                    "selection_root_vtable_rva": "0x2",
                    "rows": rows,
                }

        capture = NativeDeploymentMaskCapture(slots)
        capture.capture_available(Probe(), {
            "tick": 10,
            "players": [
                {"side": 0, "hand_deck_indices": [0, 1, 2, 3]},
                {"side": 1, "hand_deck_indices": [4, 5, 6, 7]},
            ],
        })
        mask_metadata = capture.metadata(require_complete=False)
        mask_payloads = capture.payloads
        mask_probe_count = capture.probe_rpc_count
    return SimpleNamespace(
        teacher_forced_success=success,
        failure=failure,
        accepted_actions=len(action_sequence) if success else 0,
        accepted_deploy_actions=len(action_sequence) if success else 0,
        accepted_ability_actions=0,
        action_acceptance_sequence=action_sequence,
        ability_resolutions=(),
        logic_freeze_diagnostic=None,
        collected_tick_states=collected_states,
        final_tick=80,
        terminal_validated=False,
        terminal_match=None,
        terminal_diagnostic_status="native_terminal_missing",
        source_crowns=(0, 0),
        observed_crowns=None,
        terminal_tower_hp_validated=False,
        terminal_tower_hp_match=None,
        terminal_tower_hp_diagnostic_status=(
            "native_terminal_tower_hp_missing"
        ),
        source_final_tower_hp=None,
        observed_final_tower_hp=None,
        chosen_seed=chosen_seed,
        seeds_tested=3,
        seed_search_native_resets=3,
        layout_resolution_mode="source_order_bounded_native_seed_search",
        native_ticks_advanced=70,
        tick_trace_batches=0,
        tick_trace_complete_frames=len(collected_states),
        tick_trace_incomplete_terminal_frames=0,
        tick_trace_incomplete_nonterminal_freeze_frames=0,
        deployment_mask_probe_rpc_count=mask_probe_count,
        deployment_mask_metadata=mask_metadata,
        deployment_mask_payloads=mask_payloads,
        deployment_mask_label_checks=0,
        deployment_mask_label_rejections=0,
        deployment_mask_first_label_rejection=None,
        deployment_mask_label_rejection_sequence=(),
        action_execution_tick_offset=1,
    )


class FakeCompatibleSeedSearch:
    def __init__(
        self, seed: int = 7, *, scanned: int = 3,
        seeds: list[int] | None = None,
    ) -> None:
        self.seeds = [seed] if seeds is None else list(seeds)
        self.seeds_scanned = scanned
        self.native_resets = scanned
        self.compatible_seeds_yielded = len(self.seeds)

    def __iter__(self):
        for index, seed in enumerate(self.seeds):
            yield SimpleNamespace(
                chosen_seed=seed,
                seeds_tested=min(self.seeds_scanned, index + 1),
            )


def semantic_audit(
    *, seed: int = 7, success: bool = False, prefix: int = 0
) -> dict:
    return {
        "schema_version": 2,
        "kind": "single_semantic_seed_preflight_v2",
        "maximum_compatible_seeds": 1,
        "raw_seed_scan_limit": 16,
        "raw_seeds_scanned": 3,
        "layout_compatible_candidates_tested": 1,
        "layout_compatible_candidates_found": 1,
        "layout_scan_native_resets": 3,
        "semantic_preflight_native_resets": 1,
        "selected_seed": seed,
        "selected_accepted_source_event_prefix": prefix,
        "selected_teacher_forced_success": success,
        "selection_rule": "first_layout_compatible_seed_only",
        "ability_identity_policy": "branch_required_fails_closed_no_guess",
        "candidates": [{
            "ordinal": 0,
            "seed": seed,
            "raw_seeds_scanned_when_found": 3,
            "teacher_forced_success": success,
            "accepted_source_event_prefix": prefix,
            "failure": None if success else "ability_branch_required",
            "semantics_sha256": "a" * 64,
        }],
    }


def current_result_pipeline_fields(*, success: bool, seed: int = 7) -> dict:
    return {
        "native_preflight_contract_version": 4,
        "native_execution_pipeline_mode": (
            "single_semantic_seed_preflight_then_fixed_seed_trace_v4"
        ),
        "preflight_teacher_forced_success": success,
        "preflight_chosen_seed": seed,
        "chosen_seed": seed,
        "semantic_seed_preflight": semantic_audit(
            seed=seed, success=success, prefix=0
        ),
    }


class NativeDatasetGeneratorTest(unittest.TestCase):
    def test_current_preflight_not_reached_diagnostic_has_no_artifacts(self) -> None:
        diagnostic = {
            "native_preflight_contract_version": 4,
            "native_execution_pipeline_mode": (
                "single_semantic_seed_preflight_then_fixed_seed_trace_v4"
            ),
            "preflight_teacher_forced_success": None,
            "preflight_chosen_seed": None,
            "chosen_seed": None,
            "semantic_seed_preflight": None,
            "tick_store_entry": None,
            "audit_prefix_tick_store_entry": None,
            "token_coverage_actor_evidence": [],
            "prefix_token_coverage_actor_evidence": [],
        }
        self.assertTrue(native_result_pipeline_contract_valid(diagnostic))
        for field, value in (
            ("chosen_seed", 7),
            ("semantic_seed_preflight", {}),
            ("tick_store_entry", {}),
            ("prefix_token_coverage_actor_evidence", [{}]),
        ):
            changed = dict(diagnostic)
            changed[field] = value
            self.assertFalse(native_result_pipeline_contract_valid(changed), field)

    def test_failed_preflight_runs_fixed_seed_prefix_without_mask_probes(self) -> None:
        rejected = replay_result_stub(
            success=False,
            failure="native_rejected_tick_21_codes_[4]",
            action_sequence=({
                "source_tick": 20, "execution_tick": 21,
                "source_event_index": 0, "type": "play", "side": 0,
                "accepted": False, "result_code": 4,
            },),
        )
        with patch(
            "expert_v1.native_dataset_generator.execute_plan",
            return_value=rejected,
        ) as mocked, patch(
            "expert_v1.native_dataset_generator.compatible_native_seed_search",
            return_value=FakeCompatibleSeedSearch(),
        ):
            outcome = execute_two_phase_plan(
                object(), SimpleNamespace(battle_tag="PREFIX"), {}, StagedTickSink(),
                seed=1, maximum_seeds_to_test=16, trace_batch_steps=8,
                tick_store_metadata={},
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertIsNone(outcome.full_trace)
        second = mocked.call_args_list[1]
        self.assertEqual(second.kwargs["fixed_seed"], 7)
        self.assertTrue(second.kwargs["collect_tick_states_on_failure"])
        self.assertTrue(second.kwargs["capture_deployment_masks"])
        self.assertIsNone(second.kwargs["tick_sink"])
        self.assertFalse(outcome.failure_prefix_staged)
        self.assertEqual(list(outcome.preflight_recorder.trace_history), [])
        self.assertEqual(
            outcome.preflight_recorder.native_deployment_mask_probes_attempted,
            0,
        )

    def test_successful_preflight_reuses_seed_and_matches_full_trace(self) -> None:
        actions = ({
            "source_tick": 20, "execution_tick": 21,
            "source_event_index": 0, "type": "play", "side": 0,
            "accepted": True, "result_code": 0,
        },)
        preflight = replay_result_stub(
            success=True, chosen_seed=7, action_sequence=actions
        )
        full = replay_result_stub(
            success=True, chosen_seed=7, action_sequence=actions
        )
        full.seeds_tested = 0
        full.seed_search_native_resets = 1
        full.layout_resolution_mode = "fixed_preflight_seed_replay"
        with patch(
            "expert_v1.native_dataset_generator.execute_plan",
            side_effect=[preflight, full],
        ) as mocked, patch(
            "expert_v1.native_dataset_generator.compatible_native_seed_search",
            return_value=FakeCompatibleSeedSearch(),
        ):
            outcome = execute_two_phase_plan(
                object(), SimpleNamespace(), {}, StagedTickSink(),
                seed=1, maximum_seeds_to_test=16, trace_batch_steps=8,
                tick_store_metadata={},
            )

        self.assertEqual(mocked.call_count, 2)
        second = mocked.call_args_list[1]
        self.assertEqual(second.kwargs["fixed_seed"], 7)
        self.assertTrue(second.kwargs["capture_deployment_masks"])
        self.assertIsNotNone(second.kwargs["tick_sink"])
        self.assertEqual(outcome.semantic_diff, {})

    def test_unique_mask_rejection_builds_strict_fixed_seed_censor(self) -> None:
        failure_event = {
            "source_tick": 20, "execution_tick": 21,
            "source_event_index": 0, "type": "play", "side": 0,
            "accepted": True, "result_code": 0,
        }
        states = tuple(tick_states(12))
        preflight = replay_result_stub(
            success=True, chosen_seed=7, action_sequence=(failure_event,)
        )
        masked = replay_result_stub(
            success=False,
            chosen_seed=7,
            failure="derived_deployment_mask_rejected_source_event_0",
            action_sequence=(),
            collected_states=states,
            with_masks=True,
        )
        masked.final_tick = 21
        masked.seeds_tested = 0
        masked.seed_search_native_resets = 1
        masked.layout_resolution_mode = "fixed_preflight_seed_replay"
        digest = next(iter(masked.deployment_mask_payloads))
        rejection = {
            "tick": 21, "side": 0, "deck_index": 0,
            "card_id": 26_000_000, "x": 3_500, "y": 17_501,
            "content_sha256": digest, "legal": False,
            "reasons": ["position_not_in_derived_native_mask"],
            "source_event_index": 0, "source_marker_index": 9,
            "locked_pocket": {
                "reason": "live_enemy_princess_tower_locked_pocket_v1",
                "tower_side": 1, "tower_x": 3_500,
                "tower_y": 25_500, "tower_hp": 3_052,
                "lane": 0, "row": 17, "column": 3,
            },
        }
        masked.deployment_mask_label_checks = 1
        masked.deployment_mask_label_rejections = 1
        masked.deployment_mask_first_label_rejection = rejection
        masked.deployment_mask_label_rejection_sequence = (rejection,)
        reference = replay_result_stub(
            success=True,
            chosen_seed=7,
            action_sequence=(failure_event,),
            collected_states=states,
        )
        reference.seeds_tested = 0
        reference.seed_search_native_resets = 1
        reference.layout_resolution_mode = "fixed_preflight_seed_replay"
        action = SimpleNamespace(
            source_event_index=0, source_marker_index=9, tick=20,
            side=0, logical_card_index=0, x=3_500, y=17_501,
        )
        plan = SimpleNamespace(
            battle_tag="MASK-CENSOR",
            actions=(action,),
            ability_events=(),
            sides=(
                SimpleNamespace(deck=(SimpleNamespace(card_id=26_000_000),)),
                SimpleNamespace(deck=()),
            ),
        )
        evidence = _mask_invalid_censor_evidence(
            preflight, masked, reference, plan
        )
        self.assertIsNotNone(evidence)
        self.assertFalse(evidence["failure_event_executed"])
        self.assertEqual(
            evidence["preflight_boundary_accepted_action"], failure_event
        )
        self.assertEqual(
            evidence["preflight_boundary_accepted_action_sha256"],
            hashlib.sha256(json.dumps(
                failure_event, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        )

        rejected_preflight = replay_result_stub(
            success=False,
            chosen_seed=7,
            failure="native_rejected_tick_21_source_tick_20",
            action_sequence=({**failure_event, "accepted": False},),
        )
        self.assertIsNone(_mask_invalid_censor_evidence(
            rejected_preflight, masked, reference, plan
        ))

        tampered_n = SimpleNamespace(**vars(masked))
        tampered_n.failure = "derived_deployment_mask_rejected_source_event_1"
        self.assertIsNone(
            _mask_invalid_censor_evidence(preflight, tampered_n, reference, plan)
        )
        tampered_count = SimpleNamespace(**vars(masked))
        tampered_count.deployment_mask_label_rejections = 2
        self.assertIsNone(
            _mask_invalid_censor_evidence(
                preflight, tampered_count, reference, plan
            )
        )
        changed_states = list(states)
        changed_states[2] = replace(
            changed_states[2],
            episode=replace(changed_states[2].episode, battle_flag=9),
        )
        tampered_state = SimpleNamespace(**vars(masked))
        tampered_state.collected_tick_states = tuple(changed_states)
        self.assertIsNone(
            _mask_invalid_censor_evidence(
                preflight, tampered_state, reference, plan
            )
        )

        staged = StagedTickSink()
        with patch(
            "expert_v1.native_dataset_generator.execute_plan",
            side_effect=[preflight, masked, reference],
        ) as mocked, patch(
            "expert_v1.native_dataset_generator.compatible_native_seed_search",
            return_value=FakeCompatibleSeedSearch(),
        ):
            outcome = execute_two_phase_plan(
                object(), plan, {}, StagedTickSink(), prefix_staged=staged,
                seed=1, maximum_seeds_to_test=16, trace_batch_steps=8,
                tick_store_metadata={"source_sha256": "a" * 64},
            )
        self.assertEqual(mocked.call_count, 3)
        self.assertTrue(outcome.mask_invalid_prefix)
        self.assertTrue(outcome.failure_prefix_staged)
        self.assertIsNone(outcome.semantic_diff)
        self.assertIs(outcome.maskless_reference, reference)
        self.assertIsNotNone(outcome.maskless_reference_recorder)
        self.assertGreaterEqual(outcome.maskless_reference_seconds, 0.0)
        extent = staged.episode.metadata["native_replay_extent_v1"]
        self.assertEqual(
            extent["training_admission"],
            "actor_bc_mask_invalid_censored_prefix_v1",
        )
        self.assertEqual(extent["timing_censor_tick_exclusive"], 21)
        self.assertFalse(extent["semantic_match"])
        self.assertTrue(extent["pre_censor_tick_state_parity"])
        outcome.maskless_reference_recorder.native_actions_attempted = 3
        outcome.maskless_reference_recorder.native_actions_responded = 3
        outcome.maskless_reference_recorder.native_actions_accepted = 3
        combined = _combined_phase_metrics(
            outcome.preflight_recorder,
            outcome.full_trace_recorder,
            outcome.failure_prefix_recorder,
            outcome.maskless_reference_recorder,
        )
        self.assertEqual(
            combined["native_actions_attempted"],
            sum(
                recorder.native_actions_attempted
                for recorder in (
                    outcome.preflight_recorder,
                    outcome.full_trace_recorder,
                    outcome.maskless_reference_recorder,
                )
                if recorder is not None
            ),
        )

        for name, changed in (
            (
                "zero_rejections",
                {
                    "deployment_mask_label_rejections": 0,
                    "deployment_mask_first_label_rejection": None,
                    "deployment_mask_label_rejection_sequence": (),
                },
            ),
            (
                "multiple_rejections",
                {
                    "deployment_mask_label_rejections": 2,
                    "deployment_mask_first_label_rejection": rejection,
                    "deployment_mask_label_rejection_sequence": (
                        rejection,
                        {**rejection, "source_event_index": 1},
                    ),
                },
            ),
            (
                "generic_derived_false_not_locked_pocket",
                {
                    "deployment_mask_label_rejections": 1,
                    "deployment_mask_first_label_rejection": {
                        **rejection, "locked_pocket": None,
                    },
                    "deployment_mask_label_rejection_sequence": ({
                        **rejection, "locked_pocket": None,
                    },),
                },
            ),
        ):
            with self.subTest(name=name):
                invalid = SimpleNamespace(**vars(masked))
                for field, value in changed.items():
                    setattr(invalid, field, value)
                invalid_prefix = StagedTickSink()
                with patch(
                    "expert_v1.native_dataset_generator.execute_plan",
                    side_effect=[preflight, invalid],
                ) as rejected_mock, patch(
                    "expert_v1.native_dataset_generator.compatible_native_seed_search",
                    return_value=FakeCompatibleSeedSearch(),
                ):
                    rejected = execute_two_phase_plan(
                        object(), plan, {}, StagedTickSink(),
                        prefix_staged=invalid_prefix,
                        seed=1, maximum_seeds_to_test=16,
                        trace_batch_steps=8,
                        tick_store_metadata={"source_sha256": "a" * 64},
                    )
                self.assertEqual(rejected_mock.call_count, 2)
                self.assertFalse(rejected.mask_invalid_prefix)
                self.assertFalse(rejected.failure_prefix_staged)
                self.assertIsNone(invalid_prefix.episode)

    def test_semantic_failure_prefix_is_consecutive_censored_and_audit_only(self) -> None:
        action = ({
            "source_tick": 20, "execution_tick": 21,
            "source_event_index": 0, "type": "play", "side": 0,
            "accepted": False, "result_code": 4,
        },)
        preflight = replay_result_stub(
            success=False,
            chosen_seed=7,
            failure="native_rejected_tick_21_source_tick_20_codes_[4]",
            action_sequence=action,
        )
        prefix_states = tick_states(12)
        transient_players = list(prefix_states[5].players)
        transient_players[0] = replace(
            transient_players[0], hand=(0, 1, -1, 3), refill_timer=0
        )
        prefix_states[5] = replace(
            prefix_states[5], players=tuple(transient_players)
        )
        prefix = replay_result_stub(
            success=False,
            chosen_seed=7,
            failure=preflight.failure,
            action_sequence=action,
            collected_states=tuple(prefix_states),
            with_masks=True,
        )
        prefix.seeds_tested = 0
        prefix.seed_search_native_resets = 1
        prefix.layout_resolution_mode = "fixed_preflight_seed_replay"
        staged = StagedTickSink()
        with patch(
            "expert_v1.native_dataset_generator.execute_plan",
            side_effect=[preflight, prefix],
        ), patch(
            "expert_v1.native_dataset_generator.compatible_native_seed_search",
            return_value=FakeCompatibleSeedSearch(),
        ):
            outcome = execute_two_phase_plan(
                object(), SimpleNamespace(battle_tag="PREFIX"), {}, StagedTickSink(),
                prefix_staged=staged,
                seed=1, maximum_seeds_to_test=16, trace_batch_steps=8,
                tick_store_metadata={"source_sha256": "a" * 64},
            )
        self.assertTrue(outcome.failure_prefix_staged)
        self.assertIsNotNone(staged.episode)
        extent = staged.episode.metadata["native_replay_extent_v1"]
        self.assertEqual(
            extent["training_admission"], "actor_bc_censored_prefix_v1"
        )
        self.assertEqual(extent["action_label_tick_stop_exclusive"], 21)
        self.assertEqual(extent["timing_censor_tick_exclusive"], 21)
        self.assertFalse(extent["failure_tick_has_labels"])
        self.assertEqual(extent["terminal_target"], "unknown_censored")
        self.assertEqual(
            extent["timing_target"], "right_censored_at_failure_tick_v1"
        )
        self.assertTrue(
            extent["mask_coverage"][
                "all_retained_visible_hand_slots_covered"
            ]
        )
        self.assertEqual(extent["mask_coverage"]["empty_slot_actor_ticks"], 1)
        self.assertIn("native_deployment_masks_v1", staged.episode.metadata)
        self.assertEqual(staged.episode.states[-1].tick, 21)

    def test_two_phase_action_or_terminal_difference_is_a_closed_diff(self) -> None:
        preflight = replay_result_stub(success=True, chosen_seed=7)
        full = replay_result_stub(success=True, chosen_seed=7)
        full.seeds_tested = 0
        full.seed_search_native_resets = 1
        full.layout_resolution_mode = "fixed_preflight_seed_replay"
        full.terminal_match = False
        with patch(
            "expert_v1.native_dataset_generator.execute_plan",
            side_effect=[preflight, full],
        ), patch(
            "expert_v1.native_dataset_generator.compatible_native_seed_search",
            return_value=FakeCompatibleSeedSearch(),
        ):
            outcome = execute_two_phase_plan(
                object(), SimpleNamespace(), {}, StagedTickSink(),
                seed=1, maximum_seeds_to_test=16, trace_batch_steps=8,
                tick_store_metadata={},
            )
        self.assertEqual(set(outcome.semantic_diff or {}), {"terminal"})

    def test_failure_domains_keep_infrastructure_out_of_semantic_rejections(self) -> None:
        timeout = _failure_class(
            None, TimeoutError("RPC timeout"), "native_teacher_forced_replay"
        )
        disk = _failure_class(
            None, OSError("disk full"), "immutable_tick_store_commit"
        )
        source_sha = _failure_class(
            None, RuntimeError("changed"), "source_sha_verification"
        )
        self.assertEqual(_failure_domain(timeout), "infrastructure")
        self.assertEqual(_failure_domain(disk), "infrastructure")
        self.assertEqual(_failure_domain(source_sha), "infrastructure")
        self.assertEqual(_failure_domain("native_action_rejected"), "semantic")
        protocol = _failure_class(
            SimpleNamespace(failure="native_action_count_mismatch_tick_20"),
            None,
            "first_native_difference",
        )
        hand = _failure_class(
            SimpleNamespace(failure="hand_mismatch_event_7"),
            None,
            "first_native_difference",
        )
        unknown = _failure_class(
            SimpleNamespace(failure="new_unrecognized_runner_failure"),
            None,
            "first_native_difference",
        )
        terminal = _failure_class(
            SimpleNamespace(failure="native_terminal_before_execution_tick_80"),
            None,
            "first_native_difference",
        )
        self.assertEqual(_failure_domain(protocol), "infrastructure")
        self.assertEqual(_failure_domain(hand), "source_integrity")
        self.assertEqual(_failure_domain(unknown), "infrastructure")
        self.assertEqual(_failure_domain(terminal), "semantic")
        divergence = _failure_class(
            None,
            PreflightFullTraceDivergence({"terminal": {}}),
            "preflight_full_trace_semantic_diff",
        )
        self.assertEqual(
            divergence,
            "infrastructure_preflight_full_trace_semantic_divergence",
        )
        self.assertEqual(_failure_domain(divergence), "infrastructure")
        self.assertTrue(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "infrastructure",
        }, 1))
        self.assertFalse(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "infrastructure",
        }, 3))
        self.assertFalse(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "semantic",
        }, 1))

    def test_limited_selection_is_deterministic_and_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "candidates.jsonl"
            rows = [
                candidate("P1", root / "p1.json", abilities=2),
                candidate("P2", root / "p2.json", abilities=1),
                candidate("Z1", root / "z1.json", abilities=0),
                candidate("Z2", root / "z2.json", abilities=0),
            ]
            write_candidates(queue, rows)
            first, summary = select_tasks(queue, limit=2, selection_seed="smoke")
            second, _ = select_tasks(queue, limit=2, selection_seed="smoke")
            self.assertEqual(first, second)
            self.assertEqual(sum(task.ability_positive for task in first), 1)
            self.assertEqual(sum(not task.ability_positive for task in first), 1)
            self.assertEqual(summary["selected_rows"], 2)
            self.assertNotIn("source_json", first[0].json())

    def test_explicit_stratum_quotas_are_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "candidates.jsonl"
            rows = [
                candidate("P1", root / "p1.json", abilities=2),
                candidate("P2", root / "p2.json", abilities=1),
                candidate("P3", root / "p3.json", abilities=3),
                candidate("Z1", root / "z1.json", abilities=0),
                candidate("Z2", root / "z2.json", abilities=0),
            ]
            write_candidates(queue, rows)
            selected, summary = select_tasks(
                queue, limit=4, selection_seed="quota",
                deployment_zero_quota=1, ability_exact_quota=3,
            )
            self.assertEqual(sum(not task.ability_positive for task in selected), 1)
            self.assertEqual(sum(task.ability_positive for task in selected), 3)
            self.assertEqual(summary["explicit_stratum_quotas"], {
                "authoritative_native_deployment_only": 1,
                "authoritative_native_ability_exact": 3,
            })
            with self.assertRaises(ValueError):
                select_tasks(
                    queue, limit=4, deployment_zero_quota=1,
                    ability_exact_quota=2,
                )
            with self.assertRaises(RuntimeError):
                select_tasks(
                    queue, limit=5, deployment_zero_quota=3,
                    ability_exact_quota=2,
                )

    def test_prepare_is_idempotent_and_contract_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_queue = root / "candidates.jsonl"
            write_candidates(source_queue, [
                candidate("P", root / "p.json", abilities=1),
                candidate("Z", root / "z.json", abilities=0),
            ])
            template = root / "template.json"
            template.write_text('{"battle":{}}\n', encoding="utf-8")
            output = root / "output"
            args = dict(
                candidate_queue=source_queue,
                output_root=output,
                template_path=template,
                limit=2,
                selection_seed="stable",
                seed=1,
                maximum_seeds_to_test=16,
                trace_batch_steps=8,
                episodes_per_shard=4,
            )
            first = prepare_run(**args)
            second = prepare_run(**args)
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[3]["run_contract_version"], 4)
            self.assertEqual(
                first[3]["native_execution_pipeline"]["contract_version"], 4
            )
            self.assertEqual(
                first[3]["native_execution_pipeline"]
                ["semantic_seed_preflight"]
                ["layout_compatible_candidate_limit"],
                1,
            )
            with TickStoreWorkQueue(first[2]) as queue:
                self.assertEqual(queue.counts(), {"pending": 2})
            changed = dict(args)
            changed["selection_seed"] = "different"
            with self.assertRaises(RuntimeError):
                prepare_run(**changed)
            contract_path = output / "run-contract.json"
            cap8_contract = json.loads(contract_path.read_text())
            cap8_contract["run_contract_version"] = 3
            cap8_contract["native_execution_pipeline"].update({
                "contract_version": 3,
                "mode": (
                    "bounded_semantic_seed_preflight_then_fixed_seed_trace_v3"
                ),
            })
            cap8_contract["native_execution_pipeline"].pop(
                "semantic_seed_preflight"
            )
            atomic_json(contract_path, cap8_contract)
            with self.assertRaisesRegex(RuntimeError, "resume contract changed"):
                prepare_run(**args)

    def test_staged_episode_commits_once_and_can_be_reused_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = StoredFrameRegistry(root)
            sink = WorkerShardSink(root, "worker-a", episodes_per_shard=10)
            first_stage = StagedTickSink()
            first_stage.append("BATTLE", tick_states(), {"stable": 1})
            self.assertIsNotNone(first_stage.episode)
            first = registry.commit_or_reuse(sink, first_stage.episode)  # type: ignore[arg-type]
            second_stage = StagedTickSink()
            second_stage.append("BATTLE", tick_states(), {"stable": 1})
            second = registry.commit_or_reuse(sink, second_stage.episode)  # type: ignore[arg-type]
            self.assertFalse(first["resume_reused_existing_frame"])
            self.assertTrue(second["resume_reused_existing_frame"])
            self.assertEqual(first["payload_sha256"], second["payload_sha256"])
            manifests = sink.finalize()
            self.assertEqual(manifests[0]["episode_count"], 1)

    def test_source_sha_mismatch_never_writes_a_tick_frame_or_source_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text('{"secret":"must-not-copy"}\n', encoding="utf-8")
            task = NativeDatasetTask(
                0, "b" * 64, "BAD-SHA", str(source), "a" * 64,
                3, 100, 10, 0, 0, "source_reports_zero",
                "all_card_events_raw_data_i",
                "authoritative_native_deployment_only",
            )
            shard_root = root / "shards"
            sink = WorkerShardSink(shard_root, "worker", episodes_per_shard=2)
            execution = execute_task(
                object(), task, {}, sink, StoredFrameRegistry(shard_root),
                worker_id="worker", port=1, attempt=1,
            )
            self.assertFalse(execution.record["teacher_forced_success"])
            self.assertEqual(execution.record["failure_class"], "source_sha_mismatch")
            self.assertEqual(execution.record["failure_domain"], "infrastructure")
            self.assertEqual(sink.writer.episode_count, 0)
            diagnostic_text = json.dumps(execution.diagnostic, ensure_ascii=False)
            self.assertNotIn("must-not-copy", diagnostic_text)
            self.assertFalse(execution.diagnostic["source_identity"]["source_json_copied"])
            self.assertEqual(sink.finalize(), [])

    def test_result_frame_checkpoint_reconciles_interrupted_queue_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite3"
            with TickStoreWorkQueue(queue_path) as queue:
                queue.add_tasks([{
                    "battle_tag": "CRASH-WINDOW",
                    "source_path": "source.json",
                    "source_sha256": "a" * 64,
                    "payload": {},
                }])
                claim = queue.claim(
                    "worker", limit=1, lease_seconds=300,
                    maximum_attempts=10,
                )[0]
                self.assertEqual(claim.battle_tag, "CRASH-WINDOW")
            shards = root / "shards"
            sink = WorkerShardSink(shards, "worker", episodes_per_shard=1)
            entry = sink.append("CRASH-WINDOW", tick_states(), {"stable": 1})
            sink.finalize()
            result_path = root / "results" / "CRASH_WINDOW.json"
            current_result = {
                **current_result_pipeline_fields(success=True),
                "schema_version": 1,
                "kind": "expert_authoritative_native_tick_result_v1",
                "battle_tag": "CRASH-WINDOW",
                "teacher_forced_success": True,
                "tick_store_entry": entry,
            }
            legacy_result = json.loads(json.dumps(current_result))
            legacy_result["native_preflight_contract_version"] = 3
            legacy_result["native_execution_pipeline_mode"] = (
                "bounded_semantic_seed_preflight_then_fixed_seed_trace_v3"
            )
            legacy_result["semantic_seed_preflight"].update({
                "schema_version": 1,
                "kind": "bounded_semantic_seed_preflight_v1",
                "maximum_compatible_seeds": 8,
            })
            atomic_json(result_path, legacy_result)
            with self.assertRaisesRegex(RuntimeError, "stale/invalid"):
                reconcile_result_files(root, queue_path)
            with TickStoreWorkQueue(queue_path) as queue:
                self.assertEqual(queue.counts(), {"leased": 1})
            atomic_json(result_path, current_result)
            self.assertEqual(reconcile_result_files(root, queue_path), 1)
            with TickStoreWorkQueue(queue_path) as queue:
                self.assertEqual(queue.counts(), {"done": 1})
                row = queue.connection.execute(
                    "SELECT episode_sha256 FROM tasks WHERE battle_tag=?",
                    ("CRASH-WINDOW",),
                ).fetchone()
                self.assertEqual(row["episode_sha256"], entry["payload_sha256"])
            # Reconciliation is idempotent and never appends another frame.
            self.assertEqual(reconcile_result_files(root, queue_path), 0)
            manifests = list(shards.glob("*.manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(
                json.loads(manifests[0].read_text())["episode_count"], 1
            )

    def test_prefix_frame_checkpoint_reconciles_failed_queue_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite3"
            with TickStoreWorkQueue(queue_path) as queue:
                queue.add_tasks([{
                    "battle_tag": "PREFIX-CRASH",
                    "source_path": "source.json",
                    "source_sha256": "a" * 64,
                    "payload": {},
                }])
                queue.claim(
                    "worker", limit=1, lease_seconds=300,
                    maximum_attempts=10,
                )
            prefix_root = root / "audit-prefix-shards"
            sink = WorkerShardSink(prefix_root, "prefix", episodes_per_shard=1)
            extent = {
                "kind": "cr_native_replay_extent_v1",
                "extent": "valid_prefix",
                "training_admission": "actor_bc_censored_prefix_v1",
                "source_episode_complete": False,
                "every_native_tick_present_within_extent": True,
                "fixed_seed_replay": True,
                "chosen_seed": 7,
                "preflight_semantics_sha256": "a" * 64,
                "prefix_replay_semantics_sha256": "a" * 64,
                "semantic_match": True,
                "failure_domain": "semantic",
                "failure_tick_has_labels": False,
                "terminal_target": "unknown_censored",
                "terminal_validated": False,
                "deployment_masks": "partial_native_visible_hand_complete_v1",
                "observation_tick_start": 10,
                "observation_tick_stop_exclusive": 14,
                "action_label_tick_stop_exclusive": 13,
                "timing_censor_tick_exclusive": 13,
                "timing_target": "right_censored_at_failure_tick_v1",
                "mask_coverage": {
                    "all_retained_visible_hand_slots_covered": True,
                    "retained_ticks": 3,
                    "actor_ticks": 6,
                    "visible_slot_references": 24,
                    "empty_slot_actor_ticks": 0,
                    "safe_deploy_labels": 0,
                    "checked_deploy_labels": 0,
                    "rejected_deploy_labels": 0,
                },
            }
            audit = semantic_audit(seed=7, success=False, prefix=0)
            entry = sink.append("PREFIX-CRASH", tick_states(), {
                "source_sha256": "a" * 64,
                "selection_index": 0,
                "selection_digest": "b" * 64,
                "native_replay_extent_v1": extent,
                "native_execution_pipeline": {
                    "contract_version": 4,
                    "mode": "single_semantic_seed_preflight_then_fixed_seed_trace_v4",
                    "preflight_chosen_seed": 7,
                    "preflight_semantics_sha256": "a" * 64,
                    "semantic_seed_selection": audit,
                },
            })
            sink.finalize()
            atomic_json(root / "results" / "PREFIX_CRASH.json", {
                **current_result_pipeline_fields(success=False),
                "schema_version": 1,
                "kind": "expert_authoritative_native_tick_result_v1",
                "battle_tag": "PREFIX-CRASH",
                "selection_index": 0,
                "selection_digest": "b" * 64,
                "source_sha256": "a" * 64,
                "teacher_forced_success": False,
                "failure": "native_rejected_tick_13_codes_[4]",
                "failure_domain": "semantic",
                "retry_scheduled": False,
                "audit_prefix_tick_store_entry": entry,
                "audit_prefix_extent": extent,
                "failure_prefix_tick_count": 4,
                "failure_prefix_semantic_match": True,
                "deployment_mask_label_rejections": 0,
                "deployment_mask_probe_rpc_count": 1,
                "native_deployment_mask_probes_attempted": 1,
                "native_deployment_mask_probe_exceptions": 0,
            })
            self.assertEqual(reconcile_result_files(root, queue_path), 1)
            with TickStoreWorkQueue(queue_path) as queue:
                self.assertEqual(queue.counts(), {"failed": 1})
            self.assertEqual(reconcile_result_files(root, queue_path), 0)

    def test_new_run_requeues_only_infrastructure_with_fresh_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite3"
            with TickStoreWorkQueue(queue_path) as queue:
                queue.add_tasks([
                    {
                        "battle_tag": tag, "source_path": f"{tag}.json",
                        "source_sha256": "a" * 64, "payload": {},
                    }
                    for tag in ("INFRA", "SEMANTIC")
                ])
                queue.connection.execute(
                    "UPDATE tasks SET status='failed', attempts=3"
                )
            for tag, domain in (
                ("INFRA", "infrastructure"), ("SEMANTIC", "semantic")
            ):
                atomic_json(root / "results" / f"{tag}.json", {
                    "schema_version": 1,
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": tag,
                    "teacher_forced_success": False,
                    "failure_domain": domain,
                })
            self.assertEqual(
                requeue_failed_infrastructure(root, queue_path), 1
            )
            with TickStoreWorkQueue(queue_path) as queue:
                rows = {
                    row["battle_tag"]: (row["status"], row["attempts"])
                    for row in queue.connection.execute(
                        "SELECT battle_tag,status,attempts FROM tasks"
                    )
                }
            self.assertEqual(rows["INFRA"], ("pending", 0))
            self.assertEqual(rows["SEMANTIC"], ("failed", 3))

    def test_native_action_metrics_use_attempted_not_planned_denominator(self) -> None:
        class FakeEnv:
            def joint_act(self, actions):
                return {
                    "actions": [
                        {"result": {"accepted": True, "result_code": 0}},
                        {"result": {"accepted": False, "result_code": 4}},
                    ]
                }

        recorder = RecordingCountingEnv(FakeEnv())
        recorder.joint_act([
            {"type": "play", "side": 0},
            {"type": "ability", "side": 1},
        ])
        metrics = recorder.metrics()
        self.assertEqual(metrics["native_actions_attempted"], 2)
        self.assertEqual(metrics["native_actions_accepted"], 1)
        self.assertEqual(metrics["native_actions_responded"], 2)
        self.assertEqual(metrics["native_actions_rejected"], 1)
        self.assertEqual(metrics["native_actions_no_response"], 0)
        self.assertEqual(metrics["true_attempted_acceptance_rate"], 0.5)
        self.assertEqual(metrics["native_deploy_actions_accepted"], 1)
        self.assertEqual(metrics["native_ability_actions_accepted"], 0)

        class TimeoutEnv:
            def joint_act(self, actions):
                raise TimeoutError("transport stalled")

        timed_out = RecordingCountingEnv(TimeoutEnv())
        with self.assertRaises(TimeoutError):
            timed_out.joint_act([{"type": "play", "side": 0}])
        timeout_metrics = timed_out.metrics()
        self.assertEqual(timeout_metrics["native_actions_attempted"], 1)
        self.assertEqual(timeout_metrics["native_actions_responded"], 0)
        self.assertEqual(timeout_metrics["native_actions_no_response"], 1)
        self.assertEqual(timeout_metrics["native_action_exceptions"], 1)

    def test_finalize_crash_window_rebuilds_missing_shard_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sink = WorkerShardSink(root, "worker", episodes_per_shard=1)
            sink.append("RECOVER", tick_states(), {"stable": 1})
            sink.finalize()
            manifest = next(root.glob("worker-*.manifest.json"))
            manifest.unlink()
            self.assertEqual(
                recover_unmanifested_final_shards(root, "worker"), 1
            )
            recovered = json.loads(manifest.read_text())
            self.assertTrue(recovered["recovered_after_finalize_crash"])
            self.assertEqual(recovered["episode_count"], 1)
            self.assertEqual(
                recover_unmanifested_final_shards(root, "worker"), 0
            )
            selection = root / "selection.jsonl"
            selection.write_text("{}\n", encoding="utf-8")
            build_store_manifest(
                root, source_manifest=selection,
                expected_episodes=1, expected_ticks=len(tick_states()),
            )
            self.assertEqual(
                verify_published_tick_store(root)["episodes"], 1
            )
            data = next(root.glob("worker-*.crts"))
            with data.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 0x01]))
            with self.assertRaises(RuntimeError):
                verify_published_tick_store(root)

    def test_summary_separates_semantic_rejection_from_infrastructure(self) -> None:
        task = NativeDatasetTask(
            0, "c" * 64, "A", "a.json", "d" * 64, 3, 100, 10,
            1, 1, "observed_ticks_identity_runtime_resolved",
            "all_card_events_raw_data_i", "authoritative_native_ability_exact",
        )
        base = {
            "battle_tag": "A",
            "source_sha_verified": True,
            "native_teacher_forced_profile": native_teacher_forced_profile(1),
            "coordinate_provenance": COORDINATE_PROVENANCE,
            "native_actions_attempted": 4,
            "native_actions_responded": 4,
            "native_actions_accepted": 3,
            "native_actions_rejected": 1,
            "native_actions_no_response": 0,
            "native_action_response_excess": 0,
            "native_action_exceptions": 0,
            "native_deploy_actions_attempted": 3,
            "native_deploy_actions_accepted": 3,
            "native_ability_actions_attempted": 1,
            "native_ability_actions_accepted": 0,
            "teacher_forced_success": False,
            "failure_class": "ability_branch_required",
            "failure_domain": "semantic",
            "terminal_diagnostic_status": "not_reached",
            "native_preflight_contract_version": 4,
            "native_execution_pipeline_mode": (
                "single_semantic_seed_preflight_then_fixed_seed_trace_v4"
            ),
            "preflight_teacher_forced_success": False,
            "preflight_chosen_seed": 7,
            "semantic_seed_preflight": semantic_audit(
                seed=7, success=False, prefix=0
            ),
            "full_trace_executed": False,
            "preflight_seconds": 0.5,
            "full_trace_seconds": 0.0,
            "avoided_trace_ticks": 70,
            "preflight_full_trace_semantic_match": None,
            "preflight_full_trace_semantic_diff": None,
            "tick_trace_complete_frames": 0,
            "deployment_mask_probe_rpc_count": 0,
        }
        summary = summarize_results(
            [task], [base], queue_counts={"failed": 1}, worker_reports=[{
                "worker_error": None,
            }], wall_seconds=2.0, missing_tags=[], unexpected_tags=[],
        )
        self.assertTrue(summary["infrastructure_complete"])
        self.assertFalse(summary["publication_ready"])
        self.assertEqual(summary["unframed_episodes"], 1)
        self.assertEqual(summary["true_attempted_acceptance_rate"], 0.75)
        self.assertEqual(summary["branch_required_battles"], 1)
        self.assertTrue(summary["native_action_accounting_closed"])
        self.assertTrue(summary["two_phase_preflight_integrity"])
        self.assertEqual(summary["preflight_rejections"], 1)
        self.assertEqual(summary["full_trace_executions"], 0)
        self.assertEqual(summary["preflight_seconds"], 0.5)
        self.assertEqual(summary["avoided_trace_ticks"], 70)

        infrastructure = dict(base)
        infrastructure.update({
            "failure_class": "infrastructure_native_replay_exception",
            "failure_domain": "infrastructure",
            "native_actions_responded": 3,
            "native_actions_rejected": 0,
            "native_actions_no_response": 1,
        })
        blocked = summarize_results(
            [task], [infrastructure], queue_counts={"failed": 1},
            worker_reports=[{"worker_error": None}], wall_seconds=2.0,
            missing_tags=[], unexpected_tags=[],
        )
        self.assertFalse(blocked["infrastructure_complete"])
        self.assertFalse(blocked["publication_ready"])
        self.assertEqual(
            blocked["failure_domain_counts"], {"infrastructure": 1}
        )

    def test_audit_prefix_store_is_physically_isolated_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.jsonl"
            selection.write_text("{}\n", encoding="utf-8")
            extent = {
                "schema_version": 1,
                "kind": "cr_native_replay_extent_v1",
                "extent": "valid_prefix",
                "training_admission": "actor_bc_censored_prefix_v1",
                "source_episode_complete": False,
                "every_native_tick_present_within_extent": True,
                "fixed_seed_replay": True,
                "chosen_seed": 7,
                "preflight_semantics_sha256": "a" * 64,
                "prefix_replay_semantics_sha256": "a" * 64,
                "semantic_match": True,
                "failure_class": "native_action_rejected",
                "failure_domain": "semantic",
                "failure": "native_rejected_tick_13_codes_[4]",
                "failure_source_tick": 12,
                "failure_execution_tick": 13,
                "first_invalid_source_event_index": 0,
                "safe_accepted_event_count": 0,
                "safe_accepted_action_transcript_sha256": "b" * 64,
                "observation_tick_start": 10,
                "observation_tick_stop_exclusive": 14,
                "action_label_tick_stop_exclusive": 13,
                "timing_censor_tick_exclusive": 13,
                "timing_target": "right_censored_at_failure_tick_v1",
                "failure_tick_has_labels": False,
                "terminal_target": "unknown_censored",
                "terminal_validated": False,
                "deployment_masks": "partial_native_visible_hand_complete_v1",
                "mask_coverage": {
                    "all_retained_visible_hand_slots_covered": True,
                    "retained_ticks": 3,
                    "actor_ticks": 6,
                    "visible_slot_references": 24,
                    "empty_slot_actor_ticks": 0,
                    "captured_slots": 8,
                    "safe_deploy_labels": 0,
                    "checked_deploy_labels": 0,
                    "rejected_deploy_labels": 0,
                },
                "trace_batches": 1,
                "trace_complete_frames": 4,
                "trace_incomplete_terminal_frames": 0,
                "trace_incomplete_nonterminal_freeze_frames": 0,
            }
            mask_result = replay_result_stub(
                success=False, with_masks=True
            )
            mask_store = DeploymentMaskStore(root)
            mask_store.publish_many(mask_result.deployment_mask_payloads)
            mask_manifest = mask_store.build_manifest()
            sink = WorkerShardSink(root, "prefix", episodes_per_shard=1)
            sink.append(
                "PREFIX",
                tick_states(),
                {
                    "native_replay_extent_v1": extent,
                    EPISODE_METADATA_KEY: mask_result.deployment_mask_metadata,
                },
            )
            sink.finalize()
            build_store_manifest(
                root,
                source_manifest=selection,
                expected_episodes=1,
                expected_ticks=4,
                store_kind=AUDIT_PREFIX_STORE_KIND,
                store_metadata={
                    "training_admission": "actor_bc_censored_prefix_v1",
                    "native_deployment_masks": {
                        "required": True,
                        "partial": True,
                        "schema_version": 1,
                        "dynamic_rule": DYNAMIC_RULE,
                        "manifest": "deployment-masks-v1/manifest.json",
                        "manifest_sha256": hashlib.sha256(
                            (root / "deployment-masks-v1" / "manifest.json")
                            .read_bytes()
                        ).hexdigest(),
                        "sidecars": mask_manifest["sidecars"],
                    },
                },
            )
            physical = verify_published_audit_prefix_store(root)
            self.assertEqual(physical["battle_tags"], ["PREFIX"])
            self.assertEqual(physical["deployment_mask_sidecars_referenced"], 8)
            with self.assertRaisesRegex(RuntimeError, "manifest kind"):
                verify_published_tick_store(root)


if __name__ == "__main__":
    unittest.main()
