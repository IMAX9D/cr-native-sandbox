from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.diagnose_native_replay_instrumentation_ab import (
    DiagnosticError,
    InstrumentedEnv,
    _strict_manifest_rows,
    parse_cases,
    run_four_lanes,
    validate_run_contract,
)


class FakeEnv:
    def __init__(self) -> None:
        self.tick = 10
        self.probes = 0

    def state(self) -> dict:
        return {
            "tick": self.tick,
            "coherent": True,
            "players": [
                {
                    "side": side,
                    "elixir_raw": 100_000,
                    "hand_deck_indices": [0, 1, 2, 3],
                    "next_deck_index": 4,
                    "refill_timer": 0,
                }
                for side in (0, 1)
            ],
            "entities": [],
            "episode": {
                "terminated": False,
                "truncated": False,
                "winner": None,
                "crowns": [0, 0],
                "crown_towers": [],
            },
        }

    def reset(self, _replay: dict, *, warmup_steps: int) -> dict:
        self.tick = warmup_steps
        return self.state()

    def observe_train(self) -> dict:
        return self.state()

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        self.probes += 1
        return {"side": side, "deck_index": deck_index, "rows": ["1"]}

    def joint_act(self, _actions: list[dict]) -> dict:
        return {"actions": []}

    def step(self, steps: int = 1) -> dict:
        self.tick += steps
        return {
            "tick_after": self.tick,
            "stepped": steps,
            "episode": {"terminated": False, "truncated": False},
        }

    def trace_train(self, steps: int = 1, **_kwargs: object) -> dict:
        initial = self.state()
        self.tick += steps
        final = self.state()
        return {
            "stepped": steps,
            "initial_frame": {"state": initial},
            "frames": [{"state": final}],
        }


class FakeResult(SimpleNamespace):
    def json(self) -> dict:
        return {
            "battle_tag": self.battle_tag,
            "failure": self.failure,
            "chosen_seed": self.chosen_seed,
        }


def fake_result(*, seed: int, failure: str | None) -> FakeResult:
    accepted = 1 if failure is None else 0
    return FakeResult(
        battle_tag="TAG",
        teacher_forced_success=failure is None,
        failure=failure,
        accepted_actions=accepted,
        accepted_deploy_actions=accepted,
        accepted_ability_actions=0,
        action_acceptance_sequence=tuple(
            [{"accepted": True, "source_event_index": 0}]
            if accepted else []
        ),
        final_tick=11,
        terminal_validated=False,
        terminal_match=None,
        terminal_diagnostic_status="native_terminal_missing",
        source_crowns=None,
        observed_crowns=None,
        terminal_tower_hp_validated=False,
        terminal_tower_hp_match=None,
        terminal_tower_hp_diagnostic_status="not_requested",
        source_final_tower_hp=None,
        observed_final_tower_hp=None,
        chosen_seed=seed,
        seeds_tested=0,
        seed_search_native_resets=1,
        layout_resolution_mode="fixed_preflight_seed_replay",
        tick_trace_batches=0,
        tick_trace_complete_frames=0,
        tick_trace_incomplete_terminal_frames=0,
        tick_trace_incomplete_nonterminal_freeze_frames=0,
        deployment_mask_probe_rpc_count=0,
        deployment_mask_base_probe_rpc_count=0,
        deployment_mask_dynamic_label_probe_rpc_count=0,
        deployment_mask_slots_captured=0,
        deployment_mask_capture_complete=False,
        deployment_mask_label_checks=0,
        deployment_mask_label_rejections=0,
        deployment_mask_first_label_rejection=None,
    )


class NativeReplayInstrumentationAbTests(unittest.TestCase):
    def test_case_parser_is_explicit_and_unique(self) -> None:
        self.assertEqual(parse_cases(["ABC=7"]), (("ABC", 7),))
        with self.assertRaisesRegex(DiagnosticError, "unique TAG=SEED"):
            parse_cases(["ABC=7", "ABC=8"])
        with self.assertRaisesRegex(DiagnosticError, "positive"):
            parse_cases(["ABC=0"])

    def test_probe_guard_records_same_tick_control_and_next_step(self) -> None:
        wrapped = InstrumentedEnv(FakeEnv())
        wrapped.reset({}, warmup_steps=10)
        wrapped.probe_grid(side=0, deck_index=1)
        wrapped.joint_act([])
        wrapped.step(1)
        row = wrapped.probe_checks[0]
        self.assertTrue(row["same_tick"])
        self.assertTrue(row["same_control"])
        self.assertIsNotNone(row["post_action_control_sha256"])
        self.assertIsNotNone(row["post_step_control_sha256"])

    def test_four_lanes_use_one_exact_fixed_seed_reset(self) -> None:
        calls: list[dict] = []

        def execute(env: InstrumentedEnv, _plan: object, _template: object,
                    **kwargs: object) -> FakeResult:
            calls.append(dict(kwargs))
            env.reset({}, warmup_steps=10)
            env.observe_train()
            if kwargs["capture_deployment_masks"]:
                env.probe_grid(side=0, deck_index=0)
            env.joint_act([])
            if kwargs["tick_sink"] is not None:
                env.trace_train(1)
                kwargs["tick_sink"].append("TAG", [], {})
            else:
                env.step(1)
            return fake_result(
                seed=int(kwargs["fixed_seed"]),
                failure=(
                    "derived_deployment_mask_rejected_source_event_0"
                    if kwargs["capture_deployment_masks"] else None
                ),
            )

        value = run_four_lanes(
            FakeEnv(), object(), {}, fixed_seed=61, execute=execute
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [
                (call["tick_sink"] is not None,
                 call["capture_deployment_masks"])
                for call in calls
            ],
            [(False, False), (True, False), (False, True), (True, True)],
        )
        self.assertTrue(all(call["seed"] == 61 for call in calls))
        self.assertTrue(all(call["fixed_seed"] == 61 for call in calls))
        self.assertFalse(
            value["comparisons"]["mask_only"]["semantic_equal_to_preflight"]
        )
        self.assertTrue(
            value["lanes"]["mask_only"]["instrumentation"][
                "all_probes_same_control"
            ]
        )

    def test_selection_and_run_contract_are_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps({"battle_tag": "TAG", "schema_version": 5}),
                encoding="utf-8",
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            selection = root / "selection.jsonl"
            selection.write_text(
                json.dumps({
                    "battle_tag": "TAG",
                    "source_path": str(source),
                    "source_sha256": source_sha,
                }) + "\n",
                encoding="utf-8",
            )
            contract = root / "contract.json"
            contract.write_text("{}", encoding="utf-8")
            template = root / "template.json"
            template.write_text("{}", encoding="utf-8")
            run_contract = root / "run-contract.json"
            run_contract.write_text(json.dumps({
                "kind": "expert_authoritative_native_tick_generator_v1",
                "selection_manifest": str(selection),
                "selection_manifest_sha256": hashlib.sha256(
                    selection.read_bytes()
                ).hexdigest(),
                "native_ingest_contract": {
                    "path": str(contract),
                    "file_sha256": hashlib.sha256(
                        contract.read_bytes()
                    ).hexdigest(),
                },
                "template": str(template),
                "template_sha256": hashlib.sha256(
                    template.read_bytes()
                ).hexdigest(),
            }), encoding="utf-8")
            validate_run_contract(
                run_contract,
                selection=selection,
                native_contract=contract,
                template=template,
            )
            rows = _strict_manifest_rows(selection, (("TAG", 7),))
            self.assertEqual(rows[0]["fixed_seed"], 7)
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DiagnosticError, "source SHA changed"):
                _strict_manifest_rows(selection, (("TAG", 7),))


if __name__ == "__main__":
    unittest.main()
