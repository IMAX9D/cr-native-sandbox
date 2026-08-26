from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from native_core.card_catalog import card_cost, metadata as card_metadata

from expert_v1.compile_native_bc_dataset import (
    CAPACITY_MINIMUM_RESERVE_BYTES,
    EpisodeInput,
    NativeBcCompileError,
    _acquire_capacity_reservation,
    _assign_components,
    _release_capacity_reservation,
    _stratified_capacity_sample,
    compile_planned_shards,
    create_compile_plan,
    finalize_dataset,
    load_compile_plan,
)
from expert_v1.native_ingest_contract import (
    load_native_ingest_contract,
    write_native_ingest_contract,
)
from expert_v1.native_replay_plan import compile_battle
from expert_v1.one_click_v1 import (
    evaluate_ability_positive_coverage,
    file_fingerprint,
    native_contract_binding,
    validate_native_result_records,
)
from expert_v1.tick_store_v1.deployment_masks import (
    DeploymentMaskContractError,
    DeploymentMaskStore,
    NativeDeploymentMaskCapture,
)
from expert_v1.tick_store_v1.schema import (
    EntityState,
    EpisodeState,
    PlayerPrivate,
    TickState,
    TowerState,
)
from expert_v1.tick_store_v1.shard import (
    AppendOnlyShardWriter,
    build_store_manifest,
)
from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.schema import (
    read_manifest,
    unpack_sparse_grid,
    validate_shard,
    verify_dataset_integrity,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "expert_schema5_authoritative.json"


class _Probe:
    def __init__(
        self,
        costs: dict[tuple[int, int], int],
        card_ids: dict[tuple[int, int], int],
    ) -> None:
        self.costs = costs
        self.card_ids = card_ids

    def probe_grid(self, *, side: int, deck_index: int) -> dict[str, object]:
        cost = self.costs[(side, deck_index)]
        return {
            "width": 18,
            "height": 32,
            "cell_size": 1000,
            "valid_cells": 576,
            "rows": ["1" * 18 for _ in range(32)],
            "resolved_data_id": self.card_ids[(side, deck_index)],
            "packed_selection": 0,
            "card_cost": cost,
            "card_cost_raw": cost * 10_000,
            "selection_form_index": 0,
            "selection_strategy": "canonical",
            "selection_builder_rva": "0x1",
            "selection_root_vtable_rva": "0x2",
        }


def _tower_states() -> tuple[TowerState, ...]:
    values = []
    for side in (0, 1):
        king_y = 3000 if side == 0 else 29000
        princess_y = 6500 if side == 0 else 25500
        values.extend(
            (
                TowerState(side * 3, side, 0, -1, 9000, king_y, 7728, 7728),
                TowerState(side * 3 + 1, side, 1, 0, 3500, princess_y, 4858, 4858),
                TowerState(side * 3 + 2, side, 1, 1, 14500, princess_y, 4858, 4858),
            )
        )
    return tuple(values)


def _states(
    source: dict[str, object], native_ingest_contract: object | None = None
) -> tuple[TickState, ...]:
    plan = compile_battle(
        source, native_ingest_contract=native_ingest_contract
    )
    first_card = plan.sides[0].deck[0]
    first_metadata = card_metadata(first_card.card_id)
    entity_card_id = (
        int(first_metadata["hero_form_id"])
        if first_card.form_flags & 2
        else int(first_card.card_id)
    )
    entity_has_ability = bool(first_card.form_flags & 2)
    hands = [list(plan.sides[side].cycle.initial_hand) for side in (0, 1)]
    queues = [list(plan.sides[side].cycle.initial_queue) for side in (0, 1)]
    actions: dict[tuple[int, int], int] = {
        (action.tick + 1, action.side): action.logical_card_index
        for action in plan.actions
    }
    result = []
    for tick in range(100, 281):
        players = tuple(
            PlayerPrivate(side, 100_000, tuple(hands[side]), queues[side][0])
            for side in (0, 1)
        )
        entities = ()
        if tick >= 122:
            entities = (
                EntityState(
                    5_000_001,
                    0,
                    8500,
                    10500,
                    entity_card_id,
                    14,
                    1000,
                    1500,
                    1,
                    1 if entity_has_ability else 0,
                    1 if entity_has_ability else -1,
                    1 if entity_has_ability else 0,
                    -1,
                    -1,
                    -1,
                    -1,
                ),
            )
        result.append(
            TickState(
                tick,
                players,  # type: ignore[arg-type]
                _tower_states(),
                entities,
                EpisodeState(1, 0, 1, 1, 0, 1, 0, 0, 0),
            )
        )
        for side in (0, 1):
            played = actions.get((tick, side))
            if played is None:
                continue
            slot = hands[side].index(played)
            hands[side][slot] = queues[side].pop(0)
            queues[side].append(played)
    return tuple(result)


class NativeBcCompilerTests(unittest.TestCase):
    def _inputs(
        self,
        root: Path,
        *,
        source_contract_mismatch: bool = False,
        episode_contract_mismatch: bool = False,
        source_contract_file_mismatch: bool = False,
        episode_contract_file_mismatch: bool = False,
    ) -> tuple[Path, Path, Path, Path]:
        contract = root / "native-contract.json"
        published = write_native_ingest_contract(contract)
        loaded_contract = load_native_ingest_contract(contract)
        contract_value = json.loads(contract.read_text(encoding="utf-8"))
        base = json.loads(FIXTURE.read_text(encoding="utf-8"))
        base["authoritative_native_contract"] = {
            "game_version": contract_value["game_version"],
            "contract_sha256": contract_value["contract_sha256"],
            "contract_file_sha256": published["file_sha256"],
        }
        sources = root / "sources"
        sources.mkdir()
        index = root / "schema5.jsonl"
        rows = []
        values: list[
            tuple[dict[str, object], dict[str, object], Path, str]
        ] = []
        for number in range(3):
            source = deepcopy(base)
            tag = f"SCHEMA5FIXTURE{number}"
            source["battle_tag"] = tag
            source["team_tags"] = [f"TEAM{number}"]
            source["opponent_tags"] = [f"OPP{number}"]
            source["rounds"][0]["team"][0]["player_tag"] = f"TEAM{number}"
            source["rounds"][0]["opponent"][0]["player_tag"] = f"OPP{number}"
            source["deck_metadata"]["source_list_url"] = (
                f"https://royaleapi.com/player/TEAM{number}/battles/history?before=1"
            )
            if number == 0:
                # Exercise the conditional ability head in the same compiled
                # Tick Store path (identity is resolved from the exact native
                # entity at the expert Tick, never from opponent-private data).
                source["team_deck"][0] = "knight-hero"
                team = source["rounds"][0]["team"][0]
                team["full_deck"][0] = "knight-hero"
                team["deck_cards"][0].update(
                    slug="knight-hero", base_slug="knight", form="hero"
                )
                level = team["card_levels"].pop("knight-ev1")
                team["card_levels"]["knight-hero"] = level
                source["card_plays"][0]["card_form"] = "knight-hero"
                source["ability_plays"] = [
                    {
                        "time": 8.5,
                        "time_raw": 170,
                        "side": "team",
                        "color": "blue",
                        "ability_id": None,
                        "resolution_status": "unresolved",
                        "marker_index": 16,
                        "data_i": 0,
                        "x_raw": None,
                        "y_raw": None,
                        "coordinate_status": "not_applicable",
                    }
                ]
                source["elixir_stats"]["team"]["Ability"]["count"] = 1
            valid_source = deepcopy(source)
            if source_contract_mismatch and number == 0:
                source["authoritative_native_contract"]["contract_sha256"] = "0" * 64
            if source_contract_file_mismatch and number == 0:
                source["authoritative_native_contract"]["contract_file_sha256"] = "2" * 64
            path = sources / f"{tag}.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "battle_tag": tag,
                    "source_schema_version": 5,
                    "source_path": str(path),
                    "source_sha256": digest,
                    "player_tags": [f"OPP{number}", f"TEAM{number}"],
                    "source_group": f"group-{number}",
                }
            )
            values.append((source, valid_source, path, digest))
        index.write_bytes(
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            )
        )

        tick_root = root / "tick-store"
        tick_root.mkdir()
        mask_store = DeploymentMaskStore(tick_root)
        with AppendOnlyShardWriter(tick_root, "worker-00000") as writer:
            for source, valid_source, path, source_sha in values:
                plan = compile_battle(
                    valid_source, native_ingest_contract=loaded_contract
                )
                slots = [
                    {
                        "side": side,
                        "deck_index": index_value,
                        "card_id": card.card_id,
                        "level": card.level,
                        "form_flags": card.form_flags,
                        "source_token": card.source_token,
                        "base_token": card.base_token,
                    }
                    for side in (0, 1)
                    for index_value, card in enumerate(plan.sides[side].deck)
                ]
                costs = {
                    (value["side"], value["deck_index"]): int(card_cost(value["card_id"]))
                    for value in slots
                }
                card_ids = {
                    (value["side"], value["deck_index"]): int(value["card_id"])
                    for value in slots
                }
                capture = NativeDeploymentMaskCapture(slots)
                capture.capture_available(
                    _Probe(costs, card_ids),
                    {
                        "tick": 100,
                        "players": [
                            {"side": side, "hand_deck_indices": [0, 1, 2, 3]}
                            for side in (0, 1)
                        ],
                    },
                )
                capture.capture_available(
                    _Probe(costs, card_ids),
                    {
                        "tick": 101,
                        "players": [
                            {"side": side, "hand_deck_indices": [4, 5, 6, 7]}
                            for side in (0, 1)
                        ],
                    },
                )
                mask_store.publish_many(capture.payloads)
                states = _states(valid_source, loaded_contract)
                writer.append(
                    str(source["battle_tag"]),
                    states,
                    {
                        "source_path": str(path),
                        "source_sha256": source_sha,
                        "source_schema_version": 5,
                        "action_execution_tick_offset": 1,
                        "authoritative_contract_sha256": (
                            "1" * 64
                            if episode_contract_mismatch
                            and source["battle_tag"] == "SCHEMA5FIXTURE0"
                            else contract_value["contract_sha256"]
                        ),
                        "authoritative_contract_file_sha256": (
                            "3" * 64
                            if episode_contract_file_mismatch
                            and source["battle_tag"] == "SCHEMA5FIXTURE0"
                            else published["file_sha256"]
                        ),
                        "native_deployment_masks_v1": capture.metadata(),
                    },
                )
            writer.finalize()
        mask_store.build_manifest()
        build_store_manifest(
            tick_root,
            source_manifest=index,
            expected_episodes=3,
            expected_ticks=3 * 181,
        )
        candidate_queue = root / "candidate-queue.jsonl"
        candidate_queue.write_text(
            "".join(
                json.dumps(
                    {
                        "battle_tag": f"SCHEMA5FIXTURE{number}",
                        "ability_events_observed": 1 if number == 0 else 0,
                    }
                )
                + "\n"
                for number in range(3)
            ),
            encoding="utf-8",
        )
        results_path = root / "native-results.jsonl"
        results_path.write_text(
            "".join(
                json.dumps(
                    {
                        "kind": "expert_authoritative_native_tick_result_v1",
                        "battle_tag": f"SCHEMA5FIXTURE{number}",
                        "final_attempt": True,
                        "teacher_forced_success": True,
                    }
                )
                + "\n"
                for number in range(3)
            ),
            encoding="utf-8",
        )
        audit = validate_native_result_records(
            results_path, candidate_queue, expected_rows=3
        )
        ability_coverage = evaluate_ability_positive_coverage(
            {"ability_positive": 1, "ability_zero": 2},
            audit,
            minimum_success_count=1,
            minimum_success_rate=0.10,
            waived=False,
            waiver_reason=None,
        )
        receipt = root / "native-generation-coverage.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "cr_expert_native_generation_coverage_v2",
                    "created_utc": "2026-08-27T00:00:00+00:00",
                    "frozen_manifest": file_fingerprint(index),
                    "candidate_queue": file_fingerprint(candidate_queue),
                    "results": file_fingerprint(results_path),
                    "native_contract": native_contract_binding(contract),
                    "target_battles": 3,
                    "selected_battles": 3,
                    "processed_battles": 3,
                    "teacher_forced_successes": 3,
                    "teacher_forced_failures": 0,
                    "stored_episodes": 3,
                    "success_rate": 1.0,
                    "minimum_success_rate": 0.50,
                    "ability_coverage": ability_coverage,
                    "failure_class_counts": {},
                    "failure_domain_counts": {},
                    "terminal_diagnostic_counts": {},
                    "queue_counts": {},
                    "native_actions_attempted": 0,
                    "native_actions_accepted": 0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return tick_root, index, contract, receipt

    def test_end_to_end_actor_safe_content_addressed_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            plan = create_compile_plan(
                tick_root,
                source,
                output,
                contract,
                receipt,
                maximum_rows_per_shard=10_000,
                io_workers=2,
            )
            resumed_plan = create_compile_plan(
                tick_root,
                source,
                output,
                contract,
                receipt,
                maximum_rows_per_shard=10_000,
                io_workers=2,
            )
            self.assertEqual(resumed_plan, plan)
            first = compile_planned_shards(plan, process_workers=1)
            self.assertEqual(len(first), 3)
            self.assertTrue(all(not value["resumed"] for value in first))
            resumed = compile_planned_shards(plan, process_workers=1)
            self.assertTrue(all(value["resumed"] for value in resumed))
            result = finalize_dataset(plan)
            self.assertEqual(result["battles"], 3)
            first_manifest_bytes = (output / "manifest.json").read_bytes()
            first_manifest_sha = (output / "manifest.sha256").read_bytes()
            second_result = finalize_dataset(plan)
            self.assertEqual(second_result["manifest_sha256"], result["manifest_sha256"])
            self.assertEqual((output / "manifest.json").read_bytes(), first_manifest_bytes)
            self.assertEqual((output / "manifest.sha256").read_bytes(), first_manifest_sha)
            manifest = read_manifest(output)
            verified_manifest, integrity = verify_dataset_integrity(output, workers=2)
            self.assertEqual(
                verified_manifest["dataset_content_sha256"],
                manifest["dataset_content_sha256"],
            )
            self.assertGreater(integrity["shard_files"], 0)
            self.assertTrue(manifest["native_replay_validated"])
            self.assertEqual(manifest["schema_version"], 2)
            capacity = json.loads(
                (output / "capacity-preflight.json").read_text(encoding="utf-8")
            )
            self.assertTrue(capacity["passed"])
            self.assertGreater(capacity["sample_actor_rows"], 0)
            self.assertLess(capacity["sample_bytes_per_actor_row"], 4_608)
            self.assertEqual(
                manifest["capacity_preflight"], plan["capacity_preflight"]
            )
            published_coverage = manifest["native_generation_coverage"]
            self.assertEqual(
                published_coverage["receipt_sha256"],
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                published_coverage["ability_coverage"],
                json.loads(receipt.read_text())["ability_coverage"],
            )
            self.assertEqual(manifest["quality_gates"]["split_collisions"], 0)
            assignments = [
                json.loads(line)
                for line in (output / "split-assignments.jsonl").read_text().splitlines()
            ]
            self.assertEqual({value["split"] for value in assignments}, {"train", "validation", "test"})
            ability_labels = 0
            for paths in manifest["splits"].values():
                for relative in paths:
                    validate_shard(output / relative, manifest)
                    self.assertFalse((output / relative / "enemy_hand.npy").exists())
                    ability_labels += int(
                        np.load(output / relative / "ability_label_mask.npy").sum()
                    )
            self.assertGreater(ability_labels, 0)
            dataset = NativeExpertSequenceDataset(
                output, split="train", sequence_length=64, burn_in=8
            )
            batch = collate_sequences([dataset[0]])
            self.assertIn("entity_tokens", batch)
            self.assertTrue(batch["entity_mask"].any())
            self.assertEqual(str(batch["entity_tokens"].dtype), "torch.int64")
            shard = output / manifest["splits"]["train"][0]
            grid = unpack_sparse_grid(
                np.load(shard / "grid_offsets.npy"),
                np.load(shard / "grid_indices.npy"),
                np.load(shard / "grid_values.npy"),
                start=0,
                stop=1,
                channels=int(manifest["dimensions"]["grid_channels"]),
            )
            tokens = np.load(shard / "entity_tokens.npy")
            self.assertEqual(grid.dtype, np.uint8)
            self.assertTrue(np.issubdtype(tokens.dtype, np.integer))
            self.assertGreater(int(tokens.max(initial=0)), 0)
            # Windows keeps mmap files locked until every array view is
            # released; explicitly close the test dataset before temp cleanup.
            del batch
            dataset._arrays.clear()
            del dataset

    def test_missing_mask_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            sidecar = next((tick_root / "deployment-masks-v1").glob("[0-9a-f][0-9a-f]/*.json"))
            sidecar.unlink()
            with self.assertRaises(DeploymentMaskContractError):
                create_compile_plan(
                    tick_root, source, root / "compiled", contract, receipt
                )

    def test_fully_resigned_shard_metadata_forgery_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            plan = create_compile_plan(
                tick_root, source, output, contract, receipt,
                maximum_rows_per_shard=10_000,
            )
            compile_planned_shards(plan, process_workers=1)
            shard = output / plan["shards"][0]["relative_path"]
            path = shard / "shard.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["rows"] += 1
            body = {
                key: item for key, item in value.items()
                if key != "metadata_content_sha256"
            }
            value["metadata_content_sha256"] = hashlib.sha256(
                (
                    json.dumps(body, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n"
                ).encode("utf-8")
            ).hexdigest()
            path.write_bytes(
                (
                    json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n"
                ).encode("utf-8")
            )
            with self.assertRaisesRegex(
                NativeBcCompileError, "metadata disagrees"
            ):
                finalize_dataset(plan)

    def test_capacity_preflight_is_bound_and_low_disk_fails_before_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            disk_usage = type("DiskUsage", (), {"total": 1024**4, "used": 0, "free": 1})()
            with patch(
                "expert_v1.compile_native_bc_dataset.shutil.disk_usage",
                return_value=disk_usage,
            ):
                with self.assertRaisesRegex(NativeBcCompileError, "capacity preflight failed"):
                    create_compile_plan(
                        tick_root,
                        source,
                        output,
                        contract,
                        receipt,
                        maximum_rows_per_shard=10_000,
                    )
            self.assertFalse((output / "compile-plan.json").exists())
            failed = json.loads(
                (output / "capacity-preflight.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failed["disk_gate_passed"])
            self.assertFalse(failed["passed"])

    def test_capacity_preflight_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            create_compile_plan(
                tick_root,
                source,
                output,
                contract,
                receipt,
                maximum_rows_per_shard=10_000,
            )
            path = output / "capacity-preflight.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["projected_output_bytes"] += 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(NativeBcCompileError, "capacity preflight"):
                load_compile_plan(output / "compile-plan.json")

    def test_native_coverage_receipt_sha_is_bound_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            create_compile_plan(
                tick_root, source, output, contract, receipt,
                maximum_rows_per_shard=10_000,
            )
            receipt.write_text(receipt.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(
                NativeBcCompileError, "native generation receipt changed"
            ):
                load_compile_plan(output / "compile-plan.json")

    def test_ability_failures_require_explicit_waiver_at_compiler_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            value = json.loads(receipt.read_text())
            candidate_queue = Path(value["candidate_queue"]["path"])
            results_path = Path(value["results"]["path"])
            candidate_rows = [
                {
                    "battle_tag": f"SCHEMA5FIXTURE{number}",
                    "ability_events_observed": 0,
                }
                for number in range(3)
            ] + [{"battle_tag": "ABILITYFAIL", "ability_events_observed": 1}]
            result_rows = [
                {
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": f"SCHEMA5FIXTURE{number}",
                    "final_attempt": True,
                    "teacher_forced_success": True,
                }
                for number in range(3)
            ] + [
                {
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": "ABILITYFAIL",
                    "final_attempt": True,
                    "teacher_forced_success": False,
                    "failure_class": "semantic",
                }
            ]
            candidate_queue.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows),
                encoding="utf-8",
            )
            results_path.write_text(
                "".join(json.dumps(row) + "\n" for row in result_rows),
                encoding="utf-8",
            )
            audit = validate_native_result_records(
                results_path, candidate_queue, expected_rows=4
            )
            ability = evaluate_ability_positive_coverage(
                {"ability_positive": 1, "ability_zero": 3},
                audit,
                minimum_success_count=1,
                minimum_success_rate=0.10,
                waived=False,
                waiver_reason=None,
            )
            value.update(
                target_battles=4,
                selected_battles=4,
                processed_battles=4,
                teacher_forced_successes=3,
                teacher_forced_failures=1,
                stored_episodes=3,
                success_rate=0.75,
                ability_coverage=ability,
                candidate_queue=file_fingerprint(candidate_queue),
                results=file_fingerprint(results_path),
                failure_class_counts={"semantic": 1},
            )
            receipt.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                NativeBcCompileError, "ability coverage receipt is not admitted"
            ):
                create_compile_plan(
                    tick_root, source, root / "rejected", contract, receipt,
                    maximum_rows_per_shard=10_000,
                )

            value["ability_coverage"] = evaluate_ability_positive_coverage(
                {"ability_positive": 1, "ability_zero": 3},
                audit,
                minimum_success_count=1,
                minimum_success_rate=0.10,
                waived=True,
                waiver_reason="known issue CR-ABILITY-1",
            )
            receipt.write_text(json.dumps(value) + "\n", encoding="utf-8")
            plan = create_compile_plan(
                tick_root, source, root / "waived", contract, receipt,
                maximum_rows_per_shard=10_000,
            )
            self.assertEqual(
                plan["inputs"]["native_generation_receipt_sha256"],
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
            )

    def test_fully_resigned_capacity_arithmetic_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            create_compile_plan(
                tick_root, source, output, contract, receipt,
                maximum_rows_per_shard=10_000,
            )
            capacity_path = output / "capacity-preflight.json"
            plan_path = output / "compile-plan.json"
            base_capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
            base_plan = json.loads(plan_path.read_text(encoding="utf-8"))

            def canonical(item: object) -> bytes:
                return (
                    json.dumps(item, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8") + b"\n"
                )

            mutations = {
                "safety": lambda value: value.__setitem__("safety_factor", 1.36),
                "projected": lambda value: value.__setitem__(
                    "projected_output_bytes", int(value["projected_output_bytes"]) + 1
                ),
                "safe": lambda value: value.__setitem__(
                    "projected_output_with_safety_bytes",
                    int(value["projected_output_with_safety_bytes"]) + 1,
                ),
                "reserve": lambda value: value.__setitem__(
                    "minimum_reserve_bytes", int(value["minimum_reserve_bytes"]) + 1
                ),
                "required": lambda value: value.__setitem__(
                    "required_free_bytes", int(value["required_free_bytes"]) + 1
                ),
                "episode_upper": lambda value: value["sample_episodes"][0].__setitem__(
                    "array_payload_bytes",
                    int(value["sample_episodes"][0]["array_payload_bytes"]) + 1,
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    capacity = deepcopy(base_capacity)
                    mutate(capacity)
                    capacity_body = {
                        key: item for key, item in capacity.items()
                        if key != "content_sha256"
                    }
                    capacity["content_sha256"] = hashlib.sha256(
                        canonical(capacity_body)
                    ).hexdigest()
                    capacity_raw = canonical(capacity)
                    capacity_path.write_bytes(capacity_raw)
                    plan = deepcopy(base_plan)
                    reference = plan["capacity_preflight"]
                    reference["file_sha256"] = hashlib.sha256(capacity_raw).hexdigest()
                    reference["content_sha256"] = capacity["content_sha256"]
                    for field in (
                        "sample_actor_rows", "sample_bytes_per_actor_row",
                        "sample_max_episode_bytes_per_actor_row",
                        "projected_output_bytes", "required_free_bytes",
                    ):
                        reference[field] = capacity[field]
                    plan["plan_content_sha256"] = hashlib.sha256(
                        canonical({
                            key: item for key, item in plan.items()
                            if key != "plan_content_sha256"
                        })
                    ).hexdigest()
                    plan_raw = canonical(plan)
                    plan_path.write_bytes(plan_raw)
                    (output / "compile-plan.sha256").write_text(
                        f"{hashlib.sha256(plan_raw).hexdigest()}  compile-plan.json\n",
                        encoding="ascii",
                    )
                    with self.assertRaisesRegex(
                        NativeBcCompileError, "capacity"
                    ):
                        load_compile_plan(plan_path)

    def test_capacity_sample_is_hash_duration_density_stratified(self) -> None:
        episodes = [
            EpisodeInput(
                battle_tag=f"B{index:04d}", tick_data_path="x", tick_index_path="y",
                tick_count=100 + (index % 20) * 100,
                tick_payload_sha256="a" * 64,
                tick_payload_size=(100 + (index % 20) * 100) * (5 + index % 17),
                source_path="z", source_sha256="b" * 64,
                source_group=f"g{index}", player_tags=("p", "q"),
                split="train", component_sha256="c" * 64,
            )
            for index in range(240)
        ]
        first = _stratified_capacity_sample(episodes)
        second = _stratified_capacity_sample(list(reversed(episodes)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertGreater(len({value.tick_count for value in first}), 10)
        self.assertGreater(
            len({value.tick_payload_size / value.tick_count for value in first}), 10
        )

    def test_cross_process_disk_reservations_cannot_oversell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = SimpleNamespace(
                total=100 * 1024**3,
                used=0,
                free=CAPACITY_MINIMUM_RESERVE_BYTES + 1_500,
            )
            with patch(
                "expert_v1.compile_native_bc_dataset.shutil.disk_usage",
                return_value=fake,
            ):
                first = _acquire_capacity_reservation(
                    root, relative_path="shards/a", requested_bytes=1_000
                )
                try:
                    with self.assertRaisesRegex(
                        NativeBcCompileError, "reservation failed"
                    ):
                        _acquire_capacity_reservation(
                            root, relative_path="shards/b", requested_bytes=1_000
                        )
                finally:
                    _release_capacity_reservation(root, first)
            self.assertFalse(list((root / ".capacity-reservations-v1").glob("*.json")))

    def test_worker_count_is_capacity_clamped_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            plan = create_compile_plan(
                tick_root, source, output, contract, receipt,
                maximum_rows_per_shard=10_000,
            )
            compile_planned_shards(plan, process_workers=64)
            receipt = json.loads(
                (output / "worker-receipts" / "worker-00000-of-00001.json").read_text()
            )
            self.assertEqual(receipt["requested_process_workers"], 64)
            self.assertLess(receipt["effective_process_workers"], 64)
            self.assertLessEqual(
                receipt["effective_process_workers"],
                receipt["capacity_recommended_partition_process_workers"],
            )

    def test_resigned_legacy_dense_multi_terabyte_plan_cannot_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            create_compile_plan(
                tick_root,
                source,
                output,
                contract,
                receipt,
                maximum_rows_per_shard=10_000,
            )
            path = output / "compile-plan.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["compiler"]["storage_schema"]["grid"] = (
                "legacy_dense_actor_rows_uint8_4tb_v1"
            )

            def canonical(item: object) -> bytes:
                return (
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )

            value["input_content_sha256"] = hashlib.sha256(
                canonical(
                    {
                        "inputs": value["inputs"],
                        "compiler": value["compiler"],
                    }
                )
            ).hexdigest()
            value["plan_content_sha256"] = hashlib.sha256(
                canonical(
                    {
                        key: item
                        for key, item in value.items()
                        if key != "plan_content_sha256"
                    }
                )
            ).hexdigest()
            raw = canonical(value)
            path.write_bytes(raw)
            (output / "compile-plan.sha256").write_text(
                f"{hashlib.sha256(raw).hexdigest()}  compile-plan.json\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(NativeBcCompileError, "compiler semantics"):
                load_compile_plan(path)

    def test_source_and_episode_contract_must_equal_cli_contract(self) -> None:
        for field in (
            "source_canonical",
            "source_file",
            "episode_canonical",
            "episode_file",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tick_root, source, contract, receipt = self._inputs(
                    root,
                    source_contract_mismatch=field == "source_canonical",
                    source_contract_file_mismatch=field == "source_file",
                    episode_contract_mismatch=field == "episode_canonical",
                    episode_contract_file_mismatch=field == "episode_file",
                )
                with self.assertRaisesRegex(
                    NativeBcCompileError,
                    "contract differs from CLI contract",
                ):
                    create_compile_plan(
                        tick_root, source, root / "compiled", contract, receipt,
                        io_workers=2
                    )

    def test_resigned_hand_edited_compile_plan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            output = root / "compiled"
            create_compile_plan(
                tick_root, source, output, contract, receipt, io_workers=2
            )
            path = output / "compile-plan.json"
            sidecar = output / "compile-plan.sha256"
            original = json.loads(path.read_text(encoding="utf-8"))

            def resign(value: dict[str, object]) -> None:
                raw = (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                path.write_bytes(raw)
                sidecar.write_text(
                    f"{hashlib.sha256(raw).hexdigest()}  compile-plan.json\n",
                    encoding="ascii",
                )

            mutations = {
                "created": lambda value: value.__setitem__(
                    "created_utc", "2099-01-01T00:00:00+00:00"
                ),
                "kind": lambda value: value.__setitem__("kind", "forged"),
                "schema": lambda value: value.__setitem__("schema_version", 4),
                "count": lambda value: value.__setitem__(
                    "episodes", int(value["episodes"]) + 1
                ),
                "vocab": lambda value: value["card_vocabulary"].__setitem__(1, "forged"),
                "shard": lambda value: value["shards"][0].__setitem__(
                    "estimated_rows", int(value["shards"][0]["estimated_rows"]) + 2
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = deepcopy(original)
                    mutate(changed)
                    resign(changed)
                    with self.assertRaises(NativeBcCompileError):
                        load_compile_plan(path)

    def test_player_source_graph_never_crosses_splits(self) -> None:
        rows = [
            {
                "battle_tag": "A",
                "player_tags": ("P0", "P1"),
                "source_group": "G0",
                "source_sha256": "a" * 64,
                "tick_count": 100,
            },
            {
                "battle_tag": "B",
                "player_tags": ("P1", "P2"),
                "source_group": "G1",
                "source_sha256": "b" * 64,
                "tick_count": 100,
            },
            {
                "battle_tag": "C",
                "player_tags": ("P3", "P4"),
                "source_group": "G2",
                "source_sha256": "c" * 64,
                "tick_count": 100,
            },
            {
                "battle_tag": "D",
                "player_tags": ("P5", "P6"),
                "source_group": "G3",
                "source_sha256": "d" * 64,
                "tick_count": 100,
            },
        ]
        assignments, audit = _assign_components(
            rows, seed=7, validation_fraction=0.2, test_fraction=0.2
        )
        self.assertEqual(assignments["A"], assignments["B"])
        self.assertEqual(audit["player_holdout_leaks"], 0)

    def test_source_mutation_after_tick_capture_fails_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tick_root, source, contract, receipt = self._inputs(root)
            # Make battle 1 share a player with battle 0.  The corresponding
            # Tick source SHA must change too, so this mutation correctly trips
            # content identity before a leaky split can be published.
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            path = Path(rows[1]["source_path"])
            value = json.loads(path.read_text())
            value["team_tags"] = ["TEAM0"]
            value["rounds"][0]["team"][0]["player_tag"] = "TEAM0"
            path.write_text(json.dumps(value))
            rows[1]["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(NativeBcCompileError):
                create_compile_plan(
                    tick_root, source, root / "compiled", contract, receipt
                )


if __name__ == "__main__":
    unittest.main()
