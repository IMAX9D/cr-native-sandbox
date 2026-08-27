"""Isolated current-v3 frozen-libg expert-chain smoke.

This command never mutates the authoritative DB/index or the production
one-click journal.  It snapshots a deterministic coverage-oriented subset,
then reuses the production audit, generator, Tick/Mask validation, sparse
compiler, and two-batch training-smoke implementations under an isolated root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.freeze_schema5_manifest import freeze as formal_freeze  # noqa: E402
from expert_v1.one_click_v1 import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_CRAWLER_CONFIG,
    DEFAULT_CRAWLER_PYTHON,
    DEFAULT_CRAWLER_ROOT,
    DEFAULT_TEMPLATE,
    DEFAULT_TRAINING_PYTHON,
    OneClickConfig,
    OneClickError,
    OneClickLock,
    OneClickOrchestrator,
    _atomic_bytes,
    _atomic_json,
    _canonical,
    _crawler_active,
    _read_json,
    _verify_fingerprints,
    build_frozen_source_token_coverage_receipt,
    component_fingerprints,
    file_fingerprint,
    native_contract_binding,
    sha256_file,
    value_fingerprint,
)
from expert_v1.token_coverage_v1 import (  # noqa: E402
    _contract_index,
)


DEFAULT_OUTPUT_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\smoke-current-v3"
)
DEFAULT_AUTHORITATIVE_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
    r"\authoritative-schema5-v3"
)
DEFAULT_DB = DEFAULT_CRAWLER_ROOT / "data" / "authoritative-progress.sqlite3"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_PORTS = tuple(range(38_031, 38_039))
SELECTION_KIND = "cr_expert_v3_chain_smoke_greedy_multicover_v1"
STATE_KIND = "cr_expert_v3_chain_smoke_status_v1"


class ChainSmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeConfig(OneClickConfig):
    @property
    def frozen_manifest(self) -> Path:
        return self.data_root / "snapshot" / "schema5-v3-smoke.jsonl"

    @property
    def frozen_metadata(self) -> Path:
        return self.frozen_manifest.with_suffix(".jsonl.manifest.json")

    @property
    def source_pool_manifest(self) -> Path:
        return self.data_root / "snapshot" / "live-accepted-pool.jsonl"

    @property
    def source_pool_metadata(self) -> Path:
        return self.source_pool_manifest.with_suffix(".jsonl.manifest.json")

    @property
    def training_run_id(self) -> str:
        return "expert-v3-current-chain-smoke"


def validate_output_isolation(config: SmokeConfig) -> None:
    output = config.data_root.resolve()
    authoritative = config.authoritative_root.resolve()
    production_state = Path(
        r"D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3"
    ).resolve()
    if (
        output == production_state
        or output == authoritative
        or output.is_relative_to(authoritative)
        or authoritative.is_relative_to(output)
        or config.authoritative_db.resolve().is_relative_to(output)
        or output.is_relative_to(config.crawler_root.resolve())
    ):
        raise ChainSmokeError(
            "smoke output overlaps authoritative/crawler/production state"
        )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ChainSmokeError(f"JSON root is not an object: {path}")
    return value


def _stable_rank(tag: str) -> str:
    return hashlib.sha256(f"{SELECTION_KIND}:{tag}".encode()).hexdigest()


def candidate_features(
    manifest_row: Mapping[str, Any],
    battle: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    contract_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract only source-proven coverage features; never infer identity."""

    index = dict(contract_index or _contract_index(contract))
    tag = str(manifest_row.get("battle_tag") or "")
    if not tag or battle.get("battle_tag") != tag:
        raise ChainSmokeError("snapshot row/source battle tag mismatch")
    decks = {
        "team": tuple(str(token) for token in battle.get("team_deck") or []),
        "opponent": tuple(
            str(token) for token in battle.get("opponent_deck") or []
        ),
    }
    if any(len(deck) != 8 for deck in decks.values()):
        raise ChainSmokeError(f"snapshot deck is incomplete: {tag}")
    deck_tokens = set().union(*map(set, decks.values()))
    if not deck_tokens <= index["allowed_set"]:
        raise ChainSmokeError(f"snapshot deck escaped contract: {tag}")
    played = {
        str(event.get("card_form") or "")
        for event in battle.get("card_plays") or []
    }
    if not played <= deck_tokens:
        raise ChainSmokeError(f"source play token escaped its deck: {tag}")
    ability_candidates: set[str] = set()
    ability_singletons: set[str] = set()
    ability_events = list(battle.get("ability_plays") or [])
    for event in ability_events:
        side = str(event.get("side") or "")
        if side not in decks:
            raise ChainSmokeError(f"source ability side is invalid: {tag}")
        candidates = set(decks[side]) & index["ability_set"]
        if not candidates:
            raise ChainSmokeError(f"source ability has no contract candidate: {tag}")
        ability_candidates.update(candidates)
        if len(candidates) == 1:
            ability_singletons.update(candidates)
    return {
        "battle_tag": tag,
        "manifest_row": dict(manifest_row),
        "source_path": str(manifest_row["source_path"]),
        "played_cards": frozenset(played),
        "played_forms": frozenset(played & set(index["form_tokens"])),
        "ability_candidates": frozenset(ability_candidates),
        "ability_singletons": frozenset(ability_singletons),
        "ability_positive": bool(ability_events),
        "player_tags": frozenset(str(tag) for tag in manifest_row.get("player_tags") or []),
        "stable_rank": _stable_rank(tag),
    }


def greedy_select(
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    limit: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic weighted multi-cover with positive/zero-ability balance."""

    if limit <= 0 or len(candidates) < limit:
        raise ChainSmokeError(
            f"insufficient live candidates: {len(candidates)}/{limit}"
        )
    index = _contract_index(contract)
    required: dict[tuple[str, str], int] = {
        **{("card", token): 3 for token in index["allowed"]},
        **{("form", token): 3 for token in index["form_tokens"]},
        **{("ability", token): 3 for token in index["ability"]},
    }
    weights = {"card": 1, "form": 3, "ability": 8}
    selected: list[dict[str, Any]] = []
    selected_tags: set[str] = set()
    player_tags: set[str] = set()

    def features(row: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
        yield from (("card", token) for token in row["played_cards"])
        yield from (("form", token) for token in row["played_forms"])
        yield from (("ability", token) for token in row["ability_candidates"])

    while any(value > 0 for value in required.values()) and len(selected) < limit:
        scored = []
        for row in candidates:
            if row["battle_tag"] in selected_tags:
                continue
            gain = sum(
                weights[group] * min(1, required.get((group, token), 0))
                for group, token in features(row)
            )
            singleton_gain = sum(
                4 for token in row["ability_singletons"]
                if required.get(("ability", token), 0) > 0
            )
            new_players = len(set(row["player_tags"]) - player_tags)
            scored.append((gain, singleton_gain, new_players, row["stable_rank"], row))
        if not scored:
            break
        best = max(scored, key=lambda item: (item[0], item[1], item[2], -int(item[3], 16)))
        if best[0] + best[1] <= 0:
            break
        row = dict(best[-1])
        selected.append(row)
        selected_tags.add(str(row["battle_tag"]))
        player_tags.update(row["player_tags"])
        for key in features(row):
            if required.get(key, 0) > 0:
                required[key] -= 1

    # Reserve at least one quarter of the smoke for each cohort.  The rest is
    # filled by stable player diversity, never by current DB/list order.
    cohort_floor = max(1, limit // 4)
    for positive in (False, True):
        while (
            sum(bool(row["ability_positive"]) == positive for row in selected)
            < cohort_floor
            and len(selected) < limit
        ):
            pool = [
                row for row in candidates
                if row["battle_tag"] not in selected_tags
                and bool(row["ability_positive"]) == positive
            ]
            if not pool:
                break
            row = min(
                pool,
                key=lambda item: (
                    -len(set(item["player_tags"]) - player_tags),
                    item["stable_rank"],
                ),
            )
            row = dict(row)
            selected.append(row)
            selected_tags.add(str(row["battle_tag"]))
            player_tags.update(row["player_tags"])
    while len(selected) < limit:
        pool = [row for row in candidates if row["battle_tag"] not in selected_tags]
        if not pool:
            break
        row = min(
            pool,
            key=lambda item: (
                -len(set(item["player_tags"]) - player_tags),
                item["stable_rank"],
            ),
        )
        row = dict(row)
        selected.append(row)
        selected_tags.add(str(row["battle_tag"]))
        player_tags.update(row["player_tags"])

    deficits = {
        group: sorted(token for (kind, token), count in required.items() if kind == group and count > 0)
        for group in ("card", "form", "ability")
    }
    counts = {
        "ability_positive": sum(bool(row["ability_positive"]) for row in selected),
        "ability_zero": sum(not bool(row["ability_positive"]) for row in selected),
    }
    if len(selected) != limit or any(deficits.values()) or min(counts.values()) < cohort_floor:
        raise ChainSmokeError(
            "live corpus cannot satisfy deterministic smoke coverage: "
            + json.dumps({"selected": len(selected), "deficits": deficits, "cohorts": counts})
        )
    return selected, {
        "kind": SELECTION_KIND,
        "selected": len(selected),
        "cohort_floor": cohort_floor,
        "cohorts": counts,
        "contract_counts": {
            "cards": len(index["allowed"]),
            "evolution": len(index["evolution"]),
            "hero": len(index["hero"]),
            "ability": len(index["ability"]),
        },
        "multicover": {"card": 3, "form": 3, "ability": 3},
        "deficits": deficits,
        "selected_tags_sha256": hashlib.sha256(
            _canonical([row["battle_tag"] for row in selected])
        ).hexdigest(),
    }


class V3ChainSmokeRunner:
    def __init__(
        self,
        config: SmokeConfig,
        *,
        orchestrator: OneClickOrchestrator | None = None,
    ) -> None:
        self.config = config
        validate_output_isolation(config)
        self.orchestrator = orchestrator or OneClickOrchestrator(config)

    def _preflight(self) -> None:
        config = self.config
        validate_output_isolation(config)
        if (
            config.authoritative_root.name.casefold()
            != "authoritative-schema5-v3"
            or config.avds != 2
            or config.workers_per_avd != 4
            or config.workers != 8
            or config.ports != DEFAULT_PORTS
        ):
            raise ChainSmokeError("current-v3 smoke identity/layout changed")
        for required in (
            config.crawler_config,
            config.authoritative_db,
            config.authoritative_root / "index.jsonl",
            config.native_contract,
            config.native_contract.with_suffix(
                config.native_contract.suffix + ".sha256"
            ),
            config.template,
            config.training_python,
        ):
            if not required.exists():
                raise ChainSmokeError(f"required smoke dependency missing: {required}")
        native_contract_binding(config.native_contract)

    def snapshot(self) -> None:
        config = self.config
        inputs = component_fingerprints(
            config,
            "scripts/run_expert_v3_chain_smoke.py",
            "expert_v1/freeze_schema5_manifest.py",
            "expert_v1/token_coverage_v1.py",
        ) + [
            file_fingerprint(config.native_contract),
            value_fingerprint("smoke-snapshot", {
                "db": str(config.authoritative_db.resolve()),
                "authoritative_root": str(config.authoritative_root.resolve()),
                "sample_size": config.target,
                "selection_kind": SELECTION_KIND,
            }),
        ]

        def action() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            if _crawler_active(config):
                raise ChainSmokeError(
                    "authoritative crawler must be paused before snapshot smoke"
                )
            config.source_pool_manifest.parent.mkdir(parents=True, exist_ok=True)
            formal = formal_freeze(
                db_path=config.authoritative_db,
                authoritative_root=config.authoritative_root,
                output=config.source_pool_manifest,
                target=100_000,
                allow_incomplete=True,
                native_contract_path=config.native_contract,
            )
            pool_rows = [
                json.loads(line)
                for line in config.source_pool_manifest.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]
            live_index = config.authoritative_root / "index.jsonl"
            if (
                len(pool_rows) < config.target
                or int(formal.get("accepted_battles", -1)) != len(pool_rows)
                or formal.get("manifest_sha256")
                != sha256_file(config.source_pool_manifest)
                or formal.get("authoritative_index_sha256")
                != sha256_file(live_index)
            ):
                raise ChainSmokeError(
                    "formal live snapshot/index fence is incomplete or changed"
                )
            contract = _load_json(config.native_contract)
            contract_index = _contract_index(contract)
            candidates = [
                candidate_features(
                    row,
                    _load_json(Path(row["source_path"])),
                    contract,
                    contract_index=contract_index,
                )
                for row in pool_rows
            ]
            selected, selection = greedy_select(
                candidates, contract, limit=config.target
            )
            payload = b"".join(
                json.dumps(
                    row["manifest_row"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8") + b"\n"
                for row in selected
            )
            _atomic_bytes(config.frozen_manifest, payload)
            token_receipt = build_frozen_source_token_coverage_receipt(
                config.frozen_manifest, config.native_contract
            )
            source_coverage = token_receipt["source_coverage"]
            if (
                len(source_coverage["observed_card_tokens"]) != 180
                or len(source_coverage["observed_form_tokens"]) != 58
                or len(source_coverage["observed_ability_tokens"]) != 25
            ):
                raise ChainSmokeError("selected snapshot lost full token coverage")
            _atomic_json(config.source_token_coverage_receipt, token_receipt)
            metadata = {
                "schema_version": 1,
                "kind": "cr_expert_v3_chain_smoke_snapshot_v1",
                "production_ready": True,
                "accepted_battles": config.target,
                "target_battles": config.target,
                "manifest_path": str(config.frozen_manifest.resolve()),
                "manifest_sha256": sha256_file(config.frozen_manifest),
                "native_contract_sha256": native_contract_binding(
                    config.native_contract
                )["canonical_sha256"],
                "native_contract_file_sha256": native_contract_binding(
                    config.native_contract
                )["file_sha256"],
                "source_pool": formal,
                "live_db_observation": file_fingerprint(config.authoritative_db),
                "live_index_observation": file_fingerprint(
                    live_index
                ),
                "selection": selection,
            }
            _atomic_json(config.frozen_metadata, metadata)
            if _crawler_active(config):
                raise ChainSmokeError("crawler resumed during snapshot fence")
            _verify_fingerprints(inputs)
            return [
                file_fingerprint(config.source_pool_manifest),
                file_fingerprint(config.source_pool_metadata),
                file_fingerprint(config.frozen_manifest),
                file_fingerprint(config.frozen_metadata),
                file_fingerprint(config.source_token_coverage_receipt),
            ], selection

        self.orchestrator._run_stage("freeze_schema5_v3", inputs, action)

    def run(self) -> None:
        self._preflight()
        if _crawler_active(self.config):
            raise ChainSmokeError(
                "pause authoritative collection before running the 2-AVD smoke"
            )
        self.snapshot()
        self.orchestrator.audit()
        with OneClickLock(self.config.native_hardware_lock_path):
            stopped = False
            try:
                self.orchestrator.generate_native()
                self.orchestrator.stop_workers()
                stopped = True
            finally:
                if not stopped:
                    self.orchestrator._best_effort_stop_native()
        self.orchestrator.validate_tick_store()
        self.orchestrator.compile()
        self.orchestrator.smoke()


def build_config(args: argparse.Namespace) -> SmokeConfig:
    return SmokeConfig(
        project_root=PROJECT_ROOT,
        data_root=args.output_root.resolve(),
        crawler_root=args.crawler_root.resolve(),
        crawler_python=args.crawler_python.resolve(),
        training_python=args.training_python.resolve(),
        crawler_config=args.crawler_config.resolve(),
        authoritative_db=args.authoritative_db.resolve(),
        authoritative_root=args.authoritative_root.resolve(),
        native_contract=args.native_contract.resolve(),
        template=args.template.resolve(),
        target=args.sample_size,
        workers=8,
        avds=2,
        workers_per_avd=4,
        ports=DEFAULT_PORTS,
        native_layout_reason="isolated_current_v3_chain_smoke_fixed_2x4",
        audit_workers=max(8, args.audit_workers),
        compile_io_workers=max(1, args.compile_io_workers),
        compile_process_workers=max(1, args.compile_process_workers),
    )


def status(config: SmokeConfig) -> dict[str, Any]:
    state = _read_json(config.state_path)
    stages = (state.get("stages") or {}) if isinstance(state, Mapping) else {}
    return {
        "schema_version": 1,
        "kind": STATE_KIND,
        "output_root": str(config.data_root),
        "sample_size": config.target,
        "worker_layout": {"avds": 2, "workers_per_avd": 4, "workers": 8},
        "active_stage": state.get("active_stage"),
        "last_error": state.get("last_error"),
        "stages": {
            name: (stages.get(name) or {}).get("status", "pending")
            for name in (
                "freeze_schema5_v3",
                "audit_schema5_v3",
                "generate_native_ticks",
                "stop_native_workers",
                "validate_tick_store_and_masks",
                "compile_native_bc",
                "real_data_training_smoke",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--crawler-root", type=Path, default=DEFAULT_CRAWLER_ROOT)
    parser.add_argument("--crawler-python", type=Path, default=DEFAULT_CRAWLER_PYTHON)
    parser.add_argument("--training-python", type=Path, default=DEFAULT_TRAINING_PYTHON)
    parser.add_argument("--crawler-config", type=Path, default=DEFAULT_CRAWLER_CONFIG)
    parser.add_argument("--authoritative-db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--authoritative-root", type=Path, default=DEFAULT_AUTHORITATIVE_ROOT
    )
    parser.add_argument("--native-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--audit-workers", type=int, default=20)
    parser.add_argument("--compile-io-workers", type=int, default=32)
    parser.add_argument("--compile-process-workers", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size <= 0:
        raise ChainSmokeError("--sample-size must be positive")
    config = build_config(args)
    if args.status:
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return 0
    with OneClickLock(config.lock_path):
        V3ChainSmokeRunner(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
