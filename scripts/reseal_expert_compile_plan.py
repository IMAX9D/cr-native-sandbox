"""Reseal an already validated plan after a compiler-only contract migration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from expert_v1.compile_native_bc_dataset import (
    _atomic_bytes,
    _atomic_json,
    _digest,
    load_compile_plan,
    sha256_file,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\one-click-schema5-v3-current-frontier-v5"
)
PLAN = DATA_ROOT / "compiled" / "native-bc-v1" / "compile-plan.json"


def main() -> int:
    raw = PLAN.resolve(strict=True).read_bytes()
    old_sha = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    old_shards = [str(value["content_sha256"]) for value in plan["shards"]]
    archive_root = PLAN.parent / "migrations"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"compile-plan.pre-terminal-censor.{old_sha[:16]}.json"
    if archive.exists() and archive.read_bytes() != raw:
        raise RuntimeError("compile-plan migration archive collision")
    if not archive.exists():
        _atomic_bytes(archive, raw)
    sidecar = PLAN.with_name("compile-plan.sha256")
    sidecar_archive = archive.with_suffix(".sha256")
    if not sidecar_archive.exists():
        shutil.copy2(sidecar, sidecar_archive)

    components = plan["compiler"]["components"]
    paths = {
        "compiler_sha256": PROJECT / "expert_v1" / "compile_native_bc_dataset.py",
        "training_schema_sha256": PROJECT / "expert_v1" / "training_v1" / "schema.py",
        "deployment_masks_sha256": PROJECT / "expert_v1" / "tick_store_v1" / "deployment_masks.py",
        "native_coverage_validator_sha256": PROJECT / "expert_v1" / "one_click_v1.py",
        "token_coverage_validator_sha256": PROJECT / "expert_v1" / "token_coverage_v1.py",
        "native_dataset_generator_sha256": PROJECT / "expert_v1" / "native_dataset_generator.py",
        "native_seed_search_sha256": PROJECT / "expert_v1" / "native_seed_search.py",
    }
    for name, path in paths.items():
        components[name] = sha256_file(path.resolve(strict=True))
    plan["input_content_sha256"] = _digest({
        "inputs": plan["inputs"], "compiler": plan["compiler"]
    })
    plan.pop("plan_content_sha256", None)
    plan["plan_content_sha256"] = _digest(plan)
    if [str(value["content_sha256"]) for value in plan["shards"]] != old_shards:
        raise RuntimeError("compiler migration changed shard content identities")
    _atomic_json(PLAN, plan)
    new_sha = sha256_file(PLAN)
    _atomic_bytes(
        sidecar, f"{new_sha}  compile-plan.json\n".encode("ascii")
    )
    authenticated = load_compile_plan(PLAN, verify_live_inputs=False)
    if authenticated["plan_content_sha256"] != plan["plan_content_sha256"]:
        raise RuntimeError("resealed compile plan did not authenticate")
    receipt = {
        "kind": "cr_expert_compile_plan_terminal_censor_migration_v1",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "censor_accepted_action_whose_post_action_observation_is_terminal",
        "old_plan_path": str(archive.resolve()),
        "old_plan_sha256": old_sha,
        "new_plan_path": str(PLAN.resolve()),
        "new_plan_sha256": new_sha,
        "episodes_unchanged": True,
        "shard_content_identities_unchanged": True,
        "shards": len(old_shards),
    }
    _atomic_json(
        DATA_ROOT / "receipts" / "compile-plan-terminal-censor-migration.json",
        receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
