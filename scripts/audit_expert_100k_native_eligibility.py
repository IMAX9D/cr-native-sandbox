from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.audit_native_eligibility import DEFAULT_MANIFEST, DEFAULT_OUTPUT, run_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen expert corpus native teacher-forced eligibility without libg"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-rows", type=int, default=5_000)
    parser.add_argument("--native-contract", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(
        args.manifest,
        args.output,
        workers=args.workers,
        shard_rows=args.shard_rows,
        native_contract_path=args.native_contract,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "counts": summary["counts"],
        "tick_sums": summary["tick_sums"],
        "estimate": summary["pilot_based_success_estimate_not_100k_measurement"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
