#!/usr/bin/env python3
"""Export the frozen libg/RoyaleAPI native-ingest contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_ingest_contract import (  # noqa: E402
    DEFAULT_BINDING_PATH,
    DEFAULT_CONTRACT_PATH,
    write_native_ingest_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING_PATH)
    args = parser.parse_args()
    result = write_native_ingest_contract(
        args.output, binding_path=args.binding
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

