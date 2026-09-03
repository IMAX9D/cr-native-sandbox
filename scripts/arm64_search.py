#!/usr/bin/env python3
"""Search an RVA-aligned ARM64 image for instruction-text regular expressions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("pattern")
    parser.add_argument("--start", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--stop", type=lambda value: int(value, 0))
    parser.add_argument("--context", type=int, default=3)
    args = parser.parse_args()

    stop = args.stop or args.binary.stat().st_size
    with args.binary.open("rb") as handle:
        handle.seek(args.start)
        data = handle.read(stop - args.start)
    disassembler = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    instructions = list(disassembler.disasm(data, args.start))
    matcher = re.compile(args.pattern)
    matched = 0
    for index, instruction in enumerate(instructions):
        text = f"{instruction.mnemonic} {instruction.op_str}".rstrip()
        if not matcher.search(text):
            continue
        matched += 1
        lower = max(0, index - args.context)
        upper = min(len(instructions), index + args.context + 1)
        print(f"--- match {matched} at 0x{instruction.address:x} ---")
        for nearby in instructions[lower:upper]:
            marker = ">" if nearby.address == instruction.address else " "
            print(
                f"{marker} {nearby.address:08x}: "
                f"{nearby.mnemonic:9s} {nearby.op_str}"
            )
    print(f"matches={matched}")


if __name__ == "__main__":
    main()
