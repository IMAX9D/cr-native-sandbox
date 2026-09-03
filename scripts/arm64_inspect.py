#!/usr/bin/env python3
"""Small local helper for comparing decrypted ARM64 libg instruction ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("address", type=lambda value: int(value, 0))
    parser.add_argument("--before", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument("--size", type=lambda value: int(value, 0), default=0x140)
    args = parser.parse_args()

    start = max(0, args.address - args.before)
    with args.binary.open("rb") as handle:
        handle.seek(start)
        data = handle.read(args.size)

    disassembler = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    for instruction in disassembler.disasm(data, start):
        marker = ">" if instruction.address == args.address else " "
        print(
            f"{marker} {instruction.address:08x}: "
            f"{instruction.mnemonic:9s} {instruction.op_str}"
        )


if __name__ == "__main__":
    main()
