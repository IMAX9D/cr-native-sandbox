#!/usr/bin/env python3
"""Find direct ARM64 BL/B references in a decrypted RVA-aligned image."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("target", type=lambda value: int(value, 0))
    parser.add_argument("--branch", choices=("bl", "b", "both"), default="bl")
    args = parser.parse_args()

    data = args.binary.read_bytes()
    for address in range(0, len(data) - 3, 4):
        word = struct.unpack_from("<I", data, address)[0]
        opcode = word >> 26
        kind = "bl" if opcode == 0b100101 else "b" if opcode == 0b000101 else None
        if kind is None or (args.branch != "both" and kind != args.branch):
            continue
        destination = address + sign_extend(word & 0x03FFFFFF, 26) * 4
        if destination == args.target:
            print(f"0x{address:x} {kind}")


if __name__ == "__main__":
    main()
