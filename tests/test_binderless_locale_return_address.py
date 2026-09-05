"""Execute the locale guard against a dispatcher that observes its caller."""
from __future__ import annotations

import ctypes
import mmap
from pathlib import Path
import platform
import re
import struct
import unittest


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "x86_64",
                     "executes the Linux x86_64 native host guard")
class BinderlessLocaleReturnAddressTest(unittest.TestCase):
    def test_dispatcher_sees_original_callsite_and_null_paths_return(self) -> None:
        source = (Path(__file__).resolve().parents[1] /
                  "android_probe/native/jni_bridge.cpp").read_text()

        def rva(name: str) -> int:
            return int(re.search(rf"{name} = (0x[0-9A-Fa-f]+);", source)[1], 16)

        def code(name: str) -> bytes:
            body = re.search(rf"{name} = \{{(.*?)\}};", source, re.S)[1]
            body = re.sub(r"//[^\n]*", "", body)
            return bytes(int(value, 16) for value in re.findall(r"0x[0-9A-Fa-f]+", body))

        site = rva("kBinderlessUiLocaleContainerGuardRva")
        cave = rva("kBinderlessUiLocaleContainerGuardCaveRva")
        dispatch = rva("kBinderlessUiLocaleRegistrationTargetRva")
        sentinel = 0x123456789ABCDEF0
        with mmap.mmap(-1, dispatch + 4096,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC) as memory:
            address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
            # A normal caller initializes the result and invokes the original
            # site; the dispatcher's result is the return address it received.
            caller = b"\x48\xB8" + struct.pack("<Q", sentinel)
            caller += b"\xE8" + struct.pack("<i", site - 15) + b"\xC3"
            memory[:len(caller)] = caller
            memory[site:site + 5] = code("binderless_ui_locale_container_guard_(?:call|jump)")
            memory[site + 5:site + 6] = b"\xC3"
            guard = code("binderless_ui_locale_container_cave_code")
            memory[cave:cave + len(guard)] = guard
            memory[dispatch:dispatch + 5] = b"\x48\x8B\x04\x24\xC3"
            invoke = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(address)
            registry = ctypes.c_uint64(1)
            container = ctypes.c_void_p(ctypes.addressof(registry))
            empty = ctypes.c_void_p()
            self.assertEqual(invoke(ctypes.addressof(container)), address + site + 5)
            self.assertEqual(invoke(None), sentinel)
            self.assertEqual(invoke(ctypes.addressof(empty)), sentinel)


if __name__ == "__main__":
    unittest.main()
