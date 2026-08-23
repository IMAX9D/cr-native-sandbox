# Scroll audit

Pinned upstream: `d897141dd5fb6c09753ff29761ef95c051c4655f`

## What Scroll actually does

Scroll is an Android Rust `cdylib`. `JNI_OnLoad` starts a local TCP server,
initializes resources, and accepts a patched Clash Royale client. It discovers
the already-loaded `libg.so` base from `/proc/self/maps` and calls native
functions at `base + RVA` through typed `extern "cdecl"` wrappers.

The public source provides native-layout wrappers for, among other types:

- `LogicBattle`
- `LogicSpell`, `LogicSpellDeck`, and `LogicSpellCollection`
- `LogicDataTables`, `LogicLocationData`, and `LogicSpellData`
- `LogicClientHome`, avatar objects, commands, JSON nodes, strings, and arrays

It allocates object storage with libc `malloc`, then invokes the matching libg
constructor. Battle setup uses native `set_location` and `set_spell_decks`
methods rather than reproducing their behavior in Rust.

## Important limits

- The reference targets Clash Royale v1.3.2 and ARMv7 Thumb functions.
- The repository contains only two commits and exposes training battles, not a
  complete modern PvP/training API.
- It remains an Android library and still relies on libg being loaded in the
  same process. It is not a desktop ELF runner.
- Every layout, calling convention, RVA, constructor prerequisite, and resource
  format must be rediscovered for runtime `150535029` x86_64.

## Transferable design

The reusable idea is deliberately small:

1. load the packaged original libg in a controlled process;
2. attest its identity before any call;
3. resolve a versioned binding table from module base plus audited RVAs;
4. represent native objects with ABI-accurate layouts;
5. allocate storage and call native constructors;
6. use native data loaders, battle setup, commands, and tick functions; and
7. expose a narrow external protocol without rewriting gameplay.

The old numeric offsets are explicitly non-transferable.

