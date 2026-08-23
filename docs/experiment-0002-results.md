# Experiment 0002 results

Date: 2026-08-23

## Outcome

The strict no-Surface direct-native gate passed.

The frozen x86_64 `libg.so` now cold-starts without creating, attaching, or
borrowing an Android Surface. It loads the original content, creates a standard
1v1 `BattleGameState`, advances 100 original logic ticks, and reproduces the
same public baseline state as the renderer-backed control.

No battle rule is reimplemented. Card, deployment, movement, pathfinding,
targeting, collision, damage, RNG, hand cycle, tower, and command behavior
remain original `libg` code.

## Exact native chain

The direct path uses these frozen-build entry points:

1. `GameMain::init` at RVA `0x727050`, with byte-guarded presentation-only
   call sites bypassed while the function runs.
2. The original LoadingState/DataTables chain:
   - range loader `0xE74B40` over all 158 resource descriptors;
   - task start `0xCDC5B0`;
   - task pump `0xCDC620`;
   - task completion `0xCDC5A0`;
   - LoadingState update/completion `0xCE98F0` / `0xCE9750`.
3. The original native resource request-list chain for
   `locations/training_arena.csv` and `tilemaps/tilemap.csv`:
   `0x12B6FD0 -> 0x12B7320 -> 0x12B7480`.
4. The original battlefield-cache builder at `0xE2AF80`.
5. The original outer replay loader at `0x10B85B0`. This applies `rndSeed`,
   initializes hand order and cycle, sets battle phase 4, and then calls the
   inner battle JSON loader. Calling only the inner loader produced valid
   towers but the wrong RNG and hash.
6. Core-only stepping at `0xCE2CC0`, fixed at 20 Hz (`0.05 s`) for exactly
   100 calls.

The DataTables range call changes scheduling only: all parsing and table
construction still happen in the original loader. The headless adapter holds
the completed resource gate at state 7 and supplies LoadingState's
presentation-only ready latch after both native futures reach phase 5.

## Presentation boundary

The permanent process-local shim is RVA `0x137F220`, the resource-variant
dispatcher. For variants 1 and 2 it returns an empty render resource; other
variants already return empty in the original function. Logic arena and
tilemap files are loaded separately through the original request-list API.

All other patches are byte-guarded temporary presentation call-site patches
and are restored immediately after their guarded native call. No `LogicBattle`
or command function is patched.

## Ten-process certificate

Command:

```powershell
.\scripts\accept_direct_core.ps1 -Runs 10
```

Summary evidence:

`D:\AI_data\cr-native-core\acceptance-direct-core\20260823-135630-645-acceptance-summary.json`

Every fresh process passed all assertions:

- no `surface_create` event;
- native DataTables/LoadingState completed;
- battle ready at tick 0;
- exact tick transition `0 -> 100`;
- six native crown-tower entities;
- king HP `4824 / 4824` and princess HP `3052 / 3052`;
- native battle phase 4;
- RNG state `3502570521`;
- coherent public observation;
- canonical hash `5594aa3c81dc52fa`.

There was one unique hash across all ten runs.

## Measurements

| Metric | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Fresh-process wall time | 12.973 s | 13.095 s | 13.529 s |
| Replay injection through 100-tick observation | 11.553 ms | 13.126 ms | 18.474 ms |
| Validated-path throughput | 5,413 tick/s | about 7,618 tick/s | 8,656 tick/s |

The validated-path throughput is conservative because it includes replay
injection, battle creation, and final observation in addition to 100 core
updates. The cold wall metric also includes host/device deployment checks and
a deliberate 5-second platform initialization fence; it is not a per-episode
training reset cost.

## Remaining engineering gate

This result proves direct native feasibility and determinism. It does not yet
prove production self-play throughput. The next gate is a persistent worker
that can reset battles without restarting the Android process, accept actions
through the stable ABI, batch observations, and demonstrate repeated episodes
without leaks or state carry-over. Multi-worker scaling should be measured only
after that reset gate, so cold-start orchestration is not confused with kernel
step speed.
