# Experiment 0002 results

Date: 2026-08-23

## Outcome

This experiment crossed the direct-native initialization gate but did not yet
cross the direct battle gate.

The frozen x86_64 `libg.so` now executes its original `GameMain::init` at RVA
`0x727050` with no Android Surface. The call returns normally and publishes:

- resource loader with vtable RVA `0x1924680`;
- game helper;
- stage registry;
- `GameStateManager`;
- native DataTables owner and its loader adapter.

Only presentation construction is suppressed. The shim is bound to the exact
`libg` build ID and verifies every original instruction sequence before making
a temporary patch. All instructions are restored immediately after init. No
`LogicBattle`, card, movement, targeting, collision, damage, RNG, or command
function is patched.

## Direct readiness result

Evidence:
`D:\AI_data\cr-native-core\experiment-0001\20260823-120430-843-probe-direct.log`

The probe exits normally with:

```text
status = blocked_data_tables
game_main_initialized = true
manager_initialized = true
surface_created = false
battle_data_root != 0
battle_data_object != 0
battle_data_loader != 0
battle_data_content = 0
```

The earlier crashes are now explained. `BattleGameState` obtains the table
container through `0x7DB0F0`; battle construction later reads its first table
through `0xE26E60`. The Surface-free path reaches that point with the container
allocated but the table slot null. The probe now checks this invariant before
replay injection and reports a structured blocker instead of producing a
native tombstone.

## Detached-Surface control

Evidence:
`D:\AI_data\cr-native-core\experiment-0001\20260823-120444-043-probe-detach-surface.log`

The control profile creates an offscreen Surface only for platform bootstrap,
loads the replay, pauses, destroys and releases both Java Surface objects, and
then calls the native core directly. It completed:

- tick `0 -> 100`;
- `100 / 100` requested native steps;
- six tower entities;
- king HP `4824 / 4824`;
- princess HP `3052 / 3052`;
- canonical state hash `5594aa3c81dc52fa`;
- end-to-end measured interval about `122 ms`, including the deliberate
  `100 ms` pause fence.

This proves battle simulation itself does not require a live Surface. The
remaining dependency is cold content-table loading only.

## Architecture decision

Two paths remain deliberately separate:

1. `probe-detach-surface` is the working headless-worker baseline. It is usable
   for throughput and multi-worker measurements now, without MuMu or a visible
   game window.
2. `probe-direct` is the strict no-Surface experiment. It must not borrow table
   pointers from another process, copy already-initialized heap state, or patch
   missing tables manually. The next acceptable change is to identify and call
   the original platform content-loader completion chain.

## Next gate

Map the caller that populates the `0xB70` native DataTables object after
`GameMain::init`, invoke that original loader/callback chain in the direct
profile, and require `battle_data_content != 0` before creating HomeState or
BattleGameState. After that, rerun the exact 100-tick certificate and then the
ten-fresh-process determinism gate.
