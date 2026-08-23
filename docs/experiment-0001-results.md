# Experiment 0001 results

Date: 2026-08-23

## Verdict

The current runtime can execute an exact 100-tick native battle after its
renderer is detached, but it cannot yet be cold-started by calling the native
manager constructor alone. The repository therefore **does not pass** the
first acceptance gate and makes no 100%-separated-kernel claim.

This is still a useful architectural result: Python battle-rule patching is
not involved, and the remaining dependency is now isolated to native resource
initialization before `GameStateManager`, rather than battle simulation.

## Lifecycle matrix

| Profile | Surface | Manager after 5 s | Result |
| --- | --- | ---: | --- |
| baseline | created and retained | yes | replay bootstrap was unstable in this cold slot |
| no-surface | never created | no | replay rejected: manager not ready |
| create-only | never created | no | replay rejected: manager not ready |
| minimal | never created | no | replay rejected: manager not ready |
| direct | never created | constructor entered | native null dereference in missing resource-registry chain |
| detach-surface | destroyed before stepping | yes | **100/100 native ticks completed** |

The no-Surface result rules out the idea that `nOnCreate`, `nOnStart`, and
`nOnResume` merely need time. The manager global remained zero while the
native registry map reached four entries.

## Direct-construction evidence

Static mapping shows:

- `0xCE65B0`: allocate `0xD0`, call manager constructor, publish global;
- `0xCE7000`: manager constructor;
- `0xCE6660`: state factory, with type `4` selecting BattleGameState;
- `0xCE7B40`: pending-state transition;
- `0xCE7810`: manager update.

Calling `0xCE65B0` after `CreateGameMain` but before a Surface reaches the real
constructor and then crashes at `0xCE70FC -> 0xE6F610 -> 0xE71605 ->
0x11E6FC2`. The fault is a null resource/registry receiver, which is consistent
with Scroll explicitly having a separate `resources::init()` stage. Directly
constructing Manager is therefore one stage too late; the prerequisite
resource-init chain must be mapped first.

## Successful 100-tick proof

Evidence log:
`D:\AI_data\cr-native-core\experiment-0001\20260823-110123-625-probe-detach-surface.log`

- profile: `probe-detach-surface`
- replay parsed by libg: true
- native state: BattleGameState type 4
- tick: exactly `0 -> 100`
- requested/stepped: `100 / 100`
- fixed delta: `0.05`
- battle active: true
- native phase: battle 4, logic 3, substate 1
- canonical public state hash: `a6a9ec3511559524`
- end-to-end replay-load, pause fence, Surface destruction, 100 steps, and
  observation: `135.664508 ms` (about 737 ticks/s including a deliberate
  100 ms pause fence; this is not a standalone core throughput benchmark)

The Surface had been destroyed and both Java objects released before
`nativeStep` ran. No MuMu process or visible game window was used.

## Two independent blockers exposed

1. Cold bootstrap is nondeterministic: the detach profile succeeded once in
   four fresh-process attempts on the warmed AVD. This is not acceptable for a
   training worker.
2. The native replay currently produces king HP 2400, princess HP 3052, while
   the frozen acceptance target expects king HP 4824. The observed registry
   contains four ordinary tower entities while the terminal snapshot can read
   all six. Content/tower configuration must be corrected before determinism
   certification; hiding this mismatch would invalidate training.

## Next experiment

Trace the successful high-level initialization path around `0x727050` and its
calls preceding `0x7278E1 -> 0xCE65B0`. Reproduce only the resource/data-table
initializers needed by the manager constructor, then rerun `probe-direct`.
The next gate is manager creation with no Surface and no crash; ten-run state
hash certification comes only after the tower configuration matches the frozen
target.
