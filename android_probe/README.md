# Android lifecycle host

This directory contains the Android/Java/JNI host for the current headless
`libg.so` sandbox. The authoritative architecture document is
[`docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md`](../docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md).

The Java class and JNI method names remain version-bound to
`royale.nativehost.JniHost`. The source began as an isolated lifecycle probe;
it now also owns the persistent JSON-line service used by the sandbox.

## Components

- `java/royale/nativehost/JniHost.java`: lifecycle orchestration, resource
  bootstrap, replay replacement and the JSON-line server.
- `java/royale/nativehost/HeadlessApplication.java`: minimal Application shell.
- `java/royale/nativehost/HeadlessActivity.java`: Activity-compatible shell
  without a visible window.
- `native/jni_bridge.cpp`: version guard, native calls, card/ability commands,
  Tick, observation, reset and terminal latching.
- `java/com/supercell/titan/**`: minimal Java symbols required by the frozen
  APK/native runtime.

## Lifecycle profiles

1. `probe-baseline`: create Activity and Surface, then start/resume.
2. `probe-detach-surface`: bootstrap through baseline, destroy/release the
   Surface, then advance the logic core.
3. `probe-null-surface`: issue native surface callbacks with `null`; never
   construct a Surface object.
4. `probe-no-surface`: create/start/resume without Surface callbacks.
5. `probe-create-only`: stop after `GameApp.nOnCreate()`.
6. `probe-minimal`: stop after `CreateGameMain`.
7. `probe-direct`: strict no-Surface `GameMain::init` plus DataTables, arena,
   Replay and 100-Tick attestation.
8. `serve-direct`: the same strict direct initialization, retained as a
   persistent sandbox service.

The direct profiles do not attempt Replay creation before DataTables and map
resources are ready. Failure is returned as evidence; no profile silently
falls back to another lifecycle.

## `serve-direct`

`serve-direct` exposes one request/one response JSON lines over TCP. The main
pure-sandbox operations are `ping`, `status`, `reset`, `observe`, `step`,
`step_trace`, `probe_grid`, `act`, `ability`, joint actions/transitions and
`shutdown`.

Reset uses the original manager's BattleGameState type-4 to type-4 replacement.
Card play uses the original `DoSpellCommand`; active abilities use native
command type `0x5A`. The bridge fails closed on version, pointer, schema,
entity-count, path-count or command validation errors.

## Surface rule

`probe-direct` and `serve-direct` must not create, attach, borrow or retain an
Android Surface. Presentation-only calls may be bypassed only at verified
version-specific sites; battle logic, commands, movement, attacks and damage
must never be patched.
