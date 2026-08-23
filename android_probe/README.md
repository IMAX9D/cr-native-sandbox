# Android lifecycle probe

This source tree is a frozen copy of the minimal Java host at production commit
`f99e1b3`, modified only to expose four lifecycle profiles. The class and JNI
method names stay unchanged because the observation bridge is version-bound to
`royale.nativehost.JniHost`.

Profiles, from most coupled to least coupled:

1. `probe-baseline`: create activity, create Surface, start, resume.
2. `probe-detach-surface`: bootstrap through the baseline, pause, destroy and
   release the Surface, then advance 100 logic ticks with no active renderer.
3. `probe-null-surface`: issue native surface callbacks with `null` only to
   test whether they are a Mainloop gate; no Surface object is constructed.
4. `probe-no-surface`: create activity, start, resume; no Surface object or
   surface callback.
5. `probe-create-only`: only `GameApp.nOnCreate()`.
6. `probe-minimal`: stop after `CreateGameMain`.
7. `probe-direct`: create the Java Activity shell without a Surface, invoke
   the original `GameMain::init` (`0x727050`) through a version-guarded
   presentation shim, and attest every prerequisite before replay creation.
   It exits with `blocked_data_tables` instead of entering battle while the
   native table array is empty.
8. `serve-direct`: retain the same strict no-Surface initialization as
   `probe-direct`, then expose the JSON-line battle API. Its `reset` operation
   performs the original native `BattleGameState` 4-to-4 replacement so one
   process can execute many seeded episodes.

The first six profiles attempt the same replay and exactly 100 native logic
steps. The direct profiles have an earlier readiness gate: they may attempt the
replay only after native DataTables are populated. Failure is evidence; no
profile falls back to another profile.
