# Android lifecycle probe

This source tree is a frozen copy of the minimal Java host at production commit
`f99e1b3`, modified only to expose four lifecycle profiles. The class and JNI
method names stay unchanged because the observation bridge is version-bound to
`royale.nativehost.JniHost`.

Profiles, from most coupled to least coupled:

1. `probe-baseline`: create activity, create Surface, start, resume.
2. `probe-detach-surface`: bootstrap through the baseline, pause, destroy and
   release the Surface, then advance 100 logic ticks with no active renderer.
3. `probe-no-surface`: create activity, start, resume; no Surface object or
   surface callback.
4. `probe-create-only`: only `GameApp.nOnCreate()`.
5. `probe-minimal`: stop after `CreateGameMain`.
6. `probe-direct`: stop after `CreateGameMain`, call the native manager
   singleton initializer at `0xCE65B0`, submit the replay, and pump the native
   manager/state directly. It creates no Activity lifecycle and no Surface.

Every profile attempts the same native replay load and exactly 100 native logic
steps. Failure is evidence; it must not fall back to another profile.
