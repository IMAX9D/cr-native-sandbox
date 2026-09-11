# Prebuilt match host

These two binaries are built only from this repository's Java host and C++ bridge.
They contain no game library, APK, logic data, model, account or machine configuration.
Native debug sections/build paths were removed; the executable .text section
was checked byte-identical before and after stripping.
`manifest.json` binds their bytes and LF-normalized source hashes. The bootstrap
script verifies them, backs up differing local host artifacts, and never replaces
the game library. Rebuild after modifying the bound sources.

Linux bare ART still needs framework JNI registration; Android `app_process`
already performs it and uses the explicit skip-registration switch when selecting
the resource-isolated context. The new `runtime_identity_v1` RPC is additive.
