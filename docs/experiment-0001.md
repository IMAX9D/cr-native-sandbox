# Experiment 0001: current-runtime direct battle step

## Question

Can packaged Clash Royale runtime `150535029` x86_64 execute its original
battle logic behind a minimal non-visual host, without using the Python battle
implementation?

## Pass conditions

- Exact runtime, ABI, packaged-libg SHA-256, and content fingerprint are logged.
- No MuMu dependency and no interactive Clash Royale window.
- A standard level-11 1v1 battle reaches the known six-tower bootstrap state.
- The host advances exactly 100 requested logic ticks without wall-clock pacing.
- Canonical state is complete at every requested tick.
- Ten fresh-process repetitions produce identical canonical hashes.
- Every invoked native function and accessed field is listed in a versioned
  binding manifest with evidence references.

## Failure classes

- `loader`: libg cannot be initialized outside the packaged application chain.
- `resources`: native data tables or content cannot be initialized independently.
- `battle-construction`: a valid LogicBattle cannot be created or obtained.
- `tick-control`: battle tick cannot be advanced deterministically on demand.
- `observation`: required state cannot be read without renderer/client coupling.
- `isolation`: fresh runs inherit or leak battle state.
- `performance`: direct-native execution is not materially faster than the
  existing Android host.

## Guardrail

No workaround may replace a failed native result with Python simulation. A
failure is recorded as evidence about route feasibility, not patched over.

