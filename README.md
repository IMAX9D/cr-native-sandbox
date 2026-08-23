# CR Native Core

Isolated feasibility project for driving the original Clash Royale `libg.so`
as a headless battle oracle and, if the experiment succeeds, as a high-speed
self-play kernel.

## Frozen target

- Android runtime version: `150535029`
- Content identity: `15.535.29` / packaged Core8-equivalent content
- ABI: `x86_64`
- Mode: standard 1v1
- Levels: card 11, tower 11
- Variants: base cards only; no evolution, elite level, champion, or hero
- Deck: Knight, Archers, Giant, Skeletons, Musketeer, Hog Rider, Cannon, Arrows

## Prime directive

This repository does **not** reimplement battle rules. It tests whether the
original native `Logic*` implementation can be initialized, stepped, observed,
and reset behind a small stable ABI. Python/Rust code may provide lifecycle,
input, output, batching, and training adapters only.

## Isolation boundary

- Production sandbox: `D:\Codex\E\AI ClashRoyale` (read-only reference)
- Reverse-engineering evidence: `D:\Deepseek\cr_re` (read-only reference)
- Large runtime inputs and generated evidence: `D:\AI_data`
- All experimental source code and design decisions: this repository

No feasibility experiment may modify either existing project. A successful
prototype must expose its own manifest and explicit paths to external runtime
artifacts.

## First acceptance gate: passed

The direct-native route is considered feasible only after one isolated command
can:

1. attest the exact runtime/content/libg identity;
2. initialize or obtain a standard 1v1 `LogicBattle` without a visible game UI;
3. advance an idle battle for 100 consecutive logic ticks;
4. read back tick, both players, six towers, RNG, hands, cycle, and elixir;
5. reproduce the same canonical state hash across ten fresh runs; and
6. report measured startup latency and sustained ticks per second.

All six conditions passed on 2026-08-23. The certificate is reproducible with:

```powershell
.\scripts\accept_direct_core.ps1 -Runs 10
```

## Current status

- Strict no-Surface cold start is proven. No Surface is created, attached, or
  borrowed at any point in `probe-direct`.
- Original `libg` code loads all 158 DataTables resources, loads the standard
  arena and tilemap through the native resource request list, creates the
  standard 1v1 battle, and advances `0 -> 100` logic ticks.
- Ten fresh processes produced the same RNG, hands, cycle, six tower entities,
  tower HP, and canonical state hash `5594aa3c81dc52fa`.
- Mean cold process wall time was `13.095 s`. The measured replay-injection to
  100-tick observation path averaged `13.126 ms`, or about `7,618` validated
  ticks/s for one process. Cold wall time includes deployment orchestration and
  a deliberate 5-second platform initialization fence.

See `docs/experiment-0002-results.md` for exact evidence, native call-chain
boundaries, and remaining work before this becomes a reusable training worker.

## Experiment layout

- `android_probe/`: isolated Java lifecycle probe. It intentionally keeps the
  JNI class name expected by the frozen bridge while varying only lifecycle
  calls around the original `libg.so`.
- `scripts/`: build/deploy/measurement entry points owned by this repository.
- `scripts/accept_direct_core.ps1`: fresh-process determinism and baseline
  certificate.
- `artifacts/`: generated JARs, logs, and result JSON (ignored by Git).

The probe consumes the frozen APK, packaged libraries, and JNI observation
bridge from the production repository by explicit absolute input paths. It
does not write to that repository and does not import its Python sandbox.

## Upstream reference

`third_party/Scroll` pins Reversed Rooms' experimental Clash Royale v1.3.2
server-on-libg prototype. It is architectural evidence only: its ARMv7 layouts,
RVAs, resources, and lifecycle must never be reused as current-version facts.
