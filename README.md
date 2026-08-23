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

## First acceptance gate

The direct-native route is considered feasible only after one isolated command
can:

1. attest the exact runtime/content/libg identity;
2. initialize or obtain a standard 1v1 `LogicBattle` without a visible game UI;
3. advance an idle battle for 100 consecutive logic ticks;
4. read back tick, both players, six towers, RNG, hands, cycle, and elixir;
5. reproduce the same canonical state hash across ten fresh runs; and
6. report measured startup latency and sustained ticks per second.

Until all six conditions pass, this repository makes no claim that the current
runtime has been separated into a reusable kernel.

## Experiment layout

- `android_probe/`: isolated Java lifecycle probe. It intentionally keeps the
  JNI class name expected by the frozen bridge while varying only lifecycle
  calls around the original `libg.so`.
- `scripts/`: build/deploy/measurement entry points owned by this repository.
- `artifacts/`: generated JARs, logs, and result JSON (ignored by Git).

The probe consumes the frozen APK, packaged libraries, and JNI observation
bridge from the production repository by explicit absolute input paths. It
does not write to that repository and does not import its Python sandbox.

## Upstream reference

`third_party/Scroll` pins Reversed Rooms' experimental Clash Royale v1.3.2
server-on-libg prototype. It is architectural evidence only: its ARMv7 layouts,
RVAs, resources, and lifecycle must never be reused as current-version facts.
