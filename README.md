# CR Native Core

完整中文技术路线：[`docs/TECHNICAL_ROUTE.zh-CN.md`](docs/TECHNICAL_ROUTE.zh-CN.md)

Self-Play 吞吐优化报告：
[`docs/throughput-optimization-20260823.md`](docs/throughput-optimization-20260823.md)

正式训练并发 Scaling Sweep：
[`docs/TRAINING_CONCURRENCY_SCALING.zh-CN.md`](docs/TRAINING_CONCURRENCY_SCALING.zh-CN.md)

Self-Play v0.2 可变等待动作设计：
[`docs/SELFPLAY_V0_2_ACTION_TIMING_DESIGN.zh-CN.md`](docs/SELFPLAY_V0_2_ACTION_TIMING_DESIGN.zh-CN.md)

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

## One-click training: passed

Double-click [`START_TRAINING.cmd`](START_TRAINING.cmd), or run:

```powershell
.\scripts\start_selfplay_v0_1.ps1
```

The entry point builds the Java host and JNI bridge, starts the no-window
Android x86_64 ABI containers when necessary, launches eight persistent
Surface-free `app_process` workers, attests the original battle state, and
runs recurrent PPO self-play. It does not start MuMu or the visual game.

`START_TRAINING.cmd` invokes the guarded v0.1 stage runner
`scripts/start_selfplay_v0_1.ps1`: 2 AVD / 8 Worker, 1M native ticks,
milestone checkpoints, resource monitoring, paired side-swapped evaluation
and report generation. `scripts/start_training.ps1` remains the lower-level
launcher.

To resume the frozen P010 checkpoint to the controlled 2M gate with live
telemetry, double-click [`TRAINING_DASHBOARD.cmd`](TRAINING_DASHBOARD.cmd).
The loopback-only browser page shows PPO losses, entropy/KL/explained variance,
throughput, resources, card usage, progress and ETA. It cannot hide or stop an
active training process, and the native Workers stop automatically at 2M.

All mutable output is under `D:\AI_data\cr-native-core`:

- `selfplay-v0.1\runs\<run-id>\manifest.json`: immutable run configuration;
- `selfplay-v0.1\runs\<run-id>\trajectories`: replayable episode tensors/metadata;
- `selfplay-v0.1\runs\<run-id>\logs\events.jsonl`: append-only progress events;
- `selfplay-v0.1\runs\<run-id>\checkpoints`: P000/latest/recovery state;
- `selfplay-v0.1\latest_run.json`: atomic pointer to the newest checkpoint.

For a bounded end-to-end check, double-click
[`SMOKE_TEST_TRAINING.cmd`](SMOKE_TEST_TRAINING.cmd). The acceptance requires a
native episode, a PPO backward/update pass, and a loadable checkpoint—not just
a responsive launcher.

To inspect battle logic manually, double-click
[`GAME_LOGIC_GUI.cmd`](GAME_LOGIC_GUI.cmd). The GUI exposes both sides, current
hand cards, raw/native versus final training deployment masks, single/batched
ticks, target links, path nodes, entity-native fields, state hash/RNG, terminal
state, and JSON snapshot export. The top-right clock follows the certified
3-minute regulation + 2-minute overtime schedule and labels ×1/×2/×3 elixir.

The same entry was verified from a fully stopped VM/service and with four
concurrent native workers on 2026-08-23. Training uses Emulator loopback TCP
redirection (host ports 38031+) and CUDA Graph inference by default; the ADB
ports 37031+ remain available for GUI/debug and fallback. See
[`docs/training-system.md`](docs/training-system.md).

The measured throughput sweet spot on the current machine is two 4-vCPU AVDs,
four Workers per AVD, and policy batch 16. Launch it explicitly with:

```powershell
.\scripts\start_training.ps1 -Avds 2 -Workers 8
```

The formal default is 2 AVD / 8 Worker. Because measured free RAM can fall
below 1 GiB, close unrelated memory-heavy applications before starting it.

## Native-core status

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

See `docs/experiment-0002-results.md` for exact direct-core evidence and native
call-chain boundaries.

## Experiment layout

- `android_probe/`: isolated Java lifecycle probe. It intentionally keeps the
  JNI class name expected by the frozen bridge while varying only lifecycle
  calls around the original `libg.so`.
- `scripts/`: build/deploy/measurement entry points owned by this repository.
- `scripts/accept_direct_core.ps1`: fresh-process determinism and baseline
  certificate.
- `native_core/`: persistent Worker lifecycle plus stable JSON-line client/env.
- `training/`: observation schema, action masks, recurrent actor/critic,
  self-play collection, PPO update, and atomic run storage.
- `START_TRAINING.cmd`: guarded Self-Play v0.1 2-AVD/8-Worker stage entry.
- `SMOKE_TEST_TRAINING.cmd`: bounded one-iteration acceptance entry.
- `GAME_LOGIC_GUI.cmd`: interactive native battle-logic acceptance entry.
- `artifacts/`: generated JARs, logs, and result JSON (ignored by Git).

The Worker consumes frozen APK/runtime inputs from the production repository
through explicit read-only paths. The Java host, JNI bridge, Python environment,
training stack, and all writes are owned by this repository or `D:\AI_data`.

## Upstream reference

`third_party/Scroll` pins Reversed Rooms' experimental Clash Royale v1.3.2
server-on-libg prototype. It is architectural evidence only: its ARMv7 layouts,
RVAs, resources, and lifecycle must never be reused as current-version facts.
