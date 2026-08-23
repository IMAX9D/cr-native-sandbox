# Persistent native training system

## Contract

The original x86_64 `libg.so` is the only battle-rule implementation. Python
never predicts movement, targeting, collision, damage, card cycle, elixir, or
deployment legality. It supplies actions and consumes native observations.

The frozen game target is version `15.535.29`, standard 1v1, level 11 towers
and these eight base cards: Knight, Archers, Giant, Skeletons, Musketeer, Hog
Rider, Cannon, and Arrows. Evolution/elite/champion/hero variants are excluded.

## Runtime flow

1. `scripts/start_training.ps1` rebuilds the local JAR/JNI bridge.
2. `native_core.worker` starts the no-window AVD only if it is absent.
3. Each slot launches a Surface-free `app_process` service on ports 37031+.
4. The service loads DataTables and creates one original `BattleGameState`.
5. Episode reset performs an in-process native state transition `4 -> 4`.
6. Python advances to tick 100, where native deployment becomes available.
7. For each side, the actor sees public state only. The critic additionally
   receives enemy hand/elixir during training.
8. Candidate placement cells come from `libg`'s deployment validator. Accepted
   commands use the original command constructor/selection/execute functions.
9. A canonical two-sided action RPC advances the original battle one tick and
   returns the next native observation.
10. Complete trajectories feed recurrent PPO. Run files and checkpoints are
    written atomically under `D:\AI_data`.

Cold initialization is paid once per Worker, not once per episode. The reset
stress gate completed 1,200 same-process resets with a single deterministic
hash and no monotonic RSS growth; the 1,000-reset block averaged 11.475 ms,
p95 26.961 ms.

## Entry points

Normal unattended training (defaults: two Workers, four episodes per update):

```powershell
.\scripts\start_training.ps1
```

Bounded acceptance:

```powershell
.\scripts\start_training.ps1 -Smoke
```

Interactive battle-logic acceptance:

```powershell
.\scripts\start_logic_gui.ps1
```

The GUI can switch between `libg`'s raw per-cell validator output and the final
training action mask. It also exposes target links, path nodes, collision and
attack timers, RNG/state hash, both hands/elixir pools, and snapshot export to
`D:\AI_data\cr-native-core\gui-sessions`.

Custom run:

```powershell
.\scripts\start_training.ps1 -Workers 2 -Iterations 1000 `
  -EpisodesPerIteration 8 -MaxTicks 7200 -Seed 100
```

The Python layer may also be invoked directly after the artifacts/services are
ready:

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe -m training.train `
  --workers 2 --iterations 1000 --episodes-per-iteration 8
```

## Acceptance evidence

On 2026-08-23 the complete smoke entry was run after stopping both the native
service and emulator. It automatically cold-started the no-window Worker and
wrote:

`D:\AI_data\cr-native-core\training\runs\smoke-20260823T063225Z\checkpoints\checkpoint-000001.pt`

The checkpoint contains the model, AdamW optimizer, run schema, PPO config,
metrics, episode summary, completed episode count, and environment-step count.

All eight cards were also submitted independently through the original native
command path. Every command was accepted, spent native elixir, and rotated the
native hand. Troop/building entity counts were `1,2,1,3,1,1,1`; Arrows emitted
30 classified native effects. Reproduce and overwrite the certificate with:

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe scripts\accept_eight_cards.py
```

The machine-readable certificate is
`D:\AI_data\cr-native-core\acceptance-eight-cards.json`.

A separate two-Worker acceptance completed two episodes in parallel and wrote:

`D:\AI_data\cr-native-core\training\runs\dual-worker-acceptance-20260823\checkpoints\checkpoint-000001.pt`

The concurrent sampling/update portion recorded 56 environment steps in
0.992 seconds. This short measurement is a pipeline acceptance result, not a
full-match throughput forecast.

## Operational notes

- Normal training intentionally continues until stopped or the configured
  iteration count is reached. Checkpoints are updated once per iteration.
- Restart with `--resume <latest.pt>` to restore model and optimizer state.
- Worker cold boot/deployment is much slower than the roughly 11 ms native
  episode reset, so services should remain persistent.
- Raw native tick throughput is not equivalent to policy-training throughput;
  the latter includes observation JSON, masking, GPU inference, trajectory
  storage, and PPO updates.
- The AVD is an Android ABI container, not MuMu and not a rendered game client.
  No Android Surface is created by the direct service.
