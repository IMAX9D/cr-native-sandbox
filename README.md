# CR Native Sandbox

Headless, externally controlled Clash Royale battle sandbox backed by the
original Android x86_64 `libg.so` runtime.

这是一个独立的无界面战斗沙盒仓库，只提供：

- Android x86_64 无 Surface 宿主；
- 原生 JNI Bridge；
- 标准 1v1 Replay 创建与进程内重置；
- 全卡牌、觉醒、英雄和主动技能接口；
- 持久 JSON-line TCP 协议；
- Python 外部 API；
- 原生状态、路径、目标、效果、塔血和终局观测；
- 纯沙盒验收脚本。

本仓库**不包含 GUI、桌面界面、浏览器界面、AI、模型、权重或学习代码**。

## Important

This repository does not distribute Supercell binaries or game assets.

以下内容不会上传，也不属于本仓库许可证范围：

- `libg.so` 或其他原版 `.so`；
- APK / split APK / XAPK；
- Asset Pack 和解包后的游戏 Assets；
- 游戏账号、回放数据、运行日志或本地证书；
- Supercell 商标、美术、音频或其他受版权保护内容。

使用者必须自行合法取得与绑定清单完全匹配的 Android runtime。项目仅用于
互操作性研究、自动化验证和本地开发。

## Frozen runtime

| Item | Value |
| --- | --- |
| Game content | `15.535.29` |
| Runtime version | `150535029` |
| ABI | Android `x86_64` |
| Mode | Standard 1v1 |
| Logic rate | 20 Hz / 50 ms per Tick |
| `libg.so` SHA-256 | `fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba` |
| Observation scope | `public-observe-v6` |

The JNI bindings are version-specific. A different library hash or
`JNI_OnLoad` RVA must fail closed.

## Current coverage

- 122/122 visible standard base cards accepted by the original action path.
- 41/41 evolution forms resolved by native card-cycle state.
- 16/16 hero forms resolved by the native form selector.
- Native active-ability command type `0x5A` exposed through the API.
- Representative hero and champion abilities verified for cost, charges and
  casting state.
- Standard 3-minute regulation + 2-minute overtime, ×1/×2/×3 elixir and native
  HP-drain tiebreak exported through the terminal observation.

Coverage means the corresponding native path is executable. It does not claim
that every pairwise card interaction has been exhaustively certified.

## Architecture

```text
Python / JSON client
        │ persistent UTF-8 JSON lines
        ▼
Android app_process · royale.nativehost.JniHost
        │ JNI
        ▼
libnative_core_probe.so
        │ version-guarded RVAs
        ▼
original libg.so · BattleGameState type 4
```

Android supplies the ABI, dynamic linker, Java/JNI and asset filesystem. It is
not used as a visible game client. `serve-direct` creates no Android Surface.

## Repository contents

```text
android_probe/       Java host, required Titan stubs and JNI bridge
bindings/            Frozen runtime hash/RVA manifest
docs/                Detailed sandbox and full-card documentation
examples/            Bootstrap Replay JSON examples
native_core/         Python client, environment, catalog and deployment rules
scripts/             Build, lifecycle and acceptance scripts
tests/               Runtime-independent interface tests
```

There is intentionally no GUI package or GUI launcher.

## Requirements

- Windows 10/11 x64 with hardware virtualization.
- Python 3.11+.
- PowerShell 5.1+ or PowerShell 7.
- JDK 17.
- Android SDK with:
  - platform-tools / ADB;
  - emulator;
  - Android platform 35 for compilation;
  - command-line tools with D8/R8.
- Android NDK r27d.
- A pre-provisioned Android 31 x86_64 AVD.
- A complete, legally obtained runtime matching the frozen hash.

The default AVD name is `royale_worker_api31`. You may override it from the
worker CLI.

## Install the Python package

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The runtime API itself has no third-party Python dependency.

## Configure local paths

Copy the example and edit paths for your machine:

```powershell
Copy-Item runtime.env.example.ps1 runtime.env.ps1
. .\runtime.env.ps1
```

Required variables:

| Variable | Meaning |
| --- | --- |
| `CR_SANDBOX_ANDROID_SDK` | Android SDK root |
| `CR_SANDBOX_ADB` | `adb.exe` |
| `CR_SANDBOX_ANDROID_TOOLS` | Android command-line tools root |
| `CR_SANDBOX_ANDROID_JAR` | compilation `android.jar` |
| `CR_SANDBOX_NDK` | Android NDK r27d root |
| `CR_SANDBOX_JDK` | JDK 17 root |
| `CR_SANDBOX_AVD_HOME` | directory containing the provisioned AVD |
| `CR_SANDBOX_APKS` | directory containing the complete split APK set |
| `CR_SANDBOX_RUNTIME_DIR` | directory containing `libg.so` and peer `.so` files |
| `CR_SANDBOX_BASE_APK` | matching `base.apk` |
| `CR_SANDBOX_ASSET_PACK_APK` | matching install-time asset-pack APK |
| `CR_SANDBOX_ASSETS` | extracted base assets directory |
| `CR_SANDBOX_DATA` | writable logs/certificates/cache directory |

`runtime.env.ps1` is ignored by Git.

## Build

```powershell
.\scripts\build_probe.ps1
.\scripts\build_bridge.ps1
```

Generated outputs are local-only:

```text
artifacts/lifecycle-probe.jar
artifacts/libnative_core_probe.so
```

## Start one headless service

Make sure the AVD is booted and the matching APK set is installed, then run:

```powershell
.\scripts\start_direct_service.ps1 `
  -Serial emulator-5554 `
  -Port 37031 `
  -Slot 0 `
  -BootstrapReplayJson .\examples\full-card-bootstrap.json
```

Or let the Python worker manager start the AVD and service:

```powershell
python -m native_core.worker start `
  --workers 1 `
  --base-port 37031 `
  --transport adb `
  --avd-name royale_worker_api31
```

Check status:

```powershell
python -m native_core.client --port 37031 ping
python -m native_core.client --port 37031 status
python -m native_core.client --port 37031 observe
```

Stop the service:

```powershell
.\scripts\stop_direct_service.ps1 -Port 37031 -Slot 0
```

## Build an arbitrary Replay

```powershell
python scripts\build_native_replay.py `
  --deck0 "Knight@evolution,Berserker@hero,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon" `
  --deck1 "Knight,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon,Arrows" `
  --output "$env:CR_SANDBOX_DATA\sandbox-replay.json"
```

Form encoding:

| Form | Native `el` |
| --- | ---: |
| base | 0 / omitted |
| evolution | 1 |
| hero | 2 |
| both | 3 |

## Python API

```python
from pathlib import Path
from native_core import NativeRoyaleEnv

replay = Path(r"D:\sandbox-data\sandbox-replay.json")

with NativeRoyaleEnv(port=37031) as env:
    state = env.reset(replay, warmup_steps=100)

    player = state["players"][0]
    deck_index = player["hand_deck_indices"][0]

    grid = env.deployment_grid(
        side=0,
        deck_index=deck_index,
        adjusted=True,
    )

    result = env.act(
        side=0,
        deck_index=deck_index,
        x=9000,
        y=10000,
    )

    env.step(1)
    next_state = env.observe()
```

Active ability:

```python
unit = next(
    entity for entity in next_state["entities"]
    if entity["side"] == 0 and entity["ability_available"]
)
ability_result = env.use_ability(
    side=0,
    entity_id=unit["entity_id"],
)
```

Two-sided action and Tick in one call:

```python
transition = env.joint_transition(
    [
        {"type": "play", "side": 0, "deck_index": 2, "x": 9000, "y": 10000},
        {"type": "ability", "side": 1, "entity_id": 5000012},
    ],
    steps=1,
)
```

## JSON-line protocol

One request and one response per UTF-8 line. Main operations:

```text
ping / status
reset / restart_replay / load_replay
observe / observe_compact_v1
step / step_trace
probe_grid
act / ability
joint_act / joint_transition / joint_transition_trace
shutdown
```

Mutating requests are never automatically replayed after an ambiguous network
failure. See [the technical document](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
for schemas, limits, state fields and fail-closed conditions.

## Validation

Runtime-independent tests:

```powershell
python -m unittest discover -s tests
```

Native acceptance, after starting the service:

```powershell
python scripts\accept_full_card_catalog.py --port 37031
python scripts\accept_native_card_forms.py --port 37031
python scripts\accept_match_rules.py --port 37031
```

Strict ten-process no-Surface cold-start certificate:

```powershell
.\scripts\accept_direct_core.ps1 -Runs 10
```

## Security and failure policy

- The service is intended for local use only.
- Keep host forwarding bound to loopback and do not expose the port publicly.
- Runtime hash/RVA mismatch is fatal.
- Invalid pointers, entity counts, path counts or response schemas are fatal.
- Rejected native commands remain rejected.
- Never commit runtime binaries, APKs, assets, logs, Replay datasets or account
  data.

## Documentation

- [Sandbox runtime architecture](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
- [Full-card, evolution, hero and ability interface](docs/NATIVE_FULL_CARD_RUNTIME.zh-CN.md)
- [Android lifecycle host](android_probe/README.md)

## License and trademarks

The original wrapper/interface source in this repository is licensed under the
MIT License. This license does not apply to Clash Royale, Supercell binaries,
assets, trademarks or other third-party material.

Clash Royale and Supercell are trademarks of Supercell Oy. This project is not
affiliated with, endorsed by, sponsored by or approved by Supercell.
