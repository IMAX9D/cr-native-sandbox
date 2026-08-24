# CR Native Sandbox

基于原版 Android x86_64 `libg.so` 的无界面《皇室战争》标准 1v1 沙盒。

它提供：

- 原生 20 Hz 战斗推进；
- 全卡牌、觉醒、英雄和主动技能接口；
- Replay 创建与进程内重置；
- 完整/紧凑原生观测；
- 持久 JSON-line TCP API；
- Python 外部接口；
- 自动化部署、环境诊断和原生验收。

仓库**不包含 GUI、AI、模型或学习代码**。

## 1. 使用 Runtime ZIP 一键部署

如果你已经合法取得：

```text
cr-native-sandbox-runtime-150535029.zip
```

这是推荐的部署路线。ZIP 本身不存放在本 GitHub 仓库中。

### 1.1 前置软件

先安装：Windows 10/11 x64、Git、Python 3.11+、JDK 17，以及 Android SDK
Command-line Tools。

还需要在 BIOS/UEFI 开启 VT-x/AMD-V，并在 Windows 中启用
**Windows Hypervisor Platform (WHPX)** 或 Hyper-V。

Android Command-line Tools 应放到类似目录：

```text
C:\Android\Sdk\cmdline-tools\latest
```

确保以下文件存在：

```text
C:\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat
C:\Android\Sdk\cmdline-tools\latest\bin\avdmanager.bat
C:\Android\Sdk\cmdline-tools\latest\lib\r8.jar
```

其余 SDK 包、NDK 和 AVD 可由 `bootstrap.ps1` 自动安装/创建。

### 1.2 克隆源码并安装 Python 包

```powershell
git clone https://github.com/IMAX9D/cr-native-sandbox.git
cd cr-native-sandbox

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### 1.3 校验并解压 Runtime ZIP

```powershell
$Bundle = "D:\Downloads\cr-native-sandbox-runtime-150535029.zip"
$Expected = "82b2e79eaa03aa98d83f5cfec78b053179a77f1f7c0d1fd274f8ec5c833c4310"
$Actual = (Get-FileHash -LiteralPath $Bundle -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) {
    throw "Runtime ZIP SHA-256 mismatch: $Actual"
}

Expand-Archive -LiteralPath $Bundle -DestinationPath .
Move-Item `
  -LiteralPath .\cr-native-sandbox-runtime-150535029 `
  -Destination .\runtime
```

如果 `runtime\` 已存在，请先自行备份或选择空目录，不要直接覆盖来源不明的
Runtime。

解压后应为：

```text
runtime/
├─ README.txt
├─ SHA256SUMS.txt
├─ apks/
│  ├─ base.apk
│  ├─ split_config.en.apk
│  ├─ split_config.hdpi.apk
│  ├─ split_config.x86_64.apk
│  └─ split_install_time_asset_pack.apk
├─ x86_64-libs/
│  ├─ libg.so
│  ├─ libc++_shared.so
│  ├─ libfmod.so
│  ├─ libfmodstudio.so
│  └─ 其余 10 个依赖库
└─ extracted-assets/
   ├─ csv_client/
   └─ csv_logic/
```

Bundle 内容：

| 类型 | 数量 | 用途 |
| --- | ---: | --- |
| Split APK | 5 | 安装 Android 包、提供 package context 和 Asset Pack |
| x86_64 `.so` | 14 | 在无界面 `app_process` 中加载原生核心和依赖 |
| DataTables 文件 | 383 | 初始化 `csv_client` / `csv_logic` 原生数据表 |

`SHA256SUMS.txt` 保存 Bundle 内各文件的校验值。两个地图文件仍位于 Asset Pack：

```text
assets/locations/training_arena.csv
assets/tilemaps/tilemap.csv
```

部署脚本会自动提取它们。

### 1.4 配置本机路径

```powershell
Copy-Item runtime.env.example.ps1 runtime.env.ps1
notepad runtime.env.ps1
. .\runtime.env.ps1
```

通常只需修改 Android SDK 和 JDK：

```powershell
$env:CR_SANDBOX_ANDROID_SDK = "C:\Android\Sdk"
$env:CR_SANDBOX_JDK = "C:\Program Files\Eclipse Adoptium\jdk-17"
```

示例文件默认把 Runtime 指向仓库内 Git 忽略的目录：

```text
runtime\apks
runtime\x86_64-libs
runtime\extracted-assets
```

每次打开新 PowerShell 后，都要重新执行：

```powershell
. .\runtime.env.ps1
```

### 1.5 准备 SDK、AVD 和 Runtime

```powershell
.\scripts\bootstrap.ps1
```

它会：

1. 接受 Android SDK 许可证；
2. 安装 Platform-Tools、Emulator、Android 35 Platform、Build-Tools 35、
   NDK r27d 和 Android 31 x86_64 System Image；
3. 创建可 root 的 AOSP AVD `royale_worker_api31`；
4. 固定 AVD 为 4 vCPU、4 GB RAM、10 GB 数据分区；
5. 从 APK 重新提取并核对 14 个 `.so`；
6. 提取 DataTables、Arena 和 Tilemap；
7. 生成本机冻结 Runtime Manifest。

必须使用：

```text
system-images;android-31;default;x86_64
```

不要使用 Google Play System Image，因为它不能通过 `adb root`。

### 1.6 环境诊断

```powershell
.\scripts\doctor.ps1
```

它会一次性检查：环境变量、Python/JDK/SDK/NDK、5 个 APK 和 14 个原生库的
大小与 SHA-256、DataTables、Arena、Tilemap、AVD、虚拟化、端口和磁盘空间。

必须解决所有工具链、Runtime、Assets 和 AVD 的硬性 `FAIL`。虚拟化探测和端口
占用属于环境提示：如果对应 Emulator/服务正是你主动启动的，可以忽略该端口
提示；否则应先关闭冲突进程。

### 1.7 完整冒烟验收

```powershell
.\scripts\smoke.ps1
```

它会构建宿主、启动 AVD、安装 5 个 Split APK、启动无 Surface 服务、推进到
Tick 100、检查六塔和状态哈希，然后停止服务与 AVD。

预期输出：

```text
PASS toolchain
PASS runtime hashes
PASS AVD root/access
PASS package versionCode=150535029
PASS no-Surface host
PASS six towers
PASS tick 0 -> 100
PASS public-observe-v6
PASS hash 96598dc9028e1802
```

需要验收后继续保留服务：

```powershell
.\scripts\smoke.ps1 -KeepRunning
```

## 2. Runtime ZIP 中每类文件怎么用

### `apks\`

5 个 APK 必须作为同一个 Split Package 安装。`native_core.worker` 在包缺失时会
自动执行等价于：

```powershell
adb install-multiple -r -t `
  base.apk `
  split_config.en.apk `
  split_config.hdpi.apk `
  split_config.x86_64.apk `
  split_install_time_asset_pack.apk
```

不要只安装 `base.apk`，否则会缺少 x86_64 原生库和 Asset Pack。

### `x86_64-libs\`

14 个 `.so` 不需要手工放入 AVD 系统目录。`start_direct_service.ps1` 会逐个
校验后推送到：

```text
/data/local/tmp/cr-native-direct-<slot>/
```

并通过该目录的 `LD_LIBRARY_PATH` 启动 `app_process`。`libg.so` 是战斗核心，
其余库是该版本的依赖；不能混用其他版本或 ABI 的同名文件。

### `extracted-assets\`

`csv_client` 和 `csv_logic` 用于加载原生 DataTables。启动脚本会把它们与
Arena/Tilemap 合并为本地 `artifacts/runtime-assets.tar`，随后推送并解压到：

```text
/data/local/tmp/cr-native-direct-<slot>/assets/
```

这些文件由原版 `libg.so` 解析，不是 Python 重写的战斗数据。

### `SHA256SUMS.txt`

用于核对 Bundle 内每个 APK、`.so` 和 DataTables 文件。部署时
`doctor.ps1` 还会根据
[`bindings/runtime-manifest.json`](bindings/runtime-manifest.json)
重新检查 5 个 APK 和 14 个 `.so`。

### `README.txt`

Runtime Bundle 自身的简要说明，不参与程序运行。

## 3. 日常启动与停止

已执行过 `bootstrap.ps1` 后，日常启动：

```powershell
. .\runtime.env.ps1
.\.venv\Scripts\Activate.ps1

python -m native_core.worker start `
  --workers 1 `
  --base-port 37031 `
  --transport adb `
  --avd-name royale_worker_api31
```

检查服务：

```powershell
python -m native_core.client --port 37031 ping
python -m native_core.client --port 37031 status
python -m native_core.client --port 37031 observe
```

停止服务但保留 AVD：

```powershell
python -m native_core.worker stop --workers 1 --base-port 37031
```

同时停止 AVD：

```powershell
python -m native_core.worker stop --workers 1 --base-port 37031 --stop-vm
```

也可以直接启动单个已开机 Emulator 中的服务：

```powershell
.\scripts\start_direct_service.ps1 `
  -Serial emulator-5554 `
  -Port 37031 `
  -Slot 0 `
  -BootstrapReplayJson .\examples\full-card-bootstrap.json
```

## 4. 创建自定义牌组 Replay

```powershell
python scripts\build_native_replay.py `
  --deck0 "Knight@evolution,Berserker@hero,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon" `
  --deck1 "Knight,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon,Arrows" `
  --output "$env:CR_SANDBOX_DATA\sandbox-replay.json"
```

| 写法 | 原生 `el` | 含义 |
| --- | ---: | --- |
| `Knight` | 0/省略 | 基础形态 |
| `Knight@evolution` | 1 | 开启觉醒循环 |
| `Knight@hero` | 2 | 英雄形态 |
| `Card@both` | 3 | 同时启用两种形态（目录允许时） |

## 5. Python 外部接口

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

    action = env.act(
        side=0,
        deck_index=deck_index,
        x=9000,
        y=10000,
    )

    env.step(1)
    next_state = env.observe()
```

主动技能：

```python
unit = next(
    entity for entity in next_state["entities"]
    if entity["side"] == 0 and entity["ability_available"]
)

result = env.use_ability(side=0, entity_id=unit["entity_id"])
```

双方同 Tick 动作：

```python
transition = env.joint_transition(
    [
        {"type": "play", "side": 0, "deck_index": 2,
         "x": 9000, "y": 10000},
        {"type": "ability", "side": 1, "entity_id": 5000012},
    ],
    steps=1,
)
```

完整 JSON-line 协议见 [`docs/API.md`](docs/API.md)。

## 6. 验收命令

不依赖原生 Runtime 的测试：

```powershell
python -m unittest discover -s tests
```

服务启动后的原生验收：

```powershell
python scripts\accept_full_card_catalog.py --port 37031
python scripts\accept_native_card_forms.py --port 37031
python scripts\accept_match_rules.py --port 37031
```

严格无 Surface 十进程冷启动：

```powershell
.\scripts\accept_direct_core.ps1 -Runs 10
```

当前覆盖：122/122 标准基础卡、41/41 觉醒形态、16/16 英雄形态、原生主动
技能、3+2 分钟赛程、×1/×2/×3 圣水、原生 HP drain 拼血和终局后重置。

## 7. 常见问题

### `sdkmanager.bat not found`

Android Command-line Tools 没有放在 `cmdline-tools\latest`，或
`CR_SANDBOX_ANDROID_TOOLS` 配置错误。

### `adb root` 失败

使用了 Google Play/Google APIs 镜像。删除该 AVD，改用：

```text
system-images;android-31;default;x86_64
```

### `libg.so hash mismatch`

Runtime 不是 `15.535.29 / 150535029 / x86_64`，或 ZIP 已损坏。不要绕过
版本检查。

### `Missing APK input`

确认 ZIP 被移动为仓库内 `runtime\`，并重新加载：

```powershell
. .\runtime.env.ps1
```

### `runtime assets` 失败

```powershell
.\scripts\prepare_runtime.ps1
.\scripts\freeze_runtime.ps1
.\scripts\doctor.ps1
```

### 端口被占用

默认端口：`5554/5555`（Emulator）、`37031+`（ADB 转发服务）、`38031+`
（direct transport）。关闭旧实例或传入其他端口。

### 服务启动后立即退出

```powershell
adb -s emulator-5554 shell `
  "tail -n 120 /data/local/tmp/cr-native-direct-0/service.log"
```

## 8. 冻结版本

| 项目 | 值 |
| --- | --- |
| 游戏内容 | `15.535.29` |
| Runtime Version | `150535029` |
| ABI | Android `x86_64` |
| 模式 | 标准 1v1 |
| 逻辑频率 | 20 Hz / 50 ms |
| `libg.so` SHA-256 | `fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba` |
| Runtime ZIP SHA-256 | `82b2e79eaa03aa98d83f5cfec78b053179a77f1f7c0d1fd274f8ec5c833c4310` |
| 公开观测协议 | `public-observe-v6` |

完整文件清单：[`bindings/runtime-manifest.json`](bindings/runtime-manifest.json)。

## 9. 安全与法律边界

- GitHub 仓库不分发 APK、`.so`、Assets 或 Runtime ZIP；
- Runtime 由使用者自行合法取得；
- 不要把账号信息、Replay 数据集或本地日志提交到仓库；
- 服务只应用于本机回环端口，不要暴露到公网；
- 版本、哈希、结构或原生命令不匹配时必须 fail closed。

Clash Royale 和 Supercell 是 Supercell Oy 的商标。本项目与 Supercell 无隶属、
赞助或认可关系。

## 10. 更多文档

- [沙盒 Runtime 架构](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
- [JSON-line API](docs/API.md)
- [全卡、觉醒、英雄和技能接口](docs/NATIVE_FULL_CARD_RUNTIME.zh-CN.md)
- [Android 无界面宿主](android_probe/README.md)
- [Runtime Manifest](bindings/runtime-manifest.json)

项目源码使用 MIT License；该许可证不适用于任何第三方游戏二进制、资产或商标。
