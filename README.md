# CR Native Core

基于原版 Android x86_64 `libg.so` 的无界面《皇室战争》标准 1v1
运行内核，并包含 Self-Play、专家模仿训练和可选的 MuMu 实时实验工具。

本仓库提供：

- 原生 20 Hz 战斗推进、Replay 创建与进程内 reset；
- 全卡、觉醒、英雄形态与主动技能接口；
- 完整/紧凑原生观测与稳定 JSON-line TCP API；
- Python 环境、Self-Play v0.1/v0.2 与 Expert Self-Play v1；
- 环境引导、Runtime 哈希冻结、doctor 和一键 smoke；
- 可选的人机 GUI、训练看板与 MuMu ARM64 只读观测/触控控制实验。

仓库不包含游戏 APK、`libg.so`、付费/私有数据集、模型权重或训练产物。
这些文件必须由使用者合法取得，并放入 Git 忽略目录。

## 版本兼容性

| 路线 | 已验证版本 | 状态 |
| --- | --- | --- |
| 无界面 x86_64 原生内核 | `15.535.29 / 150535029` | 已冻结并通过证书 |
| MuMu ARM64 实时实验 | `versionCode 160402002` | 严格版本锁，仅实验用途 |
| 游戏商店当前更新版 | 未重新定位/认证 | 拒绝运行，不能复用旧 RVA |

游戏更新后不能只修改版本号。新 APK/`libg.so` 必须重新冻结哈希、重定位
JNI/函数 RVA/结构偏移、重建卡牌目录，并重新通过 DataTables、Replay、六塔、
RNG、下牌、技能、reset、时间、圣水和终局证书。完整步骤见
[版本升级流程](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md#27-版本升级流程)。

## 快速开始

### 1. 前置条件

- Windows 10/11 x64；
- Git、PowerShell 5.1+、Python 3.11+、JDK 17；
- Android SDK Command-line Tools；
- BIOS/UEFI 已开启 VT-x/AMD-V；
- Windows Hypervisor Platform 或 Hyper-V；
- 与绑定清单完全匹配、由你合法取得的 `150535029` Split APK/Runtime。

### 2. 克隆和安装 Python 包

```powershell
git clone https://github.com/IMAX9D/cr-native-sandbox.git
cd cr-native-sandbox

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

只运行原生 API 不需要训练依赖。需要训练、看板和完整测试时：

```powershell
python -m pip install -e ".[training,test]"
```

Stage-2 云端效率测试使用 PyTorch 2.8 / CUDA 12.8；训练依赖下限为 PyTorch 2.8。
请先安装与显卡兼容的 PyTorch CUDA 构建，再安装本包；仅安装依赖不代表
Linux Bionic Runtime、CUDA MPS 或私有模型已准备完成。

`third_party/Scroll` 只是历史架构参考，日常构建和运行不需要初始化该子模块。

### 3. 放置 Runtime

如果已经取得配套 `cr-native-sandbox-runtime-150535029.zip`，先校验：
发布包可从 [Runtime 150535029 Release](https://github.com/IMAX9D/cr-native-sandbox/releases/tag/runtime-150535029-20260903) 下载。

```powershell
$Expected = "8d829f219455ad5cb48abba717eaac2ffa97e0d51c8c9dbc76d2ad91ed15bc28"
$Actual = (Get-FileHash .\cr-native-sandbox-runtime-150535029.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Actual -ne $Expected) { throw "Runtime ZIP SHA-256 mismatch: $Actual" }
Expand-Archive .\cr-native-sandbox-runtime-150535029.zip -DestinationPath .
```

ZIP 内以 `runtime/` 为根目录，直接解压到仓库即可。也可以手工把同一安装包的
5 个 Split APK 放到：

```text
runtime/apks/
├─ base.apk
├─ split_config.en.apk
├─ split_config.hdpi.apk
├─ split_config.x86_64.apk
└─ split_install_time_asset_pack.apk
```

不要混用不同版本、ABI 或不同安装来源的 Split。仓库不会下载或分发游戏文件。

### 4. 配置本机路径

```powershell
Copy-Item runtime.env.example.ps1 runtime.env.ps1
notepad runtime.env.ps1
. .\runtime.env.ps1
```

通常只需修改 Android SDK 与 JDK 路径。Runtime、AVD 与可写输出默认放在
`runtime/`、用户 Android AVD 目录和 `%LOCALAPPDATA%`，均不会提交到 Git。

### 5. 引导、诊断和验收

```powershell
.\scripts\bootstrap.ps1
.\scripts\doctor.ps1
.\scripts\smoke.ps1
```

- `bootstrap.ps1` 安装固定 SDK/NDK/System Image、创建 rootable AOSP AVD、
  提取 Runtime 并冻结哈希；
- `doctor.ps1` 检查工具链、Runtime、Assets、AVD、虚拟化、端口和磁盘；
- `smoke.ps1` 构建宿主并验证 Tick 0→100、六塔、无 Surface 和规范状态哈希。

`smoke.ps1` 会启动无窗口 AVD；只想做静态检查时仅运行 `doctor.ps1`。

Linux x86_64 无 AVD/KVM 部署见独立仓库：
[IMAX9D/cr-native-linux-bionic](https://github.com/IMAX9D/cr-native-linux-bionic)。

## 日常运行

加载环境后启动一个 Worker：

```powershell
. .\runtime.env.ps1
python -m native_core.worker start --workers 1
```

停止服务并关闭其 AVD：

```powershell
python -m native_core.worker stop --workers 1 --stop-vm
```

Python API 示例：

```python
from native_core import NativeRoyaleEnv

with NativeRoyaleEnv(port=37031) as env:
    state = env.reset()
    state = env.step(20)
    print(state["tick"], state["state_hash"])
```

更多 API、部署网格、技能命令与观测字段见：

- [沙盒运行时技术文档](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
- [全卡/形态/技能接口](docs/NATIVE_FULL_CARD_RUNTIME.zh-CN.md)
- [Android 生命周期探针](android_probe/README.md)

## 训练与 Expert Self-Play

仓库包含三层训练代码：

1. `training/`：历史 Self-Play v0.1；
2. `selfplay_v2/`：连续行动率 Self-Play v0.2；
3. `expert_v1/` 与 `expert_selfplay_v1/`：专家数据编译、BC、在线采样、
   Critic/PPO、Promotion 和恢复流程。

训练数据、checkpoint 和运行目录不在仓库中。旧的根目录 `.cmd` 与部分实验脚本
记录了作者机器上的具体实验入口；对外部署应优先使用参数化的 Python/PowerShell
入口，并显式传入 dataset、checkpoint、output/data-root。不要把文档中的本地
`D:\...` 证据路径当成可移植默认值。

当前实现说明：

- [Expert Self-Play v1](docs/EXPERT_SELFPLAY_V1_IMPLEMENTATION.zh-CN.md)
- [Stage-2 效率实测与限制](docs/STAGE2_EFFICIENCY_20260904.zh-CN.md)
- [Expert Training v1](docs/EXPERT_TRAINING_V1.zh-CN.md)
- [Self-Play v0.2 设计](docs/SELFPLAY_V0_2_CONTINUOUS_ACTION_RATE_DESIGN.zh-CN.md)
- [训练系统](docs/training-system.md)

### 当前 Stage-2 效率版本

`scripts/start_expert_selfplay_stage2.py` 的 `throughput` 配置使用 96 个原生 Worker、
6 个采集进程、每策略版本 2 波共 192 场，配合 BF16、批量输入缓存和准备流水线。
每个策略版本结束后回收 CUDA 进程，避免跨轮显存残留；原生 Worker 继续复用。

RTX 5090 D 上连续两次完整更新：384 场、506.10 秒，短测折算约 **6.56 万场/天**。
该计时包含采集、准备、PPO、检查与保存，不包含尚未接入的 Elo 晋级；
不是 24 小时耐久结果，也未达到 20 万场/天。历史 23.2 万场/天仅为纯采集短测。

这套配置的决策窗口为 12 个原生 Tick（600 ms），chunk batch 为 32；
相较保守配置，决策分辨率和 optimizer 更新频率也发生变化，不能只凭吞吐
认定对战水平不变。当前 Stage-2 对手仍为冻结专家基模，40/40/20 league 尚未接入。

首次运行所需文件、Canary、续训、监测与停机边界见
[Stage-2 运行说明](docs/EXPERT_SELFPLAY_V1_IMPLEMENTATION.zh-CN.md#一键入口)。
普通训练入口**不会自动关闭云实例**。编译推理、集中推理服务及滚动补位等
实验开关默认关闭，不能把实验脚本的存在视为性能验收。

## MuMu 实时实验

`native_core/mumu_live_controller.py` 是可选实验路径：只读观察游戏进程内存，
动作通过普通 Android 触控发送，不会修改 `libg` 或在在线客户端中调用内部命令。

它严格要求已验证的 `versionCode 160402002`。游戏更新后版本不一致会直接拒绝，
这是预期保护。重新适配前不要关闭版本守卫，也不要把旧偏移套到新版本。

## 发布与安全边界

提交前应确认：

- `git status --ignored` 中 APK、SO、JAR、Runtime、模型、数据与日志均被忽略；
- 没有上传账号令牌、私有路径配置或训练样本；
- `python -m pytest -q` 通过；
- 对原生 Runtime 的声明只覆盖已证实的版本和 ABI。

如只做 CPU 回归，可设置 `CUDA_VISIBLE_DEVICES=-1` 后运行测试；GPU 测试将跳过。
这不替代实际 CUDA/原生对局验收。仅加载可信来源的 checkpoint 和 rollout，
这些研究工具中的 PyTorch/pickle 反序列化不用于处理不可信文件。

本仓库只提供研究与测试基础设施。使用者需要自行遵守游戏许可、平台条款及所在地
法律。MIT 许可证仅覆盖本仓库原创代码，不覆盖 Supercell 或其他第三方资产。

## 文档导航

- [2026-09-05 效率版本整理说明](docs/RELEASE_20260905.zh-CN.md)
- [2026-09-03 发布整理说明](docs/RELEASE_20260903.zh-CN.md)
- [主技术文档](docs/SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
- [历史综合技术路线](docs/TECHNICAL_ROUTE.zh-CN.md)
- [Self-Play 吞吐优化](docs/throughput-optimization-20260823.md)
- [训练并发 Scaling](docs/TRAINING_CONCURRENCY_SCALING.zh-CN.md)
