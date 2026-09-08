# MuMu 实时接口：部署与验收

## 已验证范围

- Windows 上已有 MuMu 安卓设备；只操作明确指定的 ADB serial，不自动新建/切换设备。
- 已安装 `com.supercell.clashroyale`，versionCode `160402002`，`arm64-v8a`。
- libg SHA256 必须是 `d2c8efcfe8f77e21d6a5e219999e938a42daefd7ab32f546375fb91c01f9ca97`。
- 实测资源版本 `16.402.7`，实测 Android 显示分辨率 `1080×1920`；其它分辨率只是比例缩放，尚未实机验收。
- 已通过同帧读取、回放/暂停保护、己方识别、对局/进程重绑定、六次双击下牌回执及结束停手。
- 英雄技能接口、特殊模式、完整弹道/效果对象、新版无界面 Linux 宿主、旧专家模型的新卡词表兼容性不在本次证书内。

该工具面向授权测试。普通对战可能增减奖杯；它不会替你购买、升级卡牌或无限点击“再来一场”。

## 1. Python 与读取器

```powershell
git clone https://github.com/IMAX9D/cr-native-sandbox.git
cd cr-native-sandbox
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

探针与接口冒烟测试只用 Python 标准库，不需要 Torch 或模型。MuMu 设备需允许 ADB 连接，且能通过已授权的 `su` 只读访问游戏进程；本工具不会替你修改设备安全设置。

读取器自身是 Android x86_64 小程序，在 MuMu 中读取 ARM64 游戏进程，这不等于转换了 libg 的 ABI。

构建方式（Android NDK r27d 已验证）：

```powershell
.\scripts\build_mumu_live_private.ps1 -NdkRoot C:\Android\Sdk\ndk\27.3.13750724
```

也可设置 `CR_SANDBOX_NDK` 或 `ANDROID_NDK_ROOT`。输出为 `artifacts/mumu-live/mumu-live-reader-v2-x86_64`。
也可从 [预编译读取器 Release](https://github.com/IMAX9D/cr-native-sandbox/releases/tag/mumu-live-160402002-20260908) 下载工具包，解压到仓库根目录并核对校验和后，跳过 NDK 构建。

## 2. 只读检查

手动打开已有设备中的游戏，设置正确的 ADB 程序和 serial：

```powershell
$env:CR_MUMU_ADB = 'C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe'
& $env:CR_MUMU_ADB connect 127.0.0.1:16416
python -m native_core.mumu_live_probe --serial 127.0.0.1:16416 --seconds 15
```

默认结果在 `artifacts/mumu-live-probes/时间戳/`；`--output` 可以更换目录。
版本、ABI、libg SHA 或读取器上传哈希不符时停止。大厅可能没有有效战斗对象，这不是解析成功的空场对局。

## 3. 显式下牌测试

先运行下列命令，再手动开始一场允许测试的对局。程序不会帮你匹配下一场。

```powershell
python -m native_core.mumu_live_action_smoke --serial 127.0.0.1:16416 --execute --max-actions 6 --seconds 240
```

去掉 `--execute` 就只观察并报告候选动作，不触屏。测试只选择明确知道费用的加农炮、野猪、冰精灵、小骷髅、火枪手、冰人、火球、滚木；其它卡不会被猜测使用。

- 开场前150个native Tick不发送输入，避免在入场界面下面提前点击。
- 下牌是“点卡牌 → Android端等待50ms → 点落点”，不拖拽。
- 回执同时要求指定槽位轮转和圣水下降，不把任意手牌变化当成功。
- 校准包含side1水平镜像和一格纵向偏移修正。
- 原生生成单位有实际延迟；实测确认约1.2–1.35秒，不等于接口卡住。
- 达到动作上限后只观察，终局/失去可控状态/换局即退出；未确认的输入不盲目重复。
- 不加载专家模型，不评价胜率。

输出 `events.jsonl`、实时 `progress.json`、最终 `summary.json`；`passed=true` 要求至少3次成功、所有发送均获确认，并观察到结束/换局退出。

## 4. 可选专家模型接管

自行提供兼容且可信的模型与所需词表/数据；模型不在仓库或内核快照中。

```powershell
python -m pip install -e ".[training]"
$env:CR_EXPERT_CHECKPOINT = 'D:\models\compatible-expert-fp16.pt'
.\scripts\start_mumu_expert.ps1 -Checkpoint $env:CR_EXPERT_CHECKPOINT -Python .\.venv\Scripts\python.exe -Serial 127.0.0.1:16416 -DryRun
```

先 DryRun 看词表及观测是否正常，再主动去掉 `-DryRun` 开启触屏。未知卡牌词表不应绕过检查。
启动脚本可接受 `-Adb`、`-NdkRoot`；如果可信的预编译读取器已放好，无需每次重编译。

监控 GUI：`python -m native_core.mumu_live_monitor`。控制器和监控默认共用 `artifacts/mumu-live-expert/`；用 `CR_MUMU_LOG_ROOT` 同时覆盖。双击 CMD 入口可用 `CR_EXPERT_PYTHON` 指定你的虚拟环境解释器。

## 已知外部风险

曾在“再来一场”加载94%时发生Android ANR，主线程停在MuMu ARM转译层互斥锁等待。保存了日志，重开游戏后恢复；根因未定，不能承诺模拟器本身永不挂起。
接口遇到暂停、终局、失联、空/未验证场景不会继续下牌。异常时先停止测试并保留日志，不要关闭版本守卫。

详细证据：[指针链与实测记录](MUMU_LIVE_POINTER_ADAPTATION_20260908.zh-CN.md)。旧 `150535029/x86_64` 无界面运行包保持原样；不要用这份 ARM64 文件覆盖它。
