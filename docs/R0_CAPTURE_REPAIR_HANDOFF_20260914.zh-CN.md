# R0 Elite 数据处理修复交接（2026-09-14）

> 历史实现/交接记录：后续进展以 [v17状态](R0_ATTRIBUTION_V17_STATUS.zh-CN.md) 和 [当前发布说明](R0_ELITE_RELEASE.zh-CN.md) 为准。


## 验收对象与结论

对象是最新版 `r0-public-observation.elite-v3`：72 项实体特征及 6 项生命周期伤害统计，同时包括塔楼、卡牌、事件、关系、命令历史等输入。默认模型 36,853,407 参数，`allow_incomplete_capture=False`。不是旧 177M 专家模型，不使用简化输入。

**局部真实训练闭环通过；完整 60 万局处理尚未通过验收，未启动全量训练。** 为节省用户剩余额度，本轮停止扩展研究，保存原始证据和严格合格连续前段。不能把下列局部成功当作完整机制或历史原版本复现认证。

## 已实现

- Tick 0 开始逐 Tick 采集；每 5 Tick 编译决策。状态与随后动作对齐，重复 Tick 的命令后状态不泄漏到此前决策。
- 完整源八卡及出牌序列校验、原生兼容起手搜索；不要求每张已知卡都实际出过。兼容种子不等于原始随机种子。
- 处理契约与下载器契约分离；72000268 模式有资源直通候选依据，缺失等级仍拒绝而不猜值。
- 原生伤害、出生、治疗、攻击释放、阶段/特殊运行时读取；区域效果与 EXT 动态编号分类。
- 跨 Tick 公共事件窗口、死亡塔终态保留、实际补牌空槽、双方公平视角；飞行对象越界坐标保留而不硬夹到边缘。
- 资源 Base/EXT 继承和已验证的增量字段；非圆形或视觉效果不能伪造圆半径。
- 炸弹塔死亡响应中的父代归属；1HP 生存保护保留真实零有效伤害；国王死亡且原生命令门关闭后结束游戏样本，避免清场伤害被记给单位。
- 非 pickle NPZ 连续序列，源/采集/分片哈希，原子落盘；拒绝与未解析操作的标签屏蔽。默认严格模型完成真实梯度更新、模型/优化器/RNN 保存恢复。

## 完整主样本实证

`000YL00UC2R0`，v9 原始采集 4,288 帧、69 次成功部署、3 次成功技能、1 次未解析技能；595 条伤害回执，无原生读取拒绝、无断帧。终局 Tick 4283。

最新完整语义审计产生 1,714 个双方决策样本：11 个采集域全部通过；生命周期归因 1,222 通过，492 未通过。证据为：

`D:/Deepseek/outputs/r0-complete-capture-20260914/august-source/capture-complete-v9/000YL00UC2R0/r0-semantic-audit-final-v9.json`

报告 SHA256：`605c4a1b80a16ec2a7d871d73f8e30bb4710e2d725283a403c24c75163650a26`。

## 未完成的硬阻断

Tick 3055，区域对象 `3000005`（数据 ID `856278373`）造成 674 伤害，但唯一原生注册记录未携带可靠父代或部署命令。同 Tick 新建骷髅兵等现象指向气球英雄着地爆炸，但这是调查假设，不是已经确认的原生父链。该段及受到归因不确定性影响的后续样本仍不合格。

下一步应在该着地/延迟生成路径补原生父代证据，再完整重放这一局；不要清零错误计数、把无源伤害当环境伤害、或放宽严格模型开关。

## 覆盖边界

- 9 个机制候选中，8 个完成原生采集，1 个因精确国王等级缺失而未进入采集。候选含卡不等于实际机制激活。已实际观测进化地狱龙的累积/衰减、周期强化和治疗；抓取、钻地转移尚未获得完整活跃验收。
- 主归档 600,000 局；其中 33,740 个 schema3 源还需补列表元数据，不能只改 schema 数字。所有原始数据保留。
- 本地三套资源归档对应两个 libg 哈希；另一旧 ARM64 文件不代表额外历史更新节点。新 ARM64 资源存在不等于 Linux 采集链路已验收。
- `ruleset_resolved` 只能表示处理内核/资源/模式已明确绑定；`original_state_exact`、原始种子及历史版本完全匹配仍不能标为真。

## 复现入口

- 原始采集：`python -m data_pipeline.pilot100 capture --help`
- 数据编译：`python -m data_pipeline.r0_compile --help`
- 严格训练与恢复验证：`python -m data_pipeline.r0_training_smoke --help`
- 来源/内核覆盖审计：`python -m data_pipeline.runtime_coverage_audit --help`

使用 `D:/AI_data/runtime/venv/Scripts/python.exe`。当前本地 Linux 运行时目录 `/opt/cr-capture-repair-20260914`，三个测试端口 39531–39533。本轮未启动付费云实例，收敛时已停止三个本地测试 worker。完整代码仍在本地工作区，未擅自推送 GitHub。

收敛阶段合并回归：82 项 R0/数据编译/采集修复测试通过（8.11 秒）。原始 v9 主采集与机制采集已复制至 `A:/CR_AI/datasets/royaleapi-600k-v1/validation-repair-20260914`；主原始分片复制后 SHA256 与 D 盘一致：`2d76fadc3bbe0d8d1ae86b59d3a533fd0f900c5dcc8969a00de037279939c98c`。

最终限定前段：`compiled-prefix-v9`，使用显式 `--max-tick 1200`，原始全局未改；482 个双方决策（16 动作窗口、464 等待、末尾 2 个标签屏蔽）。最新契约 `a58c2b5b2fdc8438297e9091e6618992263e45d89f3781adb38e9e914bc7cd40`。一侧真实八步训练、模型/优化器/RNN 保存恢复再次通过，Loss 1.5948197、梯度范数 7.5083184。报告为 A 盘上述目录下 `strict-il-smoke-prefix-v9-report.json`。测试权重为随机初始化的管线 smoke 产物，**不是新专家模型**；未复制成可用专家权重，也未宣布完整数据集准入。

## 给接手智能体：工作顺序

1. 先阅读本文件、最新完整语义审计 JSON 和当前代码，不必重复下载 FirstLight 或重做已通过的基础研究。用户剩余周额度紧张，优先解决具体阻断，不扩展架构研究。
2. **先修 `3000005` 的原生出生归属。** 事件是 lifecycle sequence 674、native tick 3054、观察 tick 3055、parent=0、command=0；只有一次注册，不是漏读第二次 pending→live 注册。对应伤害序号 390、674 点，区域对象同 Tick 出生并消失，快照没有存活实体。应追实际 factory/延迟动作/父实体 tick 调用上下文；不能按附近实体、同 Tick 或名称猜父代。
3. 改动后构建新不可变 bundle、新 capture attempt；只重放主样本。检查父代代际、同局/同侧约束、非单位源与单位后代的区别，以及伤害累计截至当前 Tick 的因果性。已有 `DamageScope` 修复炸弹塔死亡生成，但不足以覆盖此次延迟路径。
4. 全段 `ReplayFeatureAssembler` 严格审计应不再因这条事件污染后续生命周期。检查全部窗口，不只挑开头做一次 forward。随后重编数据、在最新默认模型上做真实反向及保存恢复。代码/资源改变会改变契约，旧缓存不能继续冒充最新缓存。
5. 再验证已经采好的 8 个机制样本；牢笼抓取/钻地转移只有候选、缺活跃证据。原有 controlled-cage-v9 仅 1 次成功命令、115 次拒绝，**不能作为机制通过证据**。需要时先修该小测试的合法循环，再做有限验证。
6. 接着处理 schema3 缺元数据、模式/历史内核路由；扩大到约 100 局覆盖试处理后，再考虑 60 万局。原始层必须保留，训练层只允许满足契约的数据或明确合格连续片段。

### 必须保持的边界

- 不把未知值变成零、不直接赋满 `REQUIRED_CAPTURE`、不改 `allow_incomplete_capture=False`。
- 拒绝/未解析技能保留在原始层并屏蔽相应标签；不得伪造成功或把漏日志当 WAIT。
- 同局两侧、重叠片段、重复源不可跨训练/验证/测试集合。
- 72 个字段均有保留位置不代表所有机制已经采到。合法 unknown/N/A 必须有类型或资源证据；字段未实现不能说“不适用”。
- 用户授权必要启动/下载/控制，但最后要求尽快收敛省额度。本轮没开付费云；不要因历史长对话里有 SSH 凭据就擅自恢复旧训练。
- 工作区有大量既有未提交修改，不要 `reset --hard`、大范围覆盖或整体删除。此次没有 GitHub push 请求。

## 文件导航

主仓库：`D:/Deepseek/CR-Native-Core`。

| 位置（相对主仓库） | 用途 |
|---|---|
| `android_probe/native/elite_origin.h` | 线程局部原生动作/命令上下文 |
| `android_probe/native/elite_lifecycle_telemetry.h` | 注册、移除、治疗及父代环形事件流 |
| `android_probe/native/elite_telemetry.h` | 有效伤害、1HP 保护、原始详细字段 |
| `android_probe/native/elite_phase_telemetry.h` | 精确版本攻击/移动阶段钩子 |
| `android_probe/native/elite_special_snapshots.h` | 捕获、转移、周期强化、变量、弹丸快照 |
| `android_probe/native/jni_bridge.cpp` / `android_probe/java/royale/nativehost/JniHost.java` | 无头接口、分类、终局及运行工件身份 |
| `training_r0/replay_adapter.py` / `lifecycle.py` | 连续语义装配与因果伤害账本 |
| `training_r0/replay_facts.py` / `training_r0/native_projection.py` | 资源继承、字段解析、R0 专用坐标投影 |
| `data_pipeline/processing_contract.py` | 独立处理契约及 `compile_processing_battle` |
| `data_pipeline/pilot100.py` | 原生采集、兼容起手、动作回执和原子输出 |
| `data_pipeline/r0_compile.py` / `r0_training_smoke.py` | NPZ 编译、显式截止 Tick、严格训练恢复验收 |
| `data_pipeline/r0_admission.py` | 契约、分片和证据检查；不等于证书已完整签发 |

数据/工具目录：

- 源归档：`D:/AI_data/datasets/royaleapi-600k-20260914/royaleapi-600000.tar.xz`，SHA256 `6365a6270178e5ca0cbbd6ae32f49e6f3af3db3832871f148b39ad862467794c`。
- 源索引：同目录 `catalog/dataset-index.sqlite3`，只读查询；主归档 600000，另一个归档的 1 条不要混入数量。
- 本轮全部证据根：`D:/Deepseek/outputs/r0-complete-capture-20260914`。
- 完整主采集：证据根下 `august-source/capture-complete-v9/000YL00UC2R0`。
- 机制采集：证据根下 `mechanism-sources/capture-special-v9`；选择清单位于 `mechanism-sources/selection.json`。
- A 盘最终归档：`A:/CR_AI/datasets/royaleapi-600k-v1/validation-repair-20260914`。其中 `compiled-prefix-v9` 的 35 个文件复制后逐个校验一致。
- 下载器源码：`D:/Deepseek/work/royaleapi-downloader`。schema3 可用现有 `upgrade_exact_replay_body` 升级动作正文，但仍须补列表元数据；不能只改版本号。
- 完整 FirstLight 参考源码：`D:/Deepseek/research/FirstLight_CR-full-20260914`，commit `28d66cc0a5d65888515e22fdf22f11d783b65efb`。Apache-2.0 的 LICENSE/NOTICE 已保留在本项目 `third_party/FirstLight_CR`。
- 当前旧版资源根：`D:/AI_data/cr-native-core/direct-runtime-export/cr-native-direct-0`。
- 可反汇编的本地代码：`D:/Deepseek/cr_re/dumps/libg_DECRYPTED_x86_64.so`；原始 libg 的代码段加密，不要直接对加密字节推断指令。

## 最小复现（PowerShell）

```powershell
Set-Location D:\Deepseek\CR-Native-Core
& D:\AI_data\runtime\venv\Scripts\python.exe -m data_pipeline.r0_compile `
  D:\Deepseek\outputs\r0-complete-capture-20260914\august-source\capture-complete-v9\000YL00UC2R0 `
  --output D:\Deepseek\outputs\r0-next-validation --max-tick 1200 --sequence-steps 16
& D:\AI_data\runtime\venv\Scripts\python.exe -m data_pipeline.r0_training_smoke `
  D:\Deepseek\outputs\r0-next-validation --output D:\Deepseek\outputs\r0-next-smoke
```

两个输出目录必须未存在。去掉 `--max-tick` 才是整局编译，现有整局尚不能称合格。重复 smoke 只验证管线，不训练真实专家能力；无需为交接重复运行。

### 本地 Linux/构建

WSL `Ubuntu` 已安装 Bionic 用户态，IP 最后为 `172.23.89.208`，重启后需重新查询。三个 worker 已全部停止；最后 v9 PID 为 456/457/458，但不能凭旧 PID 杀进程。

可复用 bundle：证据根下 `complete-bundle-v9`，显式 manifest `BUNDLE-MANIFEST.json`。bridge SHA256 `8c5677550758a8bdeff3bc26485a0e607e7b0e64468b0be2537969e9ea5d7556`；jar SHA256 `f95517bc9b55e8d6f9bfb982cb623de26fb2d7fa55894ac3cdfad348a7e91f93`；libg SHA256 `fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba`。

Linux 启停工具在 `D:/Deepseek/CR-Native-Linux-Bionic`；参数使用 `--runtime-root /opt/cr-capture-repair-20260914 --count 1 --base-port 39531`。完整启动还需显式 `--bundle-dir` 和 `--manifest`。运行 slot 为 `/data/local/tmp/cr-native-direct-0`，模板资产在 `/opt/cr-capture-repair-20260914/worker0/assets`。

**部署注意：构建产物名 `libnative_core_probe.so`，但 JAR 实际加载 `libnative_host_bridge.so`。** 两者不能弄错；采集器现会核对 libg/JAR/bridge 与 manifest 哈希。先确认目标 worker 真已停止，再复制，不要把多个命令连在一起而忽略 stop 返回的 `all_stopped:false`。

另有已知启动器隐患：worker 正忙时 ping 超时可能被误判未启动，再次 start 会启动重复进程并覆盖 PID 记录。此次通过 `ss` 监听 PID 与 `/proc/PID/cmdline` 核对归属后清理了重复启动记录；**启动器该隐患未修复**。接手时不要对正在采集的端口再次 start。

编译器：NDK `D:/Codex/toolchains/android-ndk-r27d`；JDK `D:/Codex/toolchains/jdk-17.0.20.1+1`；SDK `D:/Codex/toolchains/android-sdk`。使用 `scripts/build_bridge.ps1`、`scripts/build_probe.ps1`，生成全新输出目录，再通过 Linux 仓库 `scripts/build_host_overlay_bundle.py` 打包。不要覆盖已有证据 bundle。
