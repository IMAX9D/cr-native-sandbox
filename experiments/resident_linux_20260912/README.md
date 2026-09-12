# 四局驻留 Linux 宿主与全链路性能实验

这是 2026-09-12 云端已实测实现的**独立源码快照**，不是默认运行时升级，也不是已上线的正式 PPO 训练系统。默认 `android_probe/` 不被这个目录覆盖。

## 已验证内容

- 单原生进程四个独立对局，单执行通道交错推进，独立历史/RNN/重置。
- 新 4.804M 模型、每 4 tick 决策；冻结策略双方对弈采集。
- 1/4/8/12/16/24 进程扩展、单槽对四槽匹配对照、288 场连续补位对照。
- 分离 CPU 墙钟/执行时间与实际 CUDA 活动，保留没有提速的共享推理实验。
- 原生 worker 退出后的隔离和新局恢复；不代表已验证学习器故障恢复或 24h 稳定性。

完整结果见 [性能报告](REPORT.zh-CN.md)。`results/` 是小型结果摘要，`SOURCE_MANIFEST.json` 记录复制的源码和结果哈希。大型 traces、模型、游戏资源、二进制及 SSH 辅助程序不在本目录。

## 文件布局

- `native/`：四局驻留 JNI 宿主、状态读取辅助头文件与 opt-in profiling。
- `harness/android_probe/java/`：匹配该 JNI 的 Java 宿主及 Android 接口桩。
- `harness/scripts/build_probe.ps1`：Java 构建脚本，需要显式提供本地 JDK/Android SDK。
- `pool_launch.py`：独立端口与 PID 所有权检查的原生进程池。
- `policy_test.py`、`split_agent.py`：合批策略和槽位适配。
- `continuous_test.py`：结束槽位立即补位、独立新局状态。
- `scale_*`、`slots_compare_suite.py`、`long_stability_suite.py`：对照驱动。
- `profile_*`、`analyze_*`、`summarize_profile.py`：采样和分析。
- `shared_policy.py` 等：**性能不达标、未采用**的共享服务原型，留作分析。

## 复现前置条件与边界

这些脚本保留实测机的 `/root/autodl-tmp/...` 和 `/data/local/tmp/...` 路径，**不能在刚 clone 的仓库直接一键运行**。运行前必须准备并核对：

1. 合法取得的版本锁定游戏资源与 Linux Android/Bionic 用户态，运行时 `cr_native_bionic` 模块、工作模板目录；不是桌面安卓模拟器。
2. 与 `native/` 配套编译的 `.so` 与 Java `.jar`。不要混用默认宿主或未经认证的新游戏版本。
3. 发布模型配套的 `hokoff_model.match_agent`、观测编码合同、示例卡组和权重；它们来自模型配套工程，不是本目录自产依赖。权重路径/hash/词表/输入结构必须匹配。
4. 检查每个脚本的 CORE/ROOT/BASE 和 runtime import 路径后再迁移。默认槽位端口从 41431 开始，不与生产/人机进程共享目录或端口。
5. 根据实际依赖重跑合法性、观测、动作序列、RNN 隔离与终局对照，再开始新压测。

保留硬件/模型合同和逻辑不变的前提下迁移路径；不要通过放宽版本校验使脚本“能跑”。后续可产品化为参数化部署入口，但这个提交只同步已验证的实验版本。

`analyze_trace.py` 的 `trace-*.json` 通配也会匹配生成的 `trace-analysis.json`：重复分析前应选择原始数字编号 trace，避免把分析文件当 trace。首次分析得到的本目录结果已校验。

## 与历史训练的关系

旧 177M Stage-2 曾执行 PPO 参数更新，见 [2026-09-04 记录](../../docs/STAGE2_EFFICIENCY_20260904.zh-CN.md)，当时对手为冻结 BASE，完整 league 尚未接通。

本目录的 4.804M 模型实验只测采集，不会调用 optimizer 更新权重。288 场对照及吞吐结果不代表训练完成，也不能证明模型强度提升。
