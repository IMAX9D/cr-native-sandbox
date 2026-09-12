# 第二轮 Linux 性能优化

状态：已完成单进程与 12×4 并发对照。384 场 A/B/B/A 动作和结果相同，吞吐提升约 21.1%。详见 [优化报告](OPTIMIZATION_REPORT.zh-CN.md)。

本目录是独立实验实现，不改变模型权重、物理 tick、下牌阈值、训练精度或奖励。
它不更新本地人机盒子，不启动 PPO 学习，也不关闭云实例。

## 三类优化

1. `fast_history.py`：保留严格过去事件、未知信息、去重及重置语义，改用 NumPy 一次组装历史张量。
2. `optimized_agent.py` / `run_variant.py`：同一不可变帧在双方 prepare 之间复用规范化结果，finish 不再重复解析；缓存只在同步 prepare 组内有效，collate 后清空，跨局 reset 清空待完成帧。
3. `dense_forward.py`：只允许 eval、T=1、所有帧有效且无监督 mask 的在线路径；直接调用同权重 LSTM，避免训练序列打包/排序及长度回传。保留输入历史检查、输出有限数检查，padding/训练输入明确拒绝。空间位置修正对全有效行直接计算，不改变任何权重。

原生宿主附加优化：`native/resident_runtime.inc` 仅在需要读取时创建 SafeMemoryReader；`CR_NATIVE_RESIDENT_BIND_CACHE=1` 对 bind/unbind 复用已验证且持有 loader 引用的同路径运行时。create/step 仍执行完整 RVA 检查，slot/range/path 检查保留。不写游戏 singleton，不改变引擎更新函数。

## 部署/运行关系

沿用上一轮 `/root/autodl-tmp/resident-linux-20260912` 的已验证基线、模型合同以及 `bc-cloud-bench-20260911/core` 依赖。
代码仍是实验环境快照，不是干净仓库一键部署包。

- `launch_control.py`：原始库/参数，端口 24431 起。
- `launch_opt.py`：新库，端口 25431 起；`--cache` 显式启用绑定缓存。
- `CR_OPT_VARIANT=baseline|history|state|combined|dense`：逐项对照；dense 为组合 CPU 优化加在线推理优化。
- `CR_OPT_VERIFY_DENSE=1`：独立复制参考模型，逐次比较输出与 hidden，仅用于正确性，不能用于测速。
- `load_suite.py`：12×4、每轮 96 场、A/B/B/A，全量动作摘要和终局必须一致；结束/异常仅停止自己拥有的进程。
- `profile_suite.py`：分离墙钟、CPU 时间及 CUDA 活动。计时、trace 文件分别保留。

此前启动端口 41431/42431 落在机器临时端口范围 32768–60999 内；一次基线启动报 EADDRINUSE。
新旧负载对照同步换成范围外的监听端口，不改系统网络设置。未确定当时具体端口占有者，不将推测写成定论。

## 适用边界

- 四槽仍在一个原生执行通道交错推进，不能并发重入同一进程 libg。
- 新模型合同固定 4 tick 决策、历史长度 4，约 4.804M 参数；接口不兼容时拒绝。
- 只测固定卡组冻结策略采集，不等于全卡、PPO 学习器并发或 24h 稳定性已认证。
- `profile_suite.py` 迁移到新端口用于后续复现；已有单进程原始测量实际使用 41431/42431，日志保留原地址。
