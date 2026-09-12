# Worker 侧的进一步优化：原生循环、多局驻留与安全边界

后续更新：实体指针批读与采集并行已进入[云端对照实测](WORKER_CLOUD_TEST_20260910.zh-CN.md)；多局驻留、fork和C++完整热循环仍未实现。下文的“未云测”描述保留的是研究阶段状态。

2026-09-10。状态：源码、既有实测和系统文档核对完成；只读审计工具已添加。**下列 worker 改造尚未执行云端 A/B，未改变游戏内核、物理时间步或奖励。**

## 1. 结论和优先级

还能优化。最值得推进的是“原生热循环＋批量安全读取”，然后才是“单进程多局驻留”。不能把更少进程、更少线程或更少内存直接等同于更高训练吞吐。

| 方向 | 具体办法 | 主要可能收益 | 先决条件 |
| --- | --- | --- | --- |
| 重复读取减负 | 一次取实体指针数组，批量读取分散结构；终局轻检查和完整观测分离 | 稳态 CPU / 系统调用成本 | 短读不能伪造成塔死亡，逐 Tick 终局语义不变 |
| 原生热循环 | ART 完成初始化后进入 C++ 服务循环，缓存已验证函数表，输出紧凑结构 | Java/JSON/JNI 往返、分配与 GC | 仍保留版本、对局代次和请求序号验证 |
| K=2/4 多局驻留 | 一条原生执行 lane 轮流推进多个 MatchContext | 共享固定成本，提供更多在途对局 | 隔离 RNG、全局状态、终局记录、队列和遥测 |
| 线程与资源布局 | 按线程 CPU 增量识别开销；核对共享 inode / PSS；再试有限线程池与局部性 | 初始化、内存、调度抖动 | 不按线程数量直接裁掉线程 |
| 快照 / fork-COW | 在已证明安全的预初始化边界复制 | 冷启动、固定只读内存 | ART / 原生库 fork 安全点，目前没有这项保证 |

## 2. 已查实的两个数字

### 普通 5-Tick 决策路径可遍历实体表 13 次

按当前非终局、无重试路径静态展开：

- `nativeStep` 开始时抓取一次 episode。
- 每个 Tick 在核心推进后、外层状态推进后各抓取一次，共 10 次。
- `nativeObserveTrain` 内又抓取一次 episode。
- 然后再遍历一遍实体构建完整观测。

合计 13 次。`capture_episode_state` 主要为了识别/更新 4～6 座皇冠塔，却扫描所有实体；每个实体至少分别读取指针和结构。假设每次都有 40 个有效实体，仅这些基础读取按代码结构即可达到约 1040 次 `pread`，还不含 HP 组件、玩家与其它状态读取。

这是**静态调用量分析，不是已测 CPU 占比，更不代表能提速 13 倍**。终局、失败和其他接口的调用次数不同。

依据：[安全读取器](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:257)、[episode 抓取](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:401)、[观测入口](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:1437)、[Tick 循环](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:2301)。

### 单局进程约 78 个线程、185MiB 私有脏页

上轮一组 6-worker 样本的线程数为 468，PSS 约 1.47GB，Private_Dirty 约 1.16GB；换算每进程约 78 个线程、233.5MiB PSS、184.8MiB 私有脏页。

但当前没有逐线程 CPU 名称分布，不能断言这 78 个都是 GC 线程，也不能认为都在工作。私有页还包含当前对局，不等于全部可共享的固定开销。

依据：[原始记录](D:/Deepseek/outputs/linux-architecture-prototype-20260910/cloud-evidence/native-p4-fast-jit/p0.json)。本机的 CPU 配额是 25 核，不是从可见 affinity 数量推算的；cgroup 的 `cpu.max` 约束的是周期内可使用的 CPU 时间。[Linux cgroup 说明](https://docs.kernel.org/admin-guide/cgroup-v2.html)

## 3. 第一优先：减少旁路工作，不改战斗推进

### 3.1 保留扫描语义，先减少系统调用

先把连续的实体指针数组一次读取，再研究将实体结构按有界分组批读。批量接口需要检查实际返回字节，失败时返回无效观测或走受控读取回退；不能把未读到的对象当成被击毁。`process_vm_readv` 可能部分读取，也不保证原子快照，仍要由单执行 lane 和生命周期验证保证一致性。[Linux 接口说明](https://man7.org/linux/man-pages/man2/process_vm_readv.2.html)

当前代码对某实体读取失败会 `continue`，初始化后未再次见到的塔会被赋 HP=0。这是优化前必须处理的边界：**读失败与确实销毁必须区分**，否则缓存或批读可能放大错误终局。

### 3.2 逐 Tick 轻检查，窗口末一次完整观测

核心推进后的部分读取只用于 Tick 是否推进、是否进入终局区间，不必每次都重建全实体表。设计上拆为：轻量 Tick/生命周期检查、塔状态抓取、完整观测三层；最后一次已成功抓取的状态可以在同一原生事务内复用。

缓存标识不能只有 Tick。一次动作可能在同 Tick 改变状态，需要包含 battle 身份、generation、mutation epoch 和 Tick。王塔激活、实体增删、指针复用、重置和读失败都需要正确失效。

“只缓存六个塔指针”不是直接可交付的捷径：睡眠王塔可能稍后才进入活动实体表，地址也可能被复用。未有可靠生命周期跟踪前，保守批读比直接删扫描更容易验证。

### 3.3 缓存运行时上下文

目前多个 JNI 入口反复解析已加载库、查符号、检查基址并打开 `/proc/self/mem`。可以建立进程级 EngineRuntime，保存经过指纹验证的库句柄、函数表和读取器；请求仍核对当前对局和代次。

JIT 已在上轮证明可以减少 Java 侧 CPU，但不会重新优化 libg 的原生机器码。下一步应减少 Java 对象和跨边界工作，不应误以为物理引擎还在被 Java 解释执行。[ART JIT 说明](https://source.android.com/docs/core/runtime/jit-compiler)

## 4. 更大的架构方向：Java 冷启动，C++ 热循环

保留目前能正确初始化 Android/framework/libg 的 ART 启动流程；完成资源加载后进入一个常驻 C++ 循环：

1. 接收带版本、slot、generation、序号和长度界限的动作请求。
2. 同一执行 lane 完成动作、固定 0.05 秒原生 Tick、终局和观测。
3. 返回有界数值数组或二进制完整快照；Java/JSON 只用于低频控制和诊断。

这可以避免每次决策都在 C++ 字符串、JNI 字符串、Java JSON 和 Python 对象之间转换。它不是删除 ART，也不是重写碰撞算法。

控制与数据通道还应分开：当前 Java server 在一条持久连接内循环读取，另开的监控连接可能排在后面。低频健康状态应读取最近一次完整遥测，而不再额外抢执行 lane 或重扫场景。

增量观测可以后置：先传完整紧凑快照，验证成熟后才拆“出生时不变字段＋每 Tick 变化字段”。增量包丢失或代次不匹配必须重同步，不能让模型继续使用过期实体。

## 5. 单进程多局：省固定成本，也可能帮助 GPU 合批

本地 FirstLight 快照的 resident 实现有最多 16 个槽位，但只有一条 libg 执行 lane。它保存 manager、generation、状态 epoch、终局和塔兵遥测；切换不是把整个战场序列化再加载。[参考源码](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/probe/resident_multimatch.inc:4)

我们不能只添加 `manager[]`。目前 C++ 的 `g_episode`、全局 manager、Java 终局闩锁和动作/观测身份都需要归属到具体 MatchContext；还要查明 RNG 和后台原生任务是否依赖全局当前对局。

它的价值不是“一条 CPU 同时计算四场”，而是：A 等模型结果时，同一 lane 可以处理已就绪的 B；只加载一份宿主和可共享资源，维持更多在途对局，给 GPU 提供更大的候选批次。中央推理仍按 Actor 哈希保留一份活跃权重，不给每局复制模型。

设 P 为进程/lane 数、K 为每进程局数、N=P×K：

> 总内存 ≈ 共享只读资源 + P×宿主固定成本 + N×对局可变状态 + 推理与轨迹缓冲。

不能直接把当前 185MiB 全视为固定成本。也不能从“24 进程改为 6 进程×4局”直接推出更快，因为 CPU 执行 lane 同时从 24 条降成了 6 条。

先验证 K=2，再到 K=4，并分别做 P 固定和 N 固定的对照。当前 90GiB 内存还有余量，增加单局 worker 数也应该保留为低工程成本对照，不能为了架构复杂而直接上多局驻留。

## 6. 看似巧妙、但不应先押注的方法

- **初始化后直接 fork/COW**：Linux fork 确实使用写时复制，但多线程进程的子进程只保留调用线程，并复制锁状态。ART/原生线程池是否有可靠的 fork 安全点目前未知，不能当作可直接续跑的快照。[fork 文档](https://man7.org/linux/man-pages/man2/fork.2.html)
- **复制后沿用旧读取句柄**：内核在打开 `/proc/PID/mem` 时保存目标 mm，后续读取用的是这个对象。结合 fork 的文件描述符继承语义，父进程先打开的 self/mem 句柄不能假定会自动转而读取子进程；这也是快照方案必须处理的边界。[Linux 源码](https://raw.githubusercontent.com/torvalds/linux/v6.8/fs/proc/base.c)
- **多个线程同时调用同一个 libg**：没有可重入和完整状态隔离的证据，可能串局。
- **直接加大 dt、按空场跳 Tick**：碰撞、冷却、延迟下牌和费用积累仍可能发生，不能作为保持原游戏语义的优化。
- **立即在 GPU 上重写物理引擎**：工程和一致性风险很高；可以作为辅助模型/搜索研究，不可未经验证替代权威训练环境。
- **盲目绑核或裁线程**：affinity 能改变允许运行的位置，不增加 CPU 配额。先测线程 CPU 增量与局部性，再决定是否限制辅助线程。[CPU affinity 文档](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)

当前已通过 native replay restart 在同一宿主内替换 BattleGameState，**并非每局重启整个 ART**，所以“做一个持久进程池”不能再次当成新收益。[重置入口](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:2693)

## 7. 已补的审计工具和后续验收

新增只读脚本 `scripts/audit_native_worker_costs.py`。它仅接受明确 PID，先验证原生宿主身份，再报告线程 CPU 增量、线程生命周期缺口、PSS/私有页和 libg backing inode；不发送信号、不注入、不调用游戏函数。

本地只对解析器和计数逻辑做了四项测试，**尚未在开机的 Linux worker 上采样**。例如后续在已获授权的云端可运行：

```text
python scripts/audit_native_worker_costs.py --pids <明确的worker_PID列表> --seconds 2 --output worker-costs.json
```

验收顺序：

1. 查明线程与系统调用成本；采样本身不与正式吞吐 A/B 混算。
2. 做批读/原生融合入口，保持同版本、同动作轨迹和 5-Tick 决策合同。
3. 覆盖王塔激活、技能、召唤、大量单位、读失败、平局决胜与跨局重置，比较每个观测点的 Tick、HP、皇冠和动作结果。
4. K=2 时单独跑 A/B 与交错跑 A/B 对照；只重置 A 不能改变 B，迟到响应不能进入新一局。
5. 最后接入真实模型测完整数据闭合吞吐。不能只用纯 Tick 数推算每天多少场。

这一轮没有改动 worker 的游戏推进代码，没有新开云机。建议先做原生读取/热循环，再按证据决定驻留多局；不是先更改奖励、减少观测字段或扩大时间步。
