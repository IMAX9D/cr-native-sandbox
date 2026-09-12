# Linux 无头内核多并发优化：研究结论与实施设计

日期：2026-09-10 UTC。状态：**此文保留最初的研究阶段记录。后续已实施并进行了同硬件实测，最新结论见 [云端实测与实现报告](LINUX_ARCHITECTURE_EXPERIMENT_20260910.zh-CN.md)。下文“本轮”均指最初的只读研究阶段，不代表后续实测状态。**

本轮没有开云机、连接云端、启动模拟器、训练或重压测，也没有修改运行时代码、R0代码、模型、奖励或决策时间合同。此前四台实例的备份与释放是已经完成的另一项工作；F59保持先前的关机安排。

## 1. 总结：可以继续优化，但优先级需要调整

推荐路线是：

1. 修复测量口径，建立可区分内核与宿主的基线。
2. 减少Java JSON往返、重复序列化及逐Tick全实体扫描；保留ART初始化，先精简热路径。
3. 集中推理、同一策略版本内的ready队列、有界缓冲和RNN状态银行。
4. 先共享确定只读的资源，再验证每进程2/4局驻留；不直接对同一libg开启多个战斗线程。

最值得立即实施的是前两项。**不建议先堆更多worker，也不建议先重写一个完全脱离ART的宿主。** 多局驻留是重要的中期方案，但其收益首先是固定内存和初始化摊销，不是凭空增加CPU算力。

## 2. 版本和结论边界

| 对象 | 本轮核对状态 | 能用于什么结论 |
| --- | --- | --- |
| 主工程 `7b91ad2` | 旧链路源码及历史记录可读；另有用户未提交R0工作，未改动 | 定位已有实现开销 |
| Linux仓库 `43001ce` | 旧版生产链＋新版ARM64移植分支 | 分开讨论两条ABI路径 |
| `150535029 / x86_64` | Linux Bionic已有对局运行记录 | 后续性能A/B的可运行基线 |
| `160402002 / ARM64` | ART入口已过，SCID初始化仍阻塞；没有Linux对局验收 | 不能作为可训练或提速结果 |
| FirstLight本地快照 `28d66cc0` | 已保存相关源码片段，不是完整可运行仓库 | 参考驻留多局与协议设计，不能照搬RVA或成绩 |
| R0 | 新设计入口；原生在线采集尚未接通 | 保留250ms/5Tick、最多两微动作、40游戏秒段的设计合同 |

旧177M/600ms采样结果不能当成新R0/250ms的性能证明。本轮也不把旧177M恢复为必须沿用的模型。

依据：[Linux兼容边界](D:/Deepseek/CR-Native-Linux-Bionic/README.md:20)、[R0当前设计](D:/Deepseek/CR-Native-Core/docs/TRAINING_CHAIN_REDESIGN_20260907.zh-CN.md:29)、[FirstLight快照完整性记录](D:/Deepseek/outputs/firstlight-cr-review-20260907/SOURCE_INTEGRITY.json)。

## 3. 历史记录复算：瓶颈不应直接叫“libg太慢”

以下是**2026-09-02旧记录**：20Hz原生规则、每16Tick一次transition、确定性合法下牌、没有模型推理、详细计时开启、约60秒测量。时间是跨并发请求累计后计算的阶段墙钟，不是CPU采样。

| worker数 | 实际cgroup CPU配额 | 聚合native Tick/s | transition p95 | 选中worker RSS合计 | 整个cgroup内存 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16核 | 6,041.85 | 70.26ms | 9.78GiB | 87.98GiB |
| 48 | 16核 | 8,161.92 | 101.30ms | 14.67GiB | 87.96GiB |
| 64 | 16核 | 7,092.21 | 142.76ms | 19.55GiB | 87.97GiB |

64比48多33.3%的worker，吞吐反而下降13.1%。这是该旧负载下的扩容退化证据，不代表任何新机器都应固定48个worker。

48-worker记录共有30,705次transition，分解为：

| 阶段 | 平均每transition墙钟 | 解释 |
| --- | ---: | --- |
| 下牌处理 | 0.617ms | native调用及宿主处理，不是纯游戏物理 |
| `step_jni` | 3.303ms | 包含推进、终局采集、内存读取和字符串构建，不是纯libg Tick |
| `observe_jni` | 0.357ms | 紧凑观测生成 |
| Java解析step JSON | 1.525ms | 字符串重新转对象 |
| Java解析observation JSON | 5.584ms | 字符串重新转对象 |
| 返回前总计时 | 11.403ms | **包含上面几项，不可再次相加** |
| 额外序列化探针 | 8.755ms | 在返回前计时之外，再序列化一次用于测量 |

Java两次解析约占“返回前总计时”的62.35%。实际发送时还有一次序列化，但旧探针没有直接测到它；不能擅自认为两次耗时必然相同。

该记录RPC总传输约3.86MB/s，平均响应约7.39KB。没有带宽饱和证据。`rpc_receive_seconds`包含等待服务端处理和调度，不能把它全归为TCP成本，也不能与服务端计时相加推算总CPU消耗。

因此，本轮结论是：**宿主数据处理是优先检查的高价值路径；现有记录不足以给纯libg内核占比或任意优化倍数。**

原始记录：[32](D:/Deepseek/CR-Native-Core/artifacts/cloud-benchmarks/native-active-20260902/tier32-v2.json)、[48](D:/Deepseek/CR-Native-Core/artifacts/cloud-benchmarks/native-active-20260902/tier48-async.json)、[64](D:/Deepseek/CR-Native-Core/artifacts/cloud-benchmarks/native-active-20260902/tier64-async.json)。复算结果及源码指纹：[离线分析JSON](D:/Deepseek/outputs/linux-concurrency-audit-20260909/OFFLINE_ANALYSIS_20260910.json)。

## 4. 本轮新增确认的测量与热路径问题

### 4.1 worker CPU记录有首次采样问题

`ResourceSampler._sample_processes()`每次采样都新建`psutil.Process(pid)`，随后立即调用`cpu_percent(interval=None)`；三档历史记录的worker CPU均为0。官方接口说明非阻塞首次调用的0没有测量意义。[源码](D:/Deepseek/CR-Native-Core/scripts/run_native_active_worker_sweep.py:192)、[psutil API](https://psutil.io/api/#psutil.Process.cpu_percent)。

改进设计：保存进程对象并预热，或直接用CPU累计时间差；以PID＋进程启动时间防止PID复用，以单调时钟计算间隔。首样本记为缺失，不能记为真实0。cgroup CPU仍可说明整个容器忙，但不能据此认定全部CPU都由选中的native worker消耗。

### 4.2 cgroup占用不等于worker私有内存

48个选中worker的RSS总计只有14.67GiB，容器约87.96GiB。两者覆盖对象不同，且RSS还会重复计算共享页，不能直接相减得到精确“其它进程内存”，更不能用88GiB除48预测每局成本。

需要补采选中进程的PSS、Private_Dirty、线程CPU，以及`memory.stat`中的anon/file/kernel、PSI和其它本任务进程占用。PSS按共享者分摊页；`memory.high`会造成回收/节流，`memory.max`是硬边界。[Linux进程内存文档](https://docs.kernel.org/filesystems/proc.html)、[cgroup v2文档](https://docs.kernel.org/admin-guide/cgroup-v2.html)。

### 4.3 详细计时模式本身多做了一次序列化

压测器强制`profile_native=True`；Java先调用一次`response.toString()`计时，再调用一次交给writer。**只对该探针路径成立，不是所有生产请求都双序列化。**

改进设计：吞吐运行关闭详细探针；定位运行只抽样。若需测最终编码，编码一次并复用结果，将本次编码耗时写入旁路指标或下一条指标，避免为了把耗时塞回当前JSON而重编码。另报探针开/关差异，不将探针收益冒充生产收益。

依据：[强制探针](D:/Deepseek/CR-Native-Core/scripts/run_native_active_worker_sweep.py:549)、[重复序列化](D:/Deepseek/CR-Native-Core/android_probe/java/royale/nativehost/JniHost.java:1103)。

### 4.4 发布版Linux宿主固定解释执行

Linux包的`build_worker_command()`固定带`-Xint`。主工程的`ensure_bionic_workers.py`已支持interpreter/jit选择，但这个能力没有进入Linux包的同一入口。

先做显式可选模式与启动清单记录，不默认切换。JIT可能降低Java处理成本，但也增加预热、代码缓存和编译活动；不代表libg机器码本身被重新优化。必须对同一轨迹验证输出及长跑内存。[Linux入口](D:/Deepseek/CR-Native-Linux-Bionic/cr_native_bionic/runtime.py:148)、[已有模式开关](D:/Deepseek/CR-Native-Core/scripts/ensure_bionic_workers.py:96)、[ART JIT说明](https://source.android.com/docs/core/runtime/jit-compiler)。

### 4.5 每个原生Tick存在重复全实体遍历

`nativeStep()`的普通非终局路径，每Tick调用`capture_episode_state()`至少两次。后者遍历整个实体表，只为识别/更新皇冠塔与终局状态；读每个实体指针和结构均经过`pread`。这部分计入`step_jni`，所以它不能当作纯引擎耗时。

现有`core_update`＋受控`state_update`已经跳过重复核心推进和显示回调；外层仍承担生命周期。不能直接删掉外层frame而不验证重置与终局，以此冒充纯内核提速。

改进设计：

- 初始化时建立带generation和生命周期验证的塔引用表；普通Tick读少量已知塔及轻量终局状态。
- 国王塔激活、对象增删、战斗身份变化、读失败时重新绑定；不要把读取失败当作塔已摧毁。
- 仅在生命周期条件已证明时消除重复全表遍历；仍需逐Tick发现权威终局，不能越过终局继续推进。
- 接受结果前按原生终局、皇冠、HP与Tick对照，不对历史RoyaleAPI比分逐场重算。

对象结构的部分读取已经合并成块，不能把“第一次合并结构读取”写成全新能力。进一步重点是减少重复遍历、合并指针数组读取，以及安全地缓存不变信息。

依据：[全实体扫描](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:401)、[Tick循环](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:2355)。

### 4.6 高层调用仍重复查找运行时、跨语言装拆对象

推进/观测会重新做`dlopen(RTLD_NOLOAD)`、`dlsym`、基址验证，临时打开`/proc/self/mem`；Java还把native JSON解析成对象后再次输出。

目标不是去掉版本守卫，而是创建进程级、经过指纹验证的`EngineRuntime`，保有库句柄、函数表和reader。每个请求仍检查对局代次、当前状态及边界条件。一个native事务完成计划入队、推进、终局与紧凑观测；JSON仅作兼容/诊断出口，不能让“新融合接口”内部仍调用旧字符串接口再解析。

`process_vm_readv`可作为批量安全读取候选，但它不保证原子快照，也可能部分读取；不能替代单一执行lane与生命周期约束。[读取器](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:257)、[推进入口](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:2301)、[Linux读取接口](https://man7.org/linux/man-pages/man2/process_vm_readv.2.html)。

### 4.7 已经有的优化不应再当作新收益

训练主客户端已经使用持久连接，两端都开启`TCP_NODELAY`，写请求在模糊失败后不自动重放。Linux包中每次新连接的`tcp_request()`主要是状态/运维工具，不能据此说训练每步都重建TCP。当前宿主一次服务一条持久连接；多局方案应由宿主级复用器统一提交slot请求，或将网络收发与单一执行队列解耦，不能给每局各开一条长连接后期待旧server并行服务。

依据：[训练客户端](D:/Deepseek/CR-Native-Core/native_core/client.py:25)、[TCP设置与重试](D:/Deepseek/CR-Native-Core/native_core/client.py:127)、[宿主连接循环](D:/Deepseek/CR-Native-Core/android_probe/java/royale/nativehost/JniHost.java:811)。

## 5. 目标进程结构

```text
CPU native宿主 P个
  每宿主：1条libg执行lane，先K=1，验收后K=2/4个驻留MatchContext
       │ 观测完成后立即ready；每局最多一条未完成请求
       ▼
有界ready队列 → 按模型hash/shape分桶的集中推理
                  固定版本模型驻留；按局维护RNN状态和采样计数
       ▲                         │ 动作计划＋身份/序号
       └──────── 原生宿主按游戏Tick执行

轨迹块 → 有界写出/准备队列 → 当前版本数据闭合 → PPO更新 → 发布下一版本
```

### 5.1 同进程多局：轮流推进，不是同一libg多线程重入

FirstLight快照的resident实现上限是16个slot，但只有一条libg执行lane，持有多个manager；切换时保存/恢复generation、telemetry epoch、塔兵运行态等。它不是每切换一次就复制整个战场或重新播放回放。

我们的桥接存在单例`g_episode`，Java也有进程级终局闩锁；仅新增一个`manager[]`不够。需逐项拆分：

| 作用域 | 应归属的内容 |
| --- | --- |
| 进程共享 | 固定版本代码、函数表、确认只读的规则/资源、协议scratch缓冲 |
| MatchContext | manager/state/battle、RNG归属、终局记录、动作队列、事件游标、可变mask缓存 |
| 对局代次 | episode ID、generation、状态epoch、实体身份映射、重置失效标记 |
| 推理序列 | 模型hash、阵营、RNN状态、决策序号、随机流与历史动作 |

首先验证K=2：单独跑A/B，与交错跑A/B使用相同原生命令轨迹对照；只重置A时B必须不变，再覆盖塔兵、技能、召唤、大量单位与终局。指针地址不作为跨进程一致性比较字段。

参考：[单lane及slot结构](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/probe/resident_multimatch.inc:4)、[上下文切换](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/probe/resident_multimatch.inc:97)、[原生单例](D:/Deepseek/CR-Native-Core/android_probe/native/jni_bridge.cpp:378)。参考测试文件主要验证客户端命令前缀，不能当作已证明我们内核不会串局的证据。

### 5.2 多局驻留的成本模型

定义P为宿主进程数，K为每进程驻留局数，N=P×K；P才提供独立的CPU执行lane。

```text
内存 ≈ 共享只读页
     + P × 宿主私有固定成本
     + N × 单局可变状态
     + 活跃模型/RNN状态
     + 有界队列/轨迹/学习器缓存
```

K增大主要减少相同N下的进程固定成本。若把48个执行lane变为12个lane×4局，CPU并行也变成12条；不能只算省内存，不算吞吐损失。先测P固定时K的增量，再测N固定时不同P/K分解。

### 5.3 共享文件，不等于共享解码后的堆

Linux安装器目前用`cp -a --reflink=auto`复制worker目录。reflink首先共享的是文件系统数据块，不能凭此宣称每个worker已共享全部驻留内存。[安装器](D:/Deepseek/CR-Native-Linux-Bionic/cr_native_bionic/runtime.py:305)、[GNU cp文档](https://www.gnu.org/s/coreutils/manual/html_node/cp-invocation.html)。

建议资源分层：同一内容hash的APK/so/assets通过同一只读文件或受控只读映射复用；cache、JIT输出、日志、PID和save data必须私有。不要整目录硬链接后让一局写入影响其它局。堆中的解码对象只有在同宿主确认为只读共享，或专门重构为不可变数据区时才能共享；ASLR指针、RNG、事件与场景状态不能直接共用。

同一已验证、确实不可变的资源blob可复用校验结果，减少逐slot反复hash的启动成本；可写文件、版本切换或身份变化必须重验。它属于冷启动/恢复优化，不算进稳态Tick收益。

## 6. 集中推理与ready调度的具体要求

### 6.1 先去掉每轮等待，不先改变PPO算法

旧collector虽然并发发送transition，仍在一轮内等各future再进入下一轮。不同worker的计算可更早进入ready队列，不必等最慢者；同一局的动作与观测仍严格有序。

保持R0的每5Tick节拍与段内固定行为版本。不改变游戏时间步、不改40游戏秒采集段、不让SGD期间偷偷采旧策略数据。按完成速度只取前N局并丢弃慢局会改变样本分布；分配明确的逐lane采集预算并保留逻辑episode边界。

依据：[同步轮屏障](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/online_collector.py:1039)、[transition等待](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/online_collector.py:1164)。

### 6.2 一个活跃权重一份推理实例，不是一局一模型

已有`remote_policy.py`可作为原型，不是成品：它已有微批，但默认队列无容量限制，使用pickle，并返回每次请求的pre-action hidden。应先补有界队列/连接超时/进程身份，再迁移张量数据面；同机Unix socket的文件权限不等于对任意pickle发送者安全。

建议：

- 根据实际需要的模型hash加载，CPU保留有限历史，GPU按活跃版本缓存；不能把对手库全部常驻GPU。
- RNN状态保留在推理服务；只有训练需要的序列锚点才成块取回，不在每次动作后往返整份状态。
- 不同权重、形状、采样模式分组；有限等待合批，不为凑满batch无限延迟。
- 默认先用直接PyTorch服务验证，不先引入整套Triton/TensorRT部署。若使用Triton，隐式有状态模型需要序列调度或外置状态管理，不能当普通无状态请求随意合批。[NVIDIA批处理说明](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)。

依据：[现有通信和状态回传](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/remote_policy.py:44)、[现有无界队列](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/remote_policy.py:251)。

### 6.3 异步化必须同步处理两个隐藏风险

**状态buffer拥有权：** 当前`_store_hidden()`保存的是batch张量切片的`detach()`视图。异步更新时，少数慢局可能让旧的整个batch storage继续存活。应使用预分配的逐slot状态银行，将新状态scatter进去；不能只把字典key改成多局ID。`detach`并不复制storage。[源码](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/batched_policy.py:690)、[PyTorch说明](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html)。这是异步重构风险，不是已经证明的历史OOM根因。

**随机流与样本归属：** 当前服务共用一个随机生成器，ready顺序变化会改变各局拿到的随机数。目标采用逐episode/阵营/决策/解码头的独立可重放随机流；原生一致性测试先用固定动作轨迹。否则只能讨论采样分布，不能宣称同seed逐动作一致。

RNN缓存key至少包含宿主启动代次、slot、episode generation、阵营和模型hash。重置与迟到结果必须互斥；更换模型后的hidden处理继续遵循R0合同，不宣称短burn-in能精确恢复新模型的整局记忆。

## 7. 数据面：先紧凑二进制，再决定共享内存

控制面保留可读JSON和低频状态查询；热路径采用有版本的定长头＋有界变长记录，复用缓冲。

一个请求/响应至少携带：runtime/schema版本、host epoch、slot/generation、request sequence、计划起始Tick、实际advanced Tick、终局/截断、观测有效性与完整性、记录数量及payload长度。RNN/策略版本在推理层另明确记录。模型输入不包含原生指针、私有敌手牌或RNG等不应公开的信息。

- 同一请求重复到达只能返回既有回执，不能重复下牌；进程重启后不能延续旧的“已执行”承诺。
- 队列满时回压，不覆盖未消费观测；重置让旧generation结果失效。
- 超出实体/事件容量时明确报溢出或使用已定义的大bucket，不截断单位来制造吞吐。
- 同机先比较紧凑Unix socket与共享内存。只有拷贝/系统调用确实显著，才上memfd/共享ring＋唤醒机制。
- 共享内存不等于GPU零拷贝。需要异步H2D时，buffer生命周期必须延续到相关拷贝完成；逐请求内存分配改为有界池。[memfd接口](https://man7.org/linux/man-pages/man2/memfd_create.2.html)。

FirstLight的另一项值得参考的细节是复用长期scratch缓冲、紧凑payload失败才分配完整JSON缓冲，而非每五Tick清零大块内存。[参考实现](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/probe/resident_multimatch.inc:1418)。

现有手牌`probe_grid`已经按side/deck index缓存，并非每帧都完整RPC扫描所有落点。优化应保留已有缓存，区分静态几何与动态圣水、手牌、塔毁、技能条件，不能全部缓存不失效。[当前缓存](D:/Deepseek/CR-Native-Core/expert_selfplay_v1/online_collector.py:394)。

## 8. 不作为首选的方案

| 方案 | 当前判断 |
| --- | --- |
| 同一libg直接多线程推进多局 | 无可重入证据；全局状态、事件和终局记录会形成隔离风险 |
| ART/CUDA预热后直接fork整机状态 | 多线程锁与运行时状态不安全，COW还会因写入退化；不作为默认 |
| glibc `dlmopen`装几十份Android libg | 本链使用Bionic linker，不是glibc的即插即用命名空间；glibc本身也有命名空间数量限制 |
| 立刻移除ART、手写全部JNI/平台服务 | 初始化和回调依赖尚未排除，工作范围远大于热路径优化 |
| 换更轻的完整Android模拟器 | 旧Linux链已无AVD/KVM，重点应是剩余用户态/桥接成本 |
| 默认KSM、THP、实时调度或强制满CPU | 无对应性能证据，不动系统级设置来冒充代码优化 |
| 多开CUDA进程＋MPS作为模型共享 | 不能替代显式模型副本/缓存预算；旧链已经有跨轮OOM记录 |
| 改大决策间隔、删观测、只留短局 | 改变训练问题，不属于同语义提速 |

fork约束见[Linux fork](https://man7.org/linux/man-pages/man2/fork.2.html)和[PyTorch多进程说明](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)；`dlmopen`的GNU属性及glibc命名空间限制见[加载器文档](https://man7.org/linux/man-pages/man3/dlopen.3.html)。

## 9. Linux运行与长期稳定性

- 按实际cgroup配额、有效cpuset/affinity及父级约束选资源，不能用宿主`os.cpu_count()`直接决定并发。
- Python编码/写盘线程池与ART/libg线程是不同对象；限制OMP/MKL/PyTorch线程不会自动减少所有native线程。用线程CPU时间定位，再限制确实存在的池。
- 只读资源按内容指纹安装一次，启动清单记录ABI、内核hash、JIT模式、协议版本和资源布局；进程重置与对局重置分开计时。
- 端口检查应读取当前Linux临时端口范围。主工程曾避开390xx冲突，Linux包仍以39031为默认；建议统一可配置端口或使用Unix socket，不直接修改宿主全局端口设置。
- 缓冲、轨迹、模型历史、RNN状态和日志全部有上限；坏局回收、进程崩溃与正常终局分开计数。
- 宿主需要回收时先停止补入新局并排空已有slot；失败影响该宿主的K局，不能伪造正常结束。恢复后换host epoch，废弃迟到回执。
- 旧Stage-2跨更新驻留曾OOM，当前逐版本CUDA进程隔离是已有保护。新方案先保留该保护，不能仅因“集中推理看上去省显存”就删除它。[旧实测与隔离策略](D:/Deepseek/CR-Native-Core/docs/STAGE2_EFFICIENCY_20260904.zh-CN.md:82)。

CPU/GPU使用率是诊断信息，验收目标是有效样本/秒、正常对局/秒、资源成本和稳定性。一直100%不是成功条件。

## 10. 对“一天20万场”的正确估算

200,000场/天约等于每秒2.315场。假设完整对局平均长度L秒，在20Hz、双方每250ms推理时：

```text
所需native Tick/s = 200000 / 86400 × L × 20
所需Actor rows/s  = 200000 / 86400 × L / 0.25 × 2
```

| 假设平均局长 | 需要native Tick/s | 需要Actor rows/s |
| ---: | ---: | ---: |
| 120秒 | 5,556 | 2,222 |
| 180秒 | 8,333 | 3,333 |
| 240秒 | 11,111 | 4,444 |
| 300秒 | 13,889 | 5,556 |

这只是需求算术，未包含编码、PPO、写盘和失败损失，不是产能承诺。对局长度、实体密度、特殊卡机制和政策质量会改变实际成本。

旧完整Stage-2记录为384场/506.10秒，折算约65,555场/天；到20万约需3.05倍同口径提升，但旧配置是600ms和旧177M。R0是250ms，且模型不同，不能直接把3.05当作新架构的剩余差距；更不能把纯采集的短测折算当完整训练吞吐。[完整更新历史](D:/Deepseek/CR-Native-Core/docs/STAGE2_EFFICIENCY_20260904.zh-CN.md:91)。

## 11. 实施拆分与验收

| 顺序 | 改动边界 | 验收与回退 |
| --- | --- | --- |
| P0 | 修CPU/PSS/队列计时，分离详细探针；统一运行清单 | 首样本非伪0；阶段口径不重叠；不改游戏输出 |
| P1a | Linux入口显式JIT对照、JSON单次编码、缓存已验证运行时句柄 | 相同版本/动作轨迹一致；预热与稳态分开；可退回interpreter/旧JSON |
| P1b | 每Tick终局读取优化、指针数组合并、融合native事务 | 少量代表性场景和重置/终局回归；不能把读取失败当空场或塔毁 |
| P2 | 有界ready调度、集中推理、RNN状态银行、按锚点回传 | 固定行为版本；无串局、无重复动作、无隐藏batch引用堆积；可回退同步轮 |
| P3 | 紧凑二进制/缓冲池，必要时共享内存 | JSON与二进制解码成同一训练输入；容量/代次/生命周期测试；保留诊断回退 |
| P4 | MatchContext拆分，K=2然后4 | 单独/交错执行一致；跨局重置隔离；内存与CPU收益分别测；可退K=1 |

P2与P3部分可交错，但不能在接口与状态隔离尚不清楚时同时改多局、模型、PPO和时机。源码落点分别是`run_native_active_worker_sweep.py`、Linux `runtime.py/cli.py`、`JniHost.java/jni_bridge.cpp`、`online_collector.py/remote_policy.py/batched_policy.py`；R0通过适配层消费同一明确合同，不直接继承旧协议的缺字段假设。

### 有界测试安排

1. 离线协议/调度测试：长度与版本错误、重复请求、迟到结果、重置代次、队列满、RNN序列、模型版本切换，不需要云算力。
2. 原生短回归：普通攻防、密集召唤、法术/建筑、技能/塔兵、塔毁、加时终局与重置。只做一次性回归及抽样哨兵，不给每场生产对局新增昂贵离线对账。
3. 固定一套版本、动作轨迹和决策间隔做交错A/B。旧16Tick仅用于历史复现；R0性能基线统一5Tick。先少数进程，再逐档增加。
4. 驻留多局分别比较固定P、固定N两组；微批候选可从16/32/64/128行、0/0.5/1/2ms等待开始，全部是待测取值。
5. 少数入选配置再测30–60分钟，报告时间序列、p95/p99、CPU节流、PSS/VRAM斜率、正常终局、失败/拒绝、重置耗时和写盘压力；最终再纳入PPO完整更新。

性能提升应超过重复测量的噪声，并且不靠减少字段、增加截断、改变采样或缩短游戏工作量获得。有限短测不能发布为24小时稳定产能。

**下一步最小实施包：P0＋P1a，再做P1b。** 本报告没有执行这些改动。后续上云须重新确认当前连接与有界开机安排，不能复用之前已结束的压测授权。
