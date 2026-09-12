# R0 第一批代码：离线策略内核与接口

日期：2026-09-07。模块：`training_r0/`。状态：**可运行的CPU离线内核；不是已经接通的正式训练系统**。

该目录是按[当前R0设计](TRAINING_CHAIN_REDESIGN_20260907.zh-CN.md)新写的本地适配代码，不依赖不完整FirstLight源码切片，也不加载旧177M权重。未复制ARM64地址/私有资源，未改旧训练入口。

## 已实现

| 文件 | 职责 |
| --- | --- |
| `config.py` | 固定20Hz/5Tick决策、250ms内5个偏移、模型尺寸、40秒采集与48步BPTT配置 |
| `catalog.py` | 复用本地版本化卡牌表；PAD与UNKNOWN分开，词表有hash |
| `history.py` | 真实确认事件、可见时刻、命令去重、UNKNOWN动作与标签有效位；不从监督占位符伪造WAIT |
| `observation.py` | 接收coherent observe_train_v1；复用公开投影；当前手牌/技能来源校验、未知实体保留、动态padding但不静默截断 |
| `model.py` | 关系注意力、空间scatter/CNN、LSTM、当前局面skip、候选指针、FiLM位置、联合两微动作解码与共享value |
| `actions.py` | 0～2微动作数据包、非递减offset、圣水/重复候选/技能/建筑shadow约束、绑定对局/阵营/Tick的决策信封 |
| `session.py` | 一份冻结推理快照服务多局；RNN按episode+side隔离，限额与释放，拒绝重复/跳过的推理Tick |
| `rollout.py` | current-current双侧、冻结角色单侧、单episode连续前缀、真实h0与版本、有效mask、terminal/truncation及bootstrap检查 |
| `learning.py` | 一个recurrent PPO minibatch；跨时间chunk保持权重、carry/detach hidden，最后一步optimizer；gate-only部分IL监督 |
| `rewards.py` | 明确的本地胜负＋对称塔血势函数控制；不发等待/出牌/溢出额外奖励 |
| `synthetic.py` / `smoke.py` | 不联网、不起原生环境的合成数据和CPU smoke入口 |

基础模型使用256宽、4层、8头、512维LSTM、64空间通道。本次实测**8,232,490个参数，包含共享value头**；词表212个条目，包含PAD/UNKNOWN及本地卡牌/形态ID。它不是原作者模型的精确参数量，也还没有补齐原作者所有静态机制/效果编码器。

## 关键接口约束

### 观察与候选

`ObservationBuilder.from_native` 只接受明确的 `libg_native_train_state_v1` coherent帧，不会在缺实体字段时自动使用空场。声明的实体数必须与实际列表一致。

复用 `actor_projection`，不把敌方手牌/精确圣水/私有技能冷却送入模型。只要敌方这些私有字段改变而公开状态不变，模型上下文必须保持一致。

候选必须由调用方显式提供：真实手牌槽、技能源实体、动态费用、合法位置mask及形态/几何known信息。当前模块**尚未实现真实native候选生产器**，不会用“全地图合法”当默认。合成fixture的mask只用于测试。

Mirror要求first-only约束；第二条动作不能超费、重复候选/手牌/排斥组，不能重复技能。建筑几何未知时对第二建筑采取保守屏蔽并报告，不把未知半径当0；该shadow仍不等于全卡原生合法性认证。

### 真实历史与标签

`ConfirmedEvent` 区分事件发生Tick和该信息变得可见的Tick。身份未知但确认发生的技能仍是ABILITY/UNKNOWN事件，不是WAIT；当前观察不接收未来才可见的事件。

事件记录与 `LabelValidity` 不相互覆盖。第一批只实现gate-only部分监督；完整候选/位置的部分标签边际化、真实回放编译和IL训练循环仍待接入。不应伪造条件候选后将未知位置当真值训练。

### 动作与执行

采样、按UID重放和PPO重算使用同一联合概率及温度。温度默认1/1/1；贪心和采样是明确不同模式。

`PolicyOutput.entropy` 是沿采样/记录条件路径的熵正则估计，与参考做法一致，不宣称计算了整棵自回归动作树的精确熵。

`command_plan` 只生成高层计划，**没有RPC执行函数**。它要求决策的episode/side/Tick与当前帧匹配，拒绝把旧输出挪到新Tick执行；运行时hash、观察schema和提交时钟原点必须显式匹配证书。同offset或目标型技能没有相应证书时拒绝，不偷偷移动第二条动作时间。

证书是接口数据类型，不代表本轮已经测得证书。原生接线还需要验证命令UID、实际hand-slot→native命令映射、同offset顺序、落地延迟及回执。

### 记忆与训练

`FrozenPolicySession` 复制一次已选模型作为不可训练快照，避免learner原地更新行为模型；不是每个worker复制模型。每个episode+side有独立状态，正常结束须release；暂不提供跨版本热切换/记忆迁移。

`RolloutSegment` 只含learner lane。current-current允许两侧，冻结对手那侧不能混入。新比赛必须新逻辑lane；padding不能充当终局，非terminal必须有可用bootstrap。

`update_minibatch` 完成全部时间chunk后才调用一次optimizer.step，按实际有效帧归一。KL超限时不更新；非有限梯度停止。完整运行的更新前checkpoint、失败回滚、发布和跨GPU/DDP尚需外层runner，不能把该函数误当作已经具有全套恢复能力。

## 运行离线检查

在主工程目录、已具备torch/numpy/pytest的环境中：

```powershell
python -m training_r0.smoke --profile base --steps 4
python -m training_r0.smoke --profile tiny --steps 160
python -m pytest -q tests/test_training_r0.py tests/test_expert_selfplay_native_observation.py tests/test_expert_selfplay_contracts.py
```

本机测试使用已有 `D:\AI_data\runtime\venv\Scripts\python.exe`，没有安装/升级训练依赖。当前包发现规则 `training*` 会包含 `training_r0`；上述命令在仓库根目录直接可用。

本轮回归包括：25项新测试与35项相关旧测试，共60项通过。基础模型CPU smoke完成前向、采样、联合logp重算与一次梯度更新；长序列tiny smoke用于验证160决策/48步BPTT调度。两者都只用合成观察和合成奖励，**不是IL效果、原生动作成功率或实际对战能力的证明**。

## 下一步尚未完成

1. 接真实native候选/mask提供器、命令计划执行与回执，取得同offset等证书。
2. 将静态卡牌机制、公开事件和己方未决命令等补入统一schema，重新冻结版本，禁止静默加字段。
3. 接回放编译器、可信前缀及全IL损失与持久化数据索引。
4. 接40秒真实采集段、短局新lane、checkpoint/RNG/采样器恢复与训练快照管理。
5. 最后做真实GPU吞吐、对手池、独立评估与长跑；此时才讨论正式一键训练。

本次未调用云主机、未运行原生libg、未读取/重处理旧训练集、未启动正式训练或修改旧模型。
