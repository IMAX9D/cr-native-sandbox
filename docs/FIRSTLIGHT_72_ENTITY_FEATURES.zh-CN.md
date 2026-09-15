# FirstLight V4：72 项子实体特征详解

依据本地保存的 FirstLight_CR 快照 `28d66cc0a5d65888515e22fdf22f11d783b65efb`。字段顺序来自 `CHILD_FEATURE_NAMES`，计算方式逐项对照 `tensorizer.py` 的 `dynamic[0]` 至 `dynamic[71]`。

这里的“子实体”不只包括兵，也可能包括建筑、投射物或范围对象。并非每种实体都适用全部字段。下表序号从 1 开始，源码数组下标为序号减 1。

编码说明：

- **状态位**：通常用 0/1 表示条件是否成立。
- **已知位**：1 表示相应信息可用；0 不等于相应数值真实为零。
- **除以常数**：主要调整数值尺度，不意味着结果一定限制在 0～1。例如年龄超过60秒时，`age_over_60000ms` 可以超过1。
- 参考实现并非给每个字段都配了独立 known 位。部分缺失字段保留初始化的0；阅读或复现时不能把所有0都解释成已知事实。
- 以下均是当前观察时刻可提供的状态，不包含未来伤害或未来比赛结果。

## 1—16：生命、位置、基础攻击与投射物

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 1 | `hitpoints_ratio` | 当前血量比例 | 当前血量÷最大血量。例如750/1000=0.75。 |
| 2 | `hitpoints_over_10000` | 当前绝对血量 | 当前血量÷10000。区分同为半血、但剩余血量不同的单位。 |
| 3 | `max_hitpoints_over_10000` | 最大血量 | 最大血量÷10000。与第2项配合表示生命规模。 |
| 4 | `shield_over_10000` | 当前护盾量 | 护盾÷10000；与普通生命值分开。 |
| 5 | `age_over_60000ms` | 单位存在时间 | 单位年龄毫秒÷60000，即年龄÷60秒。不是预计剩余寿命。 |
| 6 | `has_visible_target` | 是否有可见目标 | 状态位。只表示目标存在，目标身份/关系还可通过其他输入表达。 |
| 7 | `is_ability_source` | 是否为当前技能候选的来源单位 | 状态位。不是简单判断“这张卡理论上有没有技能”。 |
| 8 | `x_over_board_width` | 横向位置 | 转到模型视角后的格坐标x÷18。 |
| 9 | `y_over_board_height` | 纵向位置 | 转到模型视角后的格坐标y÷32。 |
| 10 | `velocity_x_over_10` | 横向移动速度 | 模型视角下、换算为格单位的横向速度÷10；方向可体现为正负。 |
| 11 | `velocity_y_over_10` | 纵向移动速度 | 模型视角下、换算为格单位的纵向速度÷10；方向可体现为正负。 |
| 12 | `attack_cooldown_remaining_over_5000ms` | 攻击冷却剩余时间 | 剩余毫秒÷5000。帮助区分刚攻击过与即将再次攻击。 |
| 13 | `attack_phase_remaining_over_5000ms` | 当前攻击阶段剩余时间 | 剩余毫秒÷5000。结合攻击阶段状态位解释。 |
| 14 | `projectile_damage_over_5000` | 投射物伤害信息 | 伤害÷5000。优先来自当前投射物；缺失时可使用已知原型静态伤害依据。**不是历史累计伤害。** |
| 15 | `projectile_radius_over_5_tiles` | 投射物半径 | 格单位半径÷5。可来自投射物状态或已知原型半径。 |
| 16 | `projectile_homing` | 是否为追踪投射物 | 状态位。表示投射物的追踪属性。 |

## 17—30：攻击阶段、蓄力、部署与冲锋

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 17 | `attack_windup` | 是否处于攻击前摇 | 状态位。攻击已经开始准备，但未必已经造成伤害。 |
| 18 | `attack_release_or_channel` | 是否正在释放或持续引导攻击 | 状态位。源码把 RELEASE 与 CHANNEL 合并到这一项。 |
| 19 | `attack_cooldown` | 是否处于攻击冷却阶段 | 状态位。与第12项的“还剩多久”配合。 |
| 20 | `attack_charging` | 是否处于蓄力阶段 | 状态位，来自攻击阶段 CHARGING。 |
| 21 | `attack_interrupted` | 攻击是否被打断 | 状态位。攻击阶段为 INTERRUPTED，或中断标记为真时置1。 |
| 22 | `attack_sequence_index_over_10` | 攻击序列编号 | 序列编号÷10。描述当前序列位置，不是单位生涯总攻击次数。 |
| 23 | `attack_charge_stage_over_10` | 蓄力阶段编号 | 阶段编号÷10。用于多阶段蓄力机制。 |
| 24 | `attack_charge_elapsed_over_5000ms` | 已经蓄力的时间 | 已过毫秒÷5000；与“剩余时间”是不同量。 |
| 25 | `attack_damage_multiplier_over_5` | 当前攻击伤害倍率 | 倍率÷5。例如倍率1编码为0.2，不能把编码值直接当伤害倍率。 |
| 26 | `attack_locked` | 攻击锁定状态 | 状态位，直接来自 `attack.locked`；不额外推断为“绝对不会换目标”。 |
| 27 | `deployment_in_progress` | 是否仍在部署过程 | 状态位，部署阶段为 `deploying` 时置1。 |
| 28 | `deployment_remaining_over_5000ms` | 部署剩余时间 | 原生部署剩余毫秒÷5000。不是模型计划的下牌偏移。 |
| 29 | `movement_effective_speed_over_1000` | 当前有效移动速度 | 原生 `effective_speed`÷1000；应沿用源字段单位，不自行解释为格/秒。 |
| 30 | `classic_charge_ready` | 经典冲锋机制是否准备好 | 状态位，`classic_charge_phase == ready`。与普通攻击蓄力分开。 |

## 31—45：可见性、可攻击性与投射物后续状态

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 31 | `visibility_hidden_or_burrowed` | 是否处于隐藏或潜地状态 | 状态位；HIDDEN 与 BURROWED 合并。字段来自投影后的可用状态，不代表允许偷看不可见敌情。 |
| 32 | `visibility_transition_remaining_over_5000ms` | 可见性切换剩余时间 | 剩余毫秒÷5000。 |
| 33 | `projectile_in_flight` | 投射物是否仍在飞行 | 状态位，投射物阶段为 IN_FLIGHT。 |
| 34 | `projectile_damage_known` | 投射物伤害是否已知 | 已知位，对应第14项；已知的原型静态伤害也可使其为1。 |
| 35 | `projectile_radius_known` | 投射物半径是否已知 | 已知位，对应第15项。 |
| 36 | `classic_charge_progress_over_10000` | 经典冲锋机制进度 | 原生 `classic_charge_progress`÷10000；不直接等同于已跑距离。 |
| 37 | `visibility_targetable` | 当前玩家能否把它选为攻击目标 | 状态位，读取 `targetable_by_owner` 中当前玩家对应的值。 |
| 38 | `visibility_targetable_known` | 可被选中状态是否已知 | 已知位。第37项为0且本项为0时，不能断言目标不可选中。 |
| 39 | `visibility_area_damage_eligible` | 是否可受到范围伤害 | 状态位。与能否被选为直接攻击目标是两回事。 |
| 40 | `visibility_area_damage_eligible_known` | 范围伤害资格是否已知 | 已知位，对应第39项。 |
| 41 | `projectile_target_x_over_board_width` | 投射物目标横向位置 | 目标格坐标x÷18，已转到模型视角。 |
| 42 | `projectile_target_y_over_board_height` | 投射物目标纵向位置 | 目标格坐标y÷32，已转到模型视角。 |
| 43 | `projectile_target_position_known` | 投射物目标位置是否已知 | 已知位，对应第41、42项。 |
| 44 | `projectile_drag_back_active` | 是否处于投射物拖回阶段 | 状态位，拖拽阶段为 DRAG_BACK_ACTIVE。不要与普通追踪飞行混为一谈。 |
| 45 | `projectile_drag_stage_known` | 投射物拖拽阶段是否已知 | 已知位。表示该阶段字段有值，不表示一定正在拖回。 |

## 46—56：额外生成、抓取与拖拽机制

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 46 | `extra_spawn_accumulator_ratio` | 额外生成机制的累积进度 | 直接使用对应资源状态的 `normalized` 值；不是已生成单位数量。 |
| 47 | `extra_spawn_accumulator_known` | 额外生成进度是否已知 | 已知位，找到 `extra_spawn_accumulator` 资源状态时置1。 |
| 48 | `capture_runtime_known` | 抓取机制运行状态是否存在/可用 | 已知位。存在抓取运行对象时置1，不等于当下正抓着目标。 |
| 49 | `capture_acquired_delay` | 是否有目标处于获取后的等待阶段 | 状态位。遍历当前抓取目标，只要存在 ACQUIRED_DELAY 阶段就置1。 |
| 50 | `capture_grab_pause` | 是否有目标处于抓取暂停阶段 | 状态位，存在 GRAB_PAUSE 目标时置1。 |
| 51 | `capture_dragging` | 是否有目标正在被拖拽 | 状态位，存在 DRAGGING 目标时置1。 |
| 52 | `capture_contained` | 是否有目标处于被容纳/收纳状态 | 状态位，存在 CONTAINED 目标时置1。具体游戏表现依机制而定。 |
| 53 | `capture_release_pending` | 是否有目标等待释放 | 状态位，存在 RELEASE_PENDING 目标时置1。 |
| 54 | `capture_phase_budget_remaining_over_5000ms` | 抓取目标当前阶段的最小剩余时间预算 | 收集各目标已知的剩余预算，取最小值，再÷5000。不是把各目标时间相加。 |
| 55 | `capture_action_cycle_progress` | 抓取动作循环进度 | `hit_accumulator_ms / hit_frequency_ms`，源码上限截为1。 |
| 56 | `capture_cooldown_remaining_ratio` | 抓取机制剩余冷却比例 | 冷却剩余时间÷该机制配置的冷却时间；配置冷却须大于0。 |

49—53 不是整行只能选一个的分类：一个实体可能同时处理多个目标，因此多个阶段标记可能同时为1。

## 57—64：按阈值触发的位置转移

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 57 | `threshold_relocation_runtime_known` | 阈值转移机制状态是否可用 | 已知位，存在对应运行对象时置1。 |
| 58 | `relocation_waiting_threshold` | 是否等待达到转移阈值 | 状态位，阶段为 WAITING_THRESHOLD。 |
| 59 | `relocation_active` | 是否正在转移 | 状态位，阶段为 RELOCATING。 |
| 60 | `relocation_exhausted` | 转移机制是否已耗尽 | 状态位，阶段为 EXHAUSTED。 |
| 61 | `relocation_burrowed` | 转移机制中的潜地标记 | 状态位，直接取该机制的 `burrowed`。与第31项的一般可见性输入来源不同。 |
| 62 | `relocation_remaining_ratio` | 当前隐藏/转移阶段剩余时间比例 | `remaining_ms / hide_duration_ms`。不是剩余血量比例。 |
| 63 | `relocation_index_ratio` | 已到达的转移索引比例 | `relocation_index / threshold_count`；不是剩余次数比例。 |
| 64 | `relocation_threshold_percent` | 当前索引对应的触发阈值 | `thresholds_percent[relocation_index] / 100`，只在索引未越过阈值列表时读取。 |

## 65—72：连续攻击进度与周期性攻击修正

| 序号 | 原始字段 | 中文含义 | 编码与理解 |
| --- | --- | --- | --- |
| 65 | `attack_sequence_progress_ratio` | 攻击序列累积进度 | `sequence_progress / sequence_progress_limit`。 |
| 66 | `attack_sequence_progress_known` | 攻击序列进度是否已知 | 已知位，对应第65项。 |
| 67 | `attack_sequence_decay_remaining_ratio` | 攻击序列进度衰减的剩余时间比例 | `sequence_decay_remaining_ms / sequence_decay_duration_ms`。 |
| 68 | `attack_sequence_decay_known` | 攻击序列衰减计时是否已知 | 已知位，对应第67项。 |
| 69 | `periodic_attack_modifier_present` | 是否存在周期性攻击修正对象 | 状态位。表示这一机制存在，不等于当前一击一定带特殊效果。 |
| 70 | `periodic_attack_modifier_progress_ratio` | 周期性攻击修正的循环进度 | `completed_attacks / period_attacks`，例如每若干次攻击产生变化的进度。 |
| 71 | `periodic_attack_modifier_source_death_linger` | 是否处于效果来源死亡后的延续阶段 | 状态位，阶段为 SOURCE_DEATH_LINGER。指来源死亡后的延续，不一定是当前实体死亡。 |
| 72 | `periodic_attack_modifier_linger_remaining_ratio` | 延续效果剩余时间比例 | `linger_remaining_ms / linger_duration_ms`。 |

## 原始72项之外，已接入的生命周期伤害分支

以下是独立扩展，不算在 FirstLight 的原始72项中：

1. 本单位累计已确认的有效对单位伤害。
2. 本单位累计已确认的有效对塔伤害。
3. 最近1秒的对单位/对塔伤害。
4. 最近3秒的对单位/对塔伤害。

按用户最新要求，删除“自上次观察以来新确认的伤害”和新增的“年龄/完整性特征”。**现在仅有6项新增伤害数值，分支宽度就是6；原72项保持原样。** 不添加累计承伤，也不把数据质量判断送进网络。

生命周期账本与输入分支已经实现并完成离线测试，真实伤害回执采集仍需接入。投射物/效果的伤害按明确父链归属攻击单位，正常召唤单位保留独立生命周期；部署总计按原始事件去重。数据缺失或采集缺口只在模型外用于校验及样本准入，不是额外输入特征。

所有数值只能统计当前观察时刻已经确认的事件。全生命周期最终伤害可以成为事后训练标签，但不能进入部署当时的输入。投射物、召唤物与分裂单位应保留部署来源关联；部署合计不能把已归属的同一伤害事件重复累计。

## 源码依据

- [字段定义](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/training/v4/config.py:85)
- [逐项编码](D:/Deepseek/outputs/firstlight-cr-review-20260907/sources/native_runner/training/v4/tensorizer.py:1082)
- [本地保存的完整字段清单](D:/Deepseek/CR-Native-Core/training_r0/data/reference_features.json)
