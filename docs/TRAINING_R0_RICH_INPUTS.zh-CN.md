# R0 rich-v2：丰富输入与分层网络

> 历史实现记录：默认模型已按用户后续要求升级为 [elite-v3](TRAINING_R0_ELITE_V3.zh-CN.md)。下列 rich-v2 参数量、字段数量及默认行为不再代表当前代码。

日期：2026-09-12。用户明确要求向 FirstLight 的丰富输入结构靠拢，并允许尝试增加结构。本次已修改 `training_r0` 默认模型，保留模型/方法来源；不是原作者权重兼容移植，也不是训练效果复现。

默认模型 **11,101,962 参数**。已通过 CPU 前向、联合动作概率重算、反向传播、一次优化器更新与长序列分段检查。**正式训练、真实丰富字段采集和对战效果评估仍未完成。**

## 已改变的信息流

```text
版本绑定的卡牌标称属性 + 已知位 + 可选机制操作集合
  → 共享卡牌语义编码
  → 己方卡组/手牌、公开敌牌、单位、候选、历史和效果来源

单位基本特征 + 动态属性/已知位 + 几何 + 活跃效果
  → 子实体表示
  → 4 个局部注意力池化槽 + 组统计/组种类 → 战斗组 token

融合摘要 + 己方卡牌 token + 公开敌牌 token + 塔 token + 战斗组 token
  → 4 层、256 维、8 头注意力
     8 项空间关系 + 显式关系边偏置

子实体 → 32 通道散射（点：双线性；范围：圆/矩形覆盖）
  + 14 显式空间通道 + 3 坐标/河流侧通道
  → 49 通道卷积入口 → 3 个残差块 → 64 通道空间图
  → 平均与最大池化 → 128 维空间摘要

场景 256 + 空间/标量/事件/命令/候选各 128 = 896
  → 输入 512 / hidden 512 的 LSTM
  → 分开的策略/价值捷径 → 512 维上下文
  → 候选、FiLM 位置、时间偏移、第二动作；共享主干 value
```

事件与候选现在会查询全局 token；技能候选可引用其真实源实体表示。原先的真实事件与监督标签分离、合法动作约束、每 5 Tick 决策和两微动作时序均保留。

## 输入结构与实际数据覆盖

| 输入 | 当前结构 | 数据状态 |
| --- | --- | --- |
| 卡牌标称属性 | 24 项数值，每项带 known 位 | 默认加载现有卡牌目录和版本绑定的标称战斗表；包括已知费用、攻击目标类型、射程、间隔等 |
| 静态机制操作 | 效果种类、量值/时长/半径 known 位、产生的卡牌身份 | 编码器与可加载目录已实现；默认数据没有完整机制操作集合，不能声称自动恢复了所有机制 |
| 实体动态信息 | 16 项属性与 known 位 | 支持速度、护盾、攻击/部署计时、投射物伤害、可被选中、飞行、塔激活等；原生扩展适配器必须逐项提供公开已知值 |
| 活跃效果 | 类型、公开可见时刻、来源卡、量值/剩余时长/层数及 known 位 | 已接入实体编码并验证非零梯度；真实采集补充仍待实现 |
| 实体分组 | 明确 group ID 与类型；组内局部池化与统计 | 不知道因果分组时用单实体组，绝不按相同卡名臆造一次召唤 |
| 关系 | 14 种本地关系词表含 none，支持攻击、召唤、承载、效果来源等 | 通过公开实体引用传入，类型和端点进入网络；原生地址数值不作为学习特征 |
| 上一动作 / 未决命令 | 独立掩码、身份、目标、费用、时间、known 位 | 已接入独立编码和循环输入；必须由当前玩家命令/回执记录提供 |
| 敌方已揭示卡牌 | 独立 token 集合 | 来自公开实体、已确认历史，或公开追踪器补充；不读取敌手牌 |

`data/nominal_combat.json` 从已有本地 `cr_nominal_static_combat_v1` 表按已知位转换，保留源 SHA 和限制说明，绑定 x86_64 `150535029` 的 libg SHA。它不是模型权重或另一版本数据。原表 180 个卡牌条目中 138 个含已知属性；转换不会用未知 0 冒充已知 0，也不会臆造 native speed 到实际移速的换算。

当前默认 210 个卡牌/形态 profile 中 168 个至少有一项已知属性，其中 138 个接入上述标称战斗属性。默认机制操作集合仍为 0，须从已验证资源继续补充；相关分支已用显式非空操作集合测试，不能把支持操作编码说成机制数据已经完整。

范围网格的 14 个显式通道使用本地字段定义，顺序见 `semantics.PLANE_NAMES`；通道数量与参考一致，不宣称逐通道语义完全等价。圆/矩形边界采用本地明确定义并有测试，仍需原生几何标定。未知几何保留 unknown 标记，并以点散射提供位置表示。

## 接口使用

原有入口继续可用；默认已是丰富模型：

```python
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig
from training_r0.model import R0Policy
from training_r0.observation import ObservationBuilder

vocabulary = CardVocabulary.from_native()
config = ModelConfig()
builder = ObservationBuilder(vocabulary, config)
model = R0Policy(vocabulary, config)
```

原生补充通过 `ObservationBuilder.from_native(..., semantics=PublicSemantics(...))` 输入。

- `PublicSemantics` 绑定 episode、actor side 和当前 Tick，坐标均按同一玩家规范视角提供。
- `EntityDetail` 的 key 必须引用当前公开实体；塔 key 使用 `actor_projection` 后的规范塔 key。半径/半宽使用格单位，命令目标使用规范化坐标。
- 只允许 `DYNAMIC_SCALES` 白名单中的属性；每项缺失均有独立 unknown 语义。
- 效果、关系、命令的 observed Tick 不能晚于当前 Tick；过时帧、错误阵营、未知端点和超容量均拒绝。
- 上一动作必须已经确认，pending 仅记录己方已提交但尚未确认的命令；同一 command ID 不能同时出现两处。
- 补充接口不会验证游戏事实本身；生产适配器负责公共可见性与字段依据。它不是把任意 debug JSON 当作安全公开输入的通道。

静态机制扩展可用 `SemanticCatalog.read(path)` 加载 `r0-public-semantic.v2` 目录，再将同一目录传给 builder 和 model 的 `semantic_catalog` 参数。所有特征名、词表、归一化、目录来源与内容都进入合同哈希。

## 对照开关

`ModelConfig` 支持：

- `use_mechanics`：卡牌静态属性与操作语义分支。
- `use_effects`：活跃效果聚合分支。
- `use_groups`：因果组聚合；关闭时变成单实体组。
- `use_relation_edges`：显式关系边偏置。
- `use_command_memory`：上一动作与未决命令分支。

默认全部开启。开关用于分别训练/评估的消融配置，每种配置有独立模型合同。参数更多本身不构成优势结论。

## 版本与权重

观察版本：`r0-public-observation.rich-v2`；模型版本：`r0-rich-relation-spatial-lstm.v2`。动作协议保留现有两微动作合同。

旧 8.232M R0 参数形状已不兼容；模型权重携带合同字节，加载不匹配合同会报错，包括误用 `strict=False`。该检查不等于优化器/云训练全套恢复。没有改写现有旧权重。

## 验证

```powershell
python -m pytest tests/test_training_r0.py tests/test_training_r0_rich.py tests/test_expert_selfplay_native_observation.py tests/test_expert_selfplay_contracts.py -q
python -m training_r0.smoke --profile base --steps 4 --rich
python -m training_r0.smoke --profile tiny --steps 160 --rich
```

本轮 74 项通过。测试覆盖各新分支确实改变上下文、关闭分支后的不变性、非空效果/多成员组/关系/命令的梯度、邻格分配与范围覆盖、补位不污染、敌方私有字段不泄漏、未来事件和错帧拒绝、联合动作概率一致，以及新权重保存/恢复和旧合同拒绝。

基础模型完成一次有效更新；tiny 长序列完成 320 个双侧有效决策、48/48/48/16 四片段，累计后仅一次优化器更新。全部使用合成观察与奖励，不是对战强度或吞吐结论。

## 后续接线

本次完成网络与输入协议升级。接下来需要为丰富语义编写真实公开字段适配器、补全机制目录、更新回放缓存编译器，并完成原生动作执行与完整训练循环。现有 compact 观察没有的字段不能被网络结构凭空恢复。接入覆盖与模型训练效果应分别统计。
