# 原生全卡 Runtime、形态与技能接口

> 本文是
> [`SANDBOX_RUNTIME_TECHNICAL.zh-CN.md`](SANDBOX_RUNTIME_TECHNICAL.zh-CN.md)
> 的全卡/形态/技能附录。

## 1. 结论

`CR-Native-Core` 的原生宿主已经不再受固定八卡限制。当前冻结版本
`15.535.29` 的结果是：

| 范围 | 数量 | 原生验收 |
| --- | ---: | --- |
| 当前可见标准 1v1 基础卡 | 122 | 122/122 部署或施法成功 |
| 映射到当前 DataTables 的卡 | 152 | 已建目录；其中 30 张为隐藏/停用内容，不进入标准卡池 |
| 觉醒形态 | 41 | 41/41 经真实循环解析到 category 13 形态 |
| 英雄形态 | 16 | 16/16 解析到 category 203 形态 |
| 基础卡主动技能形态 | 8 | 通用原生技能命令已用弓箭女皇闭环 |
| 英雄主动技能形态 | 16 | 通用原生技能命令已用狂战士英雄闭环 |

基础卡证书：
`D:\AI_data\cr-native-core\full-card-acceptance.json`。

形态/技能证书：
`D:\AI_data\cr-native-core\card-form-acceptance.json`。

这表示全卡 Runtime 输入、原生形态解析和主动技能按钮已经打通。覆盖范围只
针对本文冻结的游戏版本和标准 1v1 Runtime。

## 2. 全卡目录

生成器：`scripts/build_live_card_catalog.py`。

生成物：`native_core/data/live_card_catalog.json`。

目录来自当前版本的本地逆向结果：

- `live_card_map.json`：基础卡 ID；
- `sandbox_data.json`：基础卡、单位与建筑；
- `sandbox_data_extra.json`：觉醒/英雄 Spell 表；
- `decoded_csv/characters/**/*.toml`：基础主动技能和英雄主动技能参数。

目录对每张基础卡保存：基础 ID、类型、费用、标准卡池状态、召唤对象、觉醒
形态 ID、英雄形态 ID、循环数、主动技能名、技能费用/次数/施法时间等。

## 3. 牌组形态编码

原生回放 `battle.deck0.sp` / `deck1.sp` 的每项至少包含：

```json
{"d": 26000000, "l": 10}
```

- `d`：基础卡 ID；
- `l`：零基等级，Python API 的 1..16 写成 0..15；
- `el`：原生形态位掩码。

`el` 的静态调用链来自 `BuildCanonicalSelection (0x1048170)`，当前编码为：

| Python form | `el` | 含义 |
| --- | ---: | --- |
| `base` | 省略/0 | 基础形态 |
| `evolution` | 1 | 开启觉醒循环 |
| `hero` | 2 | 开启英雄形态 |
| `both` | 3 | 同一基础卡同时具备两类能力（仅目录允许时） |

命令行示例：

```powershell
python scripts/build_native_replay.py `
  --deck0 "Knight@evolution,Berserker@hero,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon" `
  --deck1 "Knight,Archer,Giant,Skeletons,Musketeer,HogRider,Cannon,Arrows" `
  --output D:\AI_data\cr-native-core\full-form-replay.json
```

Python 也可直接使用 `native_core.decks.build_replay()`。

## 4. 三类容易混淆的“技能”

### 4.1 主动技能按钮

冠军和当前英雄形态均使用同一个原生命令族：

- command type：`0x5A`（90）；
- constructor：`0xD8F360`；
- execute：`0xD8F3C0`。

执行链会经过原生玩家身份、存活实体、技能槽、目标/状态、圣水、次数、冷却
检查，成功后由原生逻辑扣费并触发角色技能。接口没有直接改内存状态，也没有
绕过 `libg` 的合法性判断。

当前基础主动技能卡共 8 张：Mighty Miner、Skeleton King、Archer Queen、
Golden Knight、Monk、Little Prince、Giant Buffer、Boss Bandit。

当前英雄形态共 16 个：Knight、Goblins、Giant、Balloon、Valkyrie、
Musketeer、Wizard、Mini P.E.K.K.A、Dark Prince、Bowler、Ice Golem、
Mega Minion、Elite Archer、Berserker、Tombstone、Barbarian Barrel。

### 4.2 觉醒

觉醒不是“按技能按钮”。牌组通过 `el & 1` 开启后，`libg` 按原生循环计数
决定本次出牌是基础 ID（category 26/27/28）还是觉醒 ID（category 13）。

验收中的骑士真实序列为：

```text
26000000 -> 26000000 -> 13000000
基础骑士     基础骑士     觉醒骑士
```

因此外部动作接口不应给觉醒额外增加 `ABILITY` 按钮；调用方只需读取当前
手牌的觉醒状态，并正常执行 `PLAY(card, position)`。

### 4.3 精英等级/“精英化”

当前 `card_forms` 数据只有 `BasicForm / EvoForm / HeroForm`，没有独立的
`EliteForm`，也没有第三个已证实的主动技能位。

如果“精英”指卡牌等级，则它只由 `l` 控制，不是主动技能。如果产品/UI 中的
“精英技能”实际指觉醒能力，应走上一节的 `evolution` 循环，而不是主动按钮。
任何未来数据中真正带 `AbilityData` 的实体，都可复用通用 `0x5A` 动作接口，
但不能在没有当前版本证据时虚构一个额外命令。

## 5. Python 动作接口

### 5.1 单独按技能

```python
state = env.observe()
hero = next(
    entity for entity in state["entities"]
    if entity["side"] == 0 and entity["ability_slot"] > 0
)
result = env.use_ability(side=0, entity_id=hero["entity_id"])
```

`entity_id` 是公开的 5,000,000 系列 generation key；API 不泄漏、也不接受
进程原始指针。

### 5.2 同 Tick 联合动作

原有联合动作兼容两种动作：

```python
env.joint_training_transition([
    {"type": "ability", "side": 0, "entity_id": 5000011},
    {"type": "play", "side": 1, "deck_index": 3, "x": 9000, "y": 22000},
])
```

仍保持每方同 Tick 最多一个动作、固定 side0→side1 规范顺序和一个主要 RPC。

## 6. 技能观测

完整观测和紧凑兼容观测都输出：

```text
entity_id
ability_slot
ability_state_code
ability_available
ability_cooldown_remaining_ms
ability_charges_remaining
ability_pending_ms
ability_mana_cost
```

新增字段纳入公开状态哈希，协议升级为：

```text
state_hash_scope = public-observe-v6
```

调用方应优先用 `ability_available` 做按钮可用性提示，同时保留原生
`result_code` 作为 fail-closed 防线，不能在 Python 侧自行推测可用性后绕过
原生命令。

`ability_state_code` 直接对应原生按钮状态：0 unknown、1 absent、2 ready、
3 cooldown、4 charges consumed、5 limited、6 disabled、7 not enough elixir、
8 temporarily unavailable、9 deploying、10 pending、11 casting、12 not yet
available。Python 同时补充 `ability_state_name`，但合法性仍以原生命令为准。

已证实的返回示例：

| 场景 | `result_code` | 结果 |
| --- | ---: | --- |
| 成功 | 0 | 原生扣费、消耗次数并触发技能 |
| 次数用尽 | `0x3F6` / 1014 | 拒绝 |
| 圣水不足 | `0x41A` / 1050 | 拒绝 |

其他代码保持为 `native_rejected`，先记录证据再命名，避免给未知状态强行贴标签。

## 7. 实测闭环

狂战士英雄：

- 英雄形态解析为 `203000076`；
- 圣水 2 时按键返回 1050；
- 圣水 3 时返回 0，圣水 3→0、次数 1→0；
- 下一 Tick 进入角色 `behavior_state=10`；
- 再次按键返回 1014。

弓箭女皇：

- 基础卡 ID `26000072`；
- 技能费用 1；
- 按键成功并扣 1；
- 部署完成后从 pending 进入 `behavior_state=10`。

自动复验：

```powershell
python scripts/accept_native_card_forms.py
python scripts/accept_full_card_catalog.py --port 37031
python -m unittest discover -s tests
```

当前结果：16/16 英雄形态、41/41 觉醒形态、122/122 标准基础卡、37/37
单元测试全部通过。

## 8. Runtime 边界

当前证书证明的是全卡选择/形态解析和通用技能命令，不是所有组合场景的穷举。
仍需持续扩充的纯沙盒证据包括：

1. 每个主动技能各自的目标、时序、Buff、召唤物和特殊移动场景；
2. 高实体量下的 effect/projectile 分类完整性；
3. 全卡之间的关键交互组合；
4. 塔兵、特殊模式和未来新增内容；
5. 游戏版本升级后的全部 RVA、结构和证书重建。

这些扩展仍应调用原版 `libg.so`，不为单卡在 Python 中补写战斗规则。
