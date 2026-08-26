# RoyaleAPI 回放隐藏字段与原生 Teacher-Forced 差异审计

日期：2026-08-26

## 1. 结论

本轮找到了一个足以解释 v7 大部分失败的确定性适配错误：

> `data-i` 是整局回放坐标的翻转标志，不是红蓝方标志，也不是卡牌实例号。

RoyaleAPI 回放查看器的坐标流程是：

```text
viewer = data_i == 1 ? rotate_180(raw) : raw
native side-0 canonical = rotate_180(viewer)

所以：
data_i == 0 -> native = (18000 - x_raw, 32000 - y_raw)
data_i == 1 -> native = (x_raw, y_raw)
```

旧 crawler 对所有事件都执行了第一条旋转公式，导致 `data_i == 1` 的整局动作
被再次旋转。schema-v3 的 26,566 场中，这类对局占 13,406 场，接近一半。

修复只落在 `compile_battle()`：始终从来源 `x_raw/y_raw/data_i` 重新计算原生
坐标，不信任历史派生的 `x/y`，也不改写 10 万份源 JSON。固定 100 场 v9 验收：

| 指标 | v7 | data-i v9 |
| --- | ---: | ---: |
| 完整 teacher-forced 成功 | 40/100 | **83/100** |
| code 4 首失败对局 | 41 | **10** |
| code 13 首失败对局 | 11 | **6** |
| native 提前终局 | 8 | **0** |
| 其他原生拒绝 | 0 | code 3：1 |
| 来源动作首失败前覆盖率 | 67.253% | **92.292%** |
| 已尝试部署接受率 | 98.886% | **99.737%** |

40 个 v7 成功样本全部继续成功；31 个 code 4、5 个 code 13、7 个提前终局
转为完整成功。这是同 selection、同来源文件、同 Tick、同 seed-search 语义下的
隔离 A/B，不是相关性猜测。

审计还确认未来 crawler 值得新增保存四组字段：

1. `.matchup_button[data-game-mode-id]` 的原生数字模式 ID；
2. `.hp-both-popup` 的双方最终国王塔/公主塔 HP；
3. `deck_tower_card__container .level` 的塔兵等级；
4. `data-index`、marker 内的逐卡出现序号和 timeline 的 `data-ability` 交叉证据。

RoyaleAPI 页面中仍然**完全不存在**原始 RNG seed、初始 4+4 手牌/队列、逐 Tick
实体状态、Tick 内命令相位、逐次 Spirit Empress 3/6 形态、逐次 Evo 实际形态、
技能对应的原生实体 ID。这些字段不能通过改 parser 凭空恢复。

## 2. 证据边界

本审计只使用了低频、只读证据：

- 2026-08-20 保存的真实列表页：
  `D:\Deepseek\cr_re\dumps\battlelog_latest.html`
- 同站回放静态脚本：
  `D:\Deepseek\cr_re\evidence\royaleapi\local-captures\battle_replay.min.js`
- 当前生产 parser：
  `D:\皇室战争数据集\crawler\parsers.py`
- schema-v3 原始来源：
  `D:\AI_data\cr-native-core\expert-v1\training-dataset\source-round-03-selected-schema3-20260826\raw\battles`
- 固定 v7/v9 原生输出和逐 Tag 对比。

关键证据 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `battlelog_latest.html` | `626cd3ffd09f8bbcff242e43ad84f3d5a20a4080fbc3c04aefbac536b7f207eb` |
| `battle_replay.min.js` | `f86a8428a7f7f32127c678a5a9b1dee40580ea25a93a492892c9db0451107d09` |
| 当前 crawler `parsers.py` | `63d60a6001eafbf5a3bdb397ffd8bdb753231521f85639ef16a8742e56b380df` |
| v7 `summary.json` | `5071008c283405e655c29c907920b484887ff3f884c31cb6d89f35f7036b3c2c` |
| v9 `summary.json` | `fd0e33ba99703b3a08894fc8f52fe8d08139f156a7c038232c54fd68be44fe17` |
| v9 `results.jsonl` | `b49a2f9d57c4e052b6b8677bdf42af54a8abba645ba64ae1d22be2778e7ea488` |

当前站点的直接只读访问被 Cloudflare 403 挑战拦截；本轮没有绕过挑战，也没有
批量请求。保存页距本次审计仅六天，并且 schema-v3 来源仍包含同一组 marker
字段，因此足以验证字段语义和当前数据问题。

## 3. `data-i`：已确认的关键字段

### 3.1 静态脚本语义

查看器只在 `data-i` 字符串等于 `1` 时，把 marker 的 X/Y 分别相对
`18000/32000` 翻转，然后才计算页面位置。它不读取 marker 颜色来决定旋转。

因此旧文档中“`data-i=1` 代表红方”的解释不成立。真实数据中同一局蓝红双方
共享相同 `data-i`。

### 3.2 26,566 场分布

扫描 schema-v3 全部来源：

| 局级 `data-i` | 场数 |
| --- | ---: |
| 0 | 13,160 |
| 1 | 13,406 |
| 局内混合 0/1 | **0** |

事件级共约 202 万个 marker，双方与 0/1 均衡分布。Y 坐标中位数进一步验证
了它是整局朝向：

| side / data-i | `y_raw` 中位数 |
| --- | ---: |
| team / 0 | 19,500 |
| opponent / 0 | 12,500 |
| team / 1 | 12,500 |
| opponent / 1 | 19,500 |

也就是说 `data-i` 改变的是整局坐标朝向，而不是一方的身份。

### 3.3 v7 失败与 `data-i` 的对应

修复前固定 100 场：

| data-i | success | code 4 | code 13 | 提前终局 |
| ---: | ---: | ---: | ---: | ---: |
| 0（44场） | 36 | 5 | 3 | 0 |
| 1（56场） | 4 | 36 | 8 | 8 |

36/41 个 code 4、8/11 个 code 13、8/8 个提前终局全部集中在被错误旋转的
`data-i=1` 子集。

修复后：

| data-i | success | code 4 | code 13 | code 3 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 36 | 5 | 3 | 0 |
| 1 | **47** | 5 | 3 | 1 |

逐 Tag 结果保存在：

- `D:\AI_data\cr-native-core\expert-v1\native-coordinate-ab-v7-v9\per-tag.jsonl`
- `per-tag.jsonl` SHA-256：
  `d8ae8c47693c04dbfca58eecae74df745f7f0c22e20f946d66d6a2caf393afa8`
- 汇总：
  `D:\AI_data\cr-native-core\expert-v1\native-coordinate-ab-v7-v9\summary.json`

### 3.4 代码处理

`expert_v1/native_replay_plan.py` 现在为每个计划增加：

- `coordinate_provenance`
- `coordinate_audit.transform`
- `raw_data_i_events`
- `data_i_zero_events`
- `data_i_one_events`
- `legacy_xy_fallback_events`
- `data_i_values`

schema-v3 走精确 raw/data-i 路径。旧 schema 缺失 raw 或 flag 时仍兼容读取历史
`x/y`，但明确标记：

```text
legacy_stored_xy_fallback_unverified
legacy_precomputed_coordinates_unverified
```

不会把 fallback 宣称为精确来源，也不会覆盖源文件。

## 4. `data-t` 到底是什么

可以确认：

- `data-t` 是 RoyaleAPI replay viewer 的 20 Hz 整数时间槽；
- timeline 图片与 arena marker 使用相同值；
- 页面水平位置与 `data-t / 20` 秒一致；
- crawler 原样保存为 `time_raw`，没有自行加减 Tick。

不能确认：

- 它是否是客户端 command enqueue Tick；
- 在一个 libg Tick 内位于恢复圣水之前还是之后；
- 是 server command Tick、确认 Tick，还是仅为 viewer marker Tick；
- 是否存在被页面丢弃的 sub-Tick/phase 位。

静态脚本只用 `data-t` 做时间轴布局和 hover 对应，不用它重放原生命令。因此最
严谨的命名应是 `source_marker_tick_20hz`，而不是已经证明的 `command_tick`。

v9 仍按 `time_raw + 0` 执行。余下 6 个 code 13 中有多个只差一个被动恢复 Tick，
需要单独的 +1 execution-boundary A/B；不能因为此前坐标错误制造过 code 13，
就把全量 source label 统一覆盖为 T+1。

## 5. `data-index` 与数字 game mode

列表页同一场对局有两个 sibling button：

- `.replay_button[data-index=...]`
- `.matchup_button[data-index=...][data-game-mode-id=...]`

保存页中的五场全部可以用 `data-index` 无歧义关联，并得到原生数字模式
`72000006`。当前生产 `_battle_metadata()` 只读取 replay button 和模式显示文本，
没有 join matchup button，因此 schema-v3 JSON 中缺少数字模式 ID。

`data-index` 在该保存页中也与秒级 `data-timestamp` 相等。它是列表/分页和 sibling
组件的 battle index，不是 RNG seed，也不是手牌顺序。

建议未来保存：

```json
{
  "battle_index": 1787218979,
  "numeric_game_mode_id": 72000006,
  "numeric_game_mode_provenance": "list_matchup_button_joined_by_data_index"
}
```

项目内 `expert_v1/filter_dataset.py::load_list_metadata()` 已经有 sibling join 的
旧实现，可把同一逻辑移入生产 crawler；本轮按要求没有改 crawler。

这很可能是 v9 剩余 code 4 的下一优先项。当前仅有 `Ranked/pathOfLegend` 等显示
文本，不能保证唯一对应某个 frozen `gameMode` Data ID。

## 6. 塔等级与最终 HP

列表页还包含当前 crawler 没有保存的状态锚点。

### 6.1 可新增抓取

每个玩家牌组旁边的 `deck_tower_card__container` 同时提供：

- 塔兵 slug，例如 `tower-princess`；
- 显示等级，例如 `Lvl 2`、`Lvl 4`。

当前 `_deck_record()` 只解析塔兵 slug，没有解析其等级。

`.hp-both-popup` 提供：

```text
data-team-king
data-team-princess0
data-team-princess1
data-team-total
data-oppo-king
data-oppo-princess0
data-oppo-princess1
data-oppo-total
```

这是双方**最终剩余 HP**。它可以作为 hard terminal diagnostic，并能在某座塔未
受伤时反推该塔的有效最大 HP。

### 6.2 页面没有的字段

保存页没有独立的：

- `king_level` 数字字段；
- `king_max_hp` / `princess_max_hp`；
- 每 Tick HP 曲线。

因此不能把塔兵等级无条件等同于原局 King Tower 等级。可以结合冻结版 HP 表、
未受伤塔的最终 HP 和数字模式 ID做约束求解；若仍有多个候选应分支或 fail-closed。

## 7. Evo、Hero、技能和 Spirit Empress

### 7.1 页面实际提供什么

- 列表牌组 `data-card-key` 保留 `-ev1` / `-hero` slot 后缀；
- 回放 marker 的 `data-c` 只给 base card，技能为 `_invalid`；
- marker 内 `<span>` 给同 side/base card 的出现序号；
- timeline 图片有 `data-ability=0/1`，可与 `_invalid` marker 交叉验证；
- 技能 marker 没有 card slug、entity ID、generation key 或部署 marker 关联。

出现序号可以由动作序列自行计数，因此不是新的隐藏状态，但值得保存作 HTML
完整性校验。

### 7.2 Evo

页面能证明“这个 deck slot 具有 Evo”，不能直接证明“第 N 次部署实际用了 Evo
形态”。对正常从零开始的对局，原版 libg 可以根据 Evo slot 和完整卡牌循环自行
演算实际形态；不应由 crawler 把每个 marker 都标成 `-ev1` 实体。

### 7.3 Hero 与技能

Hero slot 可由 deck 后缀确定。技能 Tick 可由 `_invalid` marker 与 timeline
`data-ability=1` 确认，但页面不指出按的是哪个在场 Hero 实体。多个同能力实体同时
合法时仍需原生分支，不能从 HTML 猜一个 entity。

### 7.4 Spirit Empress 3/6 费

marker 只写 `spirit-empress`，没有 Normal/Mounted 标志。聚合 `elixir_stats` 也
不能替代逐次形态：

- schema-v3 中包含 Spirit Empress 的 484 个文件、486 个玩家 side、1,861 次部署；
- 对其中 464 个可做简单类别代数核对的 side，网页统计对**每一次**都按 wrapper
  的 6 费计账；
- 没有一例在统计中表现为 3 费或混合 3/6；
- 余下 22 个 side 又受其他动态卡/统计分类影响而不闭合。

这说明网页 `elixir_stats` 是按静态 wrapper/catalog 生成的展示统计，不是逐 command
实际扣费账本。它不能回答第几次为 3 费地面形态。

正确做法仍是：让原生 choice-wrapper 根据当前原生资源状态选择；若要恢复“原局
究竟选了哪一形态”，只能获得原始 command/出生实体证据，或对多个合法形态做显式
分支，不能从现有 RoyaleAPI HTML 唯一恢复。

## 8. 页面完全不存在的关键状态

在列表 HTML、replay HTML、关联静态脚本和已保存 JSON 中均未发现：

| 缺失字段 | 影响 |
| --- | --- |
| 原始 RNG seed | 不能恢复原局随机流 |
| 初始四张手牌与后四张队列 | 只能从八卡循环约束/seed search 找兼容状态 |
| 原始逐 Tick 实体/HP/目标/碰撞状态 | teacher-forced 轨迹是生成态，不是来源状态快照 |
| Tick 内 command phase/sub-Tick | `data-t` 的 T/T+1 边界不能由网页证明 |
| 每次 Spirit Empress 形态 | 3/6 费事件需原生选择或分支 |
| 每次 Evo 实体形态显式标志 | 只能由 slot + 原生 cycle 演算 |
| 技能的 entity/generation identity | 多候选技能必须分支 |
| 逐 Tick 圣水曲线 | 只有类别总花费和 leaked 展示统计 |
| 战斗中间塔血锚点 | 只有最终 HP |

## 9. crawler 后续修改建议（本轮未实施）

建议升为新的来源 schema，不覆盖旧字段：

### 9.1 列表页

1. 用 `data-index` join replay/matchup button；
2. 保存 `battle_index` 和 `numeric_game_mode_id`；
3. 保存每方 `tower_troop_level`；
4. 保存双方最终 king/princess HP；
5. 可保存 matchup 的 `data-players=1v1` 作门禁交叉证据；
6. 可解析 deck QR 中的 numeric card/tower IDs，只用于 slug/alias 校验。

### 9.2 replay HTML

1. 原样保存 `x_raw/y_raw/data_i/data-t/data-s/data-c`；
2. 保存 marker 内出现序号；
3. 同时解析 timeline `data-ability` 并与 map marker 一一核对；
4. parser 只派生 `viewer_x/viewer_y`，不要把训练内核的 canonical 方向耦合进
   crawler；
5. 永远保留原始字段，所有派生坐标带 `coordinate_transform_version`。

### 9.3 旧数据

- schema-v3：已经保留 `x_raw/y_raw/data_i`，可像 v9 一样无损修复；
- schema-v1/v2：通常保留 raw X/Y 但没有 `data_i`，不能宣称精确回填；
- 对旧数据可用双方 troop/building 的 `y_raw` 分布产生高置信 orientation
  candidate，但正式 exact 数据应重新抓取或对歧义样本 fail-closed；
- 不要原地改写现有 10 万源 JSON，以版本化 materialization 生成新计划。

## 10. 下一步优先级

1. 保留本轮 `data-i` 修复并作为所有 schema-v3 原生计划的默认路径；
2. 在未来抓取中补 numeric game mode、塔兵等级和最终 HP；
3. 用这三个锚点处理 v9 剩余 10 个 code 4；
4. 单独完成 source T 与 execution T+1 A/B，解释剩余 6 个 code 13；
5. 调查新出现的 1 个 code 3，保留首差异现场；
6. 完成带技能的固定 selection 验收；
7. 只有 schema-v3/data-i 路径稳定后，再决定 schema-v1/v2 是重新抓取、分支
   orientation，还是降级为 sequence-only 数据。

当前最重要的工程结论是：v7 的大部分提前终局并不是 `libg.so` 战斗逻辑与真实
游戏不同，而是 RoyaleAPI 已提供的坐标朝向字段被 adapter 忽略。修复后 100 场
通过率从 40% 提升到 83%，证明先完整审计来源约束字段比继续给原生内核做规则
补丁更有效。
