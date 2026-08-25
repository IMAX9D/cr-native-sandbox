# schema v1 八卡循环本地升级数据

正式输出：

`D:\AI_data\cr-native-core\expert-v1\training-dataset\local-upgrades\schema1-base-cycle-v1`

文件：

- `valid-sides.jsonl`：标准八卡循环可解的玩家侧，可用于基础卡循环训练。
- `invalid-sides.jsonl`：动作序列与标准八卡循环不一致的玩家侧。
- `side-sequences.jsonl`：两者全集，供审计使用。
- `summary.json`：数量、动作覆盖率和源清单哈希。

每一行代表一场对局中的一个玩家视角。`base_deck` 定义 0～7 的卡牌索引；并行数组按该玩家的动作顺序对齐：

- `ticks`：20Hz 原生动作 Tick。
- `card_indices`：实际打出的基础卡索引。
- `actor_x/actor_y`：旋转到玩家位于下方后的原生坐标。
- `hand_masks_before`：动作前四张手牌的 8-bit mask。
- `next_card_indices_before`：动作前显示的下一张牌。

解码手牌：

```python
hand = [base_deck[i] for i in range(8) if hand_mask & (1 << i)]
next_card = base_deck[next_card_index]
```

每个有效侧的前四次动作因为初始洗牌不可唯一确定，两个字段均为 `null`；利用完整未来动作序列约束后，从第五次动作起当前手牌和下一张牌唯一。预处理使用未来信息只为恢复当时真实的隐变量，不把未来动作作为模型输入。

本层确定性恢复的是基础卡组和循环。旧 JSON 没有保存进化/英雄形态、等级、塔兵和技能 Tick，因此相关字段不能凭空补齐；有技能统计但无技能事件的记录仍可训练基础卡选择/时间/落点辅助任务，但不能冒充完整原生状态复演。
