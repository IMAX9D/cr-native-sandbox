# 专家对局数据集审计

本审计只读取 `D:\皇室战争数据集`，不会修改原始 JSON。执行：

```powershell
python -m expert_v1.filter_dataset
```

结果默认写入 `D:\AI_data\cr-native-core\expert-v1\dataset-audit`：

- `confirmed_1v1.jsonl`：本地列表 HTML 同时确认 1v1、非选卡模式和双方完整八卡。
- `uncertain_1v1.jsonl`：玩家数量、事件结构和结果均符合 1v1，但本地没有完整模式/卡组元数据。
- `rejected.jsonl`：2v2、特殊选卡模式、空壳/损坏文件或其他不满足标准八卡 1v1 的记录。
- `summary.json`：汇总统计。

`uncertain_1v1` 不是负样本，也不能直接称为已确认标准模式。它只能进入后续元数据补全队列；在完整卡组、卡牌形态、英雄技能事件和初始循环被可靠恢复前，不进入权威专家训练集。

当前历史数据丢弃了 `_invalid` 技能事件，因此含英雄/冠军主动技能的对局即使卡组完整，也不能仅靠现有 JSON 做逐 Tick 权威复演。占位卡只能生成带 `synthetic_partial` 标记的调试样本，不能混入正式监督学习数据。
