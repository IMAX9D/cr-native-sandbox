# Spirit Empress 原生双形态选择

## 结论

冻结版本 `15.535.29` 中，Spirit Empress 的手牌入口是 wrapper
`28000025 (MergeMaiden)`，实际部署形态为：

| 可用圣水 | 原生形态 | Data ID | 扣费 |
| ---: | --- | ---: | ---: |
| `3.00 .. 5.99` | 地面 `MergeMaiden_Normal` | `26000104` | 3 |
| `>= 6.00` | 飞行 `MergeMaiden_Mounted` | `26000105` | 6 |

这不是 Python 按费用替换卡牌。JNI 仍把真实 hand entry、player 和 command
selection 交给原版 `libg.so`；形态、费用、合法落点、扣费和生成实体均由原生
函数决定。

## 原生调用链

- canonical builder：`0x1048170`；
- choice-card builder：`0xD71800`；
- wrapper form predicate：`0xD71070 -> 0xD709A0`；
- selection resolver：`0xE85D40`；
- 冻结版本 choice-wrapper vtable：`0x1942898`。

canonical builder 把 wrapper form index 留为默认值，因而总是解析 forms vector
的第 0 项（Mounted）。`0xD71800` 先构造 canonical selection，再用 player 的
原生资源状态逐项执行 wrapper predicate，并把选中的 form index 编入 packed
selection 的 bits 4..6。

普通卡的 root 没有 choice forms vector，不能无条件调用 `0xD71800`。Bridge
因此先从真实 `entry + 0x10` 读取 root/vtable：只有原生 choice-wrapper 才走
`0xD71800`，其余普通、觉醒、英雄及冠军卡继续走 canonical builder。判断不含
卡名或 Data ID 特判；root 无法读取时 fail-closed。

`nativeAct` 与 `nativeProbeGrid` 共用同一 selector，避免动作按动态形态执行、
而 mask 仍按固定 6 费形态计算。诊断字段包括：

- `selection_strategy`；
- `selection_builder_rva`；
- `selection_root_vtable_rva`；
- `selection_form_index`；
- `resolved_data_id`；
- `card_cost_raw`（`probe_grid`）。

## 验收

运行：

```powershell
python scripts/accept_spirit_empress_selection.py --port 38035
```

本机隔离 Host 的证书位于：

`D:\AI_data\cr-native-core\spirit-empress-selection-acceptance.json`

实测结果：

- `37800` raw 圣水：解析 `26000104`，扣 `30000`，逐 Tick 观察到地面实体；
- `77800` raw 圣水：解析 `26000105`，扣 `60000`，逐 Tick 观察到飞行实体；
- ordinary Knight、Evo Knight 首循环、Hero Knight、Archer Queen 回归均保持
  `canonical`，部署与生成 Data ID 正确。
