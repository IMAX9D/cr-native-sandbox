# 原生数据下载契约

## 目的

下载器不再手抄卡牌、形态、塔兵或技能映射。唯一权威输入是由当前
`libg 15.535.29` 运行时组件生成的冻结 JSON：

```text
D:\AI_data\cr-native-core\expert-v1\contracts\native-ingest-v150535029.json
D:\AI_data\cr-native-core\expert-v1\contracts\native-ingest-v150535029.json.sha256
```

生成命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe scripts\export_native_ingest_contract.py
```

写入采用同目录临时文件加原子替换。JSON 内含 canonical
`contract_sha256`，旁边的 `.sha256` 则校验最终文件字节。相同源码组件重复
生成会得到相同内容；时间字段继承冻结卡表，不使用当前时间。

## 权威来源

契约由程序直接合并以下冻结组件：

- `native_core/data/live_card_catalog.json`：122 张标准 1v1 基础卡及 Evo/Hero
  原生 form ID；
- `expert_v1/native_replay_plan.py` 的 `ROYALEAPI_CARD_ALIASES`：公开回放
  slug 与冻结内部名的差异；
- `expert_v1/native_capabilities.py`：4 种已探测塔兵和技能来源；
- `bindings/runtime-150535029-x86_64.json`：ABI 与 `libg.so` SHA-256。

每个组件的单独 SHA 和聚合 `component_sha256` 都写进契约。契约还包含
`ingest_schema_sha256`，所以验证语义改变也会改变身份。

## 下载器稳定读取面

跨工程读取只需以下顶层字段：

| 字段 | 语义 |
| --- | --- |
| `schema_version` | 当前固定为 `1` |
| `kind` | `cr_native_authoritative_contract_v1` |
| `game_version` | `15.535.29` |
| `allowed_card_tokens` | 精确、全小写的 RoyaleAPI 牌/形态 token |
| `allowed_tower_troops` | 原生已有 support ID 的塔兵 slug |
| `ability_source_tokens` | 可解释技能事件的精确牌形态 token |
| `numeric_game_mode_ids` | 已有证据的标准模式 ID；当前只允许 `72000006` |
| `contract_sha256` | 移除本字段后，对整个顶层对象做 canonical JSON SHA-256 |

canonical JSON 参数为：UTF-8、`sort_keys=true`、无多余空格、
`ensure_ascii=false`。未知模式不能靠显示文本猜测；发现并实证新模式后，应先
更新主工程契约再下载。

`cards`、`tower_troops`、`ability_sources` 是详细解释层，包含 numeric card
ID、允许的 `form_flags`、Evo/Hero form ID、塔兵 support ID 以及技能原生 form
ID。下载器无需重新推导这些表。

## Fail-closed 规则

1. 卡牌必须精确出现在 `allowed_card_tokens`。例如标准版本不包含
   `party-hut`，必须记录为 `native_card_mapping_missing`，不能静默替换。
2. 已知基础牌但不存在对应 Evo/Hero 时，记录
   `native_form_mapping_missing`。例如 `giant-ev1` 不可用，而 `giant-hero`
   可用。
3. 塔兵必须出现在 `allowed_tower_troops`；当前是 `tower-princess`、
   `cannoneer`、`dagger-duchess`、`royal-chef`。
4. 若回放包含技能事件，该方牌组必须至少有一个
   `ability_source_tokens` 成员。此检查只证明静态来源存在；具体按键实体仍须在
   事件 Tick 用原生状态唯一解析，不能猜实体。
5. numeric game mode 缺失或不在白名单中时拒绝。

主工程也提供纯读取 API：

```python
from expert_v1.native_ingest_contract import (
    load_native_ingest_contract,
    validate_ingest_metadata,
)

contract = load_native_ingest_contract(CONTRACT_PATH)
issues = validate_ingest_metadata(
    contract,
    deck_tokens=deck,
    tower_troop=tower,
    numeric_game_mode_id=mode_id,
    observed_ability_events=len(ability_plays),
)
```

爬虫可以只按上述 JSON 读取面实现，避免依赖主工程 Python 包。

## schema 5 回流主工程

`expert_v1.native_replay_plan.compile_battle()` 对新的 authoritative
`schema_version=5` 再做一次独立 fail-closed 验证。来源必须携带并匹配：

- `numeric_game_mode_id`、`battle_index` 及各自列表页 join provenance；
- `authoritative_native_contract` 的 game version、canonical SHA 与文件 SHA；
- 每方 8 个 `deck_cards` 的 slot/slug/base/form/level；
- 塔兵 slug 与**独立的塔兵等级**；
- `final_tower_hp` 的 King、`princess0`、`princess1`、total；
- 完整 raw `data_i` 坐标契约与精确技能事件数组。

计划物化时直接写入来源 numeric game mode，并将来源塔兵等级转成原生的
zero-based `sc[].l`。绝不把 King level、牌组最高等级或模板 avatar level 当成
塔兵等级。schema 3 仍按原有 unanchored teacher-forced 兼容路径运行，不要求这些
schema 5 字段。

列表页只证明两个 Princess Tower 槽位 `princess0/1`，尚未证明它们各自对应
native `x=3500/14500` 的哪一侧。因此终局诊断比较每方 King HP、两个 Princess
HP 的多重集合与 total，并保留 `source_slots_unmapped` provenance；不猜左右 lane。

即使 schema 5 元数据全部通过，也仍然不声明恢复了来源精确隐藏状态：原始 RNG、
初始手牌、King Tower level、精确 game build 和逐 Tick 状态锚点仍未提供。
