# 100k 对局逐 20Hz Tick 原生状态存储 v1

## 结论

逐 Tick 数据保留，但不保存逐 Tick JSON，也不预先保存重复的 dense grid。格式采用：

```text
每 256 Tick 一个完整 Anchor
        +
中间 255 Tick 的 Player/Tower/Entity 字段 Delta
        +
zlib level-1 独立块压缩与 CRC32
```

每个块可独立校验和解码；随机读取任意 Tick 最多从前一个 Anchor 回放 255 个 Delta。训练按 shard mmap，顺序读取不会反复解析 JSON。

实现位于 [tick_store_v1](D:/Deepseek/CR-Native-Core/expert_v1/tick_store_v1)。

## 真实数据规模

当前有效清单 `version-window-20260804/accepted.jsonl` 的只读全量统计：

- 73,556 场唯一对局；
- 总时长 17,339,023 秒；
- 精确 `duration_seconds × 20`：346,780,460 Tick；
- 平均 235.725 秒，即 4,714.509 Tick/场；
- P50/P90/P95/P99/max：225/310/311/312/360 秒。

按真实均值外推 100,000 场：471,450,949 Tick。

若直接保存 `uint8[10,32,18]` dense grid，需要约 2.470 TiB；现有 compact JSON RPC 实测约 4,191 B/Tick，外推约 1.797 TiB。rich debug state 样本为 7,430–23,533 B/Tick，规模更大。

## 二进制内容

### 完整 Anchor

- Tick；
- 两方私有区：`elixir_raw`、4 个 `hand_deck_indices`、`next_deck_index`；
- 六塔逻辑身份、坐标、HP/max HP；
- 全部 Character 实体；
- 原生终局、皇冠与 audit phase。

### 每 Tick Delta

- 玩家字段 bitmask：圣水、手牌、下一张牌；
- Tower：spawn/remove/字段 bitmask；
- Entity：spawn/remove/字段 bitmask；
- Episode 字段 bitmask。

实体键使用 `category == generation_key`，真实唯一键是 `(episode_uid, generation_key)`；绝不使用原生指针 `id`。坐标通常每 Tick 变化，只写 key、bitmask 和变化值。实体销毁写 remove，生成写一次完整记录。

当前 compact `observe_train_v1` 没有 effect/projectile、路径和碰撞 debug 字段，因此 v1 是“compact native training state 的逐 Tick 无损存储”，不是 rich debug trace 的替代品。需要研究 projectile/path 时应建立独立 rich-audit store，不能偷换本格式的正确性声明。

## Actor 信息隔离

原始 store 为了生成双方训练视角，分区保存两方手牌和圣水。训练读取必须调用：

```python
reader.actor_ticks(tag, actor_side=0)
```

`ActorTick` 只返回：

- 己方手牌、圣水、下一张牌；
- 公共塔和公共实体；
- 公开 HP、卡牌身份、等级、坐标；
- 仅己方 `ability_slot/available`；
- 公开皇冠和终局。

它不返回敌方 Player、敌方技能内部状态、`behavior_state`、native phase/gate、RNG、state hash、target/path/collision 或原生地址。`observe_train_v1` 本身不是 Actor-safe，禁止直接喂模型。

## Shard、崩溃恢复和并发

一个 Worker 独占一个 append-only `.crts.partial`，不同 Worker 不争用数据文件。每场是一个带固定头、payload 长度、CRC32、battle-tag hash 的 frame：

1. 写 frame；
2. flush + fsync；
3. SQLite 标记 task done；
4. 达到 256 场后原子 finalize 为 `.crts`；
5. 重建 `.index.jsonl`，计算 data/index SHA-256。

若进程在任意字节中断，重启会扫描到最后一个完整 CRC frame，截断尾部，并从数据 frame 重建索引。若已 fsync 但 DB 还没提交，append 按 battle tag 幂等，不重复写。

`TickStoreWorkQueue` 使用 SQLite WAL + FULL synchronous：

- 全局 pending 队列；
- 原子 lease claim；
- heartbeat 延长租约；
- 任意 Worker 可窃取过期 lease；
- attempts 上限和 failed 隔离；
- complete 固化 shard/offset/size/payload SHA。

最后 `build_store_manifest()` 重新验证每个不可变 shard 的 data/index SHA，核对 episode/tick 总数，再原子发布全局 `manifest.json` 和 content digest。

## 实测原型 Benchmark

命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  -m expert_v1.tick_store_v1.benchmark `
  --ticks 5000 --entities 24 --total-ticks 471450949
```

本机 CPU、24 个持续移动 compact entity 的合成压力样本：

| 指标 | 结果 |
|---|---:|
| Tick Store | 86.33 B/Tick |
| minified JSON | 1,620.83 B/Tick |
| 10-channel dense grid | 5,760 B/Tick |
| 相对 JSON | 18.78× 更小 |
| 相对 dense grid | 66.72× 更小 |
| 编码 | 12,527 Tick/s |
| 解码 | 7,867 Tick/s |
| 100k 外推 | 37.90 GiB |

结果保存在 `D:\AI_data\cr-native-core\expert-v1\tick-store-benchmark-v1.json`。

这是 synthetic codec benchmark，不是 100k 真实 capture 容量承诺；它未覆盖真实 entity-count 长尾、频繁 spawn/despawn 和 future schema。保守容量仍按 0.07–0.25 TB 预留，必须先用 100–1,000 场真实逐 Tick capture 重新标定。解码已达到此前原生短路径约 7,000 Tick/s 的同量级。

## 训练读取

`ShardReader` mmap 数据文件，按 battle tag 定位 frame；`EpisodeReader` 支持：

- `iter_ticks()`：逐 Tick 顺序还原；
- `read_tick(tick)`：Anchor 有界随机读取；
- `actor_ticks()`：安全单方投影；
- `actor_windows(length=128, burn_in=32)`：LSTM 训练窗口。

Grid 在训练批次中从当前实体即时投影，不落盘。这样仍然是逐 Tick 状态，只是不把可确定派生的 18×32 raster 重复保存几百亿格。

## 验收与下一步

当前通过 4 项测试：

- 600 Tick Anchor/Delta bit-exact roundtrip；
- 随机 Tick seek；
- Actor 投影无对手私有区/内部 phase；
- 截断 shard 恢复、双 Worker lease 窃取。

接入 capture 前仍需：

1. 在 native 批量 trace 路径输出每个 compact Tick；当前 decision-only runner 不能满足逐 Tick。
2. 给 live compact player 补 `next_deck_index/refill_timer` 后重新部署；旧 host 中 `-1` 只能表示未观测。
3. 用 100–1,000 场真实 capture 测 entity-count、bytes/Tick、CPU、随机 seek 和端到端训练吞吐。
4. 明确标注当前源没有 seed/build/state anchor：生成流只能是 `native_generated_unanchored`，不能称为真人原始场面 100% 复现。

