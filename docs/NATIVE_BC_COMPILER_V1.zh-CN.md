# Native Tick Store → BC Dataset 编译器 v2

入口：`python -m expert_v1.compile_native_bc_dataset`。

## 输入契约

编译器只接受三类不可变输入：

1. `cr_native_tick_store_v1` 的 `manifest.json`、`.crts` 与索引；
2. `expert_v1.freeze_schema5_manifest` 生成的规范 Schema5 JSONL；
3. 每场 Tick episode 引用的 `deployment-masks-v1` 内容寻址 sidecar。

开始编译前会校验 Tick Store 的 data/index SHA、每场 payload SHA、Schema5
源文件 SHA、原生 ingest contract SHA、Mask manifest/sidecar SHA。Tick Store 中的
`source_sha256` 必须与规范 Schema5 manifest 指向的文件逐字节一致。任何缺失、
漂移或非 canonical sidecar 都会 fail-closed。

## Actor 信息边界

每场产生两个 actor sequence。红方统一旋转到己方在下的视角。输入仅含：

- 自己的手牌、下一张牌、圣水和完整己方卡组；
- 已公开的敌方出牌；
- 公共塔、公共实体、时间、皇冠和命令门；
- 从原生 mask sidecar 离线还原的合法性。

敌方手牌、敌方圣水、native RNG、未公开卡组和未来终局时长均不会写入训练
数组。实体卡牌身份写入 ragged `entity_tokens`，通过全卡牌 categorical
vocabulary embedding；逻辑上的公共 grid 仍是完全相同的 `[8,32,18] uint8`，
物理存储改为 actor-row CSR：`grid_offsets/grid_indices/grid_values`。Dataset 只在
读取 recurrent window 时逐字节还原 dense grid，禁止把 `card_id` 当连续像素。

部署落点 Mask 也不再为每个 20 Hz WAIT Tick 写两份 72-byte packed row。
`selected_position_mask` 与 `ability_position_mask` 只保存对应 `label_mask=true` 的
监督行及其 row index，未监督行在窗口读取时严格还原为全零。此变更只改变物理
存储，不改变模型 tensor、logits、loss 或标签语义。

## 标签

Schema5 的来源 Tick `T` 按 episode 内冻结的 execution offset 对齐原生 Tick。
标签为条件分解：

- timing：该 Tick 是否行动；
- action kind：deploy / ability；
- deploy：当前四手牌中的 slot 与 18×32 actor 视角落点；
- ability：按原生实体 generation key 稳定排序后的合法 ability slot；
- 不适用的条件 head 保持 `label_mask=false` 和 `-100`。

部署标签必须通过 `verify_deployment_labels`。Dynamic-choice 卡仅在 sidecar 存在
精确 play-Tick variant 时参与 card/position 监督；其他 Tick 不复用首手 base
probe 冒充精确 Mask。

## 无泄漏 Split

编译计划对 `battle_tag ↔ player_tag ↔ source_group ↔ source_sha256` 建联通分量，
整个分量只能进入一个 split。两侧 actor 永远随 battle 一起移动。最终发布前会
再次验证 battle、玩家、来源组和来源文件在 train/validation/test 间零交叉。

## 断点与并发

`compile-plan.json` 固化输入、编译器组件 SHA 和 `capacity-preflight.json` SHA。
在正式写 shard 前会按内容哈希、时长和 Tick payload/state density 分层抽取最多
100 场，实测逐场 bytes/actor-row、编译速度和峰值 RSS。容量投影使用样本中的
单场最大 bytes/actor-row（不是均值），再加 35% 安全余量。每个 shard 写入前还会
通过跨进程 reservation ledger 重新检查磁盘，并始终保留 `max(10 GiB, 文件系统
容量的 5%)`；并行编译不能相互超售空间。磁盘或内存门不通过即 fail-closed。

输出默认按约 512K actor rows 切成
确定性 shard；每个 shard 写入独立临时目录、计算全部 `.npy` SHA 后原子改名。
重启时已完成且 SHA 正确的 shard 直接复用。训练 Dataset 每个进程最多保持 8 个
shard mmap 打开，随机 shuffle 不会逐步耗尽文件句柄或地址空间。

单机自动并行：

```powershell
python -m expert_v1.compile_native_bc_dataset `
  --tick-store-root D:\path\to\shards `
  --schema5-manifest D:\path\to\schema5-frozen.jsonl `
  --native-contract D:\path\to\native-ingest-v150535029.json `
  --output-root D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1
```

多进程/多批次可先 `--plan-only`，再让 N 个进程分别使用
`--worker-count N --worker-index 0..N-1`；全部结束后执行 `--finalize-only`。
`--process-workers` 是请求上限，编译器会按容量预检的内存建议自动收敛，并把
requested/effective 数值写入 Worker receipt。
`manifest.json` 是最终可见性边界，只在所有 shard 通过训练 schema 后原子发布，
并同时生成 `manifest.sha256` 与完整 `shard_file_sha256` 覆盖。
