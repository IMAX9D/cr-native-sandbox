# RoyaleAPI `time_raw` 原生命令相位固定 100 场 A/B

## 1. 实验问题

RoyaleAPI replay marker 只提供 20Hz 整数 `data-t`。采集器原样保存为
`time_raw`，但上游没有声明它对应原生 Tick 内的哪一个 command execution
boundary。

本实验只比较：

```text
A / offset 0: source label T，native command 在 T 执行
B / offset 1: source label仍为 T，native command 在 T+1 执行
```

B 不覆盖源字段、不补圣水、不绕过原生命令，也不作为正式训练数据修复。
生产默认仍为 offset 0。

## 2. 固定条件

两分支共同使用：

- 同一批 100 场 schema-3、native-ready、zero-ability 对局；
- selection SHA-256：
  `5fa5239318ce2396934408ceab624d507ccaf9a89143752ed80f458ca0127a3a`；
- 每个 Tag 相同的 source SHA-256；
- seed-preserving bounded search，preferred seed `424242`，最多 4096 个 seed；
- 100/100 Tag 的最终 chosen seed 相同；
- 4 个 Worker，端口 `38031..38034`；
- 20Hz，compact trace batch 64；
- 已包含 Spirit Empress 原生 3/6 费动态 selector 修复 `4115417`。

输入与输出：

```text
A
D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-seed-dynamic-v7

B
D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-action-phase-plus1-v8

逐 Tag 迁移表与聚合比较
D:\AI_data\cr-native-core\expert-v1\native-action-phase-ab-v7-v8
```

关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| A `results.jsonl` | `f6c70bf2c0e8909c2438d2dd1a39058b607921fee80c35caebf6bb519bd43ab8` |
| A `summary.json` | `5071008c283405e655c29c907920b484887ff3f884c31cb6d89f35f7036b3c2c` |
| B `results.jsonl` | `eec4b18e5f95ddf95770509138ce9afe57ca64ae23e07859ccf959c41f95907b` |
| B `summary.json` | `1fb72b4c6fd23b58db33fafc8e3e1638f3d4d8608984a46fd372f7782456a4a6` |
| comparison `per-tag.jsonl` | `299cc063ac2bf1910573b70e827da970bb68d1e3402b6c37ffd3050d8beeeea2` |
| comparison `summary.json` | `13bbf88dd5b153fbb0314dae1bd625c2b0977feb361e19e4dc51ac53e07d905a` |

## 3. 可审计实现

`execute_deployment_trace()` 新增：

```python
action_execution_tick_offset: int = 0
```

只接受 `0` 或 `1`；其他值 fail-closed。CLI 对应：

```text
--action-execution-tick-offset {0,1}
```

每场 audit 保存：

- `action_execution_tick_offset`；
- 明确的 source/execution provenance 公式；
- `last_source_action_tick` 和 `last_execution_action_tick`；
- 首次拒绝的 `source_tick / execution_tick / offset`；
- Tick Store metadata 中同样保存 offset 和 provenance。

不为每场重复保存完整逐动作映射。`selection + source plan + 全局 offset` 已能
无损重建每个 event 的 source/execution Tick，可避免扩展到 100k 时制造数百万
重复 JSON 对象。

CLI 新增 `--selection-from`。它逐字节重建并校验 prior `selection.jsonl` 的
SHA-256，数量不等、Tag 重复或序列化发生变化都会停止。

## 4. 总体结果

| 指标 | A：T 执行 | B：T+1 执行 | 变化 |
|---|---:|---:|---:|
| 完整 teacher-forced 成功场 | 40 | 43 | +3 |
| 失败场 | 60 | 57 | -3 |
| code13 失败场 | 11 | 5 | -6 |
| code4 失败场 | 41 | 44 | +3 |
| terminal-before 失败场 | 8 | 8 | 0 |
| code13 拒绝动作事件 | 11 | 5 | -6 |
| code4 拒绝动作事件 | 42 | 45 | +3 |
| 已接受部署动作 | 4,703 | 4,940 | +237 |
| 源动作覆盖率 | 67.253% | 70.642% | +3.389 pp |
| 已尝试动作接受率 | 98.886% | 98.998% | +0.112 pp |

失败场数按“首个失败类型”计数；动作事件数会把同 Tick 双方都拒绝分别计数，
因此 code4 场数与事件数相差一。

## 5. 逐 Tag 迁移矩阵

| A | B | 场数 |
|---|---|---:|
| success | success | 40 |
| code13 | success | 3 |
| code13 | code13 | 5 |
| code13 | code4 | 3 |
| code4 | code4 | 41 |
| terminal-before | terminal-before | 8 |

没有 A-success 退化，但 offset +1 也没有把 11 个资源失败统一修好：

- 只让 3 场成为完整成功；
- 3 场在原 code13 处通过，之后因生成态战斗终局分叉变成 code4；
- 5 场在 T+1 仍然没有足够圣水。

### 5.1 code13 → success

| Tag | 原首拒绝 | B 终局诊断 |
|---|---|---|
| `00VYPYPQV8QC` | Goblin Drill，T=2078 | match `1:0` |
| `02QY9L89CYGV` | Royal Hogs，T=2435 | mismatch：源 `3:2`，native `1:2` |
| `09LP9JLR0U8Q` | Valkyrie Evo，T=2771 | match `3:0` |

净增的 3 场中只有 2 场终局皇冠匹配，不能把“动作全接受”等同于“源状态已
恢复”。

### 5.2 code13 → 后续 code4

| Tag | A 首拒绝 | B 后续首拒绝 |
|---|---|---|
| `00CYPPG22CPJ` | Balloon Hero T=674 code13 | Elite Barbarians T=2999/source，execution 3000 code4 |
| `00YYPPGLR8YU` | Hog Rider T=1685 code13 | Skeletons T=3381/source，execution 3382 code4 |
| `080Y8LY0PQ9L` | Night Witch T=561 code13 | Knight T=3418/source，execution 3419 code4 |

这些样本证明 +1 可以消除资源临界拒绝，但不能消除由生成态战斗结果导致的
command gate 关闭。

### 5.3 code13 → code13

其余 5 场的圣水缺口超过一个恢复 Tick，包含 Giant/Elixir Golem 资源事件时序
分歧以及其他较大缺口。统一 +1 对它们没有解释力。

## 6. 终局结果

| terminal status | A | B |
|---|---:|---:|
| match | 26 | 28 |
| mismatch | 1 | 2 |
| missing at source-duration fence | 12 | 13 |
| logic frozen at fence | 1 | 0 |
| 未评估（teacher-forced 失败） | 60 | 57 |

40 场共同成功样本中，39/40 terminal status 不变；一场由
`logic_frozen_at_source_duration_fence` 变为 `missing_at_source_duration_fence`。
新增三场贡献 2 个 match 和 1 个 mismatch。

terminal-before 的 8 场完全不变，只是 B 的诊断名称显式携带
`execution_tick=T+1` 与 `source_tick=T`。

## 7. 状态 Hash

- chosen seed：100/100 相同；
- 40 场共同成功：logical training state hash 0/40 相同；
- 全 100 场：0/100 hash 相同。

这不是随机 seed 变化，而是预期的因果结果：把所有动作推迟一个原生 Tick，
从第一次动作开始就改变实体出生、移动、碰撞、攻击与死亡时序。由于源数据没有
逐 Tick state hash，A/B hash 互不相同并不能判定哪一条更接近源真值；它只说明
offset 不是无害的标签变换。

## 8. 吞吐

| 指标 | A | B |
|---|---:|---:|
| pilot wall time | 218.44 s | 208.14 s |
| stored ticks | 164,723 | 177,851 |
| stored ticks/s | 754.07 | 854.47 |
| accepted actions/s | 21.53 | 23.73 |
| 成功 episodes/hour | 659.21 | 743.72 |

B 本次墙钟吞吐较高，但两分支的失败深度与成功场数不同，且每档只有一次运行。
这些数字只记录实验成本，不能作为选择语义的依据。

## 9. 判定

### 保持 `Unknown`，生产默认继续为 offset 0

offset +1 对这批样本呈现**单调消除 6 个首个资源拒绝**，且没有让原成功场
失败；这是支持“marker Tick 与即时 command execution boundary 可能相差一步”
的证据。

但它还不够成为正式换算规则：

1. 只净增 3/100 个完整成功；
2. 另外 3 场只是从早期 code13 迁移到后续 code4；
3. 5 个 code13 不受影响；
4. 新增成功中仍有一个终局皇冠 mismatch；
5. 全部共同成功 trajectory 的状态 hash 都变化；
6. RoyaleAPI 没有公开 `data-t` 在原生 Tick 内的 phase 契约。

因此：

- 不修改源 `time_raw`；
- 不把 offset 1 设为默认；
- 不将 v8 Tick Store 混入 exact 训练集；
- 将 v8 保留为可复现实验分支；
- 等 RoyaleAPI hidden fields / replay marker 语义审计给出独立证据后再决定。
