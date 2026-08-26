# 动作间 Native Logic Freeze 可靠性契约

## 问题

当 libg 在下一个专家动作执行边界前停止推进、但 compact episode 尚未标记
`terminated/truncated` 时，Host 会返回一个 Tick 不再增长的 incomplete suffix。

旧 decoder 在动作间看到该 suffix 会直接抛出 generic `NativeHostError`。上层因此
拿不到已经确定的 chosen seed、source/accepted action 计数和已采集 Tick 前缀，
最终 result row 只剩异常文本，并被 summary 当成未知失败。

## 当前契约

decoder 的 `allow_nonterminal_freeze=True` 现在仍只是显式低层能力。语义由调用上下文
决定：

- 最终 source-duration fence：维持原有终局诊断逻辑；
- 正在前进到未来专家动作：立即 fail-closed，返回结构化
  `native_logic_frozen_before_execution_tick`；
- 任何 freeze 都不能恢复成 complete frame，也不能写入成功 Tick Store，更不能标为
  teacher-forced success。

结构化审计保存：

```text
source_tick / execution_tick / offset
last_native_tick / missing_native_ticks_to_execution
chosen_seed
source_actions / accepted_actions_before_freeze
collected_tick_count / start / stop
trace requested / stepped calls
crowns / crown tower HP
commands_allowed / command_gate_code / native_phase
terminated / truncated / outcome / winner
```

Deployment runner 的 `TraceReplay.states` 保留完整、严格连续的已采集前缀。Ability
runner 的 `NativeReplayResult.collected_tick_states` 同样保留前缀；JSON 只输出数量和
边界，避免把数千个 Python dataclass 塞入诊断 JSON。失败轨迹不写入正式成功 shard。

## Summary 分母

原始固定 selection 的 success/failure 仍保留，不能删除样本。生产 summary 另外写：

```text
logic_freeze_before_execution_episodes/battles
phase_comparable_episodes/battles
phase_comparable_success_rate
failure_class_counts
```

因此逻辑冻结不会再混入 code4/code13 或 unknown transport failure，也不会污染命令
相位的可比较分母。

## 真实 Tag 定点回归

2026-08-26 使用当前 4-Worker libg runtime、profile v1 offset 1 复测：

```text
battle_tag                     089Y82CPYYY9
failure_class                  native_logic_frozen_before_execution_tick
failure                        native_logic_frozen_before_execution_tick_3682_
                               source_tick_3681_last_tick_3681
teacher_forced_success         false
usable_tick_trajectory         false
chosen_seed                    28
source_deployment_actions      78
accepted_deployment_actions    48
collected Tick                 10..3681，共 3672 Tick
source/action boundary         3681 -> 3682
last native Tick               3681
crowns                         [1, 0]
commands_allowed               false
command_gate_code              3
native phase                   battle=4, logic=3, substate=1, flag_1e9=0
```

该结果与历史 v9 在 Tick 3681 观察到的 hard gate 一致，但不再抛 generic host error；
同时明确证明 T+1 命令从未执行，所以该场不能归入 offset 1 的 action rejection。
deployment `execute_deployment_trace` 与 ability-aware 通用 `execute_plan` 两条路径均
得到相同 failure/seed/78→48 动作计数和 3,672 Tick 前缀；后者确认
`tick_store_entry=null`。

## 回归测试

- compact decoder/accumulator：只在显式允许时读取 nonterminal freeze，保留 complete
  prefix，统计 incomplete suffix，禁止 suffix 后恢复；
- deployment runner：返回完整 prefix states、seed、动作计数和 freeze audit；
- ability runner：同样返回 structured result/diagnostic，不写 shard、不猜技能实体；
- final fence：调用路径未改，继续采用原有终局诊断规则。
