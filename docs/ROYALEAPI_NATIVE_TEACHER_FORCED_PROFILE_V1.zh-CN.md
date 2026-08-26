# RoyaleAPI Native Teacher-Forced Profile v1

## 正式契约

当前生产专家回放统一使用：

```text
profile name    = royaleapi_native_teacher_forced
profile version = 1

source marker tick       = T（RoyaleAPI time_raw，永久保留）
native execution boundary = T + 1
```

profile v1 不改写 JSON、不把 `time_raw` 存成 `T+1`，只在把专家动作送入 libg 时
选择下一条原生命令消费边界。

共享常量位于：

```python
expert_v1.native_profile

ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_NAME
ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_VERSION
ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
```

所有未来批处理入口必须引用该共享常量，不能再复制字面量 `0` 或 `1` 作为默认值。

## 生产入口

以下入口默认均为 offset 1：

```text
scripts/pilot_expert_native.py
scripts/pilot_expert_native_abilities.py
scripts/replay_expert_native.py
```

底层执行函数同样默认采用 profile v1：

```text
execute_deployment_trace(...)
execute_plan(...)
execute_ability_task(...)
```

CLI 仍接受显式诊断覆盖：

```powershell
--action-execution-tick-offset 0
```

显式 0 只用于复现历史 A/B，不属于 profile v1 的生产执行语义；输出 metadata 会将
`diagnostic_override` 标为 `true`。只有 0 和 1 合法，其他 offset 继续 fail-closed。

## Provenance 与持久化

每个 summary、结果以及 Tick Store episode metadata 都保存：

```json
{
  "native_teacher_forced_profile": {
    "name": "royaleapi_native_teacher_forced",
    "version": 1,
    "default_action_execution_tick_offset": 1,
    "effective_action_execution_tick_offset": 1,
    "source_marker_tick_immutable": true,
    "source_marker": "royaleapi_time_raw_T",
    "native_execution_boundary": "source_tick+1",
    "diagnostic_override": false
  }
}
```

Tick Store 全局 `manifest.json` 还会在 `metadata` 下重复保存同一 profile，从而无需
打开任意 episode 就能判断整批数据的执行语义：

```json
{
  "metadata": {
    "native_teacher_forced_profile": {
      "name": "royaleapi_native_teacher_forced",
      "version": 1
    }
  }
}
```

实际对象包含上面完整的 profile 字段，不只 name/version。

同时继续保留便于旧消费者读取的字段：

```text
action_execution_tick_offset = 1
action_tick_provenance = source marker is immutable RoyaleAPI time_raw T;
                         native execution boundary is source_tick+1;
                         source label unchanged
```

## 兼容边界

- `action_execution_tick(source_tick, offset)` 仍显式接受 0/1；它是纯映射函数，不
  隐藏选择来源。
- 历史 v7/v8/v9/v10 输出不回写 profile 字段，也不重写 manifest；旧证据保持不可变。
- 新 reader 不应仅凭缺失 profile 猜测 offset。旧数据必须读取其原有
  `action_execution_tick_offset` 和 provenance。
- 一批 Tick Store 内不允许混用 effective offset；入口 summary 和 global manifest
  给出批级契约，每个 episode metadata 负责逐局复核。
- 已经以显式 `--action-execution-tick-offset 1` 启动的进程继续按启动时加载的参数
  运行，不需要中断或重启。

## 为什么正式选择 T+1

正确坐标后的固定 100 场 v9/v10 审计中，排除一场到不了 T+1 边界的原生逻辑冻结
诊断后，共有 99 场相位可比较：

| 指标 | T | T+1 |
|---|---:|---:|
| teacher-forced success | 83 | 89 |
| code13 | 6 | 0 |
| code4 | 10 | 10 |

- 83/83 原成功全部保留；
- 6/6 code13 全部转成 success；
- 10 个 code4 tag 集合完全稳定；
- 新增 6 场终局为 4 match、1 mismatch、1 missing；旧成功没有 match→mismatch。

详细证据见 `NATIVE_COORDINATE_PHASE_FINAL_100.zh-CN.md`。

## 剩余限制

profile v1 只解决统一命令相位，不宣称 action-only JSON 能复原服务端隐藏状态。
v10 按时长成功率为：

```text
<=180s     13/13
181–240s   52/55
>240s      24/32
```

剩余失败集中在长局，符合没有中局状态锚点时生成态误差持续累积的特征。后续应研究
可验证的周期状态锚点或分段 replay；不能再用另一个全局 Tick offset 掩盖长局漂移。
