# Native teacher-forced pilot 重启现场

更新时间：2026-08-26（系统重启前）

## 已恢复的运行路径

- 标准容器：`D:\Codex\toolchains\android-sdk`、API 31 x86_64 AVD
  `D:\AI_data\android\avd\royale_worker_api31.avd`；
- 原生宿主：`app_process -> royale.nativehost.JniHost serve-direct -> libg.so`；
- 当前 Windows 会话的标准 Emulator 未能启动，明确错误为
  `Android Emulator hypervisor driver is not installed on this machine`；
- 在不重启系统、不修改游戏逻辑的前提下，临时使用本机 MuMu Android 12
  x86_64 作为 ABI/进程容器，成功启动同一套无 Surface 原生宿主；
- TCP `37031..37034` 四个隔离 Worker 在重启前均通过 `ping`。

MuMu 只是 Android 容器；执行战斗的仍是工程内相同哈希的
`lifecycle-probe.jar`、`libnative_core_probe.so` 和冻结的 `libg.so`。

## 单场实测

样本：`000YL0G9L0QG`

- 源动作：78；
- libg 接受动作：78/78；
- 推进：5,430 native ticks；
- 墙钟：0.6181864 s；
- `step` 耗时：0.3961193 s；
- `observe` 耗时：0.1386042 s；
- `joint_act` 耗时：0.0761225 s；
- 全链推进约 8,784 tick/s，纯 `step` 部分约 13,708 tick/s；
- 源时长围栏处未出现原生终局，记录为
  `native_terminal_missing_at_source_end`。这不否定 teacher-forced Tick
  状态生成，但证明终局皇冠不能作为动作接受的硬门槛。

证据：

- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot\single.json`
- SHA-256：`B40915CB1257027E40B95B3D083620D5599E3B5709F4D5D9F0D750DD0F516DC8`

## Seed 差分实测

对同一场、同一兼容逻辑手牌/队列，分别使用 seed
`1 / 2 / 424242 / 2147483646`。每个 seed 先校准 libg shuffle，再双射重排
八个 native deck slot。

结果：

- 四次均接受 78/78 动作；
- 四次均推进到相同 Tick；
- 将 native deck slot 反映射回逻辑卡索引后，所有决策 Tick 的公开状态完全相同；
- 不同 seed 的差异仅出现在 shuffle 对 deck slot 的排列和含 RNG 的审计 hash。

证据：

- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot\seed-diff-single-logical.json`
- SHA-256：`2F608CD694A685FAC3DF7619FA13D55E83EFB72D0AA4BBABC8D86C7348089221`

这只证明当前实测样本，尚未扩大成多场统计。

## 已完成代码修复

- `observe_train_v1` 的 player compact JSON 补回
  `next_deck_index/refill_timer`；
- Python runner 对尚未滚动升级的旧 Host 显式写 `-1`，不会猜测字段；
- JNI bridge 已完成 x86_64 Android 编译；
- `tests.test_expert_native_replay_plan`：7/7 通过。

## 重启后的第一条命令

```powershell
cd D:\Deepseek\CR-Native-Core
D:\AI_data\runtime\venv\Scripts\python.exe -m native_core.worker start --workers 4 --transport direct
```

若标准 Emulator 仍报告 hypervisor 未加载，不安装或修改系统组件；继续使用已经
验证的 MuMu 容器恢复 `serve-direct`。

## 未完成

1. 将新增 compact per-Tick trace 接口接入 Host，避免逐 Tick RPC；
2. 运行至少 100 场、4 Worker 动态任务队列 pilot；
3. 统计动作接受率、首个拒绝原因、终局一致率、每 Tick 数据完整性和吞吐；
4. 将 teacher-forced success 与终局皇冠诊断分开；
5. 扩大 seed 差分到多场，并按逻辑卡索引规范化比较。

