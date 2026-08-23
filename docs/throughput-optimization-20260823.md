# Self-Play 吞吐优化报告（2026-08-23）

## 1. 约束与测量方法

本轮保持原版 `libg.so`、20 Hz 原生 tick、AI 每 tick 双方可决策、固定八卡、
Recurrent PPO、Reward 和网络结构不变。没有实现 action repeat、10/5 Hz、
`WAIT(N)`、新算法或 Python 战斗规则。完整 `observe`、`trace` 和 GUI/debug API
继续保留。

增量优化使用固定 horizon 真实闭环：2 Worker，每场从 tick 100 到1100，合计
2000 environment steps / 4000 policy decisions，包含观测、mask、编码、CUDA
推理、联合动作、原生 tick、next state、轨迹保存和一次 PPO update。该基准不是
`step(5000)` microbenchmark。

最终另跑2 Worker×4场、`max_ticks=7200`，要求原生终局、完整轨迹和PPO更新。

## 2. 基线 Profile

固定基线 `bench-baseline-fixed`：

| 指标 | 数值 |
|---|---:|
| environment steps | 2000 |
| 环境采样吞吐 | 109.05 steps/s |
| 含PPO总吞吐 | 101.55 steps/s |
| 环境采样墙钟 | 18.34 s |
| PPO update | 1.35 s |
| RPC calls | 2038 |
| RPC response | 25.72 MB |
| RPC total（双Worker累加） | 16.17 s |
| TCP connect（双Worker累加） | 6.95 s |
| mask build | 4.85 s |
| inference（每Worker batch=2，累加） | 9.28 s |
| encoding | 0.30 s |
| trajectory append | 0.0075 s |

源码确认：每次RPC都新建连接；每tick已经只有一个主要`joint_transition`，但它
返回完整debug observation；`probe_grid`每局每方每卡首次入手才缓存，真正的
mask热点是Python每tick重复构建镜像/塔占地/领地层；两个Worker分别做batch=2。

## 3. 分项优化结果

| 阶段 | 环境 steps/s | 相对前一步 | 相对基线 |
|---|---:|---:|---:|
| 基线 | 109.05 | — | 1.00× |
| 持久TCP + `TCP_NODELAY` | 140.49 | +28.8% | 1.29× |
| compact training transition | 152.78 | +8.7% | 1.40× |
| 精确mask分层缓存 | 257.67 | +68.7% | 2.36× |
| 跨Worker全局batch | 291.31 | +13.1% | 2.67× |
| 最终确认复测 | 289.03 | -0.8% | 2.65× |

### 3.1 持久TCP

Java Host在同一Socket循环处理多行请求；Python `JsonLineClient` 在Worker生命
周期内复用连接，并用锁保证请求/响应配对。只读请求可自动重连一次；变更状态
的请求遇到不确定I/O失败时绝不自动重放。Java/Python两端启用`TCP_NODELAY`。

第一次只做长连接时吞吐反而降到34.6 steps/s：每请求出现约40 ms Delayed
ACK。加入`TCP_NODELAY`后才提升到140.49。该失败实验没有被隐去。

| RPC指标 | Before | Persistent |
|---|---:|---:|
| connect累计 | 6.95 s | 0.00069 s |
| RPC total累计 | 16.17 s | 6.72 s |
| transition累计 | 16.61 s | 7.35 s |

### 3.2 Compact training observation

新增 `observe_train_v1` 和 `joint_training_transition_v1`。训练fast path只返回：

- tick；
- `category/side/x/y/card_id/hp/max_hp/behavior_state`；
- 双方`elixir/elixir_raw/hand_deck_indices`；
- crown towers、reward、done、winner；
- coherent/entity count。

不再传路径、碰撞/攻击debug、effects/projectiles、RNG、state hash、完整cycle和
refill metadata。固定基准响应从23.61 MB降到8.08 MB（约-66%），transition
从7.35 s降到5.81 s，吞吐从140.49升至152.78。完整debug API未改变。

### 3.3 Mask分层缓存

最终动作集合仍为：

```text
最终mask = 每卡静态原生地形层 AND 当前塔状态公共动态层
```

静态层按`(side, deck_index, card_id, native_rows)`缓存原生18×32网格、四向
镜像交集；法术保持原生全场网格。动态层只在塔签名变化时重建4×4/3×3占地、
己方半场和左右塔毁后的口袋。手牌/圣水仍每tick检查。

mask build从4.83 s降到0.26 s，吞吐从152.78升至257.67。3360组单元测试
以及220组真实/合成/随机差分均逐bit等于旧实现。

### 3.4 跨Worker全局推理

旧流程是每Worker独立batch=2；新collector汇总全部active Worker的双方视角，
一次执行`batch=2×active_workers`，再分发recurrent hidden/action并并行执行原生
transition。Worker提前终局后从active batch移除。

固定基准 inference从7.45 s降到2.78 s，吞吐从257.67升至291.31。单Worker
确定性差分运行400步/方，旧/新collector的grid、scalars、privileged、两类
mask、动作、reward、done全部逐值一致。

## 4. 无收益实验

以下尝试已按实测撤销：

1. 批量回传GPU scalar以减少`.item()`：合法性测试通过，但291.31降到287.65。
2. 跳过`nativeStep`后的episode registry scan：语义一致，但两次只有215–218；
   恢复扫描后回到289.03，可能是缓存预热或原生状态依赖。
3. 整个模型放CPU（4线程）：环境约295.21，只比CUDA约2%，但PPO从约1.3 s
   增至14.0 s，不值得切换。

无收益代码均未进入最终主线。

## 5. 完整终局 Before / After

### 5.1 Before

运行：`native8-20260823T074419Z-a190a3ae`

| 指标 | Before |
|---|---:|
| episodes | 4 |
| environment steps | 14,539 |
| 环境采样墙钟 | 143.07 s |
| PPO update | 8.59 s |
| iteration wall | 151.67 s |
| 环境吞吐 | 101.62 steps/s |
| 含PPO总吞吐 | 95.86 steps/s |
| policy decisions/s（环境阶段） | 203.24 |
| episodes/hour（含PPO） | 94.95 |

### 5.2 After

运行：`throughput-after-full`

| 指标 | After |
|---|---:|
| episodes | 4 |
| 正常终局 / 截断 | 4 / 0 |
| environment steps / native ticks | 21,142 |
| policy decisions | 42,284 |
| 环境采样墙钟 | 99.74 s |
| PPO update | 12.79 s |
| iteration wall | 112.54 s |
| 环境/原生tick吞吐 | 211.96 steps/s |
| 含PPO总吞吐 | 187.87 steps/s |
| policy decisions/s（环境阶段） | 423.93 |
| episodes/hour（含PPO） | 127.96 |

完整终局提升：环境采样`2.09×`，含PPO总吞吐`1.96×`，policy decisions/s
`2.09×`，实测完整episodes/hour `+34.8%`。After四场平均更长，所以
episodes/hour提升小于按tick归一化的吞吐提升。

## 6. 最终阶段耗时

After完整运行合计21,142 environment steps：

| 阶段 | 累计耗时 | 归一化 |
|---|---:|---:|
| RPC total（双Worker累加） | 55.25 s | 2.60 ms/request |
| RPC receive/server+wire | 52.23 s | 2.46 ms/request |
| response bytes | 88.95 MB | 4.19 KB/request |
| encoding | 2.88 s | 0.136 ms/step |
| mask build | 2.02 s | 0.096 ms/step |
| global inference | 43.10 s | 约3.83 ms/vector round |
| parallel transition wall | 36.72 s | 约3.26 ms/vector round |
| reward | 1.29 s | 0.061 ms/step |
| trajectory append | 0.089 s | 0.004 ms/step |
| PPO update | 12.79 s | 0.605 ms/step |

`nativeStep`和compact observation在同一个transition RPC中执行。尚未分别插入
JNI内部wall timer，因此两者的独立耗时为 **Unknown**。独立只读探针在7实体
状态下测得compact observe约1.13 ms、full observe约1.43 ms；它不是完整训练
分布，不能替代上表。

## 7. 当前瓶颈与Worker建议

最终完整profile中，全局batch推理约占vector round的47%，两Worker并行native
transition约占40%；encoding、mask、reward和trajectory已是次要部分。

值得做独立4 Worker基准，但暂不修改默认值。batch 4→8可能继续摊薄推理，
独立`app_process`也可并行transition；但当前AVD只有4 CPU core，每服务占数百
MB RSS，扩展可能开始CPU/内存饱和。4 Worker实际吞吐为 **Unknown**，必须固定
horizon测aggregate/per-worker steps/s、RSS、CPU、p95 RPC和失败率后再决定。

## 8. 回归验收

通过：

- 单元测试4项：持久复用、并发不串包、只读重连、变更请求fail-closed；
- mask 3360组逐bit差分；
- compact opening/deployed/combat差分；
- scalar/vector collector 400步/方差分；
- 八卡原生动作证书；
- 时间/圣水/拼血证书；
- strict no-Surface direct core隔离冷启动3/3，哈希`5594aa3c81dc52fa`；
- 一键smoke：采样、PPO和检查点重载；
- After完整4场：4/4正常终局、0截断；
- 8份轨迹：finite、最终done、card mask和position mask通过；
- 模型张量finite、optimizer可恢复。

冷启动证书第一次在两个训练服务驻留时出现一次`loading_complete=false`；关闭
竞争服务后按隔离条件重跑3/3通过。这说明冷启动验收必须与常驻训练服务隔离，
也说明增加Worker前必须做资源压力测试。

## 9. 对应提交

| 提交 | 内容 |
|---|---|
| `3e33723` | 持久TCP、自动重连、fail-closed、profile |
| `58e9bf8` | compact native training transition |
| `fc6fee4` | 精确部署mask分层缓存 |
| `9921956` | 跨Worker全局策略batch |
| `ad5d0de` | fast-path与持久连接回归测试 |
