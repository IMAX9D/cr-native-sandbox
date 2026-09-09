# fixed4 在线 BC 对战

入口是仓库根目录 `run_hokoff_fixed.py`。连接已启动的一个原生 Worker，同一份正式模型控制双方，每 4 tick 观察/决策一次，每侧最多一个操作。可以等待、下牌或点击英雄技能；技能没有位置参数。

这一步已经接通 BC 对战，不包含 PPO 更新；输出日志也不是完整的 PPO 训练轨迹。

## 启动

```bash
cd /home/lenovo/gh/cr-native-sandbox
conda activate r2dreamer
python run_hokoff_fixed.py
```

默认：

- Worker：`127.0.0.1:39031`，本地目录 `/data/local/tmp/cr-native-direct-0`。
- checkpoint：`/home/lenovo/cr-data/runs/hokoff-fixed-p4/last.pt`。
- 词表/特征清单：`/home/lenovo/cr-data/expert-dataset/native-bc-v1/manifest.json`。
- 卡组/地图：Worker 的 `bootstrap-replay.json`。只使用初始化配置，清空副本的专家 `cmd`，不修改原文件。
- CPU 推理，4 个 CPU 线程，1 局。每局原生 reset 后推进到 tick100，从零初始化双方 LSTM 和对手已公开卡牌历史。
- 使用确定性、带合法掩码的动作选择，时机阈值 `0.5`。达到阈值后，在当前合法操作中选择；无合法操作时等待。
- 最多 2500 个决策，超过上限记录失败/部分对局，不冒充正常终局。
- 输出：`/home/lenovo/cr-data/runs/hokoff-fixed-live-时间戳/`。

连续两局、保存完整观察用于诊断：

```bash
python run_hokoff_fixed.py --episodes 2 --save-observations
```

使用含英雄/进化的现有卡组：

```bash
python run_hokoff_fixed.py --replay examples/top-training-deck-control.json
```

可指定 `--checkpoint`、`--port`、`--worker-dir`、`--replay`、`--seed`、`--timing-threshold`、`--device cuda`。`--output` 必须是尚不存在的新目录。多槽时同时指定对应端口和 Worker 目录。本入口每次控制一个 Worker，两侧独立状态。

按 **Ctrl+C** 保存中断状态并退出，退出码 130；不会自动续跑，也不会关闭用户已启动的 Worker。

## 实现与边界

`hokoff_model/train_fixed.py` 只增加 fixed 模型的 `forward_stream`，复用原有编码器、时间特征、LSTM 和动作头；没有改 checkpoint 张量、训练损失或默认训练参数。旧 decision/delay 模型仍不允许通过此入口上线。

`hokoff_model/live.py` 完成：

1. 校验固定模型架构、checkpoint 对应的数据清单和词表、特征维度。
2. 复用 `NativeObservationEncoder`，只把当前己方私有信息和双方公共战场信息送入模型。对手隐藏手牌/费用不会进入己方输入；已公开卡牌在对手实际成功下牌之后更新。
3. 两侧各有独立 LSTM 状态；第一个观察的 elapsed=0，之后严格为4。拒绝漏重置、重复 tick 或改变观察节奏。
4. 下牌费用取原生 `probe_grid.card_cost_raw`，位置规则复用数据编译时的 `derive_deployment_rows`。静态选择按当前卡组槽缓存，Mirror/动态选择重新查询；每局清空缓存，塔变化重新派生位置掩码。
5. 技能使用当前观测中的实例 key，并检查原生 available、技能费用和当前费用。不给技能生成坐标。
6. 将双方操作一次提交给 Worker，并检验每个原生回执。无动作也推进世界。若原生终局锁存稍晚，只用有限次空操作补齐 tick，不重复提交模型动作。

网络使用已有持久 TCP 客户端。发生含糊的写请求超时不会自动重放命令。`--worker-dir/libg.so` 校验冻结 x86_64 版本；这属于本地文件校验，不是远程 Worker 的二进制证明。

## 日志

控制台每项指标独占一行。主要看 `accepted_actions`、`attempted_actions`、`skill_decisions`、`wait_decisions`、`native_ticks_per_second`。

输出目录包含：

- `contract.json`：模型/词表/本地 libg 摘要、阈值、运行参数。
- `decisions.jsonl`：观察 tick、真实推进 tick、双方预测、执行命令、原生回执、终局；先落盘回执再检查非法执行。
- `episode-XXX-summary.json`：每局操作数量、原生结果、CPU 推理/掩码与原生推进耗时。
- `summary.json`：正常结束/失败/中断、完成局数、checkpoint 是否保持不变。
- `observations.jsonl`：显式开启 `--save-observations` 时才保存当前帧，用于无 UI 检查。
- `error.txt`：失败原因及 traceback（失败时）。

原生 `terminated` 才算正常完成；本地决策上限、原生 `truncated`、RPC 错误不作为胜负样本。

## 2026-09-08 验证结果

正式模型 step534691，CPU 推理，4 线程：

| 测试 | 结果 |
|---|---|
| 基础卡组，seed42，threshold0.5 | 1263 次决策，59/59 次操作原生接受，tick5148 正常终局 |
| 同一进程下一局，seed43 | 1232 次决策，63/63 次操作原生接受，tick5026 正常终局 |
| 含进化/英雄的控制卡组，seed44 | 896 次决策，17/17 次操作原生接受，tick3681 正常终局 |
| 受控技能接口测试 | 部署黄金圣骑后，以实例5000006点击技能，原生接受，消耗1费，不含坐标 |

前三局都是模型自主决策；技能测试使用**显式强制 logits**覆盖技能执行路径，不能作为模型已经学会释放技能的证据。自主对局中没有释放技能。

基础两局每局约8.5秒，约582原生tick/秒。这个速度是当前单 Worker、小模型 CPU 推理的实测，不是 GPU 批量吞吐承诺。正式 checkpoint 摘要保持不变。

结果目录：

- `/home/lenovo/cr-data/runs/hokoff-fixed-live-20260908-actions`
- `/home/lenovo/cr-data/runs/hokoff-fixed-live-20260908-hero`
- `/home/lenovo/cr-data/runs/hokoff-fixed-live-20260908-ability-protocol`

另有 threshold0.7 的先行测试，两局均全程等待。threshold0.5 的基础卡组也到约tick2896才首次下牌。这表明当前时机策略仍存在明显问题；接口跑通不代表策略强，也不能根据少数同模型对战得到胜率结论。默认0.5只作为在线基线，不声称已经完成阈值校准。

PPO 现已提供独立入口 `train_hokoff_ppo.py`，包括随机动作采样、价值网络和 IL-KL 控制，见 [PPO 训练说明](PPO_TRAINING.zh-CN.md)。本 BC 入口仍是阈值贪心评估，不生成 PPO 训练轨迹。
