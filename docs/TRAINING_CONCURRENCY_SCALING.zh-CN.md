# CR-Native-Core 训练并发 Scaling Sweep

日期：2026-08-23

## 1. 结论

```text
Recommended Training Configuration:

AVD count = 2
Worker count = 8
Inference batch = 16（最大；完整终局实测平均 12.49）
vCPU per AVD = 4
Worker per AVD = 4
Transport = Direct
Native Tick = 20 Hz
```

当前机器的正式甜点位是 **2 AVD / 8 Worker**。它相对1 AVD把含PPO的有效
training throughput从288.35提高到393.22 steps/s（`+36.4%`），所有Worker、
RPC、episode和轨迹验证均为0错误。继续增加50% Worker到3 AVD / 12 Worker，
training throughput只增加`6.1%`，同时系统可用内存最低跌到约36 MiB，RPC p95
和slowest-worker barrier明显恶化。因此不运行4 AVD / 16 Worker。

2 AVD也不是“资源宽松”配置：PPO阶段系统可用内存最低约0.30 GiB。长时间训练
前应关闭额外的大内存应用；若必须与其他重负载程序并行，回退到1 AVD / 4
Worker更稳妥。

推荐启动命令：

```powershell
.\scripts\start_training.ps1 -Avds 2 -Workers 8
```

## 2. 冻结条件与方法

所有档位保持以下内容完全相同：

- 原版`libg.so 15.535.29 x86_64`；
- 20 Hz native tick、双方每tick可决策；
- 固定八卡、标准1v1；
- compact training transition、原生mask cache；
- persistent TCP、Emulator direct transport；
- global batched inference、CUDA Graph；
- 当前Recurrent PPO、网络、Reward与超参数；
- seed `424242`，`max_ticks=7200`；
- 每个Worker采一场完整原生终局，然后执行一次PPO update。

这不是`nativeStep(N)` microbenchmark。每个结果包含：observation、mask、全局
policy inference、joint action、native tick、trajectory持久化、PPO backward/
update和checkpoint重载验证。

每个AVD固定4 vCPU、4 Worker。档位依次为1/2/3 AVD；3 AVD触发停止条件后，
按预先约定没有创建或运行第4个AVD。

实测主机为Intel Core i5-13600KF（14核/20线程）、NVIDIA GeForce RTX 3080
10 GiB、约31.8 GiB系统RAM，Windows桌面常驻负载保持开启。结果代表这台机器
当前真实训练使用状态，不是清空所有桌面进程后的实验室上限。

## 3. 主结果

| AVD | Worker | Batch最大/平均 | Env steps/s | 含PPO steps/s | Decisions/s | Episodes/h | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 8 / 7.83 | 373.04 | 288.35 | 746.08 | 177.07 | 通过 |
| 2 | 8 | 16 / 12.49 | 526.99 | **393.22** | 1053.98 | 300.57 | **推荐** |
| 3 | 12 | 24 / 17.76 | 572.34 | 417.12 | 1144.68 | 336.29 | 边际收益6.1%，饱和 |
| 4 | 16 | 32 / — | — | — | — | — | 达停止条件，未运行 |

相邻档位变化：

| 扩容 | Worker变化 | Env吞吐变化 | 含PPO吞吐变化 | Episodes/h变化 |
|---|---:|---:|---:|---:|
| 1→2 AVD | +100% | +41.3% | **+36.4%** | +69.8% |
| 2→3 AVD | +50% | +8.6% | **+6.1%** | +11.9% |

主要选择标准是含PPO training steps/s。3 AVD绝对值虽比2 AVD高6.1%，但为此
增加了50% Worker和一个完整QEMU，违反了预设的20%边际收益门槛，并耗尽内存
安全余量。因此甜点位取2 AVD而不是3 AVD。

## 4. 阶段耗时

| AVD | 环境墙钟 | PPO update | 迭代墙钟 | Inference ms/round | Native transition ms/round |
|---:|---:|---:|---:|---:|---:|
| 1 | 62.86 s | 18.46 s | 81.32 s | 2.85 | 3.32 |
| 2 | 71.49 s | 24.32 s | 95.82 s | 3.11 | 4.16 |
| 3 | 93.62 s | 34.84 s | 128.46 s | 3.72 | 5.27 |

batch 8→16仍能提高总吞吐，但batch 16→24没有继续提高GPU采样利用率，反而
增加推理单轮时间。Native transition也随AVD数增加而变慢，说明主机调度和
slowest-worker barrier已经主导扩容收益。

## 5. RPC与同步Barrier

### 5.1 RPC

| AVD | RPC p50 | RPC p95 | RPC p99 | RPC失败率 |
|---:|---:|---:|---:|---:|
| 1 | 1.92 ms | 3.69 ms | 6.11 ms | 0% |
| 2 | 2.31 ms | 3.99 ms | 6.34 ms | 0% |
| 3 | 2.98 ms | 5.49 ms | 6.61 ms | 0% |

3 AVD的单次最大RPC达到354 ms；虽然p99仍为6.61 ms，这类长尾会被同步collector
的全Worker barrier放大。

### 5.2 每轮Worker transition

| AVD | Fastest均值 | Median均值 | Slowest均值 | Slowest p95 / p99 | Barrier wait均值 / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.81 ms | 2.40 ms | 3.05 ms | 4.86 / 7.30 ms | 1.24 / 2.19 ms |
| 2 | 1.67 ms | 2.65 ms | 3.82 ms | 5.63 / 11.63 ms | 2.15 / 3.89 ms |
| 3 | 1.79 ms | 3.20 ms | 4.85 ms | 6.96 / 12.00 ms | 3.06 / 4.99 ms |

fastest Worker基本不变，median、slowest和barrier wait持续上升。这是“所有
Worker等待最慢Worker”的直接证据，而不是由游戏逻辑tick变慢造成的推测。
逐round原始数据保存在每个run的`evaluations/barrier-000001.npz`，包含：

```text
wave
round_index
active_workers
policy_batch_size
fastest_seconds
median_seconds
slowest_seconds
```

## 6. CPU、GPU与内存

下表为sampling阶段；CPU是整机平均，QEMU RSS和Worker RSS是所有活动AVD/
Worker合计，VRAM为sampling阶段最大值。

| AVD | Host CPU均值/p95 | GPU均值/p95 | VRAM max | System RAM均值 | System RAM可用min | QEMU RSS | Worker RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.4% / 92.6% | 34.5% / 41.0% | 3014 MiB | 25.45 GiB | 3.77 GiB | 4764 MiB | 1880 MiB |
| 2 | 45.4% / 55.0% | 34.3% / 41.6% | 3139 MiB | 29.01 GiB | 2.14 GiB | 9384 MiB | 3741 MiB |
| 3 | 49.7% / 59.4% | 34.1% / 40.6% | 2771 MiB | 31.39 GiB | **0.036 GiB** | 13536 MiB | 5619 MiB |

Windows按QEMU进程统计的AVD CPU（100%表示一个host逻辑核）：

| AVD | QEMU CPU合计均值 | 相对已分配vCPU容量 |
|---:|---:|---:|
| 1 | 61.83% | 15.46% |
| 2 | 103.00% | 12.88% |
| 3 | 125.61% | 10.47% |

每个guest的最低`MemAvailable`仍约1.3 GiB，guest swap为0；耗尽的是Windows
系统内存，而不是单个AVD内部内存。Host CPU/GPU没有达到100%并不代表还能有效
扩容：3 AVD已经受RAM压力、RPC长尾和同步barrier限制，GPU sampling利用率也
没有随batch增大。

PPO阶段资源：

| AVD | Host CPU均值 | GPU均值 | VRAM max | System RAM可用min | QEMU CPU合计 |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.1% | 66.5% | 4951 MiB | 2.50 GiB | 10.24% |
| 2 | 41.8% | 75.3% | 4908 MiB | **0.30 GiB** | 14.41% |
| 3 | 46.1% | 73.0% | 4668 MiB | **0.035 GiB** | 24.95% |

QEMU CPU在sampling→learner期间显著下降，确认当前Worker在PPO update时基本
空闲；但2 AVD的learner内存余量已经只有约0.30 GiB。因此本轮没有冒险实现
double-buffer。将来只有在降低trajectory/host RAM占用并严格冻结policy version
后，才值得测试rollout A与learner B重叠。

## 7. 正确性与稳定性

三档合计：

- 24/24场原生正常终局，0截断；
- 48份agent trajectory，合计229,420 agent steps；
- 每步card mask、position mask合法；
- observation、log-prob、value、reward全部finite；
- 每条轨迹最终`done=true`；
- 三个checkpoint模型finite，optimizer均可恢复；
- Worker failure rate = 0；
- RPC failure rate = 0；
- episode failure rate = 0；
- seed数量与episode数量一致，没有状态串局证据。

首次从1 AVD扩到2 AVD时，第二实例在旧120秒cold-ready上限内未完成DataTables
loading，驱动器正确fail-closed且未产生训练数据。该setup failure保留在Sweep
JSON中；将部署上限独立调整为300秒后，同一2 AVD档完整通过。这是冷部署运维
成本，不计入训练墙钟，也没有被当作RPC/episode成功样本。

## 8. 原始证据

Sweep汇总：

```text
D:\AI_data\cr-native-core\scaling-sweeps\concurrency-sweep-20260823\sweep-summary.json
```

同目录保存每档：

```text
tier-N-result.json
tier-N-resource-samples.json
tier-N-resource-summary.json
tier-N-training.log
tier-N-workers-ready.json
```

训练run：

```text
D:\AI_data\cr-native-core\training\runs\concurrency-sweep-20260823-1avd-4worker
D:\AI_data\cr-native-core\training\runs\concurrency-sweep-20260823-2avd-8worker
D:\AI_data\cr-native-core\training\runs\concurrency-sweep-20260823-3avd-12worker
```

4 AVD档没有数据是预设停止规则的结果，不是缺失测量。

复现Sweep前安装只读资源监控依赖：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe -m pip install `
  -r requirements-scaling.txt
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\run_concurrency_sweep.py
```

## 9. 最终建议

日常最大吞吐训练：

```ini
AVDs=2
Workers=8
WorkersPerAVD=4
InferenceBatchMax=16
Transport=direct
CUDA_Graph=enabled
NativeTickHz=20
```

若同时运行其他大内存程序，使用：

```ini
AVDs=1
Workers=4
InferenceBatchMax=8
```

下一阶段的优先级不是增加第3/4个AVD，而是：

1. 降低PPO/trajectory的Windows RAM峰值；
2. 研究去除slowest-worker同步barrier的异步collector；
3. 内存余量恢复后，再做冻结policy version的double-buffer实验。

本结论只针对本次实测机器和当前八卡网络/PPO。网络或trajectory结构改变后应
重新执行Sweep，不能仅按CPU/GPU利用率外推。

对应实现提交：`e397159`（多AVD编排、资源监控、逐round barrier和Sweep驱动）。
