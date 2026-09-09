# BC 双方公开出牌历史

`--history-length 4` 为己方和敌方各保留最近 4 次普通出牌，最近的排在前面。
默认 0 不启用。与 `--spatial-type-dim` 独立，本轮建议关闭空间分支进行对照。
历史特征：卡牌 16 维 embedding、格子归一化 x/y、log1p(距今秒数)/log(301)、
事件存在标记、内容已知标记。8 个事件拼接后经 64 维 MLP 投影到 LSTM 输入维度，
与当前局面融合。在词表 181、LSTM 512 下新增 46,992 参数。

只支持 BC 训练和离线评估。历史模型调用 forward_stream 会明确拒绝，避免在线缺历史时
悄悄使用全零输入。未实现在线 BC 对战或 PPO 历史维护。

## 命令

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_fixed.py --history-length 4 --spatial-type-dim 0 \
  --run-dir ~/cr-data/runs/hokoff-fixed-p4-history4 \
  --width 256 --hidden-size 512 --batch-size 32 --workers 4 \
  --device cuda --precision fp16 --hours 2
```

新目录从零训练；同参数重复执行自动从 last.pt 续训。不可将旧权重直接 resume 为历史模型。
关闭历史的旧 checkpoint 仍兼容。若要对照，使用另一新目录、`--history-length 0`，
其余参数保持相同。需要比较固定更新数时，用 `--steps 10000` 替代 `--hours 2`。
已有 8 通道空间类别分支可组合启用，但不要在首轮历史对照中同时改变它。

离线诊断：

```bash
python diagnose_hokoff_fixed.py \
  --checkpoint ~/cr-data/runs/hokoff-fixed-p4-history4/last.pt \
  --mode ap --device cuda
```

## 数据与因果性

复用已解压 native-bc-v1 和 fixed4 决策缓存；无需原始 tick 文件或重放。
首次访问 shard 时构建紧凑事件索引，原子写入决策缓存的 `public-history-v1/`。
缓存键绑定历史语义、源 manifest、shard 路径和 metadata 哈希，后续访问/续训直接读取，
避免随机取样反复扫描全 shard。支持多个数据 worker 同时构建；不存储逐帧稠密历史，
不修改源文件。历史查询按帧批量计算，模型、监督、采样顺序和旧 checkpoint 均不改变。
首次访问尚无缓存的 shard 仍有构建开销；需要重启训练进程使用新代码，可按原命令续训。

从原始逐 tick play_now 及动作前 hand_tokens 提取出牌，使用真实源 tick，绝不从
fixed4 提前对齐的 label_rows 构造历史。查询严格要求 event_tick < query_tick。
双方序列配对，敌方落点转换到当前玩家视角；不会使用敌方未公开手牌或未来动作。
已识别的技能事件暂不纳入；身份或标签不完整的已记录执行占一个未知槽位，不能伪装为无出牌。
无法恢复源数据里完全未记录的事件。历史在已保存的玩家序列边界内建立，边界之前未知；
空槽代表没有可用历史记录，不证明真实对局里从未出过牌。

2026-09-09 对本机训练/留出各前 5 个 shard 的抽样：740 个玩家序列，7425 个历史事件，
均具备完整标签；仅为这 10 个 shard 的检查，不代表全量覆盖。

测试覆盖双方与坐标翻转、未知标记、严格过去、fixed4 提前标签隔离、反向传播、
历史+空间组合、断点续训和离线 AP 评估；没有证明实战收益。
