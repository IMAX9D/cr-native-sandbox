# 自博弈联赛网页监控

这是只读监控入口，不是训练启动器，也不代表新模型PPO/对手池已上线。

## 启动

在项目根目录执行：

```sh
python -m training.league_dashboard --port 19731
```

也可以运行根目录 `LEAGUE_MONITOR.cmd`。浏览器打开 `http://127.0.0.1:19731/`。无数据源时只显示已确认的BC step1037042身份和明确的待接入状态，不填假Elo。页面的“查看演示”使用合成数据，全页面保留演示提示；它不写入真实数据源。

接入真实快照：

```sh
python -m training.league_dashboard --snapshot /path/to/run/league-monitor.json
```

Linux也使用相同命令，只监听127.0.0.1。远程查看使用SSH转发，例如 `ssh -L 19731:127.0.0.1:19731 ...`；不默认暴露到公网，无账号/认证功能。服务没有任何训练修改、晋级、关机或权重下载接口。

## 页面

- 水平轨迹与策略名册：展示上游提供的Elo，不把训练胜率或压测速度转换成Elo。曲线仍是整个评估集口径，不随矩阵卡组筛选改变。
- 克制矩阵：行策略对列策略得分率 `(wins + 0.5*draws)/n`，不是只计胜场的胜率。正式评估与训练对局分开；可按卡组筛选；缺失为“—”，不足20局加虚线提示，20局不等于统计显著。
- 分支谱系：只显示亲缘关系，不把枝条位置当实力或策略聚类。
- 行为分化：每分钟部署与防守响应二维散点，不命名未经证实的策略类型，也不生成伪造综合分。
- 模型×卡组资质、计划/实际对手来源比例、事件记录、有效样本率、队列年龄、长局丢弃数。
- 心跳超过30秒标记过期；无心跳不宣称运行中。坏快照返回503，页面清空统计并显示异常，不用旧数据冒充实时。

## 发布契约

使用 `training.league_dashboard.write_snapshot(path, payload)` 原子发布小型聚合快照。每3秒轮询，未变化文件仅stat并复用已解析结果；限制2MB、64模型、20,000条聚合对战、每模型2,000个历史点。长期训练应在生产端维护增量统计和历史降采样，不让网页每次扫描所有对局JSONL。

当前没有自动读取旧177M训练日志，也没有接到尚未完成的新PPO采集器。训练/评估模块在后续闭环实现时调用发布函数即可；严禁把不同模型、引擎或评估协议的旧记录混成同一新联赛。

最小结构：

```json
{
  "schema_version": 1,
  "run_name": "自博弈实验名称",
  "status": "stopped",
  "generated_at": "2026-09-11T12:00:00Z",
  "heartbeat_at": "2026-09-11T12:00:00Z",
  "evaluation_id": "固定锚点与评估协议版本",
  "models": [
    {"id": "bc", "name": "BC基模", "role": "base", "parent": null, "elo": null, "history": []},
    {"id": "p1", "name": "候选", "role": "candidate", "parent": "bc", "elo": null, "history": []}
  ],
  "matchups": [],
  "qualifications": [{"model": "p1", "deck": "野猪循环", "status": "pending"}],
  "events": [],
  "telemetry": {}
}
```

`models[].history`是`{step, elo}`列表；可选行为字段为`deploys_per_minute`和`defense_response_seconds`，不存在就留空。

`matchups`项为`{a,b,deck,scope,wins,draws,losses}`。scope只能是`evaluation`或`training`；统计从a的视角记数。每一无序模型对、卡组、scope只发布一行，双方出生侧结果在上游统一归并；不得再额外发布反向行造成双计数。网页自动生成反向得分率。带评级/对战的快照必须声明evaluation_id；上游负责该评估集内的可比性。

`opponent_sampling`可包含`planned:{latest:0.4,history:0.4,base:0.2}`和`actual:{latest:实际次数,history:实际次数,base:实际次数}`。actual必须非负。资质status为qualified/pending/failed。

`telemetry`可包含`unique_learner_samples_per_second`、`queue_seconds`、`discarded_long_games`。它们均由上游计算，监控台不会将对手帧、padding、burn-in或重复epoch算成有效学习侧样本。

## 统计边界

本版不自动计算Elo、置信区间、显著性晋级或非传递策略聚类。Elo跨度只是已报告评分的最大最小差，不是置信区间或泛化能力指标。矩阵支持检查循环克制，但不自动证明存在某种策略类型。

跨卡组总体得分可能受抽样构成影响，不能直接比较不同对手分布。正式晋级仍应采用配对种子、换边、卡组分层与预设的统计检验。只有对战聚合数时，无法恢复配对bootstrap，应保留上游逐局评估记录。

设计参考：DeepMind [AlphaStar联赛方法说明](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)（2019-10-30），重点借鉴遗忘、非传递克制与多策略联赛的监控需求；本页面不是AlphaStar原版界面或其训练系统复刻。

## 验证

```sh
python -m unittest tests.test_league_dashboard tests.test_training_dashboard -v
```

覆盖空数据、演示隔离、重复模型对、非有限数、谱系循环、坏结构、原子发布、读取失败、只读HTTP和Host校验。页面不加载外部脚本或字体；快照文本经过HTML转义。
