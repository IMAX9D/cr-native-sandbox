# R0 归因修复与准入状态（2026-09-14，v17）

本文件取代 `R0_CAPTURE_REPAIR_HANDOFF_20260914.zh-CN.md` 中"气球英雄归因未修复"的结论。
接手时以本文件为准；旧文档的其余边界、目录与工具说明仍然有效。

## 阶段结论（不用整体百分比）

- **单局语义闭环：完成。**
- **正式数据准入：未完成。**
- **跨机制规模验证：未完成。**

## 已修复：延迟生成对象的原生归属

主样本 `000YL00UC2R0`，此前 `lifecycle` 只有 1222/1714，唯一根因是 tick 3054 的区域对象
`3000005`（`data_id 856278373`，674 伤害）无父代。现有原生证据链：

```
3000005
  ├ factory_rva      0xf70fb8   组件构造器 0xf704c0 内的注册点
  ├ apply_caller     0xf83123   wrapper(obj, flag=500) 的调用方
  ├ context_caller   0xf821f8   生成例程 0xf82790 的调用方
  ├ context_key      4000264    BalloonHero_Skeletrooper_Projectile (side 0)
  └ context_owner    5000062    BalloonHero（该实体 +0x100 的 owner 槽，经 world/世代/同侧校验）
```

`context_key/context_caller` 是**创建上下文**：生成例程在实例化该对象时收到了创建者实体。
适配器只在 `context_caller` 在场、创建者已注册、且与对象同侧时才采用它；否则维持未归属并计数。
`3000004`（tick 2495）给出 `4000188 → 5000046 (BalloonHero)`，互为印证。

被实测否证的假设（保留在探针 bundle 中可复现）：`+0x100` 回退（`owner_alt` 全 0）、
调用方 RVA 判别（`caller_rva` 恒定）、对象内 45 个槽位反查（`probes` 全空）、
效果条目 source（add 时刻 0/992 命中）。该对象类**不存**实体引用，归属只能来自上下文。

## 验证矩阵

| 项目 | 结果 |
|---|---|
| 全段审计 | 12/12 域均 1714/1714，`failing_samples_total: 0`，`unattributed_total: 0` |
| 伤害归属 | `1:390` 674 点 → `credited_to: 5000062` |
| 负向对照 | 旧 v9 采集在新适配器下**仍为** 1222/1714、492 失败样本（修复是证据门控，不是放宽） |
| 回归测试 | 88 项通过（含创建上下文正例 + 异侧/无证据两个负控、采集入口 fail-closed 用例） |
| 整局编译 | `prefix_only: false`，4288 帧连续、无坏 tick、73/73 源标记；合同 SHA `ebfda45d…` |
| 严格 smoke | loss 1.5948194、梯度范数 7.5083175、权重前后变化、checkpoint 落盘 |
| 生产/诊断分层 | 生产层（4 钩子/18 字段）与诊断层（7 钩子/38 字段）**审计结果完全一致** |
| 谱系覆盖（本局） | 992 次注册去重为 497 个实体；**3M–5M 效果/弹丸 403 个全部有谱系**（395 父指针 + 6 命令 + 2 创建上下文）；单位 94 个中 1 个无谱系（`5000021 BalloonHero`，tick 992，仅影响部署分组，不影响伤害归属） |

覆盖率的正确分母是"需要谱系但父指针与命令关联均无法解释的生成"，而不是注册事件数：
本局该分母为 403 个效果/弹丸实体，未解释数为 **0**。单位缺失谱系不影响伤害归属
（单位本身就是伤害源），只影响 `deployment_totals` 分组。

`strict_r0_forward=false` 的准确含义：smoke 已通过，但只覆盖 Tick 200–235 的 8 个决策，
未覆盖 Tick 3055 的修复现场，也尚未把证据绑定进准入清单。**不能**直接将该字段改为 true；
要做的是扩大验证范围并把结果可靠绑定到数据集。

## 采集入口已改为 fail-closed

`data_pipeline/pilot100.py`：

- `capture` 缺 `--downloader-root` 且未显式给出 `--allow-diagnostic-projection` 时**直接报错**（exit 2），
  不再静默进入 schema3 诊断投影。
- 诊断投影必须在函数层也获显式授权，否则抛错。
- 报告与 episode 行新增 `capture_mode` / `replay_tier` / `native_replay_ready`，
  采集完成状态无法再被误认为数据合格。

背景：漏掉 `--downloader-root` 会继承模板默认 `gamemode 72000007 / kt 11`，
与权威契约的 `72000006 / kt 16` 是**不同对局**（血量、伤害、技能判定全变），
而报告仍显示 `native_terminal`，唯一线索是 `ruleset_match: not_certified`。

## 契约与缓存

- 新增 `ELITE_LIFECYCLE_DIAGNOSTIC` 分层：生产层只保留适配器消费的通道
  （`context_key/context_caller` + 既有边界），诊断钩子与探测字段不进入冻结产物。
  构建：`scripts/build_bridge.ps1 -Defines ELITE_LIFECYCLE_DIAGNOSTIC`（诊断）/ 不带该参数（生产）。
- 合同 SHA 现为 `ebfda45dd7b57ea3bf34e75eaaeec58b7fe31e0bfb5b98c7f68dc9e5ddc77c8f`。
- 区分三类影响：旧训练张量缓存需重编或显式兼容验证；原始状态/事件在字段充分时可直接重编；
  **只有缺少新归因通道的旧采集才需要对局级重新复演**。60 万局尚未全部编译过，
  因此不能说"60 万局缓存全部作废"。

## 单局准入证据生产者（已完成）

`data_pipeline/r0_certify.py` 从实际工件生成检查与证书，不复制能力名称充当"通过"。
每条证据都绑定：原始采集流 sha、采集报告 sha、生产 bundle manifest sha、采集入口代码 sha、
R0 合同 sha、数据集 manifest sha、验证范围。

验收结果（`compiled-prod`，生产 bundle v17）：

| 项目 | 结果 |
|---|---|
| 12 张能力证书 | 全部 `passed`，各自带 `evidence_path` + `evidence_sha256`，由 `require_dataset_files` 逐字节复核 |
| 17 项检查 | 12 项由本模块验证，5 项仍为编译器自我断言并在 `check_provenance` 中显式标注 |
| `source_stage` | `validated_raw_capture`；`admission_pending: []` |
| 全部样本严格输入检查 | **1714/1714** 个样本经官方加载器读入并通过严格 batch 契约（108 分片、2,543,862 个 known bit 全为 0/1） |
| 无未来信息（causality） | 把原始流截断到 tick 3055 重编后，tick < 3050 的 **1220** 个决策样本**逐个张量一致，0 个变化** |
| 训练验证 | 3 个窗口全部通过真实反向 + 保存恢复：opening side0/side1、repair side0（burn-in 638 决策从 Tick 0 连续到 3185，训练窗 3190–3265） |
| 正式加载入口 | `load_admitted_dataset` 成功读取：manifest + 证书 + 分片 hash 全部通过后实际加载 108 个分片 |

训练覆盖的**准确边界**（写入证据，不夸大）：本局离修复点最近的动作样本是 tick 3040 与 3190，
相距超过一个连续窗口，因此**不存在包含 tick 3055 的梯度窗口**；修复点位于递归 burn-in 内，
梯度分别来自修复前（opening）与修复后（3190+）的窗口。

负向测试（`tests/test_r0_admission_evidence.py`，9 项）：篡改分片、篡改证书证据、缺证书、
缺检查、合同不符、阶段不符、**删除一条伤害回执**（序列空洞）、**换成不匹配的 bundle**、
模板默认规则集（未走处理契约）、异侧父代、未来时刻的生命周期摘要 —— 均必须被拒绝。
其中"删除回执"用合成流验证：序列 1,2,3 / 4,5 通过；改成 1,2,3 / 4,6 后
`continuous_events` 失败。

`5000021`（BalloonHero，无部署来源）按实体口径保留为**显式未知**：本局 3M–5M 效果/弹丸
未解释谱系 **0** 个，单位未解释 **1** 个。该状态记录在 manifest 的 `lineage_unknown` 与
`groups`/`relations` 证书中；已确认分组只作为输入特征，模仿损失使用动作目标，
因此未知谱系不会伪造分组监督，也不单独否决整局。

## 未完成（按优先级）

1. **把剩余 5 项编译器断言转成验证**：`legal_candidate_alignment`、`rejected_action_loss_masks`、
   `action_offsets`、`battle_level_split`、`shard_readback` 目前在 manifest 里由编译器自我声明，
   已在 `check_provenance` 标注为 `compiler_assertion_pending_verification`。
   在它们被独立验证之前，本数据集不应被视为生产准入完成。
2. **约 100 局分层试处理**：与证书完善并行；按局统计
   "效果/弹丸实体中未解释谱系的个数"这一分母，并**分类**统计缺父代、身份错误、规则不符、
   字段未知、动作拒绝，**不得把新失败全部归因于父代路径**。
   本局基线：3M–5M 效果/弹丸 403 个实体，未解释 0；单位 95 个，未解释 1（`5000021`）。
3. **35 个零计数 child 字段**：逐项区分"本局未触发"/"不适用"/"读取未实现"。
4. **remove 钩子零事件**：992 条事件全为 op=1、`null_id_removals` 也为 0，需要解释；
   它不直接证明死亡信息缺失（致死回执与对象集合变化仍在）。
5. schema3 列表元数据（33,740 个源）可独立推进。

## 复现入口

- 归因证据：`D:/Deepseek/outputs/r0-parent-attribution-v16/context-owner-report.json`
- 审计：`.../r0-parent-attribution-v16/instrumented-audit-v16.json`
- 生产层验证：`D:/Deepseek/outputs/r0-production-v17/instrumented-audit-prod.json`
- **已认证数据集**：`D:/Deepseek/outputs/r0-production-v17/compiled-prod`
  （`manifest.json`、`evidence/` 下 raw_stream / strict_inputs / causality / strict_training 与
  12 张能力证书）
- 负向对照：`.../r0-parent-attribution-v16/negative-control-v9.json`
- 整局编译 / smoke：`.../r0-parent-attribution-v16/compiled-full/manifest.json`、
  `.../smoke-full/report.json`

采集（必须带处理契约）：

```powershell
& D:\AI_data\runtime\venv\Scripts\python.exe -m data_pipeline.pilot100 capture `
  --output <dir> --bundle-manifest <BUNDLE-MANIFEST.json> --attempt <name> `
  --host <wsl-ip> --port 39531 --seed 1 --maximum-seeds 512 --capture-masks `
  --downloader-root D:\Deepseek\work\royaleapi-downloader --battle-tag <TAG>
```

停止 worker：

```powershell
wsl -d Ubuntu -u root -- bash -c "cd /mnt/d/Deepseek/CR-Native-Linux-Bionic && python3 -m cr_native_bionic stop --runtime-root /opt/cr-capture-repair-20260914 --count 1 --base-port 39531"
```
