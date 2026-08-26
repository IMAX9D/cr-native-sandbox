# Native Teacher-Forced 固定 100 场 v6 → v7 独立审计

## 结论

固定同一批 100 场、逐字节相同的 selection 上，v7 将 teacher-forced 成功数从
`37/100` 提升到 `40/100`：

- 6 个 v6 shuffle layout 失败全部消失；其中 2 场转为完整成功，4 场继续执行后暴露为真实的 exact-tick `code 13`。
- `Spirit Empress` 唯一的 v6 `code 13` 样本在 v7 完整成功。
- 原有 `code 4` 失败集合完全稳定：41 场、42 个被拒动作，v6/v7 tag 集合相同。
- 原有“源动作到来前原生终局”集合完全稳定：8 场，v6/v7 tag 集合相同。
- 终局 crowns 匹配数从 23 增至 26，恰好对应 3 个新增成功样本。
- v7 seed 复核为 `4/4` 逻辑训练状态 SHA-256 一致；v6 为 `3/4`。
- v7 Tick Store 的 40 个成功 episode（分布在 4 个物理 shard）共 `164,723` Tick，已经全部解码；逐 Tick 连续、episode payload SHA、data/index SHA、全局 manifest/content SHA 均通过。

因此，`77223a0` 的有界原生 seed 搜索和 `4115417` 的 Spirit Empress 原生形态选择都产生了可定位、无样本漂移的预期效果。v7 仍只有 40% 完整 teacher-forced 成功，不能据此宣称回放已完全还原；剩余失败已经从 layout/形态适配问题进一步收敛到 exact-tick 资源、原生 command gate 和提前终局三类证据。

## 审计对象与可比性

| 项目 | v6 | v7 |
| --- | --- | --- |
| 目录 | `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-compact-v6` | `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-seed-dynamic-v7` |
| Runtime provenance | `7a71345` baseline（含 `1152dd1` compact trace、`5d90e56` guard 诊断） | head `4115417`，包含 `77223a0` + `4115417` 两项语义修复 |
| selection 数 | 100 | 100 |
| selection SHA-256 | `5fa5239318ce2396934408ceab624d507ccaf9a89143752ed80f458ca0127a3a` | 同左 |
| selection 字节 | 基准 | 与 v6 逐字节相同 |
| source JSON SHA | 基准 | 100/100 tag 与 v6 相同 |

两份 `selection-summary.json` 也完全相同，SHA-256 为
`c005407d9e32fbf43e1ab7c63900109b39e92c0d58516026b6b28f2dcd4b6e20`。
因此下面的差异不是抽样变化。

准确的运行时代码来源：

- v6 baseline head：`7a713452e2aad31ba1fff159ee7174dfdad4f718`；其 compact Tick trace 来自 `1152dd1d6d7f08691d43f4c7bc1918bfdf59d377`，native guard 诊断来自 `5d90e5651ae72582838fd32df895796dc340ebc3`。
- v7 runtime head：`411541791d6ea70b537115a593f6f6893d3d603e`。

本次 v6 → v7 重点审计的两个语义修复提交：

- `77223a0204469153b6de1bdcdec6b7739a974ba4`：`fix: resolve native expert layouts by bounded seed search`
- `411541791d6ea70b537115a593f6f6893d3d603e`：`fix: honor native Spirit Empress form selection`

## 结果总表

| 分类 | v6 | v7 | 净变化 |
| --- | ---: | ---: | ---: |
| teacher-forced success | 37 | 40 | +3 |
| layout failure | 6 | 0 | -6 |
| `code 13` episode | 8 | 11 | +3 |
| `code 4` episode | 41 | 41 | 0 |
| terminal-before episode | 8 | 8 | 0 |

`code 13` 的总数从 8 增到 11 不是 layout 修复的回归：

- 原 8 场中，7 场保持 `code 13`；Spirit Empress 的 1 场转为成功。
- 原 6 场 layout 失败中，4 场在 layout 解开后才运行到各自的真实 `code 13`。
- 所以 v7 为 `7 + 4 = 11` 场。

原生拒绝动作数为：

| 原生结果码 | v6 | v7 | 解释 |
| --- | ---: | ---: | --- |
| `4` | 42 | 42 | 41 场；其中一场同 Tick 两个动作都被拒绝 |
| `13` | 8 | 11 | 见上面的下游失败揭示关系 |

v7 新增成功的 3 个 battle tag：

- `00GYPP8QCJ9V`：`code13 → success`，Spirit Empress。
- `02PYPJJRY9VG`：`layout → success`。
- `09QP9J0VRQY8`：`layout → success`。

## 重点迁移证据

### Layout 6 → 0

| battle tag | v6 | v7 | v7 首个下游证据 |
| --- | --- | --- | --- |
| `008YLPVGR8GR` | layout | code13 | Tick 2152，`arrows` |
| `022YYLPR8C0R` | layout | code13 | Tick 2821，`three-musketeers` |
| `089Y82C0PGR0` | layout | code13 | Tick 2467，`golem` |
| `08YY82CVQ92P` | layout | code13 | Tick 1714，`skeleton-dragons` |
| `02PYPJJRY9VG` | layout | success | 54/54 部署；terminal `match` |
| `09QP9J0VRQY8` | layout | success | 56/56 部署；terminal `match` |

这说明有界 seed 搜索没有把 layout 失败“改名”为成功，而是把六场都推进到了可执行的真实原生布局；其中四场随后按 fail-closed 规则停在新的资源拒绝点。

### Spirit Empress

`00GYPP8QCJ9V` 在 v6 的 Tick 3361 将 Spirit Empress 按错误的 canonical 形态解析为 6 费，原生返回 `code 13`；v7 按 native dynamic choice 选择正确形态后，42/42 部署全部接受，存储 3,672 Tick，terminal 为 `match`。

这一个 tag 是固定 100 场内唯一的 `Spirit Empress code13 → success` 迁移。

### 稳定的失败集合

`code4 → code4` 共 41 个 tag：

`002YLYVQPQVU`, `008YLPVRJ09Y`, `009YLLP0PYQ8`, `00CYPY2LLYJ2`,
`00CYPY2LV28P`, `00CYPY8VCGUJ`, `00JYPYJV09PL`, `00LYPL29Y220`,
`00LYPL9Y89JL`, `00PYLPQQ8YUQ`, `00PYLPRPCG0Y`, `00RYPPG8CQ0G`,
`00VYPY99QLV2`, `00YYPPQYJ99U`, `020YPPLRVY9L`, `028YPC22YYQY`,
`028YPC2C8JV2`, `028YPJCU999Q`, `029YPJ0CC8PY`, `02CY8PPYGUCG`,
`02GY9QRR0GY8`, `02PYPJGUC0RR`, `02QY9L0GQJCL`, `02RY9QJQ8QQR`,
`02UY8PLGYUUJ`, `02VY8YG9QU2Q`, `080Y8LPVJCLV`, `080Y8P2099GL`,
`082Y8L9VLL9L`, `089Y82CPYYY9`, `08CPVRPY9PRJ`, `08CPVRYQVUJL`,
`08GPVQR2YUPU`, `08GPVQULJ89R`, `08JPVJYPRJPJ`, `08PY829UP89G`,
`08RPVRGL0L98`, `08YY8828RQJV`, `090PPUJJJQ8G`, `098PPC9JGL8V`,
`09QP9R2RRY8L`。

`terminal_before → terminal_before` 共 8 个 tag：

`02CY8PPJ08G9`, `02PYPJGRV290`, `08GPVQV2PYYG`, `08JPVJLQVPCG`,
`08LY808UY2L2`, `08RPVRY8VYJC`, `09LP9JQ8JGCJ`, `09QP9J80VGL2`。

`code13 → code13` 共 7 个 tag：

`00CYPPG22CPJ`, `00VYPYPQV8QC`, `00YYPPGLR8YU`, `02QY9L89CYGV`,
`02YYPJRY0UGQ`, `080Y8LY0PQ9L`, `09LP9JLR0U8Q`。

`success → success` 共 37 个 tag：

`002YLYCJCUP0`, `008YLPCC8Q2U`, `008YLPUYLYG9`, `009YLLPQRV02`,
`009YLLYRRUQQ`, `00GYPP2J8JCG`, `00JYPLP9CLQP`, `00LYPL2QGPJJ`,
`00PYLPPR0CJL`, `00YYPPG808JV`, `00YYPPG82LJ8`, `02CY8PYY9829`,
`02GY9QGUYQVR`, `02PYPJGGGU8J`, `02PYPJJVRLJL`, `02RY9QRJJPCV`,
`02VY8Y28U8YQ`, `02YYPJQPVVV0`, `080Y8LYYG9QJ`, `088Y82G28CLC`,
`08CPVRY0VYGU`, `08GPVQV0JL88`, `08QPVJ8CUJLL`, `08RPVRQU9QCR`,
`08RPVRQULP2Q`, `08RPVRQY8CJR`, `08UPPCVU022P`, `08VPPCG9ULRU`,
`08VPPCGP299R`, `08YY880R8PC0`, `090PPUJGUV9L`, `092PPCQRULCP`,
`099P9RJR9PGP`, `09LP9JL9C0YL`, `09R9JJQUV29Y`, `09R9JJQVQ2VC`,
`09YP9RQ80220`。

以上分组连同 7 个非稳定迁移恰好覆盖 100/100 tag。机器报告的 `per_tag` 数组还记录了每个 tag 的 failure、结果码、拒绝卡、动作数、Tick 数、终局状态、seed 和状态 SHA。

## 终局诊断

| terminal status | v6 | v7 |
| --- | ---: | ---: |
| match | 23 | 26 |
| mismatch | 1 | 1 |
| missing at source-duration fence | 12 | 12 |
| logic frozen at source-duration fence | 1 | 1 |
| teacher-forced failure，未评估 | 63 | 60 |

终局 `match +3` 与 teacher-forced `success +3` 一一对应。其余终局分类数量不变，没有通过放宽终局判定制造新增成功。

## Seed 复核

v7 的四个 seed probe 均满足：

- 主 replay 和 alternate-preferred-seed replay 都成功；
- Tick 数相同；
- `logical_training_state_sha256` 相同；
- 不需要旧式 layout calibration。

| battle tag | Tick | alternate 搜索得到的 seed | 逻辑状态 SHA 一致 |
| --- | ---: | ---: | --- |
| `09R9JJQVQ2VC` | 6,121 | 18 | 是 |
| `02CY8PYY9829` | 3,288 | 2 | 是 |
| `08RPVRQULP2Q` | 6,106 | 20 | 是 |
| `08RPVRQU9QCR` | 3,672 | 12 | 是 |

汇总为 v7 `4/4`，而 v6 为 `3/4`。这里验证的是在可兼容布局解析后 teacher-forced 逻辑训练状态一致，不把 seed 解释成原始服务器隐藏 RNG 的复原证明。

## v7 Tick Store 完整性

v7 的 40 个成功 episode 分布在 4 个物理 shard 中：

| shard | episode | Tick | data SHA-256 | index SHA-256 |
| --- | ---: | ---: | --- | --- |
| `worker-00-00000` | 9 | 41,298 | `586c955408ffe7819e2302dd3fc7b33f548c157ac4d187abe25e25d1d7f13e52` | `bd312d8d28e3e6d219ebcb5ca6dc453ca24ec7e1f19583effd6b2bf3aea3bdd8` |
| `worker-01-00000` | 9 | 35,743 | `536a1b73dc51e6dd6a6bafee9c40aaf26c7dc2fa17faadf127d4f65509a4e190` | `6a8d4db1716f6d7bf68b0a110db7399af4f557064a0f33a040febbcbdda07d94` |
| `worker-02-00000` | 14 | 50,235 | `0d2a4878ddf1b160ff8759523727cce16f3e8fbefcdce2f8a05ab94bfb515ab2` | `151662b21942d5e84eb1adef07fffaead2f8d45caaca6a20293fccdf017c7de3` |
| `worker-03-00000` | 8 | 37,447 | `e7cf4183302cd1b91fb162c3026114a4e7ca440655316c8960defed3df7ca435` | `b28102691d8fcbb0695b6ec573df7cbf1fd8cc9bcd4b87f325be5bc7c2942851` |
| **合计** | **40** | **164,723** | — | — |

独立读取验证包括：

1. `manifest.sha256` 与 `manifest.json` 实际 SHA-256 一致；
2. 全局 manifest 中每个 data/index SHA-256 与磁盘文件一致；
3. 每个 episode frame 的 payload SHA-256 与 index 一致；
4. 逐块解压并校验 Tick chunk CRC；
5. 40 个 episode 全部逐 Tick 解码，Tick 严格 `+1` 连续；
6. 每个 episode 的首尾 Tick、数量与 index/result 一致；
7. 存储 tag 集合与 40 个 teacher-forced success tag 集合完全相同；
8. 全局 content SHA-256 重新计算一致。

关键摘要：

- global store content SHA-256：`dfbe1dc13603b50b4ed75f25b6e42e3c787cbc2ac1ca5ce4c9660293f6dc2898`
- global manifest SHA-256：`7fbc2726d4ba087c041ef8d593a94cbb22165184d3ef6d87049dcb67a0741b6b`
- source selection SHA-256：`5fa5239318ce2396934408ceab624d507ccaf9a89143752ed80f458ca0127a3a`
- store bytes：3,951,533

## 描述性吞吐变化

| 指标 | v6 | v7 | 变化 |
| --- | ---: | ---: | ---: |
| stored Tick | 153,707 | 164,723 | +7.17% |
| Tick / pilot wall second | 739.61 | 754.07 | +1.96% |
| episode / hour | 640.93 | 659.21 | +2.85% |
| bytes / Tick | 24.097 | 23.989 | -0.45% |

这些只是描述性数据。v7 多跑完 3 个 episode，并且包含 per-battle seed 搜索，所以不能把单次 100 场 wall time 当成严格性能基准。

## 可复核产物

机器可读报告：

`D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-seed-dynamic-v7\comparison-v6-v7.json`

报告 SHA-256：

`335138d2a83203ccbb308e46b8675269bbb4257d7ee95f7fad08ff7e22fafb16`

该文件包含：

- 100 个 tag 的逐条 v6/v7 分类和字段；
- 所有迁移组及 tag；
- selection/source SHA 可比性；
- seed probe 原始摘要；
- 4 个物理 shard、40 个 episode 的完整解码与 SHA 验证结果；
- 12 条固定验收断言，当前全部为 `true`。

复核命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\audit_native_pilot_v7_comparison.py `
  D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-compact-v6 `
  D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-seed-dynamic-v7 `
  --output D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-seed-dynamic-v7\comparison-v6-v7.json
```

脚本只读两个 pilot 目录；唯一写入是 comparison JSON，不会修改 runner，也不会重跑 libg。
