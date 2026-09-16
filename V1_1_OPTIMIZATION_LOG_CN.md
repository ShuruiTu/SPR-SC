# SPR-SC v1.1 性能优化计划与实验记录

更新日期：2026-09-16

开发分支：`feature/v1.1-performance-optimization`

对照版本：`v1.0`（提交 `c6b83d7`）

## 1. 文档用途

本文档统一记录 v1.0 之后的代码修改、实验配置、结果目录和结论。后续每次实现新模块或运行正式实验，都应在本文档追加记录，避免代码状态与实验结果无法对应。

所有优化模块必须满足以下约束：

1. 默认关闭，关闭后保持 v1.0 行为。
2. 可以通过独立命令行参数启用，避免不同优化相互绑定。
3. 输出中记录开关、参数、随机种子、数据集、信道和 SNR。
4. 消融实验一次只改变一个主要因素。
5. 正式结论至少使用多个 channel seed；单随机种子只作为开发验证。

## 2. v1.0 基线状态

v1.0 已包含：

- SHyRe count 主流程；
- `SHyRe-soft-count` 可选软可靠度特征；
- AWGN、Rayleigh 和 clean 信道；
- 稀疏先验后验判决；
- matched-channel training；
- `shyre`、`shyre_fast` 和严格最大团候选模式；
- 候选、存储、运行时间和超边规模分层指标；
- Max Clique、ECC、DEMON 基线；
- 图 1/2、图 3/4、图 9/10/9.5、图 11/12 的实验与绘图入口。

当前 matched-channel training 结果表明：

- P.School、H.School 在两类信道下相对稳定；
- Enron、Directors、Crime、FB-AUTO 和 JF17K 在中低 SNR 下主要表现为召回不足；
- soft-count 对 Enron 有稳定小幅收益，但对两个 School 数据集略有下降；
- Rayleigh 的恢复难度明显高于 AWGN；
- JF17K 当前采用 `shyre_fast`，结果必须与精确 SHyRe 分开标注；
- Hosts-Virus 的 matched training 存在较高内存与候选规模开销，暂不进入首轮优化实验。

## 3. 优化任务与独立开关设计

### P0：补充分阶段诊断指标

目标：区分“候选阶段丢失真实超边”和“分类器拒绝真实候选”。

计划记录：

- candidate recall、precision 和候选数量；
- 最大团候选与嵌套候选各自的正样本数量和召回率；
- 分类前后候选数量；
- 分类器 precision/recall/F1；
- 空训练子集和单类别训练子集出现次数。

现有入口：`--enable-candidate-metrics`。后续需要扩展为分候选类型的详细指标。

状态：`已实现并完成首轮验证`

### P1：验证集自适应分类阈值

目标：改善当前高 precision、低 recall 的情况。

计划开关：

- `--decision-threshold-mode fixed|validation`
- `--decision-threshold FLOAT`
- 最大团和嵌套团允许使用独立阈值。

候选阈值只能在训练/验证数据上选择，禁止使用测试集调参。

状态：`已实现并完成首轮单种子验证`

### P2：信道感知候选生成

目标：避免真实边因一次硬判决被删除后，相应超边永远无法进入候选集合。

计划模式：

- 多后验阈值构图并合并候选；
- 从边后验概率多次采样接收图并合并候选；
- 可选地把 ECC/DEMON 输出加入候选池；
- 设置候选数量上限和内存保护。

该模块必须作为新的 candidate generator，不改变原 `shyre` 和 `shyre_fast`。

状态：`已实现实验版本并完成 10 dB 单种子验证`

### P3：多信道种子与邻域 SNR 训练

目标：降低单次信道实现造成的过拟合和曲线波动。

计划开关：

- `--train-channel-seeds ...`
- `--train-snr-window FLOAT`
- `--train-snr-values ...`

对照组包括：单 seed 同 SNR、多个 seed 同 SNR、多个 seed 邻域 SNR 和混合 SNR。

状态：`待实现`

### P4：软可靠度特征 v2

目标：让软可靠度反映候选内部边置信度分布，而不只是简单聚合。

计划加入：

- 最小值、均值、标准差和分位数；
- 低可靠度边比例；
- 对数团存在概率；
- 候选缺失边数量及其后验概率；
- 独立特征归一化。

计划模式：`off|basic|distribution`，其中 `basic` 保持 v1.0 soft-count 行为。

状态：`待实现`

### P5：类别不平衡与模型比较

目标：缓解低 SNR 下正样本稀少、模型倾向全部拒绝的问题。

计划比较：样本重采样、样本权重、Logistic Regression、Random Forest、MLP 和概率校准。模型改动与候选生成改动分开测试。

状态：`待实现`

### P6：Rayleigh 专项处理

目标：缩小 Rayleigh 与 AWGN 的性能差距。

计划区分已知瞬时信道增益和仅已知统计分布两种设置，并比较均衡、边缘化后验和多衰落采样集成。

状态：`待实现`

## 4. 推荐执行顺序

1. 完成 P0，确定瓶颈位于候选还是分类阶段。
2. 完成 P1，以较低成本验证阈值调整的收益上限。
3. 根据 P0 结果完成 P2。
4. 完成 P3，提高曲线稳定性。
5. 完成 P4，并与原 soft-count 做严格消融。
6. 再开展 P5、P6 和数据集专用参数优化。

首轮开发数据集建议使用 Enron 和 Crime，SNR 使用 10 dB；确认代码正确后再扩展到 0–20 dB。非 SNR 专项实验统一使用 60 dB。

## 5. 结果记录模板

### YYYY-MM-DD：实验或修改名称

- Git 提交：
- 模块与开关：
- 数据集：
- 信道与 SNR：
- model seed / channel seed：
- 对照组：
- 结果目录：
- 主要结果：
- 资源消耗：
- 是否符合预期：
- 结论与下一步：

## 6. 变更记录

### 2026-09-16：建立 v1.1 优化分支

- 从 `v1.0` 标签对应提交 `c6b83d7` 创建 `feature/v1.1-performance-optimization`。
- 建立本优化计划与实验记录文档。
- 尚未改变 v1.0 默认算法行为。

### 2026-09-16：P0 分阶段诊断指标

- 扩展现有 `--enable-candidate-metrics`，默认仍为关闭。
- 分别记录最大团候选、嵌套候选和全部候选的覆盖率。
- 记录训练候选的正负样本数量、空子集和单类别子集状态。
- 记录候选经过分类器后的保留数量、类内召回和端到端召回。
- 已在 Enron、Crime 的 AWGN/Rayleigh 10 dB 上完成开发验证：

| 数据集 | 信道 | 最终 F1 | 候选召回 | 分类器对真实候选的保留率 |
|---|---|---:|---:|---:|
| Enron | AWGN | 0.1102 | 0.3519 | 0.1805 |
| Enron | Rayleigh | 0.0610 | 0.3862 | 0.0890 |
| Crime | AWGN | 0.1490 | 0.4844 | 0.2097 |
| Crime | Rayleigh | 0.0223 | 0.4219 | 0.0278 |

- 结论：首轮样本中的候选集合仍覆盖约 35%–48% 的真实超边，但分类器只保留其中约 3%–21%；当前中低 SNR 的主要瓶颈位于分类判决阶段，P1 的优先级高于扩大候选池。
- 结果目录：`results_v1_1_p0_diagnostics_awgn_10db/`、`results_v1_1_p0_diagnostics_rayleigh_10db/`（本地忽略，不提交）。

### 2026-09-16：P1 验证集自适应分类阈值

- 新增 `--decision-threshold-mode fixed|validation`，默认 `fixed`。
- 新增 `--decision-threshold` 和 `--threshold-validation-fraction`。
- 最大团与嵌套团分类器分别在训练候选的分层验证子集上选择 F1 最优阈值；测试标签不参与阈值选择。
- 若训练子集为空、只有单类别或样本不足，则回退到固定阈值。
- 默认固定阈值回归检查与 v1.0 完全一致。

| 数据集 | 信道 | 固定阈值 F1 | 自适应阈值 F1 | 变化 |
|---|---|---:|---:|---:|
| Enron | AWGN | 0.1102 | 0.3084 | +0.1982 |
| Enron | Rayleigh | 0.0610 | 0.2991 | +0.2381 |
| Crime | AWGN | 0.1490 | 0.1685 | +0.0195 |
| Crime | Rayleigh | 0.0223 | 0.2339 | +0.2116 |

- Enron 自适应阈值：AWGN 为最大团 0.15、嵌套团 0.05；Rayleigh 为 0.05、0.10。
- Crime 自适应阈值：AWGN 为最大团 0.45、嵌套团 0.25；Rayleigh 均为 0.15。
- 结论：结果支持 P0 的判断，固定 0.5 阈值是中低 SNR 的主要瓶颈之一；需要通过多 channel seed 和完整 SNR 曲线确认泛化与稳定性。
- 结果目录：`results_v1_1_p1_validation_threshold_awgn_10db/`、`results_v1_1_p1_validation_threshold_rayleigh_10db/`（本地忽略，不提交）。

### 2026-09-16：P1 完整单种子 SNR 曲线

- 数据集：Enron、P.School、H.School、Directors、Crime。
- SNR：0–20 dB，间隔 2 dB；AWGN 和 Rayleigh；matched-channel training。

| 数据集 | AWGN 平均 F1 变化 | Rayleigh 平均 F1 变化 |
|---|---:|---:|
| Enron | 0.1164 → 0.2538（+0.1374） | 0.0758 → 0.2626（+0.1868） |
| P.School | 0.5379 → 0.5514（+0.0135） | 0.4843 → 0.5028（+0.0185） |
| H.School | 0.5577 → 0.5655（+0.0078） | 0.5102 → 0.5124（+0.0022） |
| Directors | 0.3211 → 0.3198（-0.0014） | 0.0317 → 0.0479（+0.0162） |
| Crime | 0.3376 → 0.3855（+0.0480） | 0.1905 → 0.2732（+0.0827） |

- AWGN 高 SNR 下 Crime 有 0.009–0.030 的局部回退，后续需要多 seed 或阈值正则化确认。
- 结果目录：`results_v1_1_p1_adaptive_threshold_awgn_0_20_step2/`、`results_v1_1_p1_adaptive_threshold_rayleigh_0_20_step2/`（本地忽略，不提交）。

### 2026-09-16：P2 信道感知候选生成实验版

- 新增 `shyre_channel_aware`，原 `shyre` 行为不变。
- 使用低于 hard 判决阈值的后验边建立候选搜索图；原 hard received graph 继续用于结构特征和基线。
- 后验边按概率排序，并使用 `--channel-candidate-max-extra-edges` 控制规模。
- hard graph 与后验图的最大团、子团候选采用并集，保证不会因最大团吞并而删除原候选。
- 比较 0.40、0.45、0.50 后，0.45 在当前 10 dB 单种子实验中最稳定，因此作为实验模块的推荐初值。

| 数据集 | 信道 | P1 F1 | P2（tau=0.45）F1 | 候选召回变化 |
|---|---|---:|---:|---:|
| Enron | AWGN | 0.308 | 0.317 | 0.352 → 0.378 |
| Directors | AWGN | 0.093 | 0.177 | 0.118 → 0.147 |
| Crime | AWGN | 0.168 | 0.329 | 0.484 → 0.516 |
| Enron | Rayleigh | 0.299 | 0.295 | 0.386 → 0.415 |
| Directors | Rayleigh | 0.000 | 0.000 | 0.029 → 0.059 |
| Crime | Rayleigh | 0.234 | 0.242 | 0.422 → 0.438 |

- 结论：P2 能稳定提高候选召回，但最终 F1 仍受分类器和阈值稳定性影响；暂不作为默认模型，应在 P3 多 seed 训练完成后重新验证。
- 结果目录：`results_v1_1_p2_tau_0p45_awgn_10db/`、`results_v1_1_p2_tau_0p45_rayleigh_10db/`（本地忽略，不提交）。
