# SPR-SC v1.1 最优方案实验图与结果分析

更新时间：2026-09-17

## 1. 本次整理范围

- 主曲线采用 3 个随机种子，阴影表示标准差。
- 信道：AWGN 与 Rayleigh；SNR 为 0–20 dB，间隔 2 dB。
- 本次已纳入：Enron、P.School、H.School、Directors、Crime、FB-AUTO、JF17K、M-FB15K、WikiPeople。
- DBLP 正在断点续跑，完成前不进入图 1、图 2；绘图程序会自动跳过不完整数据，避免把部分结果误画成完整曲线。
- Hosts-Virus 与 Foursquare 按当前实验安排排除。
- JF17K、M-FB15K、WikiPeople 使用 `SHyRe-fast` 适配路径，解释时应与标准 SHyRe 数据集分开标注。

## 2. 已生成图片

| 图号 | 内容 | 文件 |
|---|---|---|
| 图 1 | AWGN 下各数据集 F1–SNR 曲线 | [fig_1_v1_1_optimal_awgn_snr.png](figures_v1_1_optimal/fig_1_v1_1_optimal_awgn_snr.png) |
| 图 2 | Rayleigh 下各数据集 F1–SNR 曲线 | [fig_2_v1_1_optimal_rayleigh_snr.png](figures_v1_1_optimal/fig_2_v1_1_optimal_rayleigh_snr.png) |
| 图 3 | Enron 10 dB 的 F1 消融 | [fig_3_v1_1_f1_ablation.png](figures_v1_1_optimal/fig_3_v1_1_f1_ablation.png) |
| 图 4 | Enron 10 dB 的候选召回率消融 | [fig_4_v1_1_candidate_recall_ablation.png](figures_v1_1_optimal/fig_4_v1_1_candidate_recall_ablation.png) |
| 图 9 | 不同保留率下的 F1 | [fig_9_v1_1_retention_f1.png](figures_v1_1_optimal/fig_9_v1_1_retention_f1.png) |
| 图 9.5 | F1–存储量折线图 | [fig_9_5_v1_1_f1_vs_storage.png](figures_v1_1_optimal/fig_9_5_v1_1_f1_vs_storage.png) |
| 图 10 | 保留率–存储量关系 | [fig_10_v1_1_retention_storage.png](figures_v1_1_optimal/fig_10_v1_1_retention_storage.png) |
| 图 11 | Enron 按超边规模分层结果 | [fig_11_v1_1_enron_size_stratified.png](figures_v1_1_optimal/fig_11_v1_1_enron_size_stratified.png) |
| 图 12 | FB-AUTO 按超边规模分层结果 | [fig_12_v1_1_fb_auto_size_stratified.png](figures_v1_1_optimal/fig_12_v1_1_fb_auto_size_stratified.png) |
| 额外消融 | Enron、Crime 在 AWGN 20 dB 下的模块消融 | [fig_ablation_enron_crime_awgn20.png](figures_v1_1_optimal/fig_ablation_enron_crime_awgn20.png) |

额外消融的原始汇总表见 [ablation_enron_crime_awgn20.csv](figures_v1_1_optimal/ablation_enron_crime_awgn20.csv)。

## 3. 图 1、图 2：主曲线分析

下表中的“平均增益”是 SPR-SC 在全部 SNR 点的平均 F1，减去同图中平均 F1 最好的传统基线；它用于快速判断整体表现，不等同于逐 SNR 显著性检验。

| 数据集 | AWGN 平均 F1 | AWGN 平均增益 | Rayleigh 平均 F1 | Rayleigh 平均增益 | 判断 |
|---|---:|---:|---:|---:|---|
| Enron | 0.2666 | +0.1589 | 0.2664 | +0.1818 | 明显有效 |
| P.School | 0.5501 | +0.5072 | 0.5255 | +0.4845 | 明显有效且稳定 |
| H.School | 0.5641 | +0.4147 | 0.5550 | +0.4059 | 明显有效且稳定 |
| FB-AUTO | 0.3669 | +0.0486 | 0.2901 | +0.0495 | 有效，但 Rayleigh 衰减较明显 |
| Crime | 0.3916 | -0.0054 | 0.3155 | -0.0179 | 与最佳基线接近 |
| JF17K-fast | 0.2438 | -0.0079 | 0.1310 | -0.0225 | 与最佳基线接近，Rayleigh 偏弱 |
| Directors | 0.3145 | -0.1021 | 0.0906 | -0.2002 | 中低 SNR 不理想 |
| M-FB15K-fast | 0.0985 | -0.0731 | 0.1062 | -0.0605 | 未达到预期 |
| WikiPeople-fast | 0.0207 | -0.4515 | 0.0542 | -0.4276 | 明显不符合预期 |

### 符合预期的现象

1. P.School、H.School 在两类信道上均显著优于基线，且随 SNR 提升总体趋于稳定，说明候选生成与分类流程对这类结构规则、重复模式较强的数据有效。
2. Enron 在 AWGN 和 Rayleigh 上都保持明显优势，说明 P1 自适应机制对该数据集有效，而非只适配单一信道。
3. FB-AUTO 总体优于基线，高 SNR 下恢复明显；此前 6 dB 的异常下降在 3 个种子的均值与标准差表达后不应被解释为确定性信道规律。
4. AWGN 通常优于 Rayleigh，符合衰落信道带来额外随机幅度扰动的预期。

### 暂不符合预期的现象

1. WikiPeople-fast 几乎全程失效，并出现非单调波动。这更像候选覆盖、标签分布或 fast 适配路径与数据结构不匹配，而不是单纯分类器能力不足。
2. M-FB15K-fast 没有超过 ECC；即使 SNR 增大，收益仍有限，说明瓶颈可能在候选空间而非信道判决。
3. Directors 在 AWGN 20 dB 可达到 1.0，但中低 SNR 很弱；Rayleigh 20 dB 也只有约 0.38，显示其结果对少量边扰动极其敏感，不能只用高 SNR 单点论证鲁棒性。
4. JF17K-fast 在 AWGN 20 dB 达到约 0.77，但全 SNR 平均仅与最佳基线相当；当前更适合表述为“高 SNR 有效”，不宜表述为全区间领先。

## 4. 图 3、图 4：候选生成与软可靠度

在 Enron 10 dB 下，SHyRe 候选模式将候选召回率从约 0.04–0.05 提升到约 0.35–0.39，并把 F1 从约 0.03–0.05 提升到约 0.30–0.32。这是目前最明确、幅度最大的正向消融结论。

- AWGN：`soft-v2 + SHyRe` 的 F1 为约 0.3203，`count + SHyRe` 为约 0.3145，软可靠度仅有小幅收益。
- Rayleigh：二者约为 0.3043 和 0.3041，差异可忽略。
- strict 候选生成无论是否加入软可靠度都明显较差。

因此，当前性能主要来自 SHyRe 候选生成；软可靠度属于小幅、依数据与信道而变的增益，不能宣称稳定全面提升。

## 5. 图 9、图 9.5、图 10：保留率与存储权衡

存储量随保留率严格线性增长：20%、40%、60%、80%、100% 分别约为 1097、2194、3290、4387、5484 个存储单元。F1 则是非线性增长：约从 0.05、0.19–0.20、0.31、0.39 跃升到 0.70。

- 60% 保留率可以节省约 40% 存储，但 F1 仅约为完整保留的 44%。
- 80% 保留率可以节省约 20% 存储，但 F1 仅约为完整保留的 55%。
- 从 80% 到 100% 的最后 20% 存储带来了最大的 F1 增益。
- count 与 soft 两条曲线几乎重合，soft 在部分保留率下还略低；当前实验不支持“软可靠度改善压缩场景”的结论。

因此，图 9.5 可用于展示存储—精度权衡，但当前最优工作点仍偏向 100% 保留；若论文需要强调压缩优势，还需要重新设计非均匀、按可靠度或信息量选择的保留策略。

## 6. 图 11、图 12：超边规模分层

### Enron

- SHyRe 在规模 2 上最好，F1 约 0.4314。
- 规模 3 和 4 分别降至约 0.0567 和 0.0262。
- 规模 5+ 回升至约 0.1147，但仍不高。

这说明 Enron 的总体优势主要由二元超边贡献，对中等规模超边的恢复仍是明显短板。

### FB-AUTO

- 规模 2 的 F1 约 0.9637，表现非常强。
- 规模 4 与 5+ 分别约 0.7356、0.7339，也较稳定。
- 规模 3 的 F1 为 0，是突出异常点。

FB-AUTO 的规模 3 失败更可能来自该层样本量、训练标签或候选覆盖不足，需要先报告每层真实超边数和候选召回率，再决定是否把该图作为主文结论。

## 7. Enron/Crime 20 dB 模块消融

| 配置 | Enron F1 | Crime F1 |
|---|---:|---:|
| Base | 0.2304 | **0.8861** |
| P1 | 0.3070 | 0.8596 |
| P1+P2 | 0.3070 | 0.8596 |
| P1+P3 | 0.2864 | 0.8588 |
| P1+P2+P3 | 0.2864 | 0.8532 |
| P1+P4 | 0.2988 | 0.8654 |
| P1+P5 Logistic Regression | **0.3080** | 0.8786 |
| P1 + upsample | 0.3064 | 0.8305 |

结论是模块收益不具有简单可加性：

- P1 对 Enron 有显著收益，但在 Crime 20 dB 下反而降低 F1。
- P2 在 20 dB 下与 P1 完全相同，高 SNR 时没有体现额外价值。
- P3 在该高 SNR 场景下有负作用；它此前在 Crime 10 dB 的收益不能推广到所有 SNR。
- P5 的 Logistic Regression 是 P1 系列中最稳妥的版本，在 Enron 最佳，在 Crime 也最接近 Base。
- upsample 明显损害 Crime，不应作为默认设置。

## 8. 总体判断与下一步

当前 v1.1 可以支持以下结论：SHyRe 候选生成是核心贡献；P1/P5 对 Enron、学校类数据有效；Rayleigh 校准能维持部分数据集的性能；系统在不同超图结构上的泛化能力仍不均衡。

当前不应支持以下强结论：软可靠度在所有数据集上稳定优于 count；所有模块叠加均产生正收益；方法在知识图谱适配数据上普遍领先；降低存储量只造成轻微精度损失。

建议后续优先级：

1. 为 WikiPeople、M-FB15K、FB-AUTO 的异常规模层增加候选召回率、正负样本量和超边规模分布审计。
2. 将 P1/P3 改成按数据集统计量或 SNR 自适应启停，而不是固定全开。
3. 设计基于后验可靠度或候选信息量的非均匀保留策略，重新验证图 9–10。
4. DBLP 完成后只需重新运行绘图脚本，即可自动把完整 DBLP 曲线加入图 1、图 2。

## 9. 扩展到其他数据集的模块消融（已完成）

实验协议为 AWGN 20 dB、3 个种子。九个数据集共 198/198 个有效实验单元已经
完成，未发现失败。由于当前没有 `shyre_fast_channel_aware`，P2 和 P1+P2+P3
在 JF17K、M-FB15K、WikiPeople 上标记为 N/A，而不是用相同配置重复运行并
冒充有效消融。

最终结果如下：

| 数据集 | Base F1 | 最佳配置 | 最佳 F1 | 相对 Base |
|---|---:|---|---:|---:|
| Enron | 0.2304 | P1+P5 Logistic Regression | 0.3080 | +0.0775 |
| P.School | 0.6156 | P1+P2 | 0.6282 | +0.0127 |
| H.School | 0.6984 | P1+P5 Logistic Regression | 0.7073 | +0.0089 |
| Directors | 1.0000 | Base（多数组合并列） | 1.0000 | 0.0000 |
| Crime | 0.8861 | Base | 0.8861 | 0.0000 |
| FB-AUTO | 0.8024 | Base | 0.8024 | 0.0000 |
| JF17K-fast | 0.7864 | Base | 0.7864 | 0.0000 |
| M-FB15K-fast | 0.1783 | P1+P5 Logistic Regression | 0.2216 | +0.0433 |
| WikiPeople-fast | 0.0324 | P1+P3 | 0.5522 | +0.5198 |

最终结果进一步证明不存在统一最优的增强配置：

- Enron 最适合 P1+P5 LR；P1、P1+P2 和 upsample 也有接近的正增益。
- P.School 的 P1、P1+P2、P1+P5 LR 差距小于随机种子标准差，实际可视为同一档。
- H.School 中 P1+P5 LR 最好且方差最低，但领先 P1 约 0.0022，优势较小。
- Directors 在 20 dB 已出现 1.0 的天花板效应，此单点无法区分模块。
- Crime 与 FB-AUTO 均以 Base 最佳；对它们强制启用自适应模块会造成负收益。
- JF17K 以 Base 最佳；P1+P5 LR 将 F1 从 0.7864 明显降低到 0.5495，说明 LR
  不能跨数据集固定启用。
- M-FB15K 中 P1+P5 LR 把 F1 从 0.1783 提升到 0.2216，同时三个种子的标准差
  从 0.0314 降到 0.0009，是可信度较高的正向结果。
- WikiPeople 的候选召回率一直约为 0.977，证明主要瓶颈不在候选生成，而在训练和
  分类。P1+P3 的三个种子 F1 分别为 0.1310、0.7882、0.7375，均值虽然达到
  0.5522，但标准差为 0.2986，表现非常不稳定，需要增加种子并排查 seed 123。
- P3 在九个数据集里有六个负收益，其正平均增益完全由 WikiPeople 的异常大提升
  拉动，而且计算成本约为单训练实例的三倍，因此不适合作为全局默认项。
- P4 软可靠度在九个数据集上正负各四个、一个持平，没有形成稳定优势。

从稳健性看，Base 在四个数据集上最佳或并列最佳。P1 的中位增益为 0，P1+P5 LR
的中位增益为 +0.0089，但后者受到 JF17K 的严重负收益影响，九数据集平均增益反而
为 -0.0088。因此，最终不推荐用任何单一增强组合替代 Base。

新增汇总文件：

- [全数据集消融增益热力图](figures_v1_1_optimal/fig_ablation_all_datasets_awgn20_delta.png)
- [各数据集最佳配置图](figures_v1_1_optimal/fig_ablation_all_datasets_awgn20_best.png)
- [完整数值表](figures_v1_1_optimal/ablation_all_datasets_awgn20.csv)

当前证据支持“按数据集自适应选择模块”，不支持把全部模块统一打开。建议配置为：

- Enron、H.School、M-FB15K：优先 P1+P5 LR；
- P.School：P1 即可，P2 的额外收益小于随机波动；
- Directors、Crime、FB-AUTO、JF17K：保持 Base；
- WikiPeople：将 P1+P3 作为待复验的专项配置，暂不纳入默认方案。
