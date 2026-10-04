# SPR-SC v1.1 实验图片与结果说明

更新时间：2026-09-18

## 1. 交付内容与范围

本文件夹是 SPR-SC v1.1 当前实验结果的独立交付包，包含论文预设图、完整数值
工作簿和消融实验数据。接收者无需访问原始日志即可查看当前结果。

本次交付遵循以下范围：

- 图 1、图 2 仅展示两个代表性数据集：Enron 与 FB-AUTO。
- 其余已完成数据集的 SNR 曲线数值不再拼入图 1、图 2，而是完整写入
  `SPR_SC_v1_1_complete_results.xlsx`。
- 主曲线包含 AWGN 与 Rayleigh 两种信道，SNR 为 0–20 dB、间隔 2 dB。
- 每个主实验点使用随机种子 123、124、125；图中曲线为均值，阴影为总体标准差。
- 主曲线比较 SHyRe、Max Clique、ECC、DEMON；Bayesian-MDL 和 CFinder 未纳入
  当前主实验。
- JF17K、M-FB15K、WikiPeople 使用有界候选生成的 `SHyRe-fast` 路径，不能与
  标准 SHyRe 路径在实现成本上直接等同。
- DBLP 仍在执行，未将不完整数据写入本次图表和 Excel。
- Hosts-Virus、Foursquare 按当前实验安排排除。

主要指标为 exact-match reconstruction F1：只有预测超边与真实超边完全一致时才
计为正确。

## 2. 文件索引

### 主曲线

- [fig_1_v1_1_optimal_awgn_snr.png](fig_1_v1_1_optimal_awgn_snr.png)：
  Enron 与 FB-AUTO 的 AWGN F1–SNR 曲线。
- [fig_2_v1_1_optimal_rayleigh_snr.png](fig_2_v1_1_optimal_rayleigh_snr.png)：
  Enron 与 FB-AUTO 的 Rayleigh F1–SNR 曲线。

### 候选生成与软可靠度消融

- [fig_3_v1_1_f1_ablation.png](fig_3_v1_1_f1_ablation.png)：
  Enron 在 10 dB 下的最终重构 F1 消融。
- [fig_4_v1_1_candidate_recall_ablation.png](fig_4_v1_1_candidate_recall_ablation.png)：
  与图 3 相同配置的候选召回率。

### 存储量与重构质量

- [fig_9_v1_1_retention_f1.png](fig_9_v1_1_retention_f1.png)：
  投影保留率与五数据集平均 F1。
- [fig_9_5_v1_1_f1_vs_storage.png](fig_9_5_v1_1_f1_vs_storage.png)：
  将图 9、图 10 合并成 F1–存储量关系。
- [fig_10_v1_1_retention_storage.png](fig_10_v1_1_retention_storage.png)：
  投影保留率与接收投影存储量。

图 9–10 的五个数据集为 Enron、P.School、H.School、Directors、Crime；信道为
AWGN 60 dB，使用三个随机种子。

### 超边规模分层

- [fig_11_v1_1_enron_size_stratified.png](fig_11_v1_1_enron_size_stratified.png)：
  Enron 按真实超边规模分层的 F1。
- [fig_12_v1_1_fb_auto_size_stratified.png](fig_12_v1_1_fb_auto_size_stratified.png)：
  FB-AUTO 按真实超边规模分层的 F1。

### 模块消融

- [fig_ablation_enron_crime_awgn20.png](fig_ablation_enron_crime_awgn20.png)：
  Enron、Crime 在 AWGN 20 dB 下的 F1 与候选召回率详细消融。
- [fig_ablation_all_datasets_awgn20_delta.png](fig_ablation_all_datasets_awgn20_delta.png)：
  九个数据集上各模块相对 Base 的 F1 增减热力图。
- [fig_ablation_all_datasets_awgn20_best.png](fig_ablation_all_datasets_awgn20_best.png)：
  每个数据集在当前搜索配置中的最佳模型。
- [ablation_enron_crime_awgn20.csv](ablation_enron_crime_awgn20.csv)：
  Enron、Crime 详细消融数值。
- [ablation_all_datasets_awgn20.csv](ablation_all_datasets_awgn20.csv)：
  九数据集完整消融数值、候选召回率、运行时间与 P2 可用性标记。

### Excel 数值工作簿

- [SPR_SC_v1_1_complete_results.xlsx](SPR_SC_v1_1_complete_results.xlsx)

## 3. Excel 工作簿说明

工作簿包含 9 个已完成数据集、2 种信道、11 个 SNR 点、3 个随机种子和 4 种方法，
共 2376 条逐种子 SNR 结果。各工作表含义如下：

| 工作表 | 内容 |
|---|---|
| `说明` | 实验协议、纳入范围、排除项和当前研究状态 |
| `SNR全部逐种子` | 每个数据集/信道/SNR/种子/方法的原始 F1 |
| `SNR均值标准差` | 对三个种子汇总后的 F1 均值、标准差和样本数 |
| `AWGN透视` | 便于直接制图的 AWGN 宽表 |
| `Rayleigh透视` | 便于直接制图的 Rayleigh 宽表 |
| `消融完整结果` | 九数据集的模块消融均值、标准差、增益、召回率和运行时间 |
| `消融最优配置` | 每个数据集的 Base、当前最佳配置及相对增益 |

工作簿仍包含 Enron 和 FB-AUTO 数值，以保证数据完整；`图1/2展示`列可以区分
图中展示数据和仅在 Excel 中提供的数据。

## 4. 图 1：AWGN 主曲线

图 1 展示加性高斯白噪声信道下，F1 随 SNR 的变化。

### Enron

- SPR-SC/SHyRe-count 在整个 SNR 区间总体优于 Max Clique、ECC、DEMON。
- F1 从 0 dB 的约 0.13 上升到 8–10 dB 的约 0.31。
- 高 SNR 区域没有继续显著上升，而是在约 0.27–0.31 间波动，说明主要瓶颈已经
  从信道错误转移到候选覆盖与分类。

### FB-AUTO

- 低 SNR 下所有方法接近失效；约 8 dB 后开始明显恢复。
- SHyRe-count 在大部分中高 SNR 点领先其他基线。
- 20 dB 附近 F1 约为 0.80，与 Max Clique 接近但仍略高。
- 曲线呈明显门限型，说明该数据集对投影边错误较敏感。

## 5. 图 2：Rayleigh 主曲线

图 2 使用 Rayleigh 衰落、完美 CSI 和 `channel_beta=1.5`。总体上 Rayleigh 比 AWGN
更困难，因为除了高斯噪声外还存在随机幅度衰落。

### Enron

- SHyRe-count 仍明显优于传统基线。
- 10 dB 附近 F1 约为 0.30，与 AWGN 接近。
- 20 dB 附近约为 0.28，说明高 SNR 下仍存在衰落引起的随机损失。

### FB-AUTO

- 中高 SNR 下保持领先，但恢复速度慢于 AWGN。
- 20 dB F1 约为 0.65，低于 AWGN 的约 0.80。
- Rayleigh 衰落对 FB-AUTO 的影响明显大于对 Enron 的影响。

## 6. 图 3、图 4：候选生成与软可靠度

实验数据集为 Enron，SNR 为 10 dB，同时比较 AWGN 与 Rayleigh。

四组配置为：

1. `count strict`：count 特征 + 严格最大团候选；
2. `count SHyRe`：count 特征 + SHyRe 候选；
3. `soft-v2 strict`：软可靠度特征 + 严格最大团候选；
4. `soft-v2 SHyRe`：软可靠度特征 + SHyRe 候选。

主要结论：

- SHyRe 候选生成将候选召回率从约 0.04–0.05 提升到约 0.35–0.39。
- 最终 F1 从约 0.03–0.05 提升到约 0.30–0.32。
- AWGN 下 soft-v2 的 F1 约 0.3203，count 约 0.3145，只有小幅提升。
- Rayleigh 下 soft-v2 与 count 分别约 0.3043、0.3041，差异可忽略。

因此，当前明确的核心收益来自 SHyRe 候选生成，而不是软可靠度特征。

## 7. 图 9、图 9.5、图 10：存储–精度权衡

投影保留率为 20%、40%、60%、80%、100% 时，平均存储量约为 1097、2194、
3290、4387、5484 个单位，基本线性增长；F1 则约为 0.05、0.19–0.20、0.31、
0.39、0.70，呈明显非线性。

- 60% 保留率节省约 40% 存储，但 F1 只有完整保留的约 44%。
- 80% 保留率节省约 20% 存储，但 F1 只有完整保留的约 55%。
- 从 80% 到 100% 的最后 20% 存储带来最大精度提升。
- count 与 soft 两条曲线接近，当前软可靠度未改善随机压缩条件下的存储–精度权衡。

图 9.5 可以用于展示当前权衡关系，但现有随机保留策略尚不能证明明显压缩优势。

## 8. 图 11、图 12：超边规模分层

### Enron

- 规模 2 的 SHyRe F1 约为 0.4314，是主要优势来源。
- 规模 3、4 分别约为 0.0567、0.0262，恢复能力较弱。
- 规模 5+ 回升到约 0.1147，但仍明显低于规模 2。

### FB-AUTO

- 规模 2 的 F1 约为 0.9637。
- 规模 4、5+ 分别约为 0.7356、0.7339。
- 规模 3 的 F1 为 0，是需要继续排查的异常层；可能与该层样本量、训练标签或
  候选覆盖有关。

## 9. 模块消融定义

模块消融使用 AWGN 20 dB 和三个随机种子。

| 标记 | 含义 |
|---|---|
| Base | 匹配信道训练、MLP、固定 0.5 判决阈值 |
| P1 | 使用训练候选验证集分别选择最大团与嵌套团阈值 |
| P2 | 在候选搜索图中加入受阈值与数量上限约束的高后验软边 |
| P3 | 使用测试 SNR 附近的 `[-2, 0, +2] dB` 多信道训练窗口 |
| P4 | 加入 distribution 级软可靠度特征 |
| P5 | 将 MLP 替换为 Logistic Regression |
| upsample | 对训练候选中的正类进行上采样 |

JF17K、M-FB15K、WikiPeople 必须使用 `SHyRe-fast`。当前尚无
`shyre_fast_channel_aware`，因此 P2、P1+P2+P3 在这三个数据集上标记为 N/A，
没有用重复运行冒充有效消融。

## 10. 消融实验结果

| 数据集 | Base F1 | 当前最佳配置 | 最佳 F1 | 相对 Base |
|---|---:|---|---:|---:|
| Enron | 0.2304 | P1+P5 LR | 0.3080 | +0.0775 |
| P.School | 0.6156 | P1+P2 | 0.6282 | +0.0127 |
| H.School | 0.6984 | P1+P5 LR | 0.7073 | +0.0089 |
| Directors | 1.0000 | Base/多配置并列 | 1.0000 | 0.0000 |
| Crime | 0.8861 | Base | 0.8861 | 0.0000 |
| FB-AUTO | 0.8024 | Base | 0.8024 | 0.0000 |
| JF17K-fast | 0.7864 | Base | 0.7864 | 0.0000 |
| M-FB15K-fast | 0.1783 | P1+P5 LR | 0.2216 | +0.0433 |
| WikiPeople-fast | 0.0324 | P1+P3 | 0.5522 | +0.5198 |

### 可支持的结论

- Enron、H.School、M-FB15K 在 P1+P5 LR 下获得正收益。
- P.School 的 P1、P1+P2、P1+P5 LR 差距很小，P2 的额外收益小于种子波动。
- Crime、FB-AUTO、JF17K 以 Base 最佳，不适合默认开启所有增强模块。
- Directors 在 20 dB 达到 F1=1.0，存在天花板效应，该点无法区分模块。
- P4 软可靠度在九个数据集上正负各四个、一个持平，没有稳定优势。
- P1+P5 LR 在 M-FB15K 上把均值从 0.1783 提升到 0.2216，并将标准差从
  0.0314 降到 0.0009，是较可靠的正向结果。

### 需要谨慎解释的结果

- P1+P5 LR 在 JF17K 上将 F1 从 0.7864 降到 0.5495，证明 LR 不能全局启用。
- WikiPeople 的候选召回率约为 0.977，主要瓶颈是训练与分类，而非候选覆盖。
- WikiPeople 的 P1+P3 三个种子 F1 为 0.1310、0.7882、0.7375，均值 0.5522，
  但标准差达到 0.2986；这是高潜力但不稳定的结果，需要更多随机种子复验。
- P3 在九个数据集里有六个负收益，正平均增益主要由 WikiPeople 拉动，而且训练
  成本约为单信道训练的三倍，不适合成为全局默认模块。

当前最合理的策略是按数据集选择模块：Enron、H.School、M-FB15K 优先考虑
P1+P5 LR；P.School 使用 P1 即可；Directors、Crime、FB-AUTO、JF17K 保持
Base；WikiPeople 将 P1+P3 作为待复验的专项配置。

## 11. 限制与后续工作

1. DBLP 尚未完成，因此本交付包不包含 DBLP 曲线。
2. WikiPeople P1+P3 需要增加随机种子并重点排查 seed 123。
3. FB-AUTO 规模 3 的分层结果需要检查样本量、候选召回与标签分布。
4. 随机投影保留策略的精度损失较大，需要研究按后验可靠度或信息量选择的策略。
5. 当前自动调节仅覆盖分类阈值；模型类型、信道参数、训练窗口和分类器参数尚未形成
   完整的训练集内部自动寻优流程。
6. 后续参数选择必须只使用训练/验证数据，锁定配置后再评估测试集，避免测试泄漏。

**消融实验的模型优化和参数寻优仍在进行，本文档与本文件夹反映的是当前阶段结果，
不是最终定型模型或最终参数。**
