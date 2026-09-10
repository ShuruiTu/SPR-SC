# SPR-SC 相对原版 SHyRe 的修改清单

> 用途：作为下一轮开发的决策底稿。本文件记录的是被替换前的
> SPR-SC 工作区所实现或规划的内容，不代表这些功能仍存在于当前
> SHyRe 代码根目录。当前可执行基线是原版 SHyRe。

## 1. 基线与当前状态

- 当前根目录：原版 SHyRe（仅做 NumPy 2 的 `np.float`/`np.int` 兼容修复，
  并跳过缺少 `graph_tool` 的 Bayesian-MDL）。
- 被替换的 SPR-SC：以 SHyRe 的“投影图 → 最大团 → 子集候选 → 分类”
  结构为起点，试图加入无线信道、可靠度、语义和实验工程。
- 建议原则：任何重新加入的模块都应先在**无信道、原始 SHyRe 数据划分、
  原始 beta**下复现当前基线，再引入一个变量并做消融。

## 2. 方法层修改

| 模块 | SPR-SC 的修改 | 相对 SHyRe 的差异 | 重新引入建议 |
|---|---|---|---|
| 信道模型 | 16-QAM bit LLR、AWGN/Rayleigh、信道估计误差、边先验、温度校准 | SHyRe 输入为确定性投影图，不建模接收信道 | 最优先；保持 SHyRe train/test 与候选器不动，只替换测试投影观测 |
| 可靠度 | 每对节点输出软概率，阈值形成 hard graph，并记录 ECE | SHyRe 仅使用二值图 | 第二优先；先只用于生成接收图，不要立即进入特征/采样 |
| 候选生成 | 软 quasi-clique、top-K 高可靠区域、可靠度筛选、接受/尝试双预算 | SHyRe 使用最大团及 rho/beta 贪心 child sampler | 风险最高；必须先完整保留 SHyRe `CliqueSampler`，再单独做候选召回消融 |
| rho | 平滑的 tuple-level rho(a,b)，并在未见 cell 上做邻近回退 | SHyRe rho 保存“可恢复训练超边集合 + 组合数成本” | 不建议直接替换；应保留 SHyRe 原 rho，另加新统计作为附加特征 |
| 特征 | 结构特征 + tuple reliability（均值、最小值、几何均值、logit 等）+ SNR + 可选实体 embedding | SHyRe count 为 9 维；motif 为原 C++/Python 特征 | 先保持 count/motif 不变，将信道统计作为单独、可关闭的附加列 |
| 分类 | 标准化、加权 Logistic/MLP、按 SNR/信道阈值校准 | SHyRe 使用 max/nested 两个 MLP，固定 0.5 标签决策 | 不建议先改；先复现双 MLP，再增加一个可选校准层 |
| 语义 | embedding cosine、方差、可用性/缺失指示 | SHyRe 无语义特征 | 最后再加；数据集无 embedding 时不得伪造语义输入 |

## 3. 数据与评估层修改

| 类别 | 原 SPR-SC 做法 | 风险/注意事项 |
|---|---|---|
| 数据加载 | 合并部分 SHyRe split 后做可控子图采样；还支持 HYPER 与 Walmart | 会破坏 SHyRe 原始 train/test 协议；新项目应默认使用原 `data/*/train.txt,test.txt` |
| 信道规模控制 | 对大图可抽样背景非边，避免全节点对 16-QAM 仿真 | 合法但必须在结果表中报告观测对比例；不能称为完整全对偶信道 |
| 指标 | 重建 P/R/F1/Jaccard、大小分层 F1、结构保真度、关系 AUC、completion/检索指标 | 主论文比较先锁定 SHyRe exact-match P/R/F1/Jaccard；其他为附加证据 |
| 消融 | 去 reliability、去 rho、去 channel feature、去 semantic feature、严格团等 | 只有基础 SHyRe 复现后才有解释价值 |
| 外部基线 | 尝试接入 Hyper-SAGNN、Bayesian-MDL、SBRF 等 | 需要相同接收图和候选协议；当前 Bayesian 已禁用，SBRF 无官方实现 |

## 4. 工程层修改

- JSON profile、manifest、JSONL 断点恢复、结果聚合与 bootstrap。
- Docker/Conda/Slurm 脚本，区分本地 smoke 与服务器全量实验。
- channel-to-SHyRe adapter：把测试图替换为接收图，并处理共享节点宇宙。
- MATLAB/Matplotlib 图表与 CSV 汇总。

这些工程能力可以重新建立，但不应先于算法复现；建议先给当前 SHyRe 添加一个
最小 runner（数据集、beta、seed、features、输出 JSON），再逐步加入 manifest 和批处理。

## 5. 推荐重新开发顺序

1. 固定当前 SHyRe count 基线及七个非 DBLP 数据集结果。
2. 增加独立 `channel.py`：仅从干净测试投影产生 hard received graph；测试在高 SNR
   回到接近原 SHyRe 结果。
3. 用该 hard received graph 调用**未改动的** `compute_cliques`、`CliqueSampler`、
   `DataLoader`、双 MLP。
4. 输出信道 SNR 曲线与无信道 SHyRe 对照；此时才确认“信道改动”本身。
5. 以开关方式添加软可靠度特征；每次只增加一组特征。
6. 最后尝试可靠度候选扩张、语义特征、下游任务和大规模数据。

## 6. 当前基准（count，Bayesian-MDL 已跳过）

| 数据集 | SHyRe F1 | Max Clique F1 | ECC F1 |
|---|---:|---:|---:|
| Enron | 0.2285 | 0.0805 | 0.1240 |
| Foursquare | 0.8521 | 0.1754 | 0.8882 |
| Hosts-Virus | 0.6593 | 0.3661 | 0.5816 |
| P.School | 0.6154 | 0.0018 | 0.0383 |
| H.School | 0.7051 | 0.0465 | 0.1288 |
| Directors | 1.0000 | 1.0000 | 1.0000 |
| Crimes | 0.8884 | 0.8812 | 0.8622 |


## 7. 2026-09-09 当前实施进度

### 已完成

1. **原 SHyRe count 基线固定**：七个非 DBLP 数据集的 clean count 基线保存在
   `baseline_count_non_dblp.json`；Bayesian-MDL 默认跳过，CFinder 通过
   `--enable_cfinder` 显式启用。
2. **hard received graph 信道**：`channel.py` 已实现归一化 16-QAM、AWGN、
   Rayleigh、精确 bit LLR、有界背景非边抽样，以及旧 SPR-SC 风格的
   `edge_prior + beta * LLR + temperature + tau_e` 后验判决。
3. **高 SNR 一致性**：Enron、Hosts-Virus 的 AWGN/Rayleigh 60 dB 测试均恢复
   clean 图边集；为避免顺序敏感基线差异，边集一致时返回 clean 图副本。四个
   当前基线（SHyRe、Max Clique、ECC、DEMON）与 clean 对照一致。
4. **SHyRe 主流程保持不变**：只替换测试投影图；`compute_cliques`、
   `CliqueSampler`、`DataLoader` 的候选来源和双 MLP 重建流程未被替换。
5. **软可靠度观测与特征（可选）**：`ChannelObservation` 保留每条传输对的
   后验可靠度。`--soft_reliability` 会保留原特征，并追加五个候选级统计：
   均值、最小值、几何均值、平均 logit、低于 `tau_e` 的边比例。训练候选/标签
   保持 clean；仅独立模拟同参数训练边可靠度以提供训练分布。
6. **模型变体**：目前支持 `SHyRe-count`、`SHyRe-soft-count`、
   `SHyRe-motif`、`SHyRe-soft-motif`。后二者的 soft 版本在原 motif 矩阵后
   拼接同一组五维可靠度特征。
7. **实验 runner**：`run_full_channel_benchmark.py` 支持信道、SNR 列表、
   数据集排除、count/motif 选择和软可靠度开关；结果按信道维度恢复。

### 已完成验证

- 60 dB soft-count：除 Foursquare 外六个数据集、AWGN 与 Rayleigh 均已运行；
  与 clean F1 最大差值为 0.005，说明高 SNR 下基本不退化。
- 10 dB soft-count 对照：相对同一后验 hard graph，Enron/AWGN 提升 0.0316，
  Crimes 有小幅提升；整体平均增益接近 0，H.School/Rayleigh 下降 0.0209。
  因此当前五维直接拼接不是普适增益方案，需继续校准和消融。
- CFinder：Enron 与 Hosts-Virus 的 AWGN 0 dB 已验证可正常结束；默认仍关闭，
  避免大图低 SNR 下的组合爆炸。

### 尚未完成 / 下一步

1. 软可靠度的温度/ECE 校准、Rayleigh 信道估计误差与观测比例 manifest。
2. 对 soft-count / soft-motif 做完整多 SNR、多 seed 消融，并输出带模型变体列的
   汇总 CSV 与曲线。
3. 可靠度驱动候选扩张、soft quasi-clique、top-K 高可靠区域和双预算采样。
4. 语义 embedding、下游任务、HYPER/Walmart 数据、外部基线和完整工程 manifest。

> 当前推荐比较对象是同一信道、同一 SNR 下的 `SHyRe-count` 与
> `SHyRe-soft-count`（或 motif 对应变体）；不要将早期纯 LLR 信道结果与当前
> 后验信道结果直接混合比较。
