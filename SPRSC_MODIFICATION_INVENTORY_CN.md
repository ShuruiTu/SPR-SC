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

