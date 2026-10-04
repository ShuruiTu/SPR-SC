# SPR-SC 1.2 版本：图片与数据交付说明

更新日期：2026-10-04

本目录集中保存当前已经完成、具备明确实验协议和可追溯数据来源的图片与数值结果。
原始结果目录保持不变；本目录是用于发送、汇报和后续论文整理的只读式交付副本。

## 1. 目录结构

```text
1.2版本/
├── 01_图片/
│   ├── v1_1主图/                 # 已成型的图1–图12及旧模块消融图
│   └── v1_2新增/                 # 最优MLP、参数对齐消融和rho(n,k)图
├── 02_数据/
│   ├── v1_1历史数据/             # v1.1 Excel、SNR与旧消融汇总
│   ├── v1_2最优MLP/              # 九数据集测试结果、排名和最优参数
│   └── v1_2参数对齐消融/         # 四数据集完整消融明细
├── 03_说明文档/                  # 历史实验说明、优化日志与参数寻优说明
├── 04_复现脚本/                  # 绘图、寻优、消融和本目录构建脚本
├── MANIFEST.csv                  # 文件大小与SHA-256校验值
└── README_RESULTS_CN.md
```

## 2. v1.1 主图

`01_图片/v1_1主图/` 保留此前已经成型的完整图组：

- 图1：Enron、FB-AUTO 的 AWGN SNR 曲线；
- 图2：Enron、FB-AUTO 的 Rayleigh SNR 曲线；
- 图3：Enron 10 dB 下的重构 F1 消融；
- 图4：与图3对应的候选召回率；
- 图9：投影保留率与 F1；
- 图9.5：F1 与存储量关系；
- 图10：接收投影与重构结果的存储量；
- 图11、图12：Enron、FB-AUTO 的超边规模分层结果；
- v1.1 多数据集模块消融图。

这些图片沿用 v1.1 固定实验协议，不能与 v1.2 的“每数据集独立最优 MLP”数值直接
拼接为同一实验曲线。原始解释参见 `03_说明文档/README_v1_1_RESULTS_CN.md`。

## 3. v1.2 新增图片

### 3.1 九数据集最优 MLP 对比

`fig_v1_2_mlp_dataset_comparison.png` 比较九个数据集上的最优 MLP SHyRe 与 Max
Clique、ECC、DEMON。协议为 AWGN 20 dB、三个随机种子；每个数据集先在训练集内部
进行 15 组 MLP 参数搜索，再用独立测试集报告结果。

主要 SHyRe F1 均值：

| 数据集 | F1 |
|---|---:|
| Enron | 0.3071 |
| FB-AUTO | 0.8073 |
| Crime | 0.8496 |
| Directors | 1.0000 |
| P.School | 0.6242 |
| H.School | 0.7073 |
| JF17K | 0.6778 |
| M-FB15K | 0.1699 |
| WikiPeople | 0.3594 |

### 3.2 参数对齐消融

参数对齐消融仅使用 Enron、FB-AUTO、P.School、H.School。每个数据集固定自身最优
MLP 参数，比较 P1–P4、匹配信道训练、类别平衡、完整组合及逐项移除；共
`4 × 13 × 3 = 156` 个实验单元，全部成功。

- `fig_v1_2_ablation_all4_delta.png`：四个数据集相对 Optimal Base 的 F1 增益热力图；
- `fig_v1_2_ablation_top2_absolute_f1.png`：按绝对最佳 F1 自动选择 FB-AUTO、H.School；
- `fig_v1_2_ablation_enron_pschool.png`：选择模块效应最清晰的 Enron、P.School。

主要结论：

- P1 自适应阈值是最稳定的有效模块；去除 P1 后 Enron 下降约 0.038、P.School
  下降约 0.107；
- P2 在 20 dB 下未提高候选召回率；
- P3 对 P.School 有约 0.0044 的小幅收益，但运行开销明显增加；
- P4 软可靠度未形成稳定正收益；
- 全部模块同时开启在四个数据集上都不是最佳配置。

### 3.3 Enron rho(n,k) 分布与对齐

`v1_2新增/rho_nk/` 保存基于当前 Enron `train.txt/test.txt` 计算的 rho(n,k)
图组，展示范围统一为 `n <= 13, k <= 13`：

- `fig_rho_enron_bubble_alignment_triptych_k13`：推荐主图，横向组合训练气泡图、
  查询气泡图与单元格对齐散点图；
- `fig_rho_enron_bubble_k13`：独立双气泡图，面积编码截断后的 log10 rho 密度，
  颜色编码查询相对训练的差值；
- `fig_rho_enron_composite_k13`：k 维密度指纹、逐 k 对齐误差、ECDF 和对齐散点；
- `fig_rho_enron_train_query`：用于和原论文定义核对的传统灰度图。

在 `n,k <= 13` 的共同有效单元格上，对齐散点的 Pearson 相关系数为 0.864，
log10 尺度 MAE 为 0.415。逐单元格原始值位于
`02_数据/v1_2_rho_nk/enron_rho_cells.csv`。

## 4. 数据文件说明

### 4.1 v1.1 历史数据

- `SPR_SC_v1_1_complete_results.xlsx`：v1.1 主图和曲线的综合工作簿；
- `six_dataset_snr_plot_data.csv`：六数据集 SNR 数值；
- 两个 `ablation_*.csv`：旧固定协议下的模块消融汇总。

### 4.2 v1.2 最优 MLP

- `all_datasets_evaluation_summary.csv`：按数据集、方法聚合的均值和标准差；
- `all_datasets_evaluation_detailed.csv`：逐数据集、逐种子详细结果；
- `逐数据集/<dataset>/best_config.json`：该数据集当前搜索空间内的最优参数；
- `ranking.csv`：15 个配置的训练集内部验证排名；
- `search_manifest.json`：完整搜索空间和配置 ID；
- `evaluation_results.jsonl`：最终测试集结构化指标。

“最优”仅表示 MLP、AWGN 20 dB、当前 15 组搜索空间和三个种子范围内最优，不表示
理论全局最优。

### 4.3 v1.2 参数对齐消融

- `ablation_summary.csv`：52 个“数据集 × 配置”聚合结果；
- `ablation_detailed.csv`：156 个逐种子实验单元；
- `ablation_results.jsonl`：结构化原始记录；
- `selected_top2.json`：Top-2 自动选择规则和结果。

### 4.4 v1.2 rho(n,k) 数据

- `enron_rho_cells.csv`：训练/查询划分中各 `(n,k)` 单元格的分子、分母、rho 与
  log10 rho；图片仅显示 `n,k <= 13`，CSV 保留完整原始统计以便复核。

## 5. 当前限制

- JF17K 和 WikiPeople 的测试结果对随机种子较敏感；
- M-FB15K 的召回率偏低；
- v1.2 参数搜索尚未联合优化所有信道和候选模块；
- P2 在 `shyre_fast` 数据集上仍不可用；
- 本目录未包含已明确放弃的 DBLP、Hosts-Virus 和 Foursquare 正式结果；
- 主曲线和参数对齐消融属于不同实验协议，引用时必须分别说明。

## 6. 建议引用方式

- 展示完整系统结果：使用 v1.1 图1、图2、图9–图12；
- 展示当前最优 MLP 横向性能：使用 `fig_v1_2_mlp_dataset_comparison.png`；
- 展示模块机制：优先使用 Enron、P.School 参数对齐消融图；
- 展示“基础配置已足够、叠加模块无益”：使用 FB-AUTO、H.School Top-2 图；
- 任何图表数值应同时引用其对应 CSV/JSON，而不是从图片反向读取。
