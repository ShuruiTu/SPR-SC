# 当前图 1--12 结果整理与符合度说明

生成日期：2026-09-11。统一图片位于 `figures_current_complete/`；图仅使用当前工程实际产生的结构化结果，不用历史论文数字补齐缺失单元。

## 图与数据范围

| 图 | 文件 | 当前数据与协议 | 状态 |
|---|---|---|---|
| Fig.1 | `fig_1_awgn_all_completed_snr.png` | Enron、Hosts-Virus、P.School、H.School、Directors、Crime、FB-AUTO、JF17K；AWGN；0:2:20 dB；单 seed | 可用 |
| Fig.2 | `fig_2_rayleigh_all_completed_snr.png` | 同 Fig.1，但使用 Rayleigh | 可用 |
| Fig.3 | `fig_3_f1_ablation.png` | Enron；AWGN/Rayleigh；10 dB；count/soft × strict/SHyRe | 可用 |
| Fig.4 | `fig_4_candidate_recall_ablation.png` | 同 Fig.3；候选召回率 | 可用 |
| Fig.9 | `fig_9_retention_f1.png` | 六个小数据集宏平均；AWGN 60 dB；α=0.2,0.4,0.6,0.8,1.0 | 可用 |
| Fig.10 | `fig_10_retention_storage.png` | 同 Fig.9；接收投影存储量 | 可用 |
| Fig.9.5 | `fig_9_5_f1_vs_storage.png` | 同 Fig.9；横轴接收投影存储量、纵轴宏平均 F1 | 可用（临时编号） |
| Fig.11 | `fig_11_enron_size_stratified.png` | Enron；AWGN 60 dB；按超边规模分层 | 可用 |
| Fig.12 | `fig_12_fb_auto_size_stratified.png` | FB-AUTO；AWGN 60 dB；按超边规模分层 | 可用 |

Fig.1--2 暂不纳入 DBLP 与 Foursquare：它们现有结果的 SNR 网格、模型变体或完成状态与此处的统一单 seed、0:2:20 dB 协议不一致。M-FB15K、WikiPeople已经完成数据导入，但尚未完成全源预检与曲线网格，也未纳入图片。

JF17K 的曲线采用明确标注的 `SHyRe-fast-count` / `SHyRe-fast-soft-count`：完整源上的原始精确 SHyRe 候选分布需要最大团×训练超边的二次包含枚举，不能扩展；该结果不可与原始精确 SHyRe 直接当作同一模型宣称。

## 是否符合预期

### 符合预期

1. **高 SNR 复原。** FB-AUTO 在 AWGN 20 dB 的 count/soft F1 为 0.801/0.805；JF17K 的 fast-count/fast-soft 为 0.791/0.786。投影在高 SNR 下能够恢复到强基线或接近强基线的水平。
2. **低 SNR 降级。** 两个新增数据集在 0 dB 的 F1 接近零；这是当前 hard posterior received graph 在低 SNR 下失去足够投影结构的预期现象，而不是绘图缺值。
3. **候选机制有效。** Enron 10 dB 下，AWGN 的候选召回从 strict 的 0.052 提升至 SHyRe 的 0.352，最终 count F1 从 0.064 提升至 0.213；Rayleigh 的候选召回从 0.034 提升至 0.386，最终 count F1 从 0.038 提升至 0.175。说明候选覆盖提升可以转化为最终性能提升。
4. **α--存储关系正确。** Fig.10 的接收投影存储随 α 基本线性增加，符合“发送前随机保留投影边”的实现定义。
5. **α--性能关系正确。** Fig.9 中六数据集宏平均 F1 随 α 单调增加（count：0.056、0.133、0.240、0.310、0.683），体现预期的信息--性能折中。
6. **分层差异可解释。** Fig.12 中 FB-AUTO 的 2 元边几乎可由最大团恢复，而 4 元及 5+ 元超边更依赖候选与分类；这与投影歧义随超边规模上升的预期一致。

### 目前不符合或仅部分符合预期

1. **Rayleigh 高 SNR 仍显著弱于 AWGN。** 例如 FB-AUTO 在 20 dB：AWGN count F1=0.801，Rayleigh count F1=0.276；JF17K 对应为 0.791 与 0.450。若论文目标是“高 SNR Rayleigh 必须回到无噪基线”，当前 hard posterior 判决尚不满足，需检查衰落补偿、阈值和信道先验。
2. **soft reliability 未形成稳定优势。** soft-count 在 FB-AUTO AWGN 20 dB 仅比 count 高 0.004，在多个格子持平或更低；它目前是可比较模块，而非已被证明优于硬特征的模块。
3. **候选召回提升不总能转化为 F1 优势。** Enron 的候选覆盖虽有提升，但最终 F1仍偏低；Crime 和 Directors 的 strict 候选已较好，SHyRe 增益很小。这符合数据结构异质性，但不支持“所有数据集必然提升”的强主张。
4. **JF17K 的 fast 模块与原始 SHyRe 不等价。** 它解决了全源可运行性，但需要在图例和正文中独立标注；不能用其结果证明精确 SHyRe 的规模表现。
5. **单 seed 只能作为初版证据。** 当前所有汇总曲线均为单 seed，没有置信区间；不宜据此作显著性或稳健性结论。
6. **Fig.11 的 Enron 单元边未绘制。** 图仅展示 2、3、4、5+ 元组；Enron 测试集中的 1 元超边不属于当前以投影边为输入的可恢复目标。该排除应在论文图注中明确。

## 推荐图注中的限制声明

> Curves report one matched random seed. JF17K uses the separately labelled scalable `SHyRe-fast` candidate sampler. DBLP, Foursquare, M-FB15K and WikiPeople are excluded from the uniform Fig.1--2 composite until the same protocol is completed.
