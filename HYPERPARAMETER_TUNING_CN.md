# SPR-SC 超参数寻优说明

## 1. 已修复的训练参数接口

原先 `--lr` 虽然出现在命令行中，但没有传入 `MLPClassifier`；MLP 的正则项、
网络结构以及 LR/RF 的主要参数也无法从实验入口设置。现在以下参数均会真实作用于模型：

- MLP：`--epochs`、`--lr`、`--mlp-hidden-layers`、`--mlp-alpha`、
  `--mlp-early-stopping`、`--mlp-validation-fraction`、
  `--mlp-n-iter-no-change`；
- Logistic Regression：`--logistic-c`、`--logistic-max-iter`、
  `--logistic-class-weight`；
- Random Forest：`--rf-n-estimators`、`--rf-max-depth`、
  `--rf-min-samples-leaf`、`--rf-max-features`、`--rf-class-weight`。

这些参数同时可以通过 `main.py` 和 `run_full_channel_benchmark.py` 使用。
默认值按修复前 sklearn 实际采用的有效值设置（例如 MLP 学习率 `1e-3`），以避免仅升级
接口就改变既有基线；显式传入的新值现在会真正生效。

## 2. 验证原则

`tune_hyperparameters.py` 不用测试集 F1 排名。每个配置在训练候选中进行外层留出：

1. 训练候选按最大团、嵌套团分别划分为 outer-fit 和 outer-validation；
2. 若启用自适应阈值，阈值只在 outer-fit 内再次划分得到；
3. 用 outer-validation 计算分类精确率与条件召回率；
4. 将训练候选覆盖率计入目标，避免“候选极少但分类容易”的配置虚高。

排名目标为：

```text
estimated_recall = candidate_recall × candidate_conditional_recall
objective_f1 = F1(validation_precision, estimated_recall)
robust_score = 多随机种子 objective_f1 均值 - λ × 标准差
```

如果某候选组只有单一标签，程序使用常量分类器完成诚实验证；如果少数类只有一个样本，
该组会被跳过，因为无法进行安全的训练/验证划分。

## 3. 推荐命令

先对 Enron 的 20 dB AWGN 环境搜索 30 组配置，每组使用三个随机种子：

```bash
conda run --no-capture-output -n SPR-SC python tune_hyperparameters.py \
  --dataset enron --channel awgn --snr-db 20 \
  --trials 30 --seeds 123 124 125 --workers 3 \
  --output results_tuning_enron_awgn_20db
```

同时搜索软可靠度、匹配噪声训练和信道后验参数：

```bash
conda run --no-capture-output -n SPR-SC python tune_hyperparameters.py \
  --dataset enron --channel awgn --snr-db 20 \
  --trials 60 --seeds 123 124 125 --workers 3 \
  --tune-modules --tune-channel \
  --output results_tuning_enron_awgn_20db_full
```

对于 DBLP、JF17K、M-FB15K、WikiPeople，`--candidate-generator auto` 会自动选择
`shyre_fast`；其他已登记数据集自动使用 `shyre`。也可以显式覆盖。

## 4. 断点续跑与输出

重复执行完全相同的命令即可续跑。脚本根据稳定的配置哈希与随机种子跳过已成功单元，
失败或超时单元会重新尝试。每完成一个单元都会刷新排名，因此服务器断连时已有结果仍然有效。

输出目录包含：

- `search_manifest.json`：搜索参数、候选配置及配置 ID；
- `trial_results.jsonl`：逐个“配置 × 种子”的追加式断点记录；
- `ranking.csv`：完整度、均值、标准差、稳健分数及配置排名；
- `best_config.json`：最优配置、内部验证命令与移除验证开关后的最终测试命令；
- `summary.json`：当前完成数量；
- `logs/`：每次运行的标准输出和结构化指标。

注意：`best_config.json` 中的 `final_evaluation_command` 才用于最终测试；寻优过程中应以
`validation_command` 为准。建议在多个数据集或 SNR 上分别寻优后，再决定采用单数据集最优参数、
按数据集配置，还是选择跨数据集平均更稳健的一组统一参数。
