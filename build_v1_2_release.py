#!/usr/bin/env python3
"""Build the self-contained SPR-SC v1.2 result delivery directory."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / '1.2版本'
DATASETS = ['enron', 'fb_auto', 'crime', 'directors', 'school', 'school2',
            'jf17k', 'm_fb15k', 'wikipeople']
DISPLAY = {
    'enron': 'Enron', 'fb_auto': 'FB-AUTO', 'crime': 'Crime',
    'directors': 'Directors', 'school': 'P.School', 'school2': 'H.School',
    'jf17k': 'JF17K', 'm_fb15k': 'M-FB15K', 'wikipeople': 'WikiPeople',
}


def copy(source, destination):
    source, destination = ROOT / source, OUT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_if_exists(source, destination):
    if (ROOT / source).exists():
        copy(source, destination)


def load_csv(path):
    with (ROOT / path).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def plot_mlp_benchmark():
    rows = load_csv(
        'results_mlp_tuning_and_evaluation_awgn20/all_datasets_evaluation_summary.csv')
    methods = ['SHyRe', 'Max Clique', 'ECC', 'DEMON']
    lookup = {(row['dataset'], row['method']): row for row in rows}
    x = np.arange(len(DATASETS)); width = 0.19
    colors = ['#1565c0', '#ef6c00', '#2e7d32', '#6a1b9a']
    fig, ax = plt.subplots(figsize=(15.5, 6.4), constrained_layout=True)
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = [float(lookup[(dataset, method)]['f1_mean']) for dataset in DATASETS]
        errors = [float(lookup[(dataset, method)]['f1_std']) for dataset in DATASETS]
        ax.bar(x + (index - 1.5) * width, values, width, yerr=errors,
               capsize=2, label=method, color=color, alpha=.9)
    ax.set_xticks(x, [DISPLAY[d] for d in DATASETS], rotation=22, ha='right')
    ax.set_ylim(0, 1.04); ax.set_ylabel('Exact-match F1')
    ax.set_title('Per-dataset best MLP evaluation at AWGN 20 dB (mean ± std, 3 seeds)')
    ax.grid(axis='y', alpha=.2); ax.legend(ncol=4, loc='upper center')
    target = OUT / '01_图片/v1_2新增/fig_v1_2_mlp_dataset_comparison.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=240, bbox_inches='tight'); plt.close(fig)


def plot_informative_ablation():
    rows = load_csv('results_aligned_mlp_ablation_awgn20/ablation_summary.csv')
    datasets = ['enron', 'school']
    variants = [row['variant'] for row in rows if row['dataset'] == 'enron']
    lookup = {(row['dataset'], row['variant']): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        means = [float(lookup[(dataset, variant)]['f1_mean']) for variant in variants]
        stds = [float(lookup[(dataset, variant)]['f1_std']) for variant in variants]
        labels = [lookup[(dataset, variant)]['label'] for variant in variants]
        bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=2,
                      color=plt.cm.Blues(np.linspace(.35, .9, len(variants))))
        ax.set_xticks(range(len(variants)), labels, rotation=55, ha='right', fontsize=8)
        ax.set_ylim(0, max(means) * 1.28); ax.set_ylabel('Exact-match F1')
        ax.set_title(DISPLAY[dataset]); ax.grid(axis='y', alpha=.2)
        ax.bar_label(bars, labels=['{:.3f}'.format(value) for value in means],
                     fontsize=6, rotation=90, padding=2)
    fig.suptitle('Parameter-aligned MLP ablation: two datasets with clearest module effects')
    target = OUT / '01_图片/v1_2新增/fig_v1_2_ablation_enron_pschool.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=240, bbox_inches='tight'); plt.close(fig)


def build_manifest():
    records = []
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or path.name == 'MANIFEST.csv':
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            'relative_path': str(path.relative_to(OUT)),
            'size_bytes': path.stat().st_size,
            'sha256': digest,
        })
    with (OUT / 'MANIFEST.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['relative_path', 'size_bytes', 'sha256'],
            lineterminator='\n')
        writer.writeheader(); writer.writerows(records)


def normalize_text_line_endings():
    """Keep generated delivery text friendly to Git and Unix tooling."""
    text_suffixes = {'.csv', '.json', '.jsonl', '.md', '.py', '.txt'}
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            content = path.read_bytes()
            normalized = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            if normalized != content:
                path.write_bytes(normalized)


def main():
    OUT.mkdir(exist_ok=True)

    # Stable v1.1 figure set retained as historical/core evidence.
    for path in sorted((ROOT / 'figures_v1_1_optimal').glob('*.png')):
        copy(path.relative_to(ROOT), Path('01_图片/v1_1主图') / path.name)

    # New v1.2 aligned tuning and ablation figures.
    copy('results_aligned_mlp_ablation_awgn20/fig_ablation_all4_delta.png',
         '01_图片/v1_2新增/fig_v1_2_ablation_all4_delta.png')
    copy('results_aligned_mlp_ablation_awgn20/fig_ablation_top2_f1.png',
         '01_图片/v1_2新增/fig_v1_2_ablation_top2_absolute_f1.png')
    plot_mlp_benchmark()
    plot_informative_ablation()

    # Final rho(n,k) visualization set (Enron, n/k <= 13).
    for name in (
        'fig_rho_enron_bubble_k13.png',
        'fig_rho_enron_bubble_k13.pdf',
        'fig_rho_enron_composite_k13.png',
        'fig_rho_enron_composite_k13.pdf',
        'fig_rho_enron_bubble_alignment_triptych_k13.png',
        'fig_rho_enron_bubble_alignment_triptych_k13.pdf',
        'fig_rho_enron_train_query.png',
        'fig_rho_enron_train_query.pdf',
    ):
        copy_if_exists(
            Path('results_rho_heatmap_preview') / name,
            Path('01_图片/v1_2新增/rho_nk') / name,
        )

    # Consolidated tables and legacy workbook.
    copy('figures_v1_1_optimal/SPR_SC_v1_1_complete_results.xlsx',
         '02_数据/v1_1历史数据/SPR_SC_v1_1_complete_results.xlsx')
    copy('figures_v1_1_optimal/ablation_all_datasets_awgn20.csv',
         '02_数据/v1_1历史数据/ablation_all_datasets_awgn20.csv')
    copy('figures_v1_1_optimal/ablation_enron_crime_awgn20.csv',
         '02_数据/v1_1历史数据/ablation_enron_crime_awgn20.csv')
    copy_if_exists('figures_current_six_datasets_snr/six_dataset_snr_plot_data.csv',
                   '02_数据/v1_1历史数据/six_dataset_snr_plot_data.csv')

    copy('results_mlp_tuning_and_evaluation_awgn20/all_datasets_evaluation_summary.csv',
         '02_数据/v1_2最优MLP/all_datasets_evaluation_summary.csv')
    copy('results_mlp_tuning_and_evaluation_awgn20/all_datasets_evaluation_detailed.csv',
         '02_数据/v1_2最优MLP/all_datasets_evaluation_detailed.csv')
    copy('results_mlp_tuning_and_evaluation_awgn20/status.json',
         '02_数据/v1_2最优MLP/status.json')
    for dataset in DATASETS:
        base = Path('results_mlp_tuning_and_evaluation_awgn20') / dataset
        copy(base / 'evaluation_results.jsonl',
             Path('02_数据/v1_2最优MLP/逐数据集') / dataset / 'evaluation_results.jsonl')
        for name in ('best_config.json', 'ranking.csv', 'summary.json',
                     'search_manifest.json'):
            copy(base / 'tuning' / name,
                 Path('02_数据/v1_2最优MLP/逐数据集') / dataset / name)

    for name in ('ablation_summary.csv', 'ablation_detailed.csv',
                 'ablation_results.jsonl', 'selected_top2.json'):
        copy(Path('results_aligned_mlp_ablation_awgn20') / name,
             Path('02_数据/v1_2参数对齐消融') / name)
    copy_if_exists('results_rho_heatmap_preview/enron_rho_cells.csv',
                   '02_数据/v1_2_rho_nk/enron_rho_cells.csv')

    # Documentation and exact scripts used to produce the delivery.
    copy('figures_v1_1_optimal/README_RESULTS_CN.md',
         '03_说明文档/README_v1_1_RESULTS_CN.md')
    for name in ('V1_1_FIGURE_RESULT_ANALYSIS_CN.md', 'V1_1_OPTIMIZATION_LOG_CN.md',
                 'HYPERPARAMETER_TUNING_CN.md', 'SPRSC_MODIFICATION_INVENTORY_CN.md'):
        copy_if_exists(name, Path('03_说明文档') / name)
    for name in ('plot_v1_1_optimal_results.py', 'export_v1_1_delivery_excel.py',
                 'tune_hyperparameters.py', 'run_mlp_tuning_all.py',
                 'run_aligned_mlp_ablation.py', 'plot_rho_heatmap.py',
                 'plot_rho_alternatives.py', 'build_v1_2_release.py'):
        copy_if_exists(name, Path('04_复现脚本') / name)

    normalize_text_line_endings()
    build_manifest()
    print('Built:', OUT)


if __name__ == '__main__':
    main()
