#!/usr/bin/env python3
"""Parameter-aligned MLP module ablation on four selected datasets."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {'enron': 1000, 'fb_auto': 20000, 'school': 350000, 'school2': 60000}
VARIANTS = {
    'optimal_base': {},
    'minus_p1': {'p1': False},
    'plus_p2': {'p2': True},
    'plus_matched': {'matched': True},
    'plus_p3': {'p3': True},
    'plus_p4': {'p4': True},
    'balance_toggle': {'balance_toggle': True},
    'full': {'full': True},
    'full_minus_p1': {'full': True, 'p1': False},
    'full_minus_p2': {'full': True, 'p2': False},
    'full_minus_p3': {'full': True, 'p3': False},
    'full_minus_p4': {'full': True, 'p4': False},
    'full_minus_balance': {'full': True, 'balance': False},
}
LABELS = {
    'optimal_base': 'Optimal base', 'minus_p1': 'Base − P1',
    'plus_p2': 'Base + P2', 'plus_matched': 'Base + matched',
    'plus_p3': 'Base + P3', 'plus_p4': 'Base + P4',
    'balance_toggle': 'Base ± balance', 'full': 'Full',
    'full_minus_p1': 'Full − P1', 'full_minus_p2': 'Full − P2',
    'full_minus_p3': 'Full − P3', 'full_minus_p4': 'Full − P4',
    'full_minus_balance': 'Full − balance',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS),
                        default=list(DATASETS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 124, 125])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--channel', choices=('awgn', 'rayleigh'), default='awgn')
    parser.add_argument('--snr-db', type=float, default=20.0)
    parser.add_argument('--best-root',
                        default='results_mlp_tuning_and_evaluation_awgn20')
    parser.add_argument('--output', default='results_aligned_mlp_ablation_awgn20')
    return parser.parse_args()


def read_jsonl(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def resolve_modules(best, changes):
    """Resolve independent module states and full-model leave-one-out states."""
    config = best['config']
    if changes.get('full'):
        states = {'p1': True, 'p2': True, 'p3': True, 'p4': True,
                  'matched': True, 'balance': True}
    else:
        states = {
            'p1': True, 'p2': False, 'p3': False, 'p4': False,
            'matched': False, 'balance': config['class_balance'] == 'upsample',
        }
    for key in ('p1', 'p2', 'p3', 'p4', 'matched', 'balance'):
        if key in changes:
            states[key] = changes[key]
    if changes.get('balance_toggle'):
        states['balance'] = not states['balance']
    # P3 is defined as neighbourhood-SNR augmentation and requires matched
    # channel training. Full-P3 removal retains a single matched 20 dB draw.
    if states['p3']:
        states['matched'] = True
    return states


def build_command(dataset, seed, best, states, args, metrics_path):
    config = best['config']
    balance_ratio = (config['upsample_positive_ratio']
                     if config['class_balance'] == 'upsample' else 1.0 / 3.0)
    command = [
        sys.executable, 'main.py', '--dataset', dataset, '--beta', str(DATASETS[dataset]),
        '--features', 'count', '--model', 'mlp', '--channel', args.channel,
        '--snr_db', str(args.snr_db), '--seed', str(seed),
        '--channel_seed', str(seed), '--max_background_pairs', '10000',
        '--candidate_generator', 'shyre_channel_aware' if states['p2'] else 'shyre',
        '--decision_threshold_mode', 'validation' if states['p1'] else 'fixed',
        '--decision_threshold', str(config['decision_threshold']),
        '--threshold_validation_fraction', '0.2',
        '--epochs', str(config['epochs']), '--lr', str(config['lr']),
        '--mlp_hidden_layers', *map(str, config['mlp_hidden_layers']),
        '--mlp_alpha', str(config['mlp_alpha']),
        '--mlp_validation_fraction', str(config['mlp_validation_fraction']),
        '--mlp_n_iter_no_change', str(config['mlp_n_iter_no_change']),
        '--class_balance', 'upsample' if states['balance'] else 'none',
        '--upsample_positive_ratio', str(balance_ratio),
        '--channel_tau_e', str(config['channel_tau_e']),
        '--channel_beta', str(config['channel_beta']),
        '--channel_temperature', str(config['channel_temperature']),
        '--enable_candidate_metrics', '--enable_runtime_metrics',
        '--metrics_jsonl', str(metrics_path),
    ]
    if config['mlp_early_stopping']:
        command.append('--mlp_early_stopping')
    if states['matched']:
        command.append('--train_channel_matched')
    if states['p3']:
        command.extend(['--train_snr_offsets', '-2', '0', '2'])
    if states['p4']:
        command.extend(['--soft_reliability_mode', 'distribution'])
    if states['p2']:
        command.extend(['--channel_candidate_tau', '0.45',
                        '--channel_candidate_max_extra_edges', '2000'])
    return command


def export_tables(output, datasets, seeds, records):
    successful = [row for row in records if row.get('status') == 'success']
    detail_fields = ['dataset', 'variant', 'label', 'seed', 'precision', 'recall',
                     'f1', 'jaccard', 'candidate_recall', 'candidate_count',
                     'runtime_seconds', 'states_json', 'log']
    temporary = output / 'ablation_detailed.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        for row in successful:
            writer.writerow({key: row.get(key) for key in detail_fields})
    temporary.replace(output / 'ablation_detailed.csv')

    summary = []
    for dataset in datasets:
        for variant in VARIANTS:
            rows = [row for row in successful
                    if row['dataset'] == dataset and row['variant'] == variant]
            if not rows:
                continue
            def mean(key): return statistics.fmean(row[key] for row in rows)
            summary.append({
                'dataset': dataset, 'variant': variant, 'label': LABELS[variant],
                'completed_seeds': len(rows), 'required_seeds': len(seeds),
                'f1_mean': mean('f1'),
                'f1_std': statistics.pstdev(row['f1'] for row in rows)
                if len(rows) > 1 else 0.0,
                'precision_mean': mean('precision'), 'recall_mean': mean('recall'),
                'candidate_recall_mean': mean('candidate_recall'),
                'runtime_mean_seconds': mean('runtime_seconds'),
            })
    base = {(row['dataset'], row['variant']): row for row in summary}
    for row in summary:
        reference = base.get((row['dataset'], 'optimal_base'))
        row['delta_vs_optimal_base'] = (
            row['f1_mean'] - reference['f1_mean'] if reference else '')
    fields = ['dataset', 'variant', 'label', 'completed_seeds', 'required_seeds',
              'f1_mean', 'f1_std', 'delta_vs_optimal_base', 'precision_mean',
              'recall_mean', 'candidate_recall_mean', 'runtime_mean_seconds']
    temporary = output / 'ablation_summary.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)
    temporary.replace(output / 'ablation_summary.csv')
    return summary


def plot_results(output, datasets, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(row['dataset'], row['variant']): row for row in summary}
    complete_datasets = [dataset for dataset in datasets if all(
        (dataset, variant) in lookup and
        lookup[(dataset, variant)]['completed_seeds'] ==
        lookup[(dataset, variant)]['required_seeds'] for variant in VARIANTS)]
    if not complete_datasets:
        return
    variants = list(VARIANTS)
    deltas = np.asarray([[lookup[(dataset, variant)]['delta_vs_optimal_base']
                          for variant in variants] for dataset in complete_datasets])
    limit = max(0.02, float(np.max(np.abs(deltas))))
    fig, ax = plt.subplots(figsize=(15, 5.8), constrained_layout=True)
    image = ax.imshow(deltas, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
    for row in range(deltas.shape[0]):
        for column in range(deltas.shape[1]):
            ax.text(column, row, '{:+.3f}'.format(deltas[row, column]),
                    ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants],
                  rotation=28, ha='right')
    ax.set_yticks(range(len(complete_datasets)), complete_datasets)
    ax.set_title('Parameter-aligned MLP ablation at {} {:.0f} dB'.format(
        'AWGN' if True else '', 20.0))
    fig.colorbar(image, ax=ax, label='Δ F1 versus per-dataset optimal base')
    fig.savefig(output / 'fig_ablation_all4_delta.png', dpi=240)
    plt.close(fig)

    best_by_dataset = {
        dataset: max(lookup[(dataset, variant)]['f1_mean'] for variant in variants)
        for dataset in complete_datasets
    }
    selected = sorted(complete_datasets, key=best_by_dataset.get, reverse=True)[:2]
    fig, axes = plt.subplots(1, len(selected), figsize=(16, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, selected):
        means = [lookup[(dataset, variant)]['f1_mean'] for variant in variants]
        stds = [lookup[(dataset, variant)]['f1_std'] for variant in variants]
        bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=2,
                      color=plt.cm.Blues(np.linspace(.35, .9, len(variants))))
        ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants],
                      rotation=55, ha='right', fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_ylabel('Exact-match F1')
        ax.set_title(dataset); ax.grid(axis='y', alpha=.2)
        ax.bar_label(bars, labels=['{:.3f}'.format(v) for v in means],
                     fontsize=6, rotation=90, padding=2)
    fig.suptitle('Top-2 datasets by best ablation F1 (mean ± std, 3 paired seeds)')
    fig.savefig(output / 'fig_ablation_top2_f1.png', dpi=240)
    plt.close(fig)
    (output / 'selected_top2.json').write_text(json.dumps({
        'selection_rule': 'two highest best-variant mean F1 among the four datasets',
        'datasets': selected, 'scores': {d: best_by_dataset[d] for d in selected},
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    output = (ROOT / args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'; logs.mkdir(exist_ok=True)
    state_path, heartbeat = output / 'ablation_results.jsonl', output / 'heartbeat.log'
    lock = threading.Lock()
    records = read_jsonl(state_path)
    done = {(row['dataset'], row['variant'], int(row['seed']))
            for row in records if row.get('status') == 'success'}
    best_by_dataset = {}
    for dataset in args.datasets:
        path = ROOT / args.best_root / dataset / 'tuning' / 'best_config.json'
        if not path.exists():
            raise SystemExit('Missing best configuration: {}'.format(path))
        best_by_dataset[dataset] = json.loads(path.read_text(encoding='utf-8'))
    cells = [(dataset, variant, seed) for dataset in args.datasets
             for variant in VARIANTS for seed in args.seeds
             if (dataset, variant, seed) not in done]

    def report(message):
        line = '{} {}\n'.format(datetime.now().astimezone().isoformat(), message)
        with lock:
            with heartbeat.open('a', encoding='utf-8') as handle: handle.write(line)
        print(line, end='', flush=True)

    def run_cell(cell):
        dataset, variant, seed = cell
        stem = '{}_{}_seed{}'.format(dataset, variant, seed)
        metrics_path = logs / '{}.metrics.jsonl'.format(stem)
        log_path = logs / '{}.log'.format(stem)
        if metrics_path.exists(): metrics_path.unlink()
        states = resolve_modules(best_by_dataset[dataset], VARIANTS[variant])
        command = build_command(
            dataset, seed, best_by_dataset[dataset], states, args, metrics_path)
        started = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(process.stdout, encoding='utf-8')
        payloads = read_jsonl(metrics_path)
        if process.returncode or not payloads:
            return {'dataset': dataset, 'variant': variant, 'seed': seed,
                    'status': 'failed', 'returncode': process.returncode,
                    'runtime_seconds': time.monotonic() - started,
                    'states_json': json.dumps(states, sort_keys=True),
                    'log': str(log_path)}
        payload = payloads[-1]; metrics = payload['outcome']['performance']['SHyRe']
        candidate = payload['outcome']['candidate']
        return {
            'dataset': dataset, 'variant': variant, 'label': LABELS[variant],
            'seed': seed, 'status': 'success', **metrics,
            'candidate_recall': candidate['candidate_recall'],
            'candidate_count': candidate['candidate_count'],
            'runtime_seconds': payload['outcome']['runtime']['wall_time_seconds'],
            'states_json': json.dumps(states, sort_keys=True), 'log': str(log_path),
        }

    report('RUN workers={} pending={} total={}'.format(
        args.workers, len(cells), len(args.datasets) * len(VARIANTS) * len(args.seeds)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, cell): cell for cell in cells}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {'dataset': cell[0], 'variant': cell[1], 'seed': cell[2],
                       'status': 'exception', 'error': '{}: {}'.format(
                           type(exc).__name__, exc)}
            with state_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            records.append(row)
            summary = export_tables(output, args.datasets, args.seeds, records)
            plot_results(output, args.datasets, summary)
            completed = sum(item.get('status') == 'success' for item in records)
            report('{} dataset={} variant={} seed={} progress={}/{}'.format(
                row['status'].upper(), cell[0], cell[1], cell[2], completed,
                len(args.datasets) * len(VARIANTS) * len(args.seeds)))
    summary = export_tables(output, args.datasets, args.seeds, records)
    plot_results(output, args.datasets, summary)
    report('COMPLETE')


if __name__ == '__main__':
    main()
