#!/usr/bin/env python3
"""Tune MLP per dataset, then evaluate each best configuration on the test set."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    'enron', 'fb_auto', 'crime', 'directors', 'school', 'school2',
    'jf17k', 'm_fb15k', 'wikipeople',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 124, 125])
    parser.add_argument('--dataset-workers', type=int, default=2)
    parser.add_argument('--trial-workers', type=int, default=2)
    parser.add_argument('--channel', choices=('awgn', 'rayleigh'), default='awgn')
    parser.add_argument('--snr-db', type=float, default=20.0)
    parser.add_argument('--output', default='results_mlp_tuning_and_evaluation_awgn20')
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def aggregate_evaluations(root, datasets):
    detailed, aggregate = [], []
    for dataset in datasets:
        path = root / dataset / 'evaluation_results.jsonl'
        records = [row for row in read_jsonl(path) if row.get('status') == 'success']
        detailed.extend(records)
        methods = sorted({row['method'] for row in records})
        for method in methods:
            method_rows = [row for row in records if row['method'] == method]
            aggregate.append({
                'dataset': dataset, 'channel': method_rows[0]['channel'],
                'snr_db': method_rows[0]['snr_db'], 'method': method,
                'n_seeds': len(method_rows),
                'precision_mean': statistics.fmean(row['precision'] for row in method_rows),
                'recall_mean': statistics.fmean(row['recall'] for row in method_rows),
                'f1_mean': statistics.fmean(row['f1'] for row in method_rows),
                'f1_std': statistics.pstdev(row['f1'] for row in method_rows)
                if len(method_rows) > 1 else 0.0,
                'jaccard_mean': statistics.fmean(row['jaccard'] for row in method_rows),
            })
    detail_fields = ['dataset', 'channel', 'snr_db', 'seed', 'trial_id', 'method',
                     'precision', 'recall', 'f1', 'jaccard', 'status', 'log']
    with (root / 'all_datasets_evaluation_detailed.csv').open(
            'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        for row in detailed:
            writer.writerow({key: row.get(key) for key in detail_fields})
    aggregate_fields = ['dataset', 'channel', 'snr_db', 'method', 'n_seeds',
                        'precision_mean', 'recall_mean', 'f1_mean', 'f1_std',
                        'jaccard_mean']
    with (root / 'all_datasets_evaluation_summary.csv').open(
            'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader(); writer.writerows(aggregate)


def main():
    args = parse_args()
    root = (ROOT / args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    status_path = root / 'status.log'

    def status(message):
        line = '{} {}\n'.format(datetime.now().astimezone().isoformat(), message)
        with lock:
            with status_path.open('a', encoding='utf-8') as handle:
                handle.write(line)
        print(line, end='', flush=True)

    def run_dataset(dataset):
        dataset_root = root / dataset
        dataset_root.mkdir(exist_ok=True)
        tuning_output = Path(args.output) / dataset / 'tuning'
        tuning_stdout = dataset_root / 'tuning.stdout.log'
        tuning_command = [
            sys.executable, 'tune_hyperparameters.py', '--dataset', dataset,
            '--channel', args.channel, '--snr-db', str(args.snr_db),
            '--trials', str(args.trials), '--models', 'mlp',
            '--seeds', *map(str, args.seeds), '--workers', str(args.trial_workers),
            '--output', str(tuning_output),
        ]
        status('TUNE_START dataset={}'.format(dataset))
        with tuning_stdout.open('a', encoding='utf-8') as output:
            process = subprocess.run(
                tuning_command, cwd=ROOT, text=True, stdout=output,
                stderr=subprocess.STDOUT)
        summary_path = ROOT / tuning_output / 'summary.json'
        if process.returncode != 0 or not summary_path.exists():
            status('TUNE_FAILED dataset={} exit={}'.format(dataset, process.returncode))
            return False
        summary = read_json(summary_path)
        if summary.get('complete_trials') != args.trials:
            status('TUNE_INCOMPLETE dataset={} complete={}/{}'.format(
                dataset, summary.get('complete_trials'), args.trials))
            return False
        best_path = ROOT / tuning_output / 'best_config.json'
        best = read_json(best_path)
        status('TUNE_DONE dataset={} trial={} mean={:.6f} robust={:.6f}'.format(
            dataset, best['trial_id'], best['mean_objective_f1'], best['robust_score']))

        evaluation_state = dataset_root / 'evaluation_results.jsonl'
        successful = {(int(row['seed']), row['method']) for row in read_jsonl(evaluation_state)
                      if row.get('status') == 'success'}
        final_args = list(best['final_evaluation_main_args'])
        for seed in args.seeds:
            # A seed is complete only when SHyRe and all default enabled baselines exist.
            required = {'SHyRe', 'Max Clique', 'ECC', 'DEMON'}
            if all((seed, method) in successful for method in required):
                status('EVAL_RESUME dataset={} seed={}'.format(dataset, seed))
                continue
            metrics_path = dataset_root / 'evaluation_seed{}.metrics.jsonl'.format(seed)
            log_path = dataset_root / 'evaluation_seed{}.log'.format(seed)
            if metrics_path.exists():
                metrics_path.unlink()
            command = [sys.executable, 'main.py', *final_args, '--seed', str(seed),
                       '--metrics_jsonl', str(metrics_path)]
            status('EVAL_START dataset={} seed={}'.format(dataset, seed))
            process = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            log_path.write_text(process.stdout, encoding='utf-8')
            payloads = read_jsonl(metrics_path)
            if process.returncode != 0 or not payloads:
                status('EVAL_FAILED dataset={} seed={} exit={}'.format(
                    dataset, seed, process.returncode))
                return False
            performance = payloads[-1]['outcome']['performance']
            with evaluation_state.open('a', encoding='utf-8') as handle:
                for method, metrics in performance.items():
                    record = {
                        'dataset': dataset, 'channel': args.channel,
                        'snr_db': args.snr_db, 'seed': seed,
                        'trial_id': best['trial_id'], 'method': method,
                        'precision': metrics['precision'], 'recall': metrics['recall'],
                        'f1': metrics['f1'], 'jaccard': metrics['jaccard'],
                        'status': 'success', 'log': str(log_path),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            status('EVAL_DONE dataset={} seed={} shyre_f1={:.6f}'.format(
                dataset, seed, performance['SHyRe']['f1']))
        status('DATASET_COMPLETE dataset={}'.format(dataset))
        return True

    status('RUN_START datasets={} trials={} seeds={} dataset_workers={} trial_workers={}'.format(
        ','.join(args.datasets), args.trials, args.seeds,
        args.dataset_workers, args.trial_workers))
    results = {}
    with ThreadPoolExecutor(max_workers=args.dataset_workers) as pool:
        futures = {pool.submit(run_dataset, dataset): dataset for dataset in args.datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                results[dataset] = bool(future.result())
            except Exception as exc:
                results[dataset] = False
                status('DATASET_EXCEPTION dataset={} error={}:{}'.format(
                    dataset, type(exc).__name__, exc))
            with lock:
                aggregate_evaluations(root, args.datasets)
                write_json(root / 'status.json', {
                    'updated_at': datetime.now().astimezone().isoformat(),
                    'datasets': results,
                    'complete': len(results) == len(args.datasets) and all(results.values()),
                })
    status('RUN_COMPLETE success={}/{}'.format(sum(results.values()), len(args.datasets)))


if __name__ == '__main__':
    main()
