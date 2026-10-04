#!/usr/bin/env python3
"""Resumable SHyRe hyperparameter search using training-only validation.

Every trial invokes ``main.py --tuning-validation-only``.  Ranking therefore
uses an outer holdout from training candidates and never test-set F1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_BETA = {
    'dblp': 1000000, 'enron': 1000, 'foursquare': 20000, 'hosts': 6000,
    'school': 350000, 'school2': 60000, 'directors': 800, 'crime': 1000,
    'fb_auto': 20000, 'jf17k': 200000, 'm_fb15k': 1000000,
    'wikipeople': 1000000,
}
FAST_DATASETS = {'dblp', 'jf17k', 'm_fb15k', 'wikipeople'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--beta', type=int, default=None)
    parser.add_argument('--data-dir', default='./data/')
    parser.add_argument('--features', choices=('count', 'motif'), default='count')
    parser.add_argument('--channel', choices=('clean', 'awgn', 'rayleigh'), default='awgn')
    parser.add_argument('--snr-db', type=float, default=20.0)
    parser.add_argument('--channel-seed', type=int, default=123)
    parser.add_argument('--candidate-generator', default='auto',
                        choices=('auto', 'shyre', 'shyre_fast', 'shyre_channel_aware',
                                 'strict_max_clique', 'random', 'head', 'tail'))
    parser.add_argument('--max-background-pairs', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 124, 125])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--search-seed', type=int, default=20260918)
    parser.add_argument('--models', nargs='+', choices=('mlp', 'lr', 'rf'),
                        default=['mlp', 'lr', 'rf'])
    parser.add_argument('--validation-fraction', type=float, default=0.25)
    parser.add_argument('--threshold-validation-fraction', type=float, default=0.2)
    parser.add_argument('--std-penalty', type=float, default=0.5,
                        help='ranking score = mean objective - penalty * population std')
    parser.add_argument('--tune-modules', action='store_true',
                        help='also search soft reliability and matched-channel training')
    parser.add_argument('--tune-channel', action='store_true',
                        help='also search posterior threshold, LLR scale and temperature')
    parser.add_argument('--output', default='results_hyperparameter_tuning')
    parser.add_argument('--timeout', type=float, default=0.0,
                        help='per-run timeout in seconds; 0 disables timeout')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1 or not args.seeds:
        parser.error('--trials, --workers and --seeds must be non-empty/positive')
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error('--validation-fraction must be in (0, 0.5)')
    if args.channel == 'clean' and args.tune_channel:
        parser.error('--tune-channel is meaningful only for AWGN/Rayleigh')
    return args


def canonical_id(config):
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def base_config(model):
    return {
        'model': model,
        'class_balance': 'none', 'upsample_positive_ratio': 1.0 / 3.0,
        'decision_threshold': 0.5,
        'epochs': 2000, 'lr': 1e-3, 'mlp_hidden_layers': [100],
        'mlp_alpha': 1e-4, 'mlp_early_stopping': False,
        'mlp_validation_fraction': 0.1, 'mlp_n_iter_no_change': 10,
        'logistic_c': 1.0, 'logistic_max_iter': 100,
        'logistic_class_weight': 'none',
        'rf_n_estimators': 100, 'rf_max_depth': 0,
        'rf_min_samples_leaf': 1, 'rf_max_features': 'sqrt',
        'rf_class_weight': 'none',
        'soft_reliability_mode': 'off', 'train_channel_matched': False,
        'train_snr_offsets': [0.0],
        'channel_tau_e': 0.58, 'channel_beta': 1.0,
        'channel_temperature': 1.0,
    }


def random_config(rng, args):
    model = rng.choice(args.models)
    config = base_config(model)
    config['class_balance'] = rng.choice(['none', 'upsample'])
    config['upsample_positive_ratio'] = rng.choice([0.2, 1.0 / 3.0, 0.5, 1.0])
    config['decision_threshold'] = rng.choice([0.4, 0.5, 0.6])
    if model == 'mlp':
        config.update({
            'epochs': rng.choice([500, 1000, 2000]),
            'lr': rng.choice([1e-4, 3e-4, 1e-3]),
            'mlp_hidden_layers': list(rng.choice([(64,), (100,), (128,), (128, 64)])),
            'mlp_alpha': rng.choice([1e-5, 1e-4, 1e-3]),
            'mlp_early_stopping': rng.choice([False, True]),
            'mlp_n_iter_no_change': rng.choice([10, 20, 40]),
        })
    elif model == 'lr':
        config.update({
            'logistic_c': rng.choice([0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            'logistic_max_iter': rng.choice([500, 1000, 2000]),
            'logistic_class_weight': rng.choice(['none', 'balanced']),
        })
    else:
        config.update({
            'rf_n_estimators': rng.choice([100, 200, 400]),
            'rf_max_depth': rng.choice([0, 10, 20, 40]),
            'rf_min_samples_leaf': rng.choice([1, 2, 5]),
            'rf_max_features': rng.choice(['sqrt', 'log2', 'all']),
            'rf_class_weight': rng.choice(['none', 'balanced', 'balanced_subsample']),
        })
    if args.tune_modules and args.channel != 'clean':
        config['soft_reliability_mode'] = rng.choice(['off', 'basic', 'distribution'])
        config['train_channel_matched'] = rng.choice([False, True])
        if config['train_channel_matched']:
            config['train_snr_offsets'] = list(rng.choice([(0.0,), (-2.0, 0.0, 2.0)]))
    if args.tune_channel:
        config['channel_tau_e'] = rng.choice([0.45, 0.52, 0.58, 0.65])
        config['channel_beta'] = rng.choice([0.75, 1.0, 1.5])
        config['channel_temperature'] = rng.choice([0.75, 1.0, 1.25])
    return config


def build_trials(args):
    configs = []
    seen = set()
    for model in args.models:
        config = base_config(model)
        identifier = canonical_id(config)
        if identifier not in seen:
            configs.append(config); seen.add(identifier)
    rng = random.Random(args.search_seed)
    attempts = 0
    while len(configs) < args.trials and attempts < args.trials * 100:
        config = random_config(rng, args)
        identifier = canonical_id(config)
        if identifier not in seen:
            configs.append(config); seen.add(identifier)
        attempts += 1
    return configs[:args.trials]


def config_cli(config):
    command = [
        '--model', config['model'], '--class_balance', config['class_balance'],
        '--upsample_positive_ratio', str(config['upsample_positive_ratio']),
        '--decision_threshold_mode', 'validation',
        '--decision_threshold', str(config['decision_threshold']),
        '--epochs', str(config['epochs']), '--lr', str(config['lr']),
        '--mlp_hidden_layers', *map(str, config['mlp_hidden_layers']),
        '--mlp_alpha', str(config['mlp_alpha']),
        '--mlp_validation_fraction', str(config['mlp_validation_fraction']),
        '--mlp_n_iter_no_change', str(config['mlp_n_iter_no_change']),
        '--logistic_c', str(config['logistic_c']),
        '--logistic_max_iter', str(config['logistic_max_iter']),
        '--logistic_class_weight', config['logistic_class_weight'],
        '--rf_n_estimators', str(config['rf_n_estimators']),
        '--rf_max_depth', str(config['rf_max_depth']),
        '--rf_min_samples_leaf', str(config['rf_min_samples_leaf']),
        '--rf_max_features', config['rf_max_features'],
        '--rf_class_weight', config['rf_class_weight'],
        '--soft_reliability_mode', config['soft_reliability_mode'],
        '--channel_tau_e', str(config['channel_tau_e']),
        '--channel_beta', str(config['channel_beta']),
        '--channel_temperature', str(config['channel_temperature']),
    ]
    if config['mlp_early_stopping']:
        command.append('--mlp_early_stopping')
    if config['train_channel_matched']:
        command.append('--train_channel_matched')
        command.extend(['--train_snr_offsets', *map(str, config['train_snr_offsets'])])
    return command


def load_records(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a final partial line after an abrupt disconnect
    return records


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def completed_map(records):
    return {(item['trial_id'], int(item['seed'])): item for item in records
            if item.get('status') == 'success'}


def ranking_rows(configs, seeds, records, std_penalty):
    successful = completed_map(records)
    rows = []
    for config in configs:
        trial_id = canonical_id(config)
        cells = [successful.get((trial_id, seed)) for seed in seeds]
        scores = [cell['objective_f1'] for cell in cells if cell is not None]
        runtimes = [cell['runtime_seconds'] for cell in cells if cell is not None]
        complete = len(scores) == len(seeds)
        mean = statistics.fmean(scores) if scores else None
        deviation = statistics.pstdev(scores) if len(scores) > 1 else (0.0 if scores else None)
        rows.append({
            'trial_id': trial_id, 'model': config['model'],
            'completed_seeds': len(scores), 'required_seeds': len(seeds),
            'complete': complete, 'mean_objective_f1': mean,
            'std_objective_f1': deviation,
            'robust_score': mean - std_penalty * deviation if complete else None,
            'mean_runtime_seconds': statistics.fmean(runtimes) if runtimes else None,
            'config': config,
        })
    rows.sort(key=lambda row: (
        row['robust_score'] is not None,
        row['robust_score'] if row['robust_score'] is not None else -1.0,
        row['mean_objective_f1'] if row['mean_objective_f1'] is not None else -1.0,
    ), reverse=True)
    return rows


def export_ranking(output, configs, args, records, common_command):
    rows = ranking_rows(configs, args.seeds, records, args.std_penalty)
    fields = ['rank', 'trial_id', 'model', 'complete', 'completed_seeds',
              'required_seeds', 'mean_objective_f1', 'std_objective_f1',
              'robust_score', 'mean_runtime_seconds', 'config_json']
    temporary = output / 'ranking.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({
                'rank': rank, **{key: row[key] for key in fields[1:-1]},
                'config_json': json.dumps(row['config'], sort_keys=True),
            })
    temporary.replace(output / 'ranking.csv')
    complete = [row for row in rows if row['complete']]
    summary = {
        'updated_at': datetime.now().astimezone().isoformat(),
        'dataset': args.dataset, 'channel': args.channel, 'snr_db': args.snr_db,
        'validation_protocol': 'nested training-only holdout; no test F1 used',
        'trial_count': len(configs), 'complete_trials': len(complete),
        'successful_cells': len(completed_map(records)),
        'required_cells': len(configs) * len(args.seeds),
    }
    atomic_json(output / 'summary.json', summary)
    if complete:
        best = complete[0]
        validation_args = common_command + config_cli(best['config'])
        final_args = []
        skip_value = False
        for value in common_command:
            if skip_value:
                skip_value = False
                continue
            if value == '--tuning_validation_only':
                continue
            if value == '--tuning_validation_fraction':
                skip_value = True
                continue
            final_args.append(value)
        final_args += config_cli(best['config'])
        atomic_json(output / 'best_config.json', {
            **summary, 'trial_id': best['trial_id'],
            'mean_objective_f1': best['mean_objective_f1'],
            'std_objective_f1': best['std_objective_f1'],
            'robust_score': best['robust_score'], 'config': best['config'],
            'validation_main_args': validation_args,
            'validation_command': shlex.join(
                [sys.executable, 'main.py', *validation_args]),
            'final_evaluation_main_args': final_args,
            'final_evaluation_command': shlex.join(
                [sys.executable, 'main.py', *final_args]),
        })


def main():
    args = parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir(exist_ok=True)
    beta = args.beta if args.beta is not None else DATASET_BETA.get(args.dataset)
    if beta is None:
        raise SystemExit('Unknown dataset: supply --beta explicitly')
    candidate_generator = args.candidate_generator
    if candidate_generator == 'auto':
        candidate_generator = 'shyre_fast' if args.dataset in FAST_DATASETS else 'shyre'
    common = [
        '--dataset', args.dataset, '--beta', str(beta), '--data_dir', args.data_dir,
        '--features', args.features, '--channel', args.channel,
        '--snr_db', str(args.snr_db), '--channel_seed', str(args.channel_seed),
        '--candidate_generator', candidate_generator,
        '--max_background_pairs', str(args.max_background_pairs),
        '--tuning_validation_only',
        '--tuning_validation_fraction', str(args.validation_fraction),
        '--threshold_validation_fraction', str(args.threshold_validation_fraction),
    ]
    run_context = {
        'dataset': args.dataset, 'beta': beta, 'data_dir': args.data_dir,
        'features': args.features, 'channel': args.channel,
        'snr_db': args.snr_db, 'channel_seed': args.channel_seed,
        'candidate_generator': candidate_generator,
        'max_background_pairs': args.max_background_pairs,
        'validation_fraction': args.validation_fraction,
        'threshold_validation_fraction': args.threshold_validation_fraction,
    }
    manifest_path = output / 'search_manifest.json'
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        previous_context = previous_manifest.get('run_context')
        if previous_context is not None and previous_context != run_context:
            raise SystemExit(
                'Output directory belongs to a different tuning context; '
                'choose another --output to prevent mixed rankings.')
    configs = build_trials(args)
    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'arguments': vars(args), 'resolved_beta': beta,
        'resolved_candidate_generator': candidate_generator,
        'run_context': run_context,
        'validation_rule': 'Only outcome.tuning_validation.objective_f1 is ranked.',
        'trials': [{'trial_id': canonical_id(config), 'config': config}
                   for config in configs],
    }
    atomic_json(manifest_path, manifest)
    state_path = output / 'trial_results.jsonl'
    records = load_records(state_path)
    done = completed_map(records)
    cells = [(config, seed) for config in configs for seed in args.seeds
             if (canonical_id(config), seed) not in done]
    export_ranking(output, configs, args, records, common)
    print('Tuning cells: total={}, resumed={}, pending={}'.format(
        len(configs) * len(args.seeds), len(done), len(cells)), flush=True)
    if args.dry_run:
        return

    def run_cell(config, seed):
        trial_id = canonical_id(config)
        stem = '{}_seed{}'.format(trial_id, seed)
        metrics_path = output / 'logs' / '{}.metrics.jsonl'.format(stem)
        log_path = output / 'logs' / '{}.log'.format(stem)
        if metrics_path.exists():
            metrics_path.unlink()
        command = [sys.executable, 'main.py', *common, '--seed', str(seed),
                   '--metrics_jsonl', str(metrics_path), *config_cli(config)]
        started = time.monotonic()
        try:
            process = subprocess.run(
                command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout if args.timeout > 0 else None)
            output_text = process.stdout
            returncode = process.returncode
            status = 'failed'
            objective = None
            validation = None
            if returncode == 0 and metrics_path.exists():
                payloads = load_records(metrics_path)
                if payloads:
                    validation = payloads[-1].get('outcome', {}).get('tuning_validation')
                    if validation is not None:
                        objective = float(validation['objective_f1'])
                        status = 'success'
        except subprocess.TimeoutExpired as exc:
            output_text = (exc.stdout or '') + '\nTIMEOUT\n'
            returncode = 124
            status, objective, validation = 'timeout', None, None
        log_path.write_text(output_text, encoding='utf-8')
        return {
            'timestamp': datetime.now().astimezone().isoformat(),
            'trial_id': trial_id, 'seed': seed, 'status': status,
            'objective_f1': objective, 'validation': validation,
            'runtime_seconds': time.monotonic() - started,
            'returncode': returncode, 'config': config,
            'command': command, 'log': str(log_path),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, config, seed): (config, seed)
                   for config, seed in cells}
        for future in as_completed(futures):
            config, seed = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    'timestamp': datetime.now().astimezone().isoformat(),
                    'trial_id': canonical_id(config), 'seed': seed,
                    'status': 'exception', 'objective_f1': None,
                    'runtime_seconds': 0.0, 'returncode': -1,
                    'config': config, 'error': '{}: {}'.format(type(exc).__name__, exc),
                }
            with state_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            export_ranking(output, configs, args, records, common)
            print('{} trial={} seed={} objective={}'.format(
                record['status'].upper(), record['trial_id'], seed,
                record.get('objective_f1')), flush=True)
    print('COMPLETE: {}'.format(output), flush=True)


if __name__ == '__main__':
    main()
