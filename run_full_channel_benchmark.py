"""Parallel, resumable SHyRe channel benchmark."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {"dblp": 1000000, "enron": 1000, "foursquare": 20000, "hosts": 6000,
            "school": 350000, "school2": 60000, "directors": 800, "crime": 1000,
            "fb_auto": 20000, "jf17k": 200000, "m_fb15k": 1000000, "wikipeople": 1000000}
SNRS = list(range(0, 21, 2))
PATTERN = re.compile(r"(?:(?P<shyre>Our Performance)|Baseline: (?P<baseline>Max Clique|ECC|Demon|CFinder \(k=\d+\))).*?f1 (?P<f1>[0-9.]+)")
REQUIRED_METHODS = {"SHyRe", "Max Clique", "ECC", "DEMON"}
FIELDS = ["dataset", "channel", "snr_db", "seed", "method", "f1", "returncode", "log"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default="results_channel_full_awgn_0_20_step2")
    parser.add_argument("--channel", choices=("awgn", "rayleigh"), default="awgn")
    parser.add_argument("--exclude-datasets", nargs="*", default=())
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="explicit dataset subset; defaults to all registered datasets")
    parser.add_argument("--snrs", type=int, nargs="*", default=SNRS)
    parser.add_argument("--soft-reliability", action="store_true")
    parser.add_argument("--soft-reliability-mode", choices=("off", "basic", "distribution"), default="off")
    parser.add_argument("--train-channel-matched", action="store_true",
                        help="apply an independent same-SNR channel draw to the training projection")
    parser.add_argument("--train-channel-replicates", type=int, default=1)
    parser.add_argument("--train-snr-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--enable-cfinder", action="store_true")
    parser.add_argument("--features", choices=("count", "motif"), default="count")
    parser.add_argument("--model", choices=("mlp", "lr", "rf"), default="mlp")
    parser.add_argument("--class-balance", choices=("none", "upsample"), default="none")
    parser.add_argument("--upsample-positive-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--projection-retention", type=float, default=None)
    parser.add_argument("--max-background-pairs", type=int, default=10000)
    parser.add_argument("--channel-tau-e", type=float, default=0.58)
    parser.add_argument("--channel-beta", type=float, default=1.0)
    parser.add_argument("--channel-temperature", type=float, default=1.0)
    parser.add_argument("--rayleigh-csi-mode", choices=("perfect", "estimated", "statistical"), default="perfect")
    parser.add_argument("--rayleigh-csi-error-variance", type=float, default=0.1)
    parser.add_argument("--candidate-generator", choices=("shyre", "shyre_fast", "shyre_channel_aware", "strict_max_clique", "random", "head", "tail"), default="shyre")
    parser.add_argument("--channel-candidate-tau", type=float, default=0.45)
    parser.add_argument("--channel-candidate-max-extra-edges", type=int, default=2000)
    parser.add_argument("--enable-candidate-metrics", action="store_true")
    parser.add_argument("--enable-storage-metrics", action="store_true")
    parser.add_argument("--enable-runtime-metrics", action="store_true")
    parser.add_argument("--export-metrics", action="store_true", help="write one structured metrics JSONL per cell")
    parser.add_argument("--enable-size-stratified-metrics", action="store_true",
                        help="write Fig. 11/12 size-binned metrics")
    parser.add_argument("--seed", type=int, default=123, help="model RNG seed; channel seed defaults to it")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="matched seed grid; overrides --seed when supplied")
    parser.add_argument("--channel-seed", type=int, default=None)
    parser.add_argument("--decision-threshold-mode", choices=("fixed", "validation"), default="fixed")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_background_pairs < 0:
        raise SystemExit('--max-background-pairs must be non-negative')
    if not 0.0 < args.channel_tau_e < 1.0:
        raise SystemExit('--channel-tau-e must be in (0, 1)')
    if args.channel_beta <= 0.0:
        raise SystemExit('--channel-beta must be positive')
    if args.channel_temperature <= 0.0:
        raise SystemExit('--channel-temperature must be positive')
    if args.rayleigh_csi_error_variance <= 0.0:
        raise SystemExit('--rayleigh-csi-error-variance must be positive')
    if args.channel != 'rayleigh' and args.rayleigh_csi_mode != 'perfect':
        raise SystemExit('non-perfect --rayleigh-csi-mode requires --channel rayleigh')
    output = ROOT / args.output; output.mkdir(exist_ok=True)
    csv_path, heartbeat = output / "snr_baselines.csv", output / "heartbeat.log"
    lock = threading.Lock(); rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as h:
            rows = [r for r in csv.DictReader(h) if str(r["returncode"]) == "0"]
    successful_methods = {}
    for row in rows:
        key = (row["dataset"], row["channel"].upper(), int(row["snr_db"]), int(row.get("seed", args.seed)))
        successful_methods.setdefault(key, set()).add(row["method"])
    required_methods = REQUIRED_METHODS | ({"CFinder"} if args.enable_cfinder else set())
    done = {key for key, methods in successful_methods.items() if required_methods <= methods}
    def persist(message):
        with lock:
            stamp = datetime.now().astimezone().isoformat()
            with heartbeat.open("a", encoding="utf-8") as h: h.write(f"{stamp} {message}\n")
            with csv_path.open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        print(f"{stamp} {message}", flush=True)

    def run(cell):
        dataset, snr, seed = cell
        beta = DATASETS[dataset]
        suffix = f"{dataset}_{args.channel}_{snr}dB_seed{seed}"
        log = output / f"{suffix}.log"
        persist(f"START dataset={dataset} channel={args.channel} snr_db={snr} seed={seed}")
        command = [sys.executable, "main.py", "--dataset", dataset, "--beta", str(beta),
                   "--features", args.features, "--model", args.model,
                   "--channel", args.channel, "--snr_db", str(snr),
                   "--seed", str(seed), "--channel_seed", str(seed if args.channel_seed is None else args.channel_seed),
                   "--max_background_pairs", str(args.max_background_pairs),
                   "--channel_tau_e", str(args.channel_tau_e),
                   "--channel_beta", str(args.channel_beta),
                   "--channel_temperature", str(args.channel_temperature),
                   "--rayleigh_csi_mode", args.rayleigh_csi_mode,
                   "--rayleigh_csi_error_variance", str(args.rayleigh_csi_error_variance)]
        if args.class_balance != 'none':
            command.extend(['--class_balance', args.class_balance])
        if args.upsample_positive_ratio != 1.0 / 3.0:
            command.extend(['--upsample_positive_ratio', str(args.upsample_positive_ratio)])
        if args.enable_cfinder:
            command.append("--enable_cfinder")
        if args.soft_reliability:
            command.append("--soft_reliability")
        if args.soft_reliability_mode != 'off':
            command.extend(['--soft_reliability_mode', args.soft_reliability_mode])
        if args.train_channel_matched:
            command.append("--train_channel_matched")
        if args.train_channel_replicates != 1:
            command.extend(['--train_channel_replicates', str(args.train_channel_replicates)])
        if args.train_snr_offsets != [0.0]:
            command.append('--train_snr_offsets')
            command.extend(map(str, args.train_snr_offsets))
        if args.decision_threshold_mode != 'fixed':
            command.extend(['--decision_threshold_mode', args.decision_threshold_mode])
        if args.decision_threshold != 0.5:
            command.extend(['--decision_threshold', str(args.decision_threshold)])
        if args.threshold_validation_fraction != 0.2:
            command.extend(['--threshold_validation_fraction', str(args.threshold_validation_fraction)])
        if args.projection_retention is not None:
            command.extend(['--enable_projection_retention', '--projection_retention', str(args.projection_retention)])
        if args.candidate_generator != 'shyre':
            command.extend(['--candidate_generator', args.candidate_generator])
        if args.candidate_generator == 'shyre_channel_aware':
            command.extend([
                '--channel_candidate_tau', str(args.channel_candidate_tau),
                '--channel_candidate_max_extra_edges', str(args.channel_candidate_max_extra_edges),
            ])
        if args.enable_candidate_metrics:
            command.append('--enable_candidate_metrics')
        if args.enable_storage_metrics:
            command.append('--enable_storage_metrics')
        if args.enable_runtime_metrics:
            command.append('--enable_runtime_metrics')
        if args.enable_size_stratified_metrics:
            command.append('--enable_size_stratified_metrics')
        if args.export_metrics:
            command.extend(['--metrics_jsonl', str(output / f'{suffix}.metrics.jsonl')])
        proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(proc.stdout, encoding="utf-8")
        parsed = []
        for match in PATTERN.finditer(proc.stdout):
            if match.group("shyre"):
                method = "SHyRe"
            else:
                method = match.group("baseline").replace("Demon", "DEMON")
                if method.startswith("CFinder"):
                    method = "CFinder"
            parsed.append({"dataset": dataset, "channel": args.channel.upper(), "snr_db": snr, "seed": seed, "method": method,
                           "f1": float(match.group("f1")), "returncode": proc.returncode, "log": str(log)})
        with lock:
            rows.extend(parsed)
        persist(f"DONE dataset={dataset} channel={args.channel} snr_db={snr} seed={seed} exit={proc.returncode} metrics={len(parsed)}")

    excluded = set(args.exclude_datasets)
    unknown = excluded - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets to exclude: {sorted(unknown)}")
    requested = list(DATASETS) if args.datasets is None else args.datasets
    unknown_requested = set(requested) - set(DATASETS)
    if unknown_requested:
        raise ValueError(f"Unknown datasets: {sorted(unknown_requested)}")
    datasets = [dataset for dataset in requested if dataset not in excluded]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    cells = [(dataset, snr, seed) for dataset in datasets for snr in args.snrs for seed in seeds
             if (dataset, args.channel.upper(), snr, seed) not in done]
    persist(f"RUN workers={args.workers} scheduled_cells={len(cells)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, cell) for cell in cells]
        for future in as_completed(futures):
            try: future.result()
            except Exception as exc: persist(f"ERROR {type(exc).__name__}: {exc}")
    persist("COMPLETE")


if __name__ == "__main__": main()
