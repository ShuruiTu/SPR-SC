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
            "school": 350000, "school2": 60000, "directors": 800, "crime": 1000}
SNRS = list(range(0, 21, 2))
PATTERN = re.compile(r"(?:(?P<shyre>Our Performance)|Baseline: (?P<baseline>Max Clique|ECC|Demon|CFinder \(k=\d+\))).*?f1 (?P<f1>[0-9.]+)")
REQUIRED_METHODS = {"SHyRe", "Max Clique", "ECC", "DEMON"}
FIELDS = ["dataset", "channel", "snr_db", "method", "f1", "returncode", "log"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default="results_channel_full_awgn_0_20_step2")
    parser.add_argument("--channel", choices=("awgn", "rayleigh"), default="awgn")
    parser.add_argument("--exclude-datasets", nargs="*", default=())
    parser.add_argument("--snrs", type=int, nargs="*", default=SNRS)
    parser.add_argument("--soft-reliability", action="store_true")
    parser.add_argument("--enable-cfinder", action="store_true")
    parser.add_argument("--features", choices=("count", "motif"), default="count")
    return parser.parse_args()


def main():
    args = parse_args(); output = ROOT / args.output; output.mkdir(exist_ok=True)
    csv_path, heartbeat = output / "snr_baselines.csv", output / "heartbeat.log"
    lock = threading.Lock(); rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as h:
            rows = [r for r in csv.DictReader(h) if str(r["returncode"]) == "0"]
    successful_methods = {}
    for row in rows:
        key = (row["dataset"], row["channel"].upper(), int(row["snr_db"]))
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
        dataset, snr = cell; beta = DATASETS[dataset]
        log = output / f"{dataset}_{args.channel}_{snr}dB.log"
        persist(f"START dataset={dataset} channel={args.channel} snr_db={snr}")
        command = [sys.executable, "main.py", "--dataset", dataset, "--beta", str(beta), "--features", args.features,
                   "--channel", args.channel, "--snr_db", str(snr)]
        if args.enable_cfinder:
            command.append("--enable_cfinder")
        if args.soft_reliability:
            command.append("--soft_reliability")
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
            parsed.append({"dataset": dataset, "channel": args.channel.upper(), "snr_db": snr, "method": method,
                           "f1": float(match.group("f1")), "returncode": proc.returncode, "log": str(log)})
        with lock: rows.extend(parsed)
        persist(f"DONE dataset={dataset} channel={args.channel} snr_db={snr} exit={proc.returncode} metrics={len(parsed)}")

    excluded = set(args.exclude_datasets)
    unknown = excluded - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets to exclude: {sorted(unknown)}")
    datasets = [dataset for dataset in DATASETS if dataset not in excluded]
    cells = [(dataset, snr) for dataset in datasets for snr in args.snrs
             if (dataset, args.channel.upper(), snr) not in done]
    persist(f"RUN workers={args.workers} scheduled_cells={len(cells)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, cell) for cell in cells]
        for future in as_completed(futures):
            try: future.result()
            except Exception as exc: persist(f"ERROR {type(exc).__name__}: {exc}")
    persist("COMPLETE")


if __name__ == "__main__": main()
