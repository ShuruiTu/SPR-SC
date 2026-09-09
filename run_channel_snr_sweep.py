"""Run and plot native SHyRe baselines under a received-graph SNR sweep."""
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {"enron": 1000, "hosts": 6000, "school": 350000, "school2": 60000}
SNRS = [10, 20, 60]
PATTERN = re.compile(
    r"(?:(?P<shyre>Our Performance)|Baseline: (?P<baseline>Max Clique|ECC|Demon|CFinder \(k=\d+\))).*?f1 (?P<f1>[0-9.]+)"
)


def main():
    output = ROOT / "results_channel_snr"
    output.mkdir(exist_ok=True)
    rows = []
    fields = ["dataset", "channel", "snr_db", "method", "f1", "returncode", "log"]
    csv_path = output / "snr_baselines.csv"
    heartbeat_path = output / "heartbeat.log"

    def heartbeat(message):
        line = f"{datetime.now().astimezone().isoformat()} {message}\n"
        with heartbeat_path.open("a", encoding="utf-8") as handle:
            handle.write(line); handle.flush()
        print(line, end="", flush=True)
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            for row in rows:
                row["snr_db"] = int(row["snr_db"])
                row["f1"] = float(row["f1"])
                row["returncode"] = int(row["returncode"])
    completed = {(r["dataset"], r["snr_db"]) for r in rows if r["method"] == "SHyRe"}
    for dataset, beta in DATASETS.items():
        for snr in SNRS:
            if (dataset, snr) in completed:
                heartbeat(f"SKIP dataset={dataset} snr_db={snr} reason=already_complete")
                continue
            log = output / f"{dataset}_awgn_{snr}dB.log"
            command = [sys.executable, "main.py", "--dataset", dataset, "--beta", str(beta),
                       "--features", "count", "--channel", "awgn", "--snr_db", str(snr)]
            heartbeat(f"START dataset={dataset} snr_db={snr}")
            try:
                proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
            except subprocess.TimeoutExpired as exc:
                text = exc.stdout or ""
                if isinstance(text, bytes): text = text.decode("utf-8", errors="replace")
                log.write_text(text, encoding="utf-8")
                heartbeat(f"TIMEOUT dataset={dataset} snr_db={snr} timeout_s=900")
                continue
            log.write_text(proc.stdout, encoding="utf-8")
            for match in PATTERN.finditer(proc.stdout):
                method = "SHyRe" if match.group("shyre") else match.group("baseline").replace("Demon", "DEMON")
                rows.append({"dataset": dataset, "channel": "AWGN", "snr_db": snr, "method": method,
                             "f1": float(match.group("f1")), "returncode": proc.returncode, "log": str(log)})
            # Persist after every expensive cell so an interrupted sweep keeps
            # all completed results and can be resumed manually.
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            heartbeat(f"DONE dataset={dataset} snr_db={snr} exit={proc.returncode}")
    with (output / "snr_baselines.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
        for dataset in DATASETS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in sorted({r["method"] for r in rows if r["dataset"] == dataset}):
                cell = sorted((r for r in rows if r["dataset"] == dataset and r["method"] == method), key=lambda r: r["snr_db"])
                ax.plot([r["snr_db"] for r in cell], [r["f1"] for r in cell], marker="o", label=method)
            ax.set(title=dataset, xlabel="AWGN SNR (dB)", ylabel="Exact-match F1", ylim=(0, 1)); ax.legend(); ax.grid(alpha=.25)
            fig.tight_layout(); fig.savefig(output / f"{dataset}_awgn_snr_f1.png", dpi=180); plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
