"""Plot SNR--F1 curves for the six completed non-DBLP/non-Foursquare datasets."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOFT_CSV = ROOT / "results_soft_count_no_cfinder_small_0_20_step2" / "snr_baselines.csv"
COUNT_CSV = ROOT / "results_count_no_cfinder_small_0_20_step2" / "snr_baselines.csv"
OUT_DIR = ROOT / "figures_current_six_datasets_snr"

DATASETS = [
    ("enron", "Enron"),
    ("hosts", "Hosts-Virus"),
    ("school", "P.School"),
    ("school2", "H.School"),
    ("directors", "Directors"),
    ("crime", "Crime"),
]
METHODS = [
    ("SHyRe-soft-count", "#d62728", "o"),
    ("SHyRe-count", "#1f77b4", "s"),
    ("Max Clique", "#2ca02c", "^"),
    ("ECC", "#9467bd", "D"),
    ("DEMON", "#ff7f0e", "P"),
]


def read_rows(path: Path, shyre_name: str, include_baselines: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = shyre_name if row["method"] == "SHyRe" else row["method"]
            if row["method"] != "SHyRe" and not include_baselines:
                continue
            rows.append({
                "dataset": row["dataset"], "channel": row["channel"],
                "snr_db": int(row["snr_db"]), "method": method,
                "f1": float(row["f1"]), "source": path.name,
            })
    return rows


def plot_channel(rows: list[dict[str, object]], channel: str, figure_id: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, (dataset, title) in zip(axes.flat, DATASETS):
        for method, color, marker in METHODS:
            points = sorted(
                (row for row in rows if row["dataset"] == dataset and row["channel"] == channel and row["method"] == method),
                key=lambda row: row["snr_db"],
            )
            if points:
                axis.plot([row["snr_db"] for row in points], [row["f1"] for row in points],
                          label=method, color=color, marker=marker, linewidth=1.8, markersize=4.5)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlim(0, 20)
        axis.set_xticks(range(0, 21, 4))
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25, linewidth=0.7)
    for axis in axes[-1, :]:
        axis.set_xlabel("SNR (dB)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Reconstruction F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{figure_id}: Reconstruction performance under {channel}", y=1.09, fontsize=14, fontweight="bold")
    stem = OUT_DIR / f"{figure_id.lower().replace(' ', '_')}_{channel.lower()}_six_datasets"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=250 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = read_rows(SOFT_CSV, "SHyRe-soft-count", include_baselines=False)
    rows += read_rows(COUNT_CSV, "SHyRe-count", include_baselines=True)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["channel"]), int(row["snr_db"]), str(row["method"])))
    with (OUT_DIR / "six_dataset_snr_plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "channel", "snr_db", "method", "f1", "source"])
        writer.writeheader(); writer.writerows(rows)
    plot_channel(rows, "AWGN", "Fig 1")
    plot_channel(rows, "RAYLEIGH", "Fig 2")
    print(f"Wrote figures and {len(rows)} plotting rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
