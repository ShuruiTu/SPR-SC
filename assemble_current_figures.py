#!/usr/bin/env python3
"""Assemble the current reproducible Fig. 1--12 evidence set.

The script intentionally reads only completed local result directories.  It
does not fill missing cells or mix the historical interrupted DBLP/Foursquare
runs into the uniform 0:2:20 dB composite.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_current_complete"
SMALL = ("enron", "hosts", "school", "school2", "directors", "crime")
NEW = ("fb_auto", "jf17k")
DISPLAY = {"enron": "Enron", "hosts": "Hosts-Virus", "school": "P.School", "school2": "H.School",
           "directors": "Directors", "crime": "Crime", "fb_auto": "FB-AUTO", "jf17k": "JF17K"}
COLORS = {"SHyRe-count": "#1565c0", "SHyRe-soft-count": "#00897b", "SHyRe-fast-count": "#6a1b9a",
          "SHyRe-fast-soft-count": "#ab47bc", "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#c62828"}


def metric_files(directory: Path):
    return sorted(directory.glob("*.metrics.jsonl"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def variant(directory: Path, dataset: str) -> str:
    name = directory.name
    fast = dataset == "jf17k"
    soft = "soft_count" in name
    return "SHyRe{}{}-count".format("-fast" if fast else "", "-soft" if soft else "")


def snr_sources():
    sources = []
    for channel in ("awgn", "rayleigh"):
        for soft in (False, True):
            base = "results_soft_count_no_cfinder_small_0_20_step2" if soft else "results_count_no_cfinder_small_0_20_step2"
            sources.append((ROOT / base, SMALL, channel, soft))
    for dataset in NEW:
        for channel in ("awgn", "rayleigh"):
            for soft in (False, True):
                suffix = "soft_count" if soft else "count"
                sources.append((ROOT / f"results_fig1_2_{dataset}_{suffix}_{channel}_0_20_step2", (dataset,), channel, soft))
    return sources


def load_snr():
    values = defaultdict(list)
    for directory, datasets, channel, soft in snr_sources():
        csv_path = directory / "snr_baselines.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("returncode") != "0" or row["dataset"] not in datasets:
                    continue
                method = row["method"]
                if method == "SHyRe":
                    method = variant(directory, row["dataset"])
                key = (row["dataset"], row["channel"].upper(), int(row["snr_db"]), method)
                values[key].append(float(row["f1"]))
    return {key: float(np.mean(items)) for key, items in values.items()}


def fig_snr(values, channel: str, figure: int):
    datasets = [dataset for dataset in (*SMALL, *NEW) if any(k[0] == dataset and k[1] == channel for k in values)]
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes.ravel(), datasets):
        methods = [method for method in COLORS if any(k[:2] == (dataset, channel) and k[3] == method for k in values)]
        for method in methods:
            points = sorted((key[2], value) for key, value in values.items() if key[:2] == (dataset, channel) and key[3] == method)
            ax.plot(*zip(*points), marker="o", ms=2.7, lw=1.35, label=method, color=COLORS[method])
        ax.set_title(DISPLAY[dataset]); ax.grid(alpha=.2); ax.set_ylim(-.02, 1.02)
    for ax in axes.ravel()[len(datasets):]: ax.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.supxlabel("SNR (dB)"); fig.supylabel("Exact-match reconstruction F1")
    fig.suptitle(f"Fig. {figure}. {channel}: completed datasets, single seed", y=1.025)
    fig.savefig(OUT / f"fig_{figure}_{channel.lower()}_all_completed_snr.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def fig34_records():
    records = []
    pattern = re.compile(r"results_fig34_(awgn|rayleigh)_60db_(count|soft)_(shyre|strict_max_clique)")
    for directory in ROOT.glob("results_fig34_*"):
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        channel, family, generator = match.groups()
        for path in metric_files(directory):
            payload = read_json(path)
            records.append((payload["dataset"], channel.upper(), family, generator, payload))
    return records


def fig34(records):
    datasets = list(SMALL); x = np.arange(len(datasets)); width = .18
    for figure, metric, ylabel in ((3, "f1", "F1"), (4, "candidate_recall", "Candidate recall")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True, constrained_layout=True)
        styles = [("count", "strict_max_clique", "count strict"), ("count", "shyre", "count SHyRe"),
                  ("soft", "strict_max_clique", "soft strict"), ("soft", "shyre", "soft SHyRe")]
        for ax, channel in zip(axes, ("AWGN", "RAYLEIGH")):
            for index, (family, generator, label) in enumerate(styles):
                ys = []
                for dataset in datasets:
                    cells = [payload for ds, ch, fam, gen, payload in records if ds == dataset and ch == channel and fam == family and gen == generator]
                    payload = cells[0]
                    ys.append(payload["outcome"]["performance"]["SHyRe"][metric] if metric == "f1" else payload["outcome"]["candidate"][metric])
                ax.bar(x + (index - 1.5) * width, ys, width, label=label)
            ax.set_xticks(x, [DISPLAY[d] for d in datasets], rotation=20, ha="right")
            ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel); ax.set_title(f"{channel}, 60 dB")
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(ncols=2, frameon=False)
        fig.suptitle(f"Fig. {figure}. {ylabel} ablation")
        fig.savefig(OUT / f"fig_{figure}_{'f1' if metric == 'f1' else 'candidate_recall'}_ablation.png", dpi=230)
        plt.close(fig)


def fig34_enron10():
    """Current Fig. 3/4 protocol: one representative dataset at 10 dB."""
    records = {}
    pattern = re.compile(r"results_fig34_enron_(awgn|rayleigh)_10db_(count|soft)_(shyre|strict_max_clique)")
    for directory in ROOT.glob("results_fig34_enron_*"):
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        records[match.groups()] = read_json(metric_files(directory)[0])
    styles = [("count", "strict_max_clique", "count strict"), ("count", "shyre", "count SHyRe"),
              ("soft", "strict_max_clique", "soft strict"), ("soft", "shyre", "soft SHyRe")]
    for figure, metric, ylabel in ((3, "f1", "Exact-match F1"), (4, "candidate_recall", "Candidate recall")):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True, constrained_layout=True)
        for ax, channel in zip(axes, ("awgn", "rayleigh")):
            values = []
            for family, generator, label in styles:
                payload = records[(channel, family, generator)]
                value = (payload["outcome"]["performance"]["SHyRe"]["f1"] if metric == "f1"
                         else payload["outcome"]["candidate"]["candidate_recall"])
                values.append(value)
            bars = ax.bar(range(len(styles)), values, color=["#90caf9", "#1565c0", "#80cbc4", "#00897b"])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            ax.set_xticks(range(len(styles)), [label.replace(" ", "\n") for _, _, label in styles])
            ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.2); ax.set_title(f"{channel.upper()}, 10 dB")
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"Fig. {figure}. Enron at 10 dB: {'reconstruction' if metric == 'f1' else 'candidate coverage'}")
        fig.savefig(OUT / f"fig_{figure}_{'f1' if metric == 'f1' else 'candidate_recall'}_ablation.png", dpi=230)
        plt.close(fig)


def fig910():
    rows = []
    pattern = re.compile(r"results_fig910_awgn_60db_(count|soft)_alpha(.*)")
    for directory in ROOT.glob("results_fig910_*"):
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        family, raw_alpha = match.groups(); alpha = float(raw_alpha.replace("p", "."))
        payloads = [read_json(path) for path in metric_files(directory)]
        rows.append((family, alpha, payloads))
    rows.sort(key=lambda row: (row[0], row[1]))
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for family, color in (("count", COLORS["SHyRe-count"]), ("soft", COLORS["SHyRe-soft-count"])):
        subset = [(alpha, np.mean([p["outcome"]["performance"]["SHyRe"]["f1"] for p in payloads])) for fam, alpha, payloads in rows if fam == family]
        ax.plot(*zip(*subset), marker="o", color=color, label=f"SHyRe{'-soft' if family == 'soft' else ''}-count")
    ax.set(xlabel="Projection retention ratio α", ylabel="Macro F1 across six datasets", ylim=(-.02, 1.02), title="Fig. 9. Retention–reconstruction trade-off (AWGN, 60 dB)")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.savefig(OUT / "fig_9_retention_f1.png", dpi=230); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for family, color in (("count", COLORS["SHyRe-count"]), ("soft", COLORS["SHyRe-soft-count"])):
        subset = [(alpha, np.mean([p["outcome"]["storage"]["received_projection_storage_units"] for p in payloads])) for fam, alpha, payloads in rows if fam == family]
        ax.plot(*zip(*subset), marker="o", color=color, label=f"SHyRe{'-soft' if family == 'soft' else ''}-count")
    ax.set(xlabel="Projection retention ratio α", ylabel="Mean received projection storage units", title="Fig. 10. Retention–storage trade-off (AWGN, 60 dB)")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.savefig(OUT / "fig_10_retention_storage.png", dpi=230); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for family, color in (("count", COLORS["SHyRe-count"]), ("soft", COLORS["SHyRe-soft-count"])):
        subset = []
        for fam, alpha, payloads in rows:
            if fam != family:
                continue
            f1 = np.mean([p["outcome"]["performance"]["SHyRe"]["f1"] for p in payloads])
            storage = np.mean([p["outcome"]["storage"]["received_projection_storage_units"] for p in payloads])
            subset.append((storage, f1, alpha))
        subset.sort()
        line = ax.plot([item[0] for item in subset], [item[1] for item in subset], marker="o", color=color,
                       label=f"SHyRe{'-soft' if family == 'soft' else ''}-count")[0]
        for storage, f1, alpha in subset:
            ax.annotate(f"α={alpha:.1f}", (storage, f1), xytext=(3, 5), textcoords="offset points",
                        fontsize=8, color=line.get_color())
    ax.set(xlabel="Mean received projection storage units", ylabel="Macro F1 across six datasets", ylim=(-.02, 1.02),
           title="Fig. 9.5. Reconstruction quality versus received storage (AWGN, 60 dB)")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(OUT / "fig_9_5_f1_vs_storage.png", dpi=230); plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); OUT = args.output; OUT.mkdir(parents=True, exist_ok=True)
    snr = load_snr(); fig_snr(snr, "AWGN", 1); fig_snr(snr, "RAYLEIGH", 2)
    fig34_enron10(); fig910()


if __name__ == "__main__":
    main()
