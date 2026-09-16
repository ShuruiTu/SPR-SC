#!/usr/bin/env python3
"""Render the matched-channel-training rerun of Fig. 1/2 from JSONL metrics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY = {"enron": "Enron", "hosts": "Hosts-Virus", "school": "P.School", "school2": "H.School",
           "directors": "Directors", "crime": "Crime", "fb_auto": "FB-AUTO", "jf17k": "JF17K"}
COLORS = {"SHyRe-matched-train-count": "#1565c0", "SHyRe-matched-train-soft-count": "#00897b",
          "SHyRe-fast-matched-train-count": "#6a1b9a", "SHyRe-fast-matched-train-soft-count": "#ab47bc",
          "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#c62828"}


def load(root: Path):
    records = defaultdict(list)
    metric_paths = list(root.glob("results_fig1_2_matched_train_*_0_20_step2/*.metrics.jsonl"))
    # FB-AUTO was already complete except AWGN 2 dB; its repaired point is
    # intentionally kept in a small standalone result directory.
    metric_paths.extend(root.glob("results_fig1_2_matched_train_fb_auto_repair_*/*.metrics.jsonl"))
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        method = payload["model_variant"]
        for name, metric in payload["outcome"]["performance"].items():
            label = method if name == "SHyRe" else name
            records[(payload["dataset"], payload["channel"].upper(), int(payload["snr_db"]), label)].append(metric["f1"])
    return {key: sum(values) / len(values) for key, values in records.items()}


def draw(values, channel: str, figure: int, output: Path):
    datasets = [dataset for dataset in DISPLAY if any(key[:2] == (dataset, channel) for key in values)]
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes.ravel(), datasets):
        methods = [name for name in COLORS if any(key[:2] == (dataset, channel) and key[3] == name for key in values)]
        for method in methods:
            points = sorted((key[2], value) for key, value in values.items() if key[:2] == (dataset, channel) and key[3] == method)
            ax.plot(*zip(*points), marker="o", ms=2.7, lw=1.35, label=method, color=COLORS[method])
        ax.set_title(DISPLAY[dataset]); ax.grid(alpha=.2); ax.set_ylim(-.02, 1.02)
    for ax in axes.ravel()[len(datasets):]: ax.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("SNR (dB)"); fig.supylabel("Exact-match reconstruction F1")
    fig.suptitle(f"Fig. {figure}. {channel}: matched channel training, single seed", y=1.025)
    fig.savefig(output / f"fig_{figure}_{channel.lower()}_matched_train_snr.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("figures_fig1_2_matched_train"))
    args = parser.parse_args(); values = load(args.root); args.output.mkdir(parents=True, exist_ok=True)
    if not values:
        raise SystemExit("No matched-training metric files found")
    draw(values, "AWGN", 1, args.output); draw(values, "RAYLEIGH", 2, args.output)


if __name__ == "__main__":
    main()
