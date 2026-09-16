#!/usr/bin/env python3
"""Create a comparable size-stratified overview for Fig. 11/12 selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("SHyRe", "Max Clique", "ECC", "DEMON")
LABELS = {"SHyRe": "SHyRe-count", "Max Clique": "Max Clique", "ECC": "ECC", "DEMON": "DEMON"}
COLORS = {"SHyRe": "#1565c0", "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#c62828"}
DISPLAY = {"enron": "Enron", "fb_auto": "FB-AUTO", "hosts": "Hosts-Virus", "school": "P.School",
           "school2": "H.School", "directors": "Directors", "crime": "Crime", "jf17k": "JF17K (fast)"}


def load(paths):
    records = {}
    for directory in paths:
        for path in directory.glob("*.metrics.jsonl"):
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            if "size_stratified" in payload["outcome"]:
                records[payload["dataset"]] = payload["outcome"]["size_stratified"]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("figures_size_stratified_candidates"))
    args = parser.parse_args(); records = load(args.inputs); args.output.mkdir(parents=True, exist_ok=True)
    datasets = [key for key in DISPLAY if key in records]
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10.4), sharey=True, constrained_layout=True)
    width = .19
    for ax, dataset in zip(axes.ravel(), datasets):
        record = records[dataset]
        groups = [group for group in ("2", "3", "4", "5+") if group in record["SHyRe"]]
        x = np.arange(len(groups))
        for index, method in enumerate(METHODS):
            ax.bar(x + (index - 1.5) * width, [record[method][group]["f1"] for group in groups], width,
                   color=COLORS[method], label=LABELS[method])
        counts = [record["SHyRe"][group]["truth_count"] for group in groups]
        ax.set_xticks(x, [f"{group}\n(n={count})" for group, count in zip(groups, counts)])
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.2); ax.set_title(DISPLAY[dataset])
    for ax in axes.ravel()[len(datasets):]: ax.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.supxlabel("True hyperedge size (test split)"); fig.supylabel("Exact-match F1")
    fig.suptitle("60 dB AWGN: candidate datasets for Fig. 11/12", y=1.025)
    fig.savefig(args.output / "size_stratified_candidate_overview.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
