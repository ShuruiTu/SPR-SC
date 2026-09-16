#!/usr/bin/env python3
"""Render Fig. 11--12 from structured size-stratified metrics JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("SHyRe", "Max Clique", "ECC", "DEMON")
LABELS = {"SHyRe": "SHyRe-count", "Max Clique": "Max Clique", "ECC": "ECC", "DEMON": "DEMON"}
COLORS = {"SHyRe": "#1565c0", "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#8e24aa"}


def load(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return payload["dataset"], payload["outcome"]["size_stratified"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures_fig11_12"))
    args = parser.parse_args()
    records = dict(load(path) for path in sorted(args.input.glob("*.metrics.jsonl")))
    wanted = ("enron", "fb_auto")
    missing = [name for name in wanted if name not in records]
    if missing:
        raise SystemExit(f"Missing size-stratified metrics for: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    for figure_number, dataset in enumerate(wanted, start=11):
        record = records[dataset]
        groups = [group for group in ("2", "3", "4", "5+") if group in record["SHyRe"]]
        x = np.arange(len(groups)); width = 0.19
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for index, method in enumerate(METHODS):
            values = [record[method][group]["f1"] for group in groups]
            ax.bar(x + (index - 1.5) * width, values, width, label=LABELS[method], color=COLORS[method])
        counts = [record["SHyRe"][group]["truth_count"] for group in groups]
        ax.set_xticks(x, [f"{group}\n(n={count})" for group, count in zip(groups, counts)])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Exact-match F1")
        ax.set_xlabel("True hyperedge size (test split)")
        ax.set_title(f"Fig. {figure_number}. {dataset}: size-stratified recovery (AWGN, 60 dB)")
        ax.grid(axis="y", alpha=.22); ax.legend(ncols=2, frameon=False)
        fig.savefig(args.output / f"fig_{figure_number}_{dataset}_size_stratified.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
