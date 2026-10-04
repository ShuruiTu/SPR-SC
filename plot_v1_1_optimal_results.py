#!/usr/bin/env python3
"""Plot the v1.1 three-seed optimal curves and the AWGN-20 dB ablation."""
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
CURVE_DIRS = (
    "results_v1_1_optimal_core_awgn_0_20_step2",
    "results_v1_1_optimal_core_rayleigh_0_20_step2",
    "results_v1_1_optimal_fb_auto_awgn_0_20_step2",
    "results_v1_1_optimal_fb_auto_rayleigh_0_20_step2",
    "results_v1_1_optimal_hosts_fast_awgn_0_20_step2",
    "results_v1_1_optimal_hosts_fast_rayleigh_0_20_step2",
    "results_v1_1_optimal_jf17k_fast_awgn_0_20_step2",
    "results_v1_1_optimal_jf17k_fast_rayleigh_0_20_step2",
    "results_v1_1_optimal_m_fb15k_fast_awgn_0_20",
    "results_v1_1_optimal_m_fb15k_fast_rayleigh_0_20",
    "results_v1_1_optimal_wikipeople_fast_awgn_0_20",
    "results_v1_1_optimal_wikipeople_fast_rayleigh_0_20",
    "results_v1_1_optimal_dblp_fast_awgn_0_20",
    "results_v1_1_optimal_dblp_fast_rayleigh_0_20",
)
ABLATIONS = (
    ("base", "Base\n(fixed τ)"),
    ("p1", "+P1\nthreshold"),
    ("p1_p2", "+P1+P2\ncandidates"),
    ("p1_p3", "+P1+P3\nSNR window"),
    ("p1_p2_p3", "+P1+P2+P3"),
    ("p1_p4", "+P1+P4\nsoft-v2"),
    ("p1_p5_lr", "+P1+P5\nLR"),
    ("p1_upsample", "+P1\nupsample"),
)
DISPLAY = {
    "enron": "Enron", "hosts": "Hosts-Virus", "school": "P.School",
    "school2": "H.School", "directors": "Directors", "crime": "Crime",
    "fb_auto": "FB-AUTO", "jf17k": "JF17K",
    "foursquare": "Foursquare", "m_fb15k": "M-FB15K",
    "wikipeople": "WikiPeople", "dblp": "DBLP",
}
ORDER = tuple(DISPLAY)
COLORS = {
    "SHyRe-count": "#1565c0", "SHyRe-fast-count": "#6a1b9a",
    "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#c62828",
}


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def load_curves(root: Path):
    values = defaultdict(list)
    for name in CURVE_DIRS:
        directory = root / name
        path = directory / "snr_baselines.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = 165 if "_core_" in name else 33
        completed = {
            (row["dataset"], row["channel"], row["snr_db"], row.get("seed", "123"))
            for row in rows if row["method"] == "SHyRe" and row["returncode"] == "0"
        }
        # Never draw an interrupted dataset as if it were a complete SNR curve.
        if len(completed) < expected:
            continue
        fast = "_fast_" in name
        for row in rows:
            if row["returncode"] != "0":
                continue
            method = row["method"]
            if method == "SHyRe":
                method = "SHyRe-fast-count" if fast else "SHyRe-count"
            values[(row["dataset"], row["channel"].upper(), int(row["snr_db"]), method)].append(float(row["f1"]))
    return values


def plot_curves(values, output: Path):
    for figure, channel in ((1, "AWGN"), (2, "RAYLEIGH")):
        # Delivery figures intentionally retain only two representative
        # datasets. Complete numeric results for every other finished dataset
        # are exported to the companion Excel workbook.
        datasets = [d for d in ("enron", "fb_auto")
                    if any(k[0] == d and k[1] == channel for k in values)]
        columns = max(1, len(datasets)); rows = 1
        fig, axes = plt.subplots(rows, columns, figsize=(10.8, 5.0), sharex=True,
                                 sharey=True, squeeze=False)
        for ax, dataset in zip(axes.ravel(), datasets):
            for method, color in COLORS.items():
                points = []
                for key, samples in values.items():
                    if key[0] == dataset and key[1] == channel and key[3] == method:
                        mean, std = mean_std(samples)
                        points.append((key[2], mean, std))
                if not points:
                    continue
                points.sort(); x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points]); s = np.array([p[2] for p in points])
                ax.plot(x, y, marker="o", ms=2.8, lw=1.4, color=color, label=method)
                ax.fill_between(x, np.maximum(0, y - s), np.minimum(1, y + s), color=color, alpha=.13)
            ax.set_title(DISPLAY[dataset]); ax.grid(alpha=.2); ax.set_ylim(-.02, 1.02)
        for ax in axes.ravel()[len(datasets):]:
            ax.remove()
        handles, labels = [], []
        for ax in axes.ravel()[:len(datasets)]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle); labels.append(label)
        # Keep the global legend on its own row.  ``constrained_layout`` does
        # not reserve space for a figure-level legend and used to place it on
        # top of the title in the publication montage.
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .915),
                   ncol=5, frameon=False)
        fig.supxlabel("SNR (dB)", y=.025); fig.supylabel("Exact-match reconstruction F1", x=.018)
        suffix = "; β=1.5, perfect CSI" if channel == "RAYLEIGH" else ""
        fig.suptitle(f"Fig. {figure}. v1.1 matched-channel results ({channel}{suffix}; mean ± std, 3 seeds)",
                     y=.995)
        fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.79,
                            hspace=.16, wspace=.08)
        fig.savefig(output / f"fig_{figure}_v1_1_optimal_{channel.lower()}_snr.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def load_ablation(root: Path):
    result = defaultdict(lambda: defaultdict(list))
    for key, _ in ABLATIONS:
        directory = root / f"results_v1_1_ablation_awgn20_{key}"
        for path in directory.glob("*.metrics.jsonl"):
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            dataset = payload["dataset"]
            result[(key, dataset)]["f1"].append(payload["outcome"]["performance"]["SHyRe"]["f1"])
            result[(key, dataset)]["candidate_recall"].append(payload["outcome"]["candidate"]["candidate_recall"])
            result[(key, dataset)]["runtime"].append(payload["outcome"]["runtime"]["wall_time_seconds"])
    return result


def plot_ablation(values, output: Path):
    datasets = ("enron", "crime")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2), constrained_layout=True)
    rows = []
    for column, dataset in enumerate(datasets):
        for row_index, (metric, ylabel) in enumerate((("f1", "Exact-match F1"), ("candidate_recall", "Candidate recall"))):
            means, stds = [], []
            for key, _ in ABLATIONS:
                mean, std = mean_std(values[(key, dataset)][metric]); means.append(mean); stds.append(std)
            ax = axes[row_index, column]
            bars = ax.bar(np.arange(len(ABLATIONS)), means, yerr=stds, capsize=3,
                          color=plt.cm.Blues(np.linspace(.35, .88, len(ABLATIONS))))
            ax.set_xticks(np.arange(len(ABLATIONS)), [label for _, label in ABLATIONS], rotation=22, ha="right")
            ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.2); ax.set_ylabel(ylabel)
            ax.set_title(f"{DISPLAY[dataset]} — {ylabel}")
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=2, fontsize=7)
        for key, label in ABLATIONS:
            for metric in ("f1", "candidate_recall", "runtime"):
                mean, std = mean_std(values[(key, dataset)][metric])
                rows.append({"dataset": dataset, "module": key, "label": label.replace("\n", " "),
                             "metric": metric, "mean": mean, "std": std, "seeds": len(values[(key, dataset)][metric])})
    fig.suptitle("Module ablation at AWGN 20 dB (mean ± std, seeds 123/124/125)")
    fig.savefig(output / "fig_ablation_enron_crime_awgn20.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    with (output / "ablation_enron_crime_awgn20.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "module", "label", "metric", "mean", "std", "seeds"))
        writer.writeheader(); writer.writerows(rows)


def plot_all_dataset_ablation(values, output: Path):
    """Summarize complete standard and bounded-generator ablations.

    P2 is unavailable for SHyRe-fast, so those two cells are deliberately N/A.
    """
    fast_datasets = {"jf17k", "m_fb15k", "wikipeople"}
    p2_keys = {"p1_p2", "p1_p2_p3"}
    def applicable(dataset, key):
        return not (dataset in fast_datasets and key in p2_keys)
    datasets = [dataset for dataset in ORDER if all(
        not applicable(dataset, key) or len(values[(key, dataset)]["f1"]) >= 3
        for key, _ in ABLATIONS
    )]
    if not datasets:
        return

    keys = [key for key, _ in ABLATIONS]
    labels = [label.replace("\n", " ") for _, label in ABLATIONS]
    means = np.array([[mean_std(values[(key, dataset)]["f1"])[0]
                       if applicable(dataset, key) else np.nan
                       for key in keys] for dataset in datasets])
    deltas = means - means[:, [0]]

    fig, ax = plt.subplots(figsize=(13.2, max(4.8, .58 * len(datasets) + 2.2)),
                           constrained_layout=True)
    limit = max(.05, float(np.nanmax(np.abs(deltas))))
    palette = plt.get_cmap("RdYlGn").copy(); palette.set_bad(color="#d9d9d9")
    image = ax.imshow(deltas, cmap=palette, vmin=-limit, vmax=limit, aspect="auto")
    for row in range(len(datasets)):
        for column in range(len(keys)):
            value = deltas[row, column]
            color = "white" if np.isfinite(value) and abs(value) > .58 * limit else "black"
            label = f"{value:+.3f}" if np.isfinite(value) else "N/A"
            ax.text(column, row, label, ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(len(keys)), labels, rotation=24, ha="right")
    ax.set_yticks(range(len(datasets)), [DISPLAY[d] for d in datasets])
    ax.set_title("AWGN 20 dB module ablation: F1 change relative to Base (mean of 3 seeds)")
    fig.colorbar(image, ax=ax, label="Δ exact-match F1 versus Base", shrink=.82)
    fig.savefig(output / "fig_ablation_all_datasets_awgn20_delta.png", dpi=240)
    plt.close(fig)

    best_indices = np.nanargmax(means, axis=1)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, .55 * len(datasets) + 2.0)),
                           constrained_layout=True)
    y = np.arange(len(datasets))
    bars = ax.barh(y, means[np.arange(len(datasets)), best_indices], color="#1565c0")
    ax.set_yticks(y, [DISPLAY[d] for d in datasets]); ax.invert_yaxis()
    ax.set_xlim(0, 1.02); ax.set_xlabel("Best exact-match F1")
    ax.set_title("Best AWGN 20 dB ablation configuration by dataset")
    for bar, index, value in zip(bars, best_indices, means[np.arange(len(datasets)), best_indices]):
        if value > .82:
            x, alignment = value - .012, "right"
        else:
            x, alignment = value + .012, "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{labels[index]}  ({value:.3f})", va="center", ha=alignment, fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "fig_ablation_all_datasets_awgn20_best.png", dpi=240)
    plt.close(fig)

    with (output / "ablation_all_datasets_awgn20.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "dataset", "module", "label", "f1_mean", "f1_std", "delta_vs_base",
            "candidate_recall_mean", "runtime_mean_seconds", "seeds", "p2_applicable"))
        writer.writeheader()
        for row, dataset in enumerate(datasets):
            for column, (key, label) in enumerate(ABLATIONS):
                if not applicable(dataset, key):
                    writer.writerow({
                        "dataset": dataset, "module": key, "label": label.replace("\n", " "),
                        "seeds": 0, "p2_applicable": False,
                    })
                    continue
                f1_mean, f1_std = mean_std(values[(key, dataset)]["f1"])
                recall = mean_std(values[(key, dataset)]["candidate_recall"])[0]
                runtime = mean_std(values[(key, dataset)]["runtime"])[0]
                writer.writerow({
                    "dataset": dataset, "module": key, "label": label.replace("\n", " "),
                    "f1_mean": f1_mean, "f1_std": f1_std,
                    "delta_vs_base": deltas[row, column],
                    "candidate_recall_mean": recall, "runtime_mean_seconds": runtime,
                    "seeds": len(values[(key, dataset)]["f1"]),
                    "p2_applicable": True,
                })


def plot_candidate_ablation(root: Path, output: Path):
    pattern = re.compile(r"results_v1_1_fig34_(awgn|rayleigh)_10db_(count|soft)_(shyre|strict_max_clique)")
    records = defaultdict(list)
    for directory in root.glob("results_v1_1_fig34_*"):
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        channel, family, generator = match.groups()
        for path in directory.glob("*.metrics.jsonl"):
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            records[(channel, family, generator, "f1")].append(payload["outcome"]["performance"]["SHyRe"]["f1"])
            records[(channel, family, generator, "candidate_recall")].append(payload["outcome"]["candidate"]["candidate_recall"])
    styles = (("count", "strict_max_clique", "count strict"), ("count", "shyre", "count SHyRe"),
              ("soft", "strict_max_clique", "soft-v2 strict"), ("soft", "shyre", "soft-v2 SHyRe"))
    for figure, metric, ylabel in ((3, "f1", "Exact-match F1"), (4, "candidate_recall", "Candidate recall")):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, constrained_layout=True)
        for ax, channel in zip(axes, ("awgn", "rayleigh")):
            means, stds = [], []
            for family, generator, _ in styles:
                mean, std = mean_std(records[(channel, family, generator, metric)])
                means.append(mean); stds.append(std)
            bars = ax.bar(range(4), means, yerr=stds, capsize=3,
                          color=("#90caf9", "#1565c0", "#80cbc4", "#00897b"))
            ax.set_xticks(range(4), [label.replace(" ", "\n") for _, _, label in styles])
            ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.2); ax.set_title(channel.upper())
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=2, fontsize=8)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"Fig. {figure}. Enron at 10 dB ({ylabel}; mean ± std, 3 seeds)")
        fig.savefig(output / f"fig_{figure}_v1_1_{metric}_ablation.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def load_retention(root: Path):
    pattern = re.compile(r"results_v1_1_fig910_awgn60_(count|soft)_alpha(.*)")
    values = defaultdict(list)
    for directory in root.glob("results_v1_1_fig910_*"):
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        family, raw_alpha = match.groups(); alpha = float(raw_alpha.replace("p", "."))
        by_seed = defaultdict(lambda: defaultdict(list))
        for path in directory.glob("*.metrics.jsonl"):
            payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            seed = int(payload["seed"])
            by_seed[seed]["f1"].append(payload["outcome"]["performance"]["SHyRe"]["f1"])
            by_seed[seed]["storage"].append(payload["outcome"]["storage"]["received_projection_storage_units"])
        for seed, metrics in by_seed.items():
            for metric, samples in metrics.items():
                values[(family, alpha, metric)].append(float(np.mean(samples)))
    return values


def plot_retention(values, output: Path):
    colors = {"count": "#1565c0", "soft": "#00897b"}
    labels = {"count": "SHyRe-count", "soft": "SHyRe-soft-v2-count"}
    for figure, metric, ylabel, filename in (
            (9, "f1", "Macro F1 across five datasets", "fig_9_v1_1_retention_f1.png"),
            (10, "storage", "Mean received projection storage units", "fig_10_v1_1_retention_storage.png")):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        for family in ("count", "soft"):
            points = []
            for key, samples in values.items():
                if key[0] == family and key[2] == metric:
                    mean, std = mean_std(samples); points.append((key[1], mean, std))
            points.sort(); x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points]); s = np.array([p[2] for p in points])
            ax.plot(x, y, marker="o", color=colors[family], label=labels[family])
            ax.fill_between(x, y - s, y + s, color=colors[family], alpha=.14)
        ax.set_xlabel("Projection retention ratio α"); ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(frameon=False)
        if metric == "f1": ax.set_ylim(-.02, 1.02)
        ax.set_title(f"Fig. {figure}. Retention trade-off at AWGN 60 dB (mean ± std, 3 seeds)")
        fig.savefig(output / filename, dpi=240); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for family in ("count", "soft"):
        points = []
        alphas = sorted({key[1] for key in values if key[0] == family})
        for alpha in alphas:
            storage = mean_std(values[(family, alpha, "storage")])[0]
            f1 = mean_std(values[(family, alpha, "f1")])[0]
            points.append((storage, f1, alpha))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=colors[family], label=labels[family])
        for storage, f1, alpha in points:
            ax.annotate(f"α={alpha:.1f}", (storage, f1), xytext=(3, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean received projection storage units", ylabel="Macro F1 across five datasets",
           ylim=(-.02, 1.02), title="Fig. 9.5. Reconstruction quality versus storage (AWGN, 60 dB)")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(output / "fig_9_5_v1_1_f1_vs_storage.png", dpi=240); plt.close(fig)


def plot_size_stratified(root: Path, output: Path):
    records = defaultdict(list)
    for path in (root / "results_v1_1_fig11_12_size_awgn60").glob("*.metrics.jsonl"):
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        for method, groups in payload["outcome"]["size_stratified"].items():
            for group, metrics in groups.items():
                records[(payload["dataset"], method, group, "f1")].append(metrics["f1"])
                records[(payload["dataset"], method, group, "truth_count")].append(metrics["truth_count"])
    methods = ("SHyRe", "Max Clique", "ECC", "DEMON")
    labels = {"SHyRe": "SHyRe-count", "Max Clique": "Max Clique", "ECC": "ECC", "DEMON": "DEMON"}
    colors = {"SHyRe": "#1565c0", "Max Clique": "#ef6c00", "ECC": "#2e7d32", "DEMON": "#8e24aa"}
    for figure, dataset in ((11, "enron"), (12, "fb_auto")):
        groups = [g for g in ("2", "3", "4", "5+") if (dataset, "SHyRe", g, "f1") in records]
        x = np.arange(len(groups)); width = .19
        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        for index, method in enumerate(methods):
            means, stds = zip(*(mean_std(records[(dataset, method, group, "f1")]) for group in groups))
            ax.bar(x + (index - 1.5) * width, means, width, yerr=stds, capsize=2,
                   label=labels[method], color=colors[method])
        counts = [int(round(mean_std(records[(dataset, "SHyRe", group, "truth_count")])[0])) for group in groups]
        ax.set_xticks(x, [f"{group}\n(n={count})" for group, count in zip(groups, counts)])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Exact-match F1"); ax.set_xlabel("True hyperedge size")
        ax.set_title(f"Fig. {figure}. {DISPLAY[dataset]} size-stratified recovery (AWGN, 60 dB; 3 seeds)")
        ax.grid(axis="y", alpha=.2); ax.legend(ncols=2, frameon=False)
        fig.savefig(output / f"fig_{figure}_v1_1_{dataset}_size_stratified.png", dpi=240); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "figures_v1_1_optimal")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    curves = load_curves(args.root)
    if curves:
        plot_curves(curves, args.output)
    ablation = load_ablation(args.root)
    if ablation:
        plot_ablation(ablation, args.output)
        plot_all_dataset_ablation(ablation, args.output)
    plot_candidate_ablation(args.root, args.output)
    retention = load_retention(args.root)
    if retention:
        plot_retention(retention, args.output)
    plot_size_stratified(args.root, args.output)


if __name__ == "__main__":
    main()
