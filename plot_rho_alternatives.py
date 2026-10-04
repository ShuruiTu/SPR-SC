#!/usr/bin/env python3
"""Create alternative visualizations for train/query rho(n,k) alignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def load_cells(path: Path, max_n: int, max_k: int, log_floor: float):
    records: dict[str, dict[tuple[int, int], float]] = {"train": {}, "test": {}}
    valid_n: dict[str, set[int]] = {"train": set(), "test": set()}
    with path.open("r", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            n = int(row["max_clique_size_n"])
            k = int(row["subset_size_k"])
            if n > max_n or k > max_k:
                continue
            value = float(row["rho"])
            valid_n[split].add(n)
            records[split][(n, k)] = max(np.log10(value), log_floor) if value > 0 else log_floor
    return records, valid_n


def matrices(records, valid_n, max_n: int, max_k: int):
    output = {}
    for split in ("train", "test"):
        matrix = np.full((max_n, max_k), np.nan)
        for n in valid_n[split]:
            for k in range(1, min(n, max_k) + 1):
                matrix[n - 1, k - 1] = records[split].get((n, k), np.nan)
        output[split] = matrix
    common = np.isfinite(output["train"]) & np.isfinite(output["test"])
    delta = np.full_like(output["train"], np.nan)
    delta[common] = output["test"][common] - output["train"][common]
    return output["train"], output["test"], delta, common


def format_matrix_axis(axis, max_n: int, max_k: int, title: str):
    axis.set_xticks(np.arange(max_k), np.arange(1, max_k + 1))
    axis.set_yticks(np.arange(max_n), np.arange(1, max_n + 1))
    axis.set_xlabel("Subset size k")
    axis.set_ylabel("Maximal-clique size n")
    axis.set_title(title, fontweight="bold")
    axis.tick_params(labelsize=8, length=2)


def plot_composite(train, test, delta, common, out: Path, log_floor: float):
    max_n, max_k = train.shape
    fig = plt.figure(figsize=(13.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    # (a) A k-wise statistical fingerprint: median and interquartile range
    # over all available maximal-clique sizes n. This retains the dominant
    # column-wise structure without drawing another matrix heatmap.
    ks = np.arange(1, max_k + 1)
    for matrix, label, color, marker in [
        (train, "Training", "#2878B5", "o"),
        (test, "Query", "#E07B39", "s"),
    ]:
        medians, lower, upper = [], [], []
        for k in ks:
            values = matrix[:, k - 1]
            values = values[np.isfinite(values)]
            medians.append(np.median(values) if values.size else np.nan)
            lower.append(np.quantile(values, 0.25) if values.size else np.nan)
            upper.append(np.quantile(values, 0.75) if values.size else np.nan)
        medians = np.asarray(medians)
        axes[0].plot(ks, medians, marker=marker, lw=2, ms=5,
                     color=color, label=label)
        axes[0].fill_between(ks, lower, upper, color=color, alpha=0.16)
    axes[0].set_title("(a) k-wise density fingerprint", fontweight="bold")
    axes[0].set_xlabel("Subset size k")
    axes[0].set_ylabel(r"Median $\log_{10}\rho$ (IQR band)")
    axes[0].set_xticks(ks)
    axes[0].set_ylim(log_floor - 0.1, 0.1)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    # (b) Error decomposition by k: magnitude and signed direction.
    maes, biases, counts = [], [], []
    for k in ks:
        mask = common[:, k - 1]
        errors = test[mask, k - 1] - train[mask, k - 1]
        maes.append(np.mean(np.abs(errors)) if errors.size else np.nan)
        biases.append(np.mean(errors) if errors.size else np.nan)
        counts.append(errors.size)
    axes[1].bar(ks, maes, color="#70A5D8", edgecolor="white", label="MAE")
    axes[1].plot(ks, biases, "o-", color="#B33A3A", lw=1.8, ms=5,
                 label="Signed bias")
    axes[1].axhline(0, color="0.25", lw=0.8)
    axes[1].set_title("(b) Alignment error by subset size", fontweight="bold")
    axes[1].set_xlabel("Subset size k")
    axes[1].set_ylabel(r"Error in $\log_{10}\rho$")
    axes[1].set_xticks(ks)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)

    # (c) Global empirical cumulative distributions, avoiding spatial matrix
    # encoding while showing whether both splits have similar rho statistics.
    for matrix, label, color in [
        (train, "Training", "#2878B5"), (test, "Query", "#E07B39")
    ]:
        values = np.sort(matrix[np.isfinite(matrix)])
        probability = np.arange(1, len(values) + 1) / len(values)
        axes[2].step(values, probability, where="post", lw=2.2,
                     color=color, label=f"{label} (median={np.median(values):.2f})")
    axes[2].set_title("(c) Global density distribution (ECDF)", fontweight="bold")
    axes[2].set_xlabel(r"$\log_{10}\rho(n,k)$")
    axes[2].set_ylabel("Cumulative probability")
    axes[2].set_xlim(log_floor - 0.1, 0.1)
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=False, loc="upper left")

    x = train[common]
    y = test[common]
    coords = np.argwhere(common)
    colors = coords[:, 1] + 1
    sizes = 22 + 5 * (coords[:, 0] + 1)
    scatter = axes[3].scatter(x, y, c=colors, s=sizes, cmap="turbo", alpha=0.78,
                              edgecolors="white", linewidths=0.35)
    axes[3].plot([log_floor, 0], [log_floor, 0], "--", color="0.25", lw=1.2,
                 label="perfect alignment")
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
    axes[3].text(0.04, 0.96, f"Pearson r = {pearson:.3f}\nMAE = {mae:.3f}",
                 transform=axes[3].transAxes, va="top",
                 bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85})
    axes[3].set_xlim(log_floor - 0.1, 0.1)
    axes[3].set_ylim(log_floor - 0.1, 0.1)
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].set_xlabel(r"Training $\log_{10}\rho(n,k)$")
    axes[3].set_ylabel(r"Query $\log_{10}\rho(n,k)$")
    axes[3].set_title("(d) Cell-wise alignment", fontweight="bold")
    axes[3].grid(alpha=0.18)
    cb3 = fig.colorbar(scatter, ax=axes[3], shrink=0.82, pad=0.02)
    cb3.set_label("Subset size k")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bubbles(train, test, delta, common, out: Path, log_floor: float):
    max_n, max_k = train.shape
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True,
                             sharex=True, sharey=True)
    mean_density = (train + test) / 2
    # Use a tighter diverging range so meaningful deviations are saturated,
    # while compressing bubble areas prevents dense early columns from
    # becoming visually dominant blocks.
    norm = TwoSlopeNorm(vmin=-1.25, vcenter=0.0, vmax=1.25)
    for axis, split_matrix, title in zip(
        axes, [train, test], ["(a) Training", "(b) Query"]
    ):
        mask = np.isfinite(split_matrix)
        coords = np.argwhere(mask)
        density_scaled = np.clip((split_matrix[mask] - log_floor) / -log_floor, 0, 1)
        # Keep the smallest observations clearly visible, while restoring a
        # wider area range so density ratios remain easy to distinguish.
        min_area, max_area = 96.0, 450.0
        sizes = min_area + (max_area - min_area) * density_scaled ** 1.08
        cell_delta = np.array([delta[n, k] if common[n, k] else np.nan
                               for n, k in coords])
        colors = np.nan_to_num(cell_delta, nan=0.0)
        # Draw high-deviation cells last so red/blue anomalies are never
        # visually buried underneath the numerous near-neutral cells.
        order = np.argsort(np.abs(colors))
        bubble_cmap = plt.colormaps["RdBu_r"]
        face_rgba = bubble_cmap(norm(colors[order]))
        edge_rgba = face_rgba.copy()
        edge_rgba[:, :3] *= 0.62
        edge_rgba[:, 3] = 0.88
        scatter = axis.scatter(
            coords[order, 1] + 1,
            coords[order, 0] + 1,
            s=sizes[order],
            c=colors[order],
            cmap=bubble_cmap,
            norm=norm,
            edgecolors=edge_rgba,
            linewidths=0.65,
            alpha=0.94,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Subset size k")
        axis.set_xticks(range(1, max_k + 1))
        axis.set_yticks(range(1, max_n + 1))
        axis.grid(alpha=0.15)
        axis.invert_yaxis()
    axes[0].set_ylabel("Maximal-clique size n")
    cb = fig.colorbar(scatter, ax=axes, shrink=0.82, pad=0.02)
    cb.set_label(r"Query $-$ training: $\Delta\log_{10}\rho$")
    fig.text(0.5, -0.02,
             r"Bubble area encodes clipped $\log_{10}\rho$ density; saturated color highlights alignment deviations.",
             ha="center", fontsize=10)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bubble_alignment_triptych(
    train, test, delta, common, out: Path, log_floor: float
):
    """Combine train/query bubble fields and the cell-wise alignment plot."""
    max_n, max_k = train.shape
    fig, axes = plt.subplots(
        1, 3, figsize=(18.2, 5.7), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.92]},
    )
    norm = TwoSlopeNorm(vmin=-1.25, vcenter=0.0, vmax=1.25)
    bubble_cmap = plt.colormaps["RdBu_r"]
    bubble_scatter = None

    for axis, split_matrix, title in zip(
        axes[:2], [train, test], ["(a) Training", "(b) Query"]
    ):
        mask = np.isfinite(split_matrix)
        coords = np.argwhere(mask)
        density_scaled = np.clip(
            (split_matrix[mask] - log_floor) / -log_floor, 0, 1
        )
        min_area, max_area = 96.0, 450.0
        sizes = min_area + (max_area - min_area) * density_scaled ** 1.08
        cell_delta = np.array([
            delta[n, k] if common[n, k] else np.nan for n, k in coords
        ])
        colors = np.nan_to_num(cell_delta, nan=0.0)
        order = np.argsort(np.abs(colors))
        face_rgba = bubble_cmap(norm(colors[order]))
        edge_rgba = face_rgba.copy()
        edge_rgba[:, :3] *= 0.62
        edge_rgba[:, 3] = 0.88
        bubble_scatter = axis.scatter(
            coords[order, 1] + 1,
            coords[order, 0] + 1,
            s=sizes[order],
            c=colors[order],
            cmap=bubble_cmap,
            norm=norm,
            edgecolors=edge_rgba,
            linewidths=0.65,
            alpha=0.94,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Subset size k")
        axis.set_xticks(range(1, max_k + 1))
        axis.set_yticks(range(1, max_n + 1))
        axis.set_xlim(0.4, max_k + 0.6)
        axis.set_ylim(0.4, max_n + 0.6)
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Maximal-clique size n")
    axes[1].tick_params(labelleft=False)

    bubble_cb = fig.colorbar(
        bubble_scatter, ax=axes[:2], shrink=0.78, pad=0.018, location="bottom"
    )
    bubble_cb.set_label(r"Query $-$ training: $\Delta\log_{10}\rho$")

    x = train[common]
    y = test[common]
    coords = np.argwhere(common)
    k_colors = coords[:, 1] + 1
    point_sizes = 24 + 5 * (coords[:, 0] + 1)
    align_scatter = axes[2].scatter(
        x, y, c=k_colors, s=point_sizes, cmap="turbo", alpha=0.8,
        edgecolors="white", linewidths=0.35,
    )
    axes[2].plot(
        [log_floor, 0], [log_floor, 0], "--", color="0.25", lw=1.2
    )
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
    axes[2].text(
        0.04, 0.96, f"Pearson r = {pearson:.3f}\nMAE = {mae:.3f}",
        transform=axes[2].transAxes, va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.88},
    )
    axes[2].set_xlim(log_floor - 0.1, 0.1)
    axes[2].set_ylim(log_floor - 0.1, 0.1)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"Training $\log_{10}\rho(n,k)$")
    axes[2].set_ylabel(r"Query $\log_{10}\rho(n,k)$")
    axes[2].set_title("(c) Cell-wise alignment", fontweight="bold")
    axes[2].grid(alpha=0.18)
    align_cb = fig.colorbar(align_scatter, ax=axes[2], shrink=0.78, pad=0.02)
    align_cb.set_label("Subset size k")

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_profiles(train, test, out: Path, ks: list[int]):
    max_n, _ = train.shape
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True,
                             sharex=True, sharey=True)
    for axis, k in zip(axes.flat, ks):
        n = np.arange(1, max_n + 1)
        axis.plot(n, train[:, k - 1], "o-", lw=1.7, ms=4, label="Training")
        axis.plot(n, test[:, k - 1], "s--", lw=1.7, ms=4, label="Query")
        axis.set_title(f"Subset size k = {k}", fontweight="bold")
        axis.set_xlabel("Maximal-clique size n")
        axis.set_ylabel(r"$\log_{10}\rho(n,k)$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path,
                        default=Path("results_rho_heatmap_preview/enron_rho_cells.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results_rho_heatmap_preview"))
    parser.add_argument("--max-k", type=int, default=13)
    parser.add_argument("--max-n", type=int, default=13)
    parser.add_argument("--log-floor", type=float, default=-5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, valid_n = load_cells(
        args.cells, args.max_n, args.max_k, args.log_floor
    )
    train, test, delta, common = matrices(
        records, valid_n, args.max_n, args.max_k
    )
    plot_composite(train, test, delta, common,
                   args.output_dir / "fig_rho_enron_composite_k13.png", args.log_floor)
    plot_bubbles(train, test, delta, common,
                 args.output_dir / "fig_rho_enron_bubble_k13.png", args.log_floor)
    plot_bubble_alignment_triptych(
        train, test, delta, common,
        args.output_dir / "fig_rho_enron_bubble_alignment_triptych_k13.png",
        args.log_floor,
    )
    plot_profiles(train, test, args.output_dir / "fig_rho_enron_profiles_k13.png",
                  [2, 3, 4, 5])
    print(
        "Generated composite, bubble, and profile figures with n <=",
        args.max_n, "and k <=", args.max_k,
    )


if __name__ == "__main__":
    main()
