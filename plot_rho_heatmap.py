#!/usr/bin/env python3
"""Plot paper-style rho(n, k) heatmaps from a hypergraph train/test split.

rho(n, k) is the probability that a uniformly sampled size-k subset of a
size-n maximal clique is a unique observed hyperedge:

    rho(n, k) = |E_{n,k}| / (|M_n| * C(n, k)).

The script uses each split's clean clique projection, matching the definition
in Figure 4 of "From Graphs to Hypergraphs" (ICLR 2024).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_hyperedges(path: Path) -> set[tuple[int, ...]]:
    hyperedges: set[tuple[int, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            nodes = tuple(sorted({int(node) for node in line.split()}))
            if nodes:
                hyperedges.add(nodes)
    return hyperedges


def project(hyperedges: set[tuple[int, ...]]) -> nx.Graph:
    graph = nx.Graph()
    for edge in hyperedges:
        graph.add_nodes_from(edge)
        graph.add_edges_from(combinations(edge, 2))
    return graph


def compute_rho(
    hyperedges: set[tuple[int, ...]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]], int]:
    graph = project(hyperedges)
    maximal_cliques = [frozenset(c) for c in nx.find_cliques(graph)]

    cliques_by_size: dict[int, list[frozenset[int]]] = defaultdict(list)
    edges_by_size: dict[int, list[frozenset[int]]] = defaultdict(list)
    for clique in maximal_cliques:
        cliques_by_size[len(clique)].append(clique)
    for edge in hyperedges:
        edges_by_size[len(edge)].append(frozenset(edge))

    rho: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], tuple[int, int]] = {}
    for n, n_cliques in sorted(cliques_by_size.items()):
        for k in range(1, n + 1):
            contained = {
                edge
                for edge in edges_by_size.get(k, [])
                if any(edge <= clique for clique in n_cliques)
            }
            denominator = len(n_cliques) * math.comb(n, k)
            numerator = len(contained)
            rho[(n, k)] = numerator / denominator if denominator else 0.0
            counts[(n, k)] = (numerator, denominator)

    maximum_n = max(cliques_by_size, default=0)
    return rho, counts, maximum_n


def rho_matrix(
    rho: dict[tuple[int, int], float], maximum_n: int, maximum_k: int,
    log_floor: float
) -> np.ndarray:
    matrix = np.full((maximum_n, maximum_k), np.nan, dtype=float)
    for (n, k), value in rho.items():
        if n <= maximum_n and k <= maximum_k and value > 0:
            matrix[n - 1, k - 1] = max(math.log10(value), log_floor)
    return matrix


def write_cells(
    path: Path,
    dataset: str,
    split: str,
    rho: dict[tuple[int, int], float],
    counts: dict[tuple[int, int], tuple[int, int]],
) -> None:
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(
                ["dataset", "split", "max_clique_size_n", "subset_size_k",
                 "unique_hyperedges_E_nk", "sample_space_Q_nk", "rho", "log10_rho"]
            )
        for n, k in sorted(rho):
            value = rho[(n, k)]
            numerator, denominator = counts[(n, k)]
            writer.writerow(
                [dataset, split, n, k, numerator, denominator, value,
                 math.log10(value) if value > 0 else ""]
            )


def plot_pair(
    train: np.ndarray,
    query: np.ndarray,
    output_png: Path,
    output_pdf: Path,
    log_floor: float,
) -> None:
    maximum_n = max(train.shape[0], query.shape[0])
    maximum_k = max(train.shape[1], query.shape[1])

    def pad(matrix: np.ndarray) -> np.ndarray:
        padded = np.full((maximum_n, maximum_k), np.nan)
        padded[: matrix.shape[0], : matrix.shape[1]] = matrix
        return padded

    cmap = plt.colormaps["Greys"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    images = []
    titles = [
        rf"$\rho(n,k)$ on training ($\mathcal{{H}}_0,\mathcal{{G}}_0$)",
        rf"$\rho(n,k)$ on query ($\mathcal{{H}}_1,\mathcal{{G}}_1$)",
    ]
    for axis, matrix, title in zip(axes, [pad(train), pad(query)], titles):
        image = axis.imshow(
            matrix,
            origin="upper",
            cmap=cmap,
            vmin=log_floor,
            vmax=0.0,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        x_ticks = np.arange(maximum_k)
        y_ticks = np.arange(maximum_n)
        axis.set_xticks(x_ticks, np.arange(1, maximum_k + 1), fontsize=8)
        axis.set_yticks(y_ticks, np.arange(1, maximum_n + 1), fontsize=8)
        axis.set_xlabel("subset size (k)", fontsize=11)
        axis.set_ylabel("maximal clique size (n)", fontsize=11)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.tick_params(length=2.5, width=0.7)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label(r"$\log_{10}\rho(n,k)$", fontsize=10)
    colorbar.ax.text(1.9, 0.98, "dense", transform=colorbar.ax.transAxes,
                     ha="left", va="top", fontsize=9)
    colorbar.ax.text(1.9, 0.02, "sparse", transform=colorbar.ax.transAxes,
                     ha="left", va="bottom", fontsize=9)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="enron")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results_rho_heatmap_preview"))
    parser.add_argument("--log-floor", type=float, default=-5.0)
    parser.add_argument("--max-k", type=int, default=13)
    parser.add_argument("--max-n", type=int, default=13)
    args = parser.parse_args()

    dataset_dir = args.data_dir / args.dataset
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = output_dir / f"{args.dataset}_rho_cells.csv"
    if cells_csv.exists():
        cells_csv.unlink()

    matrices = {}
    stats = {}
    for split in ("train", "test"):
        hyperedges = read_hyperedges(dataset_dir / f"{split}.txt")
        rho, counts, maximum_n = compute_rho(hyperedges)
        matrices[split] = rho_matrix(
            rho, min(maximum_n, args.max_n), args.max_k, args.log_floor
        )
        stats[split] = (len(hyperedges), maximum_n, len(rho))
        write_cells(cells_csv, args.dataset, split, rho, counts)

    output_png = output_dir / f"fig_rho_{args.dataset}_train_query.png"
    output_pdf = output_dir / f"fig_rho_{args.dataset}_train_query.pdf"
    plot_pair(
        matrices["train"], matrices["test"],
        output_png, output_pdf, args.log_floor,
    )
    print(f"PNG: {output_png}")
    print(f"PDF: {output_pdf}")
    print(f"CSV: {cells_csv}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
