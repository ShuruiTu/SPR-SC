"""Optional experiment telemetry and pre-channel projection retention.

All helpers are side-effect free except ``write_metrics_jsonl``.  They are kept
outside the SHyRe reconstruction path so individual measurement modules can be
switched on for ablations without changing model behavior.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable

import networkx as nx


def retain_projection_edges(graph: nx.Graph, retention: float, seed: int) -> tuple[nx.Graph, dict]:
    """Keep an exact random fraction of projection edges while preserving nodes.

    ``retention=1`` returns a graph copy, deliberately retaining the original
    insertion order used by some legacy baselines.
    """
    if not 0.0 <= retention <= 1.0:
        raise ValueError("projection_retention must be in [0, 1]")
    original_edges = graph.number_of_edges()
    if retention == 1.0:
        retained = graph.copy()
    else:
        keep = int(round(retention * original_edges))
        selected = random.Random(seed).sample(list(graph.edges()), keep)
        retained = nx.Graph()
        retained.add_nodes_from(graph.nodes())
        retained.add_edges_from(selected)
    return retained, {
        "projection_edges_original": original_edges,
        "projection_edges_transmitted": retained.number_of_edges(),
        "projection_retention_requested": retention,
        "projection_retention_actual": retained.number_of_edges() / max(original_edges, 1),
    }


def hypergraph_storage_units(hyperedges: Iterable[Iterable[object]]) -> int:
    """Endpoint-incidence storage units; deliberately independent of serialization."""
    return sum(len(hyperedge) for hyperedge in hyperedges)


def graph_storage_units(graph: nx.Graph) -> int:
    """Two endpoint identifiers per undirected projected edge."""
    return 2 * graph.number_of_edges()


def storage_metrics(graphs: dict, reconstructed: Iterable[Iterable[object]]) -> dict:
    original = hypergraph_storage_units(graphs["simplicies_test"])
    received = graph_storage_units(graphs["G_test"])
    reconstruction = hypergraph_storage_units(reconstructed)
    return {
        "original_hypergraph_storage_units": original,
        "received_projection_storage_units": received,
        "reconstruction_storage_units": reconstruction,
        "received_projection_over_original": received / max(original, 1),
        "reconstruction_over_original": reconstruction / max(original, 1),
    }


def candidate_metrics(candidates: Iterable[tuple], truth: set[tuple]) -> dict:
    candidate_set = set(candidates)
    covered = candidate_set & truth
    return {
        "candidate_count": len(candidate_set),
        "candidate_true_count": len(covered),
        "truth_count": len(truth),
        "candidate_recall": len(covered) / max(len(truth), 1),
        "candidate_precision": len(covered) / max(len(candidate_set), 1),
    }


def candidate_diagnostics(candidates: list[tuple], truth: set[tuple],
                          max_candidate_count: int, labels=None) -> dict:
    """Report candidate coverage separately for maximal and nested cliques."""
    max_candidates = candidates[:max_candidate_count]
    nested_candidates = candidates[max_candidate_count:]
    result = candidate_metrics(candidates, truth)
    result["by_type"] = {
        "max_clique": candidate_metrics(max_candidates, truth),
        "nested_clique": candidate_metrics(nested_candidates, truth),
    }
    if labels is not None:
        labels = list(labels)

        def label_stats(group_labels):
            positives = sum(int(value > 0.5) for value in group_labels)
            count = len(group_labels)
            return {
                "sample_count": count,
                "positive_count": positives,
                "negative_count": count - positives,
                "positive_rate": positives / max(count, 1),
                "empty": count == 0,
                "single_class": count > 0 and (positives == 0 or positives == count),
            }

        result["label_distribution"] = {
            "all": label_stats(labels),
            "max_clique": label_stats(labels[:max_candidate_count]),
            "nested_clique": label_stats(labels[max_candidate_count:]),
        }
    return result


def classifier_diagnostics(predictions, candidates: list[tuple], truth: set[tuple],
                           max_candidate_count: int) -> dict:
    """Measure how much candidate coverage survives classifier selection."""
    predictions = list(predictions)

    def group_stats(group_predictions, group_candidates):
        candidate_set = set(group_candidates)
        true_candidates = candidate_set & truth
        selected = {
            clique for prediction, clique in zip(group_predictions, group_candidates)
            if prediction > 0.5
        }
        selected_true = selected & truth
        return {
            "candidate_count": len(candidate_set),
            "candidate_true_count": len(true_candidates),
            "selected_count": len(selected),
            "selected_true_count": len(selected_true),
            "selection_precision": len(selected_true) / max(len(selected), 1),
            "selection_recall_within_candidates": len(selected_true) / max(len(true_candidates), 1),
            "end_to_end_recall": len(selected_true) / max(len(truth), 1),
        }

    return {
        "all": group_stats(predictions, candidates),
        "max_clique": group_stats(
            predictions[:max_candidate_count], candidates[:max_candidate_count]),
        "nested_clique": group_stats(
            predictions[max_candidate_count:], candidates[max_candidate_count:]),
    }


def runtime_metrics(start_time: float) -> dict:
    # Linux ru_maxrss is KiB. Import locally to retain Windows compatibility.
    try:
        import resource
        peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        peak_rss_mib = None
    return {"wall_time_seconds": perf_counter() - start_time, "peak_rss_mib": peak_rss_mib}


def write_metrics_jsonl(path: str, payload: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
