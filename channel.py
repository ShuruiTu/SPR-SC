"""Optional 16-QAM received-graph channel for SHyRe test projections.

The module deliberately has no dependency on SHyRe training: it transforms a
clean projected test graph into a hard received graph, while train.txt and all
candidate/feature/classifier logic remain untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import random
from itertools import chain

import networkx as nx


_LEVEL = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
_CONSTELLATION = [(complex(_LEVEL[(b[0], b[1])], _LEVEL[(b[2], b[3])]) / math.sqrt(10.0), b)
                  for b in ((a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1))]

Edge = Tuple[object, object]


@dataclass(frozen=True)
class ChannelObservation:
    graph: nx.Graph
    reliability: Dict[Edge, float]
    metadata: Dict[str, Any]


def _logsumexp(values):
    pivot = max(values)
    return pivot + math.log(sum(math.exp(v - pivot) for v in values))


def _llr(y: complex, variance: float) -> float:
    metric = lambda x: -abs(y - x) ** 2 / max(variance, 1e-12)
    one = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 1]
    zero = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 0]
    return _logsumexp(one) - _logsumexp(zero)


def _rayleigh_csi_llr(y: complex, h_estimate: complex, n0: float,
                      error_variance: float, statistical_only: bool = False) -> float:
    """Return a 16-QAM bit LLR after marginalizing Rayleigh CSI uncertainty."""
    if statistical_only:
        rho = 0.0
        conditional_variance = 1.0
    else:
        rho = 1.0 / (1.0 + error_variance)
        conditional_variance = error_variance / (1.0 + error_variance)

    def metric(x):
        variance = max(n0 + conditional_variance * abs(x) ** 2, 1e-12)
        mean = rho * h_estimate * x
        return -abs(y - mean) ** 2 / variance - math.log(variance)

    one = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 1]
    zero = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 0]
    return _logsumexp(one) - _logsumexp(zero)


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def simulate_channel(clean_graph: nx.Graph, snr_db: float, fading: str = "awgn", tau_e: float = 0.58,
                     seed: int = 123, max_background_pairs: int = 10000, beta: float = 1.0,
                     temperature: float = 1.0, edge_prior: float | None = None,
                     rayleigh_csi_mode: str = "perfect",
                     rayleigh_csi_error_variance: float = 0.1) -> ChannelObservation:
    """Return a hard received graph and posterior reliability per transmitted pair."""
    rng = random.Random(seed)
    estimation_rng = random.Random(seed ^ 0x5DEECE66D)
    nodes = list(clean_graph.nodes())
    clean = {tuple(sorted(edge)) for edge in clean_graph.edges()}
    possible_pairs = len(nodes) * (len(nodes) - 1) // 2
    nonedge_count = max(0, possible_pairs - len(clean))
    if edge_prior is None:
        edge_prior = len(clean) / max(possible_pairs, 1)
    edge_prior = min(max(edge_prior, 1e-4), 1.0 - 1e-4)
    prior_logit = math.log(edge_prior / (1.0 - edge_prior))
    target = min(max(0, int(max_background_pairs or 0)), nonedge_count)
    background_pairs = set(); attempts = 0
    max_attempts = max(100, target * 30)
    while len(background_pairs) < target and attempts < max_attempts and len(nodes) >= 2:
        u, v = rng.sample(nodes, 2)
        edge = (u, v) if u < v else (v, u)
        if edge not in clean:
            background_pairs.add(edge)
        attempts += 1

    n0 = 10.0 ** (-snr_db / 10.0)
    received = nx.Graph(); received.add_nodes_from(nodes)
    reliability: Dict[Edge, float] = {}
    for edge in chain(clean, background_pairs):
        bit = int(edge in clean)
        bits = (bit, rng.randrange(2), rng.randrange(2), rng.randrange(2))
        x = next(point for point, label in _CONSTELLATION if label == bits)
        h = (complex(rng.gauss(0, 1 / math.sqrt(2)), rng.gauss(0, 1 / math.sqrt(2)))
             if fading.lower() == "rayleigh" else 1.0 + 0.0j)
        noise = complex(rng.gauss(0, math.sqrt(n0 / 2)), rng.gauss(0, math.sqrt(n0 / 2)))
        y = h * x + noise
        if fading.lower() != "rayleigh" or rayleigh_csi_mode == "perfect":
            equalized = y / h if abs(h) > 1e-10 else y
            sigma_eff = n0 / max(abs(h) ** 2, 1e-12)
            llr = _llr(equalized, sigma_eff)
        elif rayleigh_csi_mode == "estimated":
            estimation_noise = complex(
                estimation_rng.gauss(0, math.sqrt(rayleigh_csi_error_variance / 2)),
                estimation_rng.gauss(0, math.sqrt(rayleigh_csi_error_variance / 2)))
            llr = _rayleigh_csi_llr(
                y, h + estimation_noise, n0, rayleigh_csi_error_variance)
        elif rayleigh_csi_mode == "statistical":
            llr = _rayleigh_csi_llr(y, 0.0j, n0, 0.0, statistical_only=True)
        else:
            raise ValueError('rayleigh CSI mode must be perfect, estimated, or statistical')
        posterior = _sigmoid((prior_logit + beta * llr) / max(temperature, 1e-4))
        reliability[edge] = posterior
        if posterior >= tau_e:
            received.add_edge(*edge)
    if {tuple(sorted(edge)) for edge in received.edges()} == clean:
        received = clean_graph.copy()
    return ChannelObservation(received, reliability, {
        "channel_type": fading.lower(),
        "channel_snr_db": snr_db,
        "channel_tau_e": tau_e,
        "channel_beta": beta,
        "channel_temperature": temperature,
        "channel_csi": rayleigh_csi_mode if fading.lower() == "rayleigh" else "not_applicable",
        "rayleigh_csi_error_variance": (rayleigh_csi_error_variance
                                          if fading.lower() == "rayleigh" and rayleigh_csi_mode == "estimated"
                                          else 0.0),
        "channel_edge_prior": edge_prior,
        "channel_input_edges": len(clean),
        "background_pairs_sampled": len(background_pairs),
        "channel_symbol_count": len(clean) + len(background_pairs),
        "channel_symbol_bits": 4 * (len(clean) + len(background_pairs)),
        "received_projection_edges": received.number_of_edges(),
    })


def received_graph(clean_graph: nx.Graph, snr_db: float, fading: str = "awgn", tau_e: float = 0.58,
                   seed: int = 123, max_background_pairs: int = 10000, beta: float = 1.0,
                   temperature: float = 1.0, edge_prior: float | None = None,
                   rayleigh_csi_mode: str = "perfect",
                   rayleigh_csi_error_variance: float = 0.1) -> nx.Graph:
    """Backward-compatible hard-graph adapter."""
    return simulate_channel(clean_graph, snr_db, fading, tau_e, seed, max_background_pairs,
                            beta, temperature, edge_prior, rayleigh_csi_mode,
                            rayleigh_csi_error_variance).graph


def clean_reliability(clean_graph: nx.Graph) -> Dict[Edge, float]:
    return {tuple(sorted(edge)): 1.0 for edge in clean_graph.edges()}


def posterior_candidate_graph(hard_graph: nx.Graph, reliability: Dict[Edge, float],
                              threshold: float, max_extra_edges: int) -> tuple[nx.Graph, dict]:
    """Augment only the candidate-search graph with high-posterior soft edges.

    The hard received graph remains untouched and continues to drive structural
    features and baselines.  A deterministic posterior ranking plus an explicit
    cap prevents low thresholds from creating an intractably dense graph.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError('candidate posterior threshold must be in (0, 1)')
    if max_extra_edges < 0:
        raise ValueError('candidate max extra edges must be non-negative')
    candidate_graph = hard_graph.copy()
    hard_edges = {tuple(sorted(edge)) for edge in hard_graph.edges()}
    eligible = [
        (posterior, edge) for edge, posterior in reliability.items()
        if edge not in hard_edges and posterior >= threshold
    ]
    eligible.sort(key=lambda item: (-item[0], item[1]))
    selected = eligible[:max_extra_edges]
    candidate_graph.add_edges_from(edge for _, edge in selected)
    return candidate_graph, {
        'candidate_posterior_threshold': threshold,
        'candidate_soft_edges_eligible': len(eligible),
        'candidate_soft_edges_added': len(selected),
        'candidate_graph_edges': candidate_graph.number_of_edges(),
    }
