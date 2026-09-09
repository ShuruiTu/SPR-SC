"""Optional 16-QAM received-graph channel for SHyRe test projections.

The module deliberately has no dependency on SHyRe training: it transforms a
clean projected test graph into a hard received graph, while train.txt and all
candidate/feature/classifier logic remain untouched.
"""
from __future__ import annotations

import math
import random
from itertools import chain

import networkx as nx


_LEVEL = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
_CONSTELLATION = [(complex(_LEVEL[(b[0], b[1])], _LEVEL[(b[2], b[3])]) / math.sqrt(10.0), b)
                  for b in ((a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1))]


def _logsumexp(values):
    pivot = max(values)
    return pivot + math.log(sum(math.exp(v - pivot) for v in values))


def _llr(y: complex, variance: float) -> float:
    metric = lambda x: -abs(y - x) ** 2 / max(variance, 1e-12)
    one = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 1]
    zero = [metric(x) for x, bits in _CONSTELLATION if bits[0] == 0]
    return _logsumexp(one) - _logsumexp(zero)


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def received_graph(clean_graph: nx.Graph, snr_db: float, fading: str = "awgn", tau_e: float = 0.58,
                   seed: int = 123, max_background_pairs: int = 10000, beta: float = 1.0,
                   temperature: float = 1.0, edge_prior: float | None = None) -> nx.Graph:
    """Transmit projection edges through 16-QAM and return a hard received graph.

    This is the SHyRe adapter for the original SPR-SC posterior decision rule:
    the exact QAM LLR is combined with a sparse edge prior, beta scaling and
    temperature calibration before thresholding at ``tau_e``.  Only a bounded
    random sample of non-edges is transmitted; it never enumerates all pairs.
    """
    rng = random.Random(seed)
    nodes = list(clean_graph.nodes())
    clean = {tuple(sorted(edge)) for edge in clean_graph.edges()}
    possible_pairs = len(nodes) * (len(nodes) - 1) // 2
    nonedge_count = max(0, possible_pairs - len(clean))
    if edge_prior is None:
        edge_prior = len(clean) / max(possible_pairs, 1)
    edge_prior = min(max(edge_prior, 1e-4), 1.0 - 1e-4)
    prior_logit = math.log(edge_prior / (1.0 - edge_prior))

    # ``None`` deliberately means no sampled background pairs: all-pairs
    # transmission is not permitted in this scalable received-graph adapter.
    target = min(max(0, int(max_background_pairs or 0)), nonedge_count)
    background_pairs = set()
    attempts = 0
    max_attempts = max(100, target * 30)
    while len(background_pairs) < target and attempts < max_attempts and len(nodes) >= 2:
        u, v = rng.sample(nodes, 2)
        edge = (u, v) if u < v else (v, u)
        if edge not in clean:
            background_pairs.add(edge)
        attempts += 1

    n0 = 10.0 ** (-snr_db / 10.0)
    received = nx.Graph(); received.add_nodes_from(nodes)
    for edge in chain(clean, background_pairs):
        bit = int(edge in clean)
        bits = (bit, rng.randrange(2), rng.randrange(2), rng.randrange(2))
        x = next(point for point, label in _CONSTELLATION if label == bits)
        if fading.lower() == "rayleigh":
            h = complex(rng.gauss(0, 1 / math.sqrt(2)), rng.gauss(0, 1 / math.sqrt(2)))
        else:
            h = 1.0 + 0.0j
        noise = complex(rng.gauss(0, math.sqrt(n0 / 2)), rng.gauss(0, math.sqrt(n0 / 2)))
        y = h * x + noise
        equalized = y / h if abs(h) > 1e-10 else y
        sigma_eff = n0 / max(abs(h) ** 2, 1e-12)
        posterior = _sigmoid((prior_logit + beta * _llr(equalized, sigma_eff)) / max(temperature, 1e-4))
        if posterior >= tau_e:
            received.add_edge(*edge)
    received_edges = {tuple(sorted(edge)) for edge in received.edges()}
    if received_edges == clean:
        # Preserve the clean graph's insertion order for order-sensitive baselines.
        return clean_graph.copy()
    return received
