"""Optional 16-QAM received-graph channel for SHyRe test projections.

The module deliberately has no dependency on SHyRe training: it transforms a
clean projected test graph into a hard received graph, while train.txt and all
candidate/feature/classifier logic remain untouched.
"""
from __future__ import annotations

import math
import random
from itertools import chain, combinations

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


def received_graph(clean_graph: nx.Graph, snr_db: float, fading: str = "awgn", threshold: float = 0.5,
                   seed: int = 123, max_background_pairs: int = 10000) -> nx.Graph:
    """Transmit projection-edge presence through normalized 16-QAM.

    Every clean edge is transmitted; a reproducible sample of non-edges models
    false positives without quadratic work on large SHyRe datasets.  At high
    SNR the hard graph converges to the clean projection.
    """
    rng = random.Random(seed)
    nodes = list(clean_graph.nodes())
    clean = {tuple(sorted(e)) for e in clean_graph.edges()}
    # Do not materialize every O(|V|^2) non-edge: this can exhaust memory on
    # large projections before any SHyRe reconstruction work begins.
    possible_pairs = len(nodes) * (len(nodes) - 1) // 2
    nonedge_count = max(0, possible_pairs - len(clean))
    if max_background_pairs is None or max_background_pairs >= nonedge_count:
        background_pairs = (pair for pair in combinations(nodes, 2) if pair not in clean)
    else:
        target = min(max_background_pairs, nonedge_count)
        sampled = set()
        attempts = 0
        max_attempts = max(target * 40, 1000)
        while len(sampled) < target and attempts < max_attempts:
            u, v = rng.sample(nodes, 2)
            pair = (u, v) if u < v else (v, u)
            if pair not in clean:
                sampled.add(pair)
            attempts += 1
        if len(sampled) < target:
            for pair in combinations(nodes, 2):
                if pair not in clean:
                    sampled.add(pair)
                    if len(sampled) == target:
                        break
        background_pairs = sampled
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
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, _llr(equalized, n0 / max(abs(h) ** 2, 1e-12))))))
        if probability >= threshold:
            received.add_edge(*edge)
    return received
