#!/usr/bin/env python3
"""Import HYPER knowledge-hyperedge datasets into SHyRe's integer format.

The source format stores a relation label in column zero.  SPR-SC represents
only the participating entities/qualifiers as a hyperedge, so that column is
deliberately removed.  ``valid.txt`` is merged into the training split, which
keeps the supplied test split strictly held out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {
    "fb_auto": "FB-AUTO",
    "jf17k": "JF17K",
    "m_fb15k": "M-FB15K",
    "wikipeople": "Wikipeople",
}


def read_edges(paths: list[Path]) -> list[tuple[str, ...]]:
    edges, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                edge = tuple(sorted(set(tokens[1:])))  # remove relation column
                if len(edge) < 2 or edge in seen:
                    continue
                seen.add(edge)
                edges.append(edge)
    return edges


def write_edges(path: Path, edges: list[tuple[int, ...]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(" ".join(map(str, edge)) + "\n")


def digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                value.update(block)
    return value.hexdigest()


def import_dataset(source_root: Path, destination_root: Path, key: str) -> dict:
    source_name = DATASETS[key]
    source = source_root / source_name
    train_sources = [source / "train.txt", source / "valid.txt"]
    test_sources = [source / "test.txt"]
    missing = [str(path) for path in train_sources + test_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))

    train_raw, test_raw = read_edges(train_sources), read_edges(test_sources)
    vocabulary = sorted({node for edge in train_raw + test_raw for node in edge})
    node_to_id = {node: index for index, node in enumerate(vocabulary)}
    train = [tuple(node_to_id[node] for node in edge) for edge in train_raw]
    test = [tuple(node_to_id[node] for node in edge) for edge in test_raw]
    output = destination_root / key
    output.mkdir(parents=True, exist_ok=True)
    write_edges(output / "train.txt", train)
    write_edges(output / "test.txt", test)
    metadata = {
        "dataset": key,
        "source_dataset": source_name,
        "source_root": str(source),
        "conversion": "drop relation column; deduplicate within split; merge valid into train; global integer reindex",
        "source_sha256": digest(train_sources + test_sources),
        "nodes": len(vocabulary),
        "train_hyperedges": len(train),
        "test_hyperedges": len(test),
        "train_mean_size": sum(map(len, train)) / len(train),
        "test_mean_size": sum(map(len, test)) / len(test),
        "max_hyperedge_size": max(map(len, train + test)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/tushurui/SPRSC_20260828/handoff_20260828_sprsc/06_REPRODUCTION_CODE/external/HYPER-data/hypergraph_dataset"))
    parser.add_argument("--destination-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=tuple(DATASETS))
    args = parser.parse_args()
    for key in args.datasets:
        metadata = import_dataset(args.source_root, args.destination_root, key)
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
