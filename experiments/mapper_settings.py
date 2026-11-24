"""Compare Mapper clustering settings on SciFact: the table behind the defaults in the README (#4).

Usage (from the repository root, after experiments/prepare_data.py):

    python experiments/mapper_settings.py

It first prints the median cosine distance from a chunk to its nearest neighbour and between
two chunks drawn at random. For each setting it then builds the graph of the corpus and prints
its size, how many chunks sit only in one-chunk nodes, and the largest node. It then simulates an ingestion (20 new chunks,
each a slightly perturbed copy of a random existing chunk, seed 0), rebuilds the graph with the
same fitted parameters, and counts the nodes holding a new chunk ("touched") and the nodes that
hold no new chunk yet whose chunks differ from every node of the graph before ("side effects").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapper_rag_audit.data import load_corpus
from mapper_rag_audit.mapper import MapperConfig, build_mapper_graph, fit_mapper, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [
    MapperConfig(n_intervals=15, overlap=0.5, linkage="single", threshold=t)
    for t in (0.14, 0.16, 0.18, 0.20)
] + [
    MapperConfig(n_intervals=n, overlap=o, linkage="average", threshold=t)
    for t in (0.25, 0.27, 0.30, 0.35)
    for n, o in ((10, 0.5), (15, 0.3), (15, 0.5), (20, 0.5))
]
N_NEW, NOISE, SEED = 20, 0.01, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    x = corpus.embeddings
    ids = [c.chunk_id for c in corpus.chunks]
    rng = np.random.default_rng(SEED)
    new = x[rng.choice(len(x), N_NEW, replace=False)]
    new = new + rng.normal(scale=NOISE, size=new.shape)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    x_after = np.vstack([x, new])
    new_ids = [f"new#{i}" for i in range(N_NEW)]

    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    nearest = 1 - sims.max(axis=1)
    pairs = rng.choice(len(x), size=(20_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    random_pairs = 1 - np.einsum("ij,ij->i", x[pairs[:, 0]], x[pairs[:, 1]])
    print(f"median cosine distance to the nearest neighbour: {np.median(nearest):.2f}")
    print(f"median cosine distance between two random chunks: {np.median(random_pairs):.2f}")
    print()

    header = (
        "linkage  threshold intervals overlap  nodes edges components "
        "alone_chunks max_node touched side_effects"
    )
    print(header)
    for config in SETTINGS:
        params = fit_mapper(x, config)
        before = build_mapper_graph(x, ids, params)
        after = build_mapper_graph(x_after, ids + new_ids, params)
        s = summarize(before, len(ids))
        in_shared = set().union(*(c for c in before.nodes.values() if len(c) > 1))
        old_sets = set(before.nodes.values())
        touched = [c for c in after.nodes.values() if c & set(new_ids)]
        side = [c for c in after.nodes.values() if not c & set(new_ids) and c not in old_sets]
        print(
            f"{config.linkage:8s} {config.threshold:9.2f} {config.n_intervals:9d} "
            f"{config.overlap:7.1f} {s['nodes']:6d} {s['edges']:5d} {s['components']:10d} "
            f"{1 - len(in_shared) / len(ids):12.1%} {s['max_node_size']:8d} "
            f"{len(touched):7d} {len(side):12d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
