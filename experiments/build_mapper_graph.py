"""Build the Mapper graph of the SciFact corpus and save it with an interactive HTML view (#4).

Usage (from the repository root, after experiments/prepare_data.py):

    python experiments/build_mapper_graph.py
    python experiments/build_mapper_graph.py --threshold 0.3 --n-intervals 15   # try other settings

Writes to data/mapper/:
- graph.json: the settings, the summary counts, and each node with its chunk ids and label.
- graph.html: the interactive view. Open it in a browser; hover a node to see its id and label
  as a tooltip, and its chunk titles in the side panel. Colour is the mean lens value of the node.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mapper_rag_audit.data import load_corpus
from mapper_rag_audit.mapper import (
    MapperConfig,
    build_mapper_graph,
    fit_mapper,
    interval_bounds,
    node_labels,
    save_html,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    default = MapperConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--lens", choices=["pca", "centroid"], default=default.lens)
    parser.add_argument("--n-intervals", type=int, default=default.n_intervals)
    parser.add_argument("--overlap", type=float, default=default.overlap)
    parser.add_argument(
        "--linkage", choices=["average", "complete", "single"], default=default.linkage
    )
    parser.add_argument("--threshold", type=float, default=default.threshold)
    parser.add_argument("--no-html", action="store_true", help="skip the HTML view")
    args = parser.parse_args()

    config = MapperConfig(args.lens, args.n_intervals, args.overlap, args.linkage, args.threshold)
    corpus = load_corpus(args.data_dir)
    chunk_ids = [c.chunk_id for c in corpus.chunks]
    texts = {c.chunk_id: c.text for c in corpus.chunks}

    params = fit_mapper(corpus.embeddings, config)
    graph = build_mapper_graph(corpus.embeddings, chunk_ids, params)
    summary = summarize(graph, len(chunk_ids))
    labels = node_labels(graph, texts)

    out = args.data_dir / "mapper"
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "config": asdict(config),
        "lens_range": [params.low, params.high],
        "intervals": interval_bounds(params),
        "summary": summary,
        "nodes": {
            n: {"label": labels[n], "chunks": sorted(graph.nodes[n])} for n in sorted(graph.nodes)
        },
        "edges": sorted(sorted(e) for e in graph.edges),
    }
    with (out / "graph.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)

    print(f"config: {config}")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    sizes = sorted(((len(c), n) for n, c in graph.nodes.items()), reverse=True)
    print("largest nodes:")
    for size, n in sizes[:8]:
        print(f"  {n} ({size} chunks): {labels[n]}")

    if not args.no_html:
        titles = [c.text.split("\n", 1)[0][:120] for c in corpus.chunks]
        save_html(
            corpus.embeddings,
            chunk_ids,
            params,
            labels,
            np.array(titles),
            out / "graph.html",
            title=f"SciFact Mapper graph ({config.lens}, {config.n_intervals} intervals, "
            f"{config.overlap:.0%} overlap, {config.linkage} linkage at {config.threshold})",
        )
        print(f"wrote {out / 'graph.json'} and {out / 'graph.html'}")


if __name__ == "__main__":
    main()
