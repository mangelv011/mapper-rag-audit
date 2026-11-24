"""Mapper graph of a set of chunk embeddings, built with KeplerMapper.

Mapper in four steps:
1. Lens: map each embedding to one number (default: its coordinate on the first principal axis).
2. Cover: cut the range of the lens into `n_intervals` intervals that overlap by `overlap`.
3. Cluster: in each interval, cluster the chunks whose lens value falls in it. Each cluster is
   a node, holding a set of chunk ids.
4. Nerve: join two nodes with an edge when they share at least one chunk.

Fitting and building are separate, so the graph after an ingestion can reuse what was fitted on
the corpus before it. `fit_mapper` learns the lens and the interval bounds; `build_mapper_graph`
only applies them. Refitting on the new corpus would move every interval a little and reshuffle
the whole graph, and the comparison of the two graphs (#5) would then see changes everywhere.
A lens value outside the fitted range is clipped to it, so the point lands in an end interval.

Node ids are `i<interval>r<row>`: the interval index, counting all intervals (empty ones
included), and the first row of `embeddings` in the node. The interval is zero-padded to the
same width for every id of a graph (at least two digits), so sorting the ids sorts by interval;
`node_interval` reads it back. A node with the same chunks in two graphs built from the same
parameters has the same id, as long as the rows of the old chunks come first in the same order
(new chunks appended at the end).

The id names a set of chunks, not a region of the graph. Average linkage re-runs on the whole
interval after an ingestion, so a new chunk can change how nearby clusters merge: a node that
holds no new chunk can still gain or lose chunks, and if it loses its first row its id changes
too. On SciFact, 20 new chunks change about 20 such nodes (README, "The cost").
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from mapper_rag_audit.data import Corpus

LensName = Literal["pca", "centroid"]
Linkage = Literal["average", "complete", "single"]


@dataclass(frozen=True)
class MapperGraph:
    nodes: dict[str, frozenset[str]]  # node id -> chunk ids
    edges: frozenset[frozenset[str]]  # each edge is a set of two node ids


@dataclass(frozen=True)
class MapperConfig:
    """Mapper settings. The defaults are the ones used for SciFact; see the README.

    - `lens`: "pca" is the coordinate on the first principal axis of the embeddings. "centroid"
      is the cosine distance to the mean embedding, an eccentricity: small in the dense middle
      of the corpus, large for outlying topics.
    - `n_intervals`, `overlap`: the cover. `overlap` is the fraction of an interval shared with
      its neighbour, strictly between 0 and 1.
    - `linkage`, `threshold`: hierarchical clustering with cosine distance inside each interval,
      cut where merging would join groups farther apart than `threshold`. Every chunk lands in
      at least one node. "average" measures the mean distance between two groups; "single"
      the closest pair, which chains text embeddings into a few giant clusters (see README).
    """

    lens: LensName = "pca"
    n_intervals: int = 15
    overlap: float = 0.5
    linkage: Linkage = "average"
    threshold: float = 0.27

    def __post_init__(self) -> None:
        if self.lens not in ("pca", "centroid"):
            raise ValueError(f"unknown lens {self.lens!r}")
        if self.n_intervals < 1:
            raise ValueError(f"n_intervals must be at least 1, got {self.n_intervals}")
        if not 0 < self.overlap < 1:
            raise ValueError(f"overlap must be strictly between 0 and 1, got {self.overlap}")
        if self.linkage not in ("average", "complete", "single"):
            raise ValueError(f"unknown linkage {self.linkage!r}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")


class _Clusterer(ClusterMixin, BaseEstimator):
    """Hierarchical clustering cut at a distance threshold, that also accepts a single point.

    `min_samples=1` tells KeplerMapper to cluster intervals holding one point instead of
    dropping them, so a lone new chunk at the edge of the range still gets a node.
    """

    def __init__(self, linkage: str = "average", threshold: float = 0.27, min_samples: int = 1):
        self.linkage = linkage
        self.threshold = threshold
        self.min_samples = min_samples

    def fit_predict(self, X, y=None):
        if len(X) == 1:
            return np.zeros(1, dtype=int)
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            linkage=self.linkage,
            metric="cosine",
        ).fit_predict(X)


@dataclass(frozen=True, eq=False)
class MapperParams:
    """What `fit_mapper` learned: the lens as an affine map, and the range of the cover.

    Both lenses have the form `lens(x) = offset + sign * (x - center) @ axis`:
    - pca: center = mean embedding, axis = first principal axis, sign = 1, offset = 0.
    - centroid: center = 0, axis = c / |c| with c the mean embedding, sign = -1, offset = 1,
      which is 1 - cos(x, c) for unit-norm x.
    `low` and `high` are the range of the lens on the fitted embeddings.

    Compared by identity (`eq=False`): the generated `==` would compare numpy arrays and raise.
    """

    config: MapperConfig
    center: np.ndarray
    axis: np.ndarray
    sign: float
    offset: float
    low: float
    high: float

    def lens(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        return self.offset + self.sign * ((x - self.center) @ self.axis)


def fit_mapper(embeddings: np.ndarray, config: MapperConfig = MapperConfig()) -> MapperParams:
    """Fit the lens and the cover range on `embeddings` (the corpus before an ingestion)."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got shape {x.shape}")
    if config.lens == "pca":
        pca = PCA(n_components=1, svd_solver="full").fit(x)
        center, axis, sign, offset = pca.mean_, pca.components_[0], 1.0, 0.0
    else:
        mean = x.mean(axis=0)
        center, axis, sign, offset = np.zeros_like(mean), mean / np.linalg.norm(mean), -1.0, 1.0
    params = MapperParams(config, center, axis, sign, offset, 0.0, 0.0)
    values = params.lens(x)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("the lens is constant on these embeddings")
    return MapperParams(config, center, axis, sign, offset, low, high)


def _cover(params: MapperParams):
    import kmapper as km

    cover = km.Cover(
        n_cubes=params.config.n_intervals,
        perc_overlap=params.config.overlap,
        limits=np.array([[params.low, params.high]]),
    )
    cover.fit(np.array([[0, params.low], [1, params.high]]))  # column 0 is KeplerMapper's index
    return cover


def interval_bounds(params: MapperParams) -> list[tuple[float, float]]:
    """The closed lens intervals of the cover; interval i holds the nodes `i<i>r...`."""
    cover = _cover(params)
    r = float(cover.radius_[0])
    return [(float(c[0]) - r, float(c[0]) + r) for c in cover.centers_]


def _kmapper_graph(embeddings: np.ndarray, params: MapperParams) -> tuple[dict, np.ndarray]:
    """Run KeplerMapper. Returns its graph dict, with renamed node ids, and the clipped lens."""
    import kmapper as km

    cfg = params.config
    x = np.asarray(embeddings, dtype=np.float64)
    lens = np.clip(params.lens(x), params.low, params.high).reshape(-1, 1)
    cover = _cover(params)
    clusterer = _Clusterer(cfg.linkage, cfg.threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # its "limits" check misfires on clipping
        graph = km.KeplerMapper(verbose=0).map(lens, x, clusterer=clusterer, cover=cover)

    # KeplerMapper numbers only the non-empty intervals ("cube<k>"); map k to the interval index.
    indexed = np.c_[np.arange(len(lens)), lens]
    non_empty = [i for i, c in enumerate(cover.centers_) if len(cover.transform_single(indexed, c))]
    # Name a cluster by its first row, not by the clusterer's label or a running count: those
    # depend on the merge tree and on the other clusters, so adding one point could rename
    # clusters it never touches. Clusters of one interval are disjoint, so the name is unique.
    width = max(2, len(str(cfg.n_intervals - 1)))
    rename = {}
    for old, rows in graph["nodes"].items():
        cube = int(old.removeprefix("cube").split("_cluster")[0])
        rename[old] = f"i{non_empty[cube]:0{width}d}r{min(rows)}"
    graph["nodes"] = {rename[k]: v for k, v in graph["nodes"].items()}
    graph["links"] = {rename[k]: [rename[t] for t in v] for k, v in graph["links"].items()}
    graph["simplices"] = [[rename[n] for n in s] for s in graph["simplices"]]
    return graph, lens[:, 0]


def node_interval(node_id: str) -> int:
    """The interval index of a node, read from its id `i<interval>r<row>`."""
    return int(node_id[1:].split("r", 1)[0])


def build_mapper_graph(
    embeddings: np.ndarray, chunk_ids: Sequence[str], params: MapperParams
) -> MapperGraph:
    """Mapper graph of `embeddings` (one row per id in `chunk_ids`) with fitted `params`."""
    if len(chunk_ids) != len(embeddings):
        raise ValueError(f"{len(chunk_ids)} chunk ids for {len(embeddings)} embeddings")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate chunk ids")
    graph, _ = _kmapper_graph(embeddings, params)
    return _to_mapper_graph(graph, chunk_ids)


def _to_mapper_graph(graph: dict, chunk_ids: Sequence[str]) -> MapperGraph:
    nodes = {n: frozenset(chunk_ids[i] for i in rows) for n, rows in graph["nodes"].items()}
    edges = frozenset(
        frozenset((a, b)) for a, targets in graph["links"].items() for b in targets if a != b
    )
    return MapperGraph(nodes, edges)


# Adapters with the signatures that `graph_diff.MapperSelector` takes as `fit` and `build`.
def fit_corpus(corpus: Corpus, config: MapperConfig = MapperConfig()) -> MapperParams:
    return fit_mapper(corpus.embeddings, config)


def build_corpus_graph(corpus: Corpus, params: MapperParams) -> MapperGraph:
    return build_mapper_graph(corpus.embeddings, [c.chunk_id for c in corpus.chunks], params)


# --------------------------------------------------------------------------------------------
# Reading the graph: summary numbers, node labels and the interactive HTML view
# --------------------------------------------------------------------------------------------
def connected_components(graph: MapperGraph) -> list[set[str]]:
    """Connected components, largest first (ties by smallest node id)."""
    neighbours: dict[str, set[str]] = {n: set() for n in graph.nodes}
    for edge in graph.edges:
        a, b = tuple(edge)
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen: set[str] = set()
    components = []
    for start in sorted(graph.nodes):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n not in component:
                component.add(n)
                stack.extend(neighbours[n] - component)
        seen |= component
        components.append(component)
    return sorted(components, key=lambda c: (-len(c), min(c)))


def summarize(graph: MapperGraph, n_chunks: int) -> dict[str, float]:
    """Counts reported in the README."""
    sizes = sorted(len(c) for c in graph.nodes.values())
    covered = set().union(*graph.nodes.values()) if graph.nodes else set()
    components = connected_components(graph)
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "components": len(components),
        "largest_component_nodes": len(components[0]) if components else 0,
        "singleton_nodes": sum(1 for s in sizes if s == 1),
        "median_node_size": float(np.median(sizes)) if sizes else 0.0,
        "max_node_size": sizes[-1] if sizes else 0,
        "chunks_in_some_node": len(covered) / n_chunks if n_chunks else 0.0,
    }


def node_labels(graph: MapperGraph, texts: dict[str, str], n_terms: int = 3) -> dict[str, str]:
    """A few representative words per node: the top terms of its chunks' mean TF-IDF vector.

    TF-IDF is fitted on all chunks in `texts`, so a word common to the whole corpus scores low.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(texts)
    row = {c: i for i, c in enumerate(ids)}
    vectorizer = TfidfVectorizer(
        stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b", min_df=2
    )
    tfidf = vectorizer.fit_transform([texts[c] for c in ids])
    vocab = vectorizer.get_feature_names_out()
    labels = {}
    for node, chunks in graph.nodes.items():
        mean = np.asarray(tfidf[[row[c] for c in sorted(chunks)]].mean(axis=0)).ravel()
        top = np.argsort(-mean, kind="stable")[:n_terms]
        labels[node] = ", ".join(vocab[i] for i in top if mean[i] > 0)
    return labels


# Added to KeplerMapper's page. Node names stay plain ids, because its script puts them in CSS
# selectors ("#node-" + name) and a label with spaces or commas would make every hover fail.
# The labels travel in a separate map: each node gets an SVG <title> (the browser shows it as a
# tooltip right over the node), and the side panel shows the label under "Cluster Details".
_LABEL_SCRIPT = """
<script>
(function () {
  const LABELS = %s;
  function addTitles() {
    for (const [id, label] of Object.entries(LABELS)) {
      const node = document.getElementById("node-" + id);
      if (node && !node.querySelector(":scope > title")) {
        const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
        t.textContent = id + ": " + label;
        node.prepend(t);
      }
    }
  }
  function addPanelLabel() {
    const idSpan = document.querySelector("#tooltip_content_focus_node .node_id");
    if (!idSpan) return;
    const p = document.createElement("p");
    p.className = "node_label";
    p.style.cssText = "font-size:1.2em;font-weight:bold;margin:0.3em 0";
    idSpan.closest("h2").after(p);
    new MutationObserver(() => { p.textContent = LABELS[idSpan.textContent] || ""; })
      .observe(idSpan, { childList: true, characterData: true, subtree: true });
  }
  window.addEventListener("load", () => { addTitles(); addPanelLabel(); setTimeout(addTitles, 1000); });
})();
</script>
"""


def save_html(
    embeddings: np.ndarray,
    chunk_ids: Sequence[str],
    params: MapperParams,
    labels: dict[str, str],
    tooltips: Sequence[str],
    path,
    title: str = "Mapper graph",
) -> None:
    """Interactive KeplerMapper view, with each node's label shown when the mouse is over it."""
    import json

    import kmapper as km

    graph, lens = _kmapper_graph(embeddings, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        html = km.KeplerMapper(verbose=0).visualize(
            graph,
            save_file=False,
            title=title,
            custom_tooltips=np.array(list(tooltips)),
            color_values=lens,
            color_function_name="lens",
            include_searchbar=True,
        )
    node_labels_json = json.dumps({n: labels.get(n, "") for n in graph["nodes"]})
    script = _LABEL_SCRIPT % node_labels_json.replace("</", "<\\/")
    html = html.replace("</body>", script + "</body>", 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
