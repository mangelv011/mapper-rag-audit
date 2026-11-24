"""Mapper graph builder (#4), on small synthetic point clouds."""

import numpy as np
import pytest

from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus
from mapper_rag_audit.mapper import (
    MapperConfig,
    MapperGraph,
    build_corpus_graph,
    build_mapper_graph,
    connected_components,
    fit_corpus,
    fit_mapper,
    interval_bounds,
    node_interval,
    node_labels,
    save_html,
    summarize,
)

DIM = 8
CONFIG = MapperConfig(n_intervals=5, overlap=0.3, threshold=0.05)


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def blobs(bridge: bool, n_blob: int = 60, n_bridge: int = 40, seed: int = 0) -> np.ndarray:
    """Two tight blobs of unit vectors around e0 and e1, optionally joined by points on the arc."""
    rng = np.random.default_rng(seed)
    e0, e1 = np.eye(DIM)[0], np.eye(DIM)[1]
    parts = [
        e0 + rng.normal(scale=0.02, size=(n_blob, DIM)),
        e1 + rng.normal(scale=0.02, size=(n_blob, DIM)),
    ]
    if bridge:
        t = np.linspace(0, np.pi / 2, n_bridge)[:, None]
        parts.append(np.cos(t) * e0 + np.sin(t) * e1)
    return _unit(np.vstack(parts))


def ids(n: int) -> list[str]:
    return [f"c{i}" for i in range(n)]


def graph_of(x, config=CONFIG):
    return build_mapper_graph(x, ids(len(x)), fit_mapper(x, config))


def test_two_blobs_with_a_bridge_give_one_component_spanning_both_blobs():
    x = blobs(bridge=True)
    graph = graph_of(x)
    blob_a, blob_b = set(ids(len(x))[:60]), set(ids(len(x))[60:120])

    nodes_a = {n for n, c in graph.nodes.items() if c & blob_a}
    nodes_b = {n for n, c in graph.nodes.items() if c & blob_b}
    assert nodes_a and nodes_b and not nodes_a & nodes_b
    components = connected_components(graph)
    assert len(components) == 1
    assert nodes_a | nodes_b <= components[0]


def test_two_blobs_without_a_bridge_give_two_components():
    graph = graph_of(blobs(bridge=False))
    components = connected_components(graph)
    assert len(components) == 2
    chunks = [set().union(*(graph.nodes[n] for n in comp)) for comp in components]
    assert sorted(map(len, chunks)) == [60, 60]


def test_an_edge_joins_exactly_the_nodes_that_share_a_chunk():
    graph = graph_of(blobs(bridge=True))
    names = sorted(graph.nodes)
    expected = {
        frozenset((a, b))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if graph.nodes[a] & graph.nodes[b]
    }
    assert graph.edges == expected
    assert graph.edges, "the bridge should produce edges"


def test_every_chunk_lands_in_a_node():
    x = blobs(bridge=True)
    graph = graph_of(x)
    assert set().union(*graph.nodes.values()) == set(ids(len(x)))


def test_building_is_deterministic():
    x = blobs(bridge=True)
    assert graph_of(x) == graph_of(x)


def test_node_ids_name_the_interval_the_chunks_fall_in():
    # Lens values only near both ends: the middle intervals are empty, and the ids must still
    # count them, so KeplerMapper's own numbering (non-empty intervals only) is not used.
    x = blobs(bridge=False)
    params = fit_mapper(x, MapperConfig(n_intervals=10, overlap=0.2, threshold=0.05))
    graph = build_mapper_graph(x, ids(len(x)), params)
    bounds = interval_bounds(params)
    lens = dict(zip(ids(len(x)), params.lens(x)))
    used = set()
    for node, chunks in graph.nodes.items():
        i = node_interval(node)
        used.add(i)
        low, high = bounds[i]
        assert all(low <= lens[c] <= high for c in chunks), node
    assert max(used) == 9, "the far end must be interval 9 even with empty intervals between"


def test_cover_spans_the_fitted_range_with_the_requested_overlap():
    params = fit_mapper(blobs(bridge=True), MapperConfig(n_intervals=4, overlap=0.25))
    bounds = interval_bounds(params)
    assert len(bounds) == 4
    # KeplerMapper spaces the centres range/n apart, from low + range/2n; the end intervals
    # stick out a little past the range.
    step = (params.high - params.low) / 4
    centres = [(a + b) / 2 for a, b in bounds]
    assert centres == pytest.approx([params.low + step * (i + 0.5) for i in range(4)])
    assert bounds[0][0] <= params.low and bounds[-1][1] >= params.high
    length = bounds[0][1] - bounds[0][0]
    shared = bounds[0][1] - bounds[1][0]
    assert shared / length == pytest.approx(0.25)


def beyond_the_range() -> np.ndarray:
    """A unit vector along the PCA axis of `blobs`, past blob e0: off the fitted lens range."""
    return _unit(np.eye(DIM)[0] - np.eye(DIM)[1])


def test_a_point_beyond_the_fitted_range_goes_to_an_end_interval():
    x = blobs(bridge=True)
    params = fit_mapper(x, CONFIG)
    far = beyond_the_range()
    value = params.lens(far[None])[0]
    assert not params.low <= value <= params.high, "the test point must lie off the range"

    after = build_mapper_graph(np.vstack([x, far]), ids(len(x)) + ["new"], params)
    intervals = {node_interval(n) for n, c in after.nodes.items() if "new" in c}
    assert intervals <= {0, CONFIG.n_intervals - 1}


def test_refitting_on_the_new_corpus_would_move_the_cover():
    """Why fit and build are separate: refitting changes the range, so the intervals move."""
    x = blobs(bridge=True)
    refit = fit_mapper(np.vstack([x, beyond_the_range()]), CONFIG)
    moved = np.ravel(interval_bounds(refit)) - np.ravel(interval_bounds(fit_mapper(x, CONFIG)))
    assert np.abs(moved).min() > 0.01


def test_corpus_adapters_use_chunk_ids():
    x = blobs(bridge=True, n_blob=10, n_bridge=10)
    chunks = [Chunk(f"d{i}#0", f"d{i}", f"text {i}") for i in range(len(x))]
    corpus = Corpus(chunks, x.astype(np.float32), [], np.zeros((0, DIM), np.float32), {})
    graph = build_corpus_graph(corpus, fit_corpus(corpus, CONFIG))
    assert set().union(*graph.nodes.values()) == {c.chunk_id for c in chunks}


def test_rows_and_ids_must_match():
    x = blobs(bridge=True)
    params = fit_mapper(x, CONFIG)
    with pytest.raises(ValueError):
        build_mapper_graph(x, ids(len(x) - 1), params)
    with pytest.raises(ValueError):
        build_mapper_graph(x[:2], ["a", "a"], params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lens": "umap"},
        {"n_intervals": 0},
        {"overlap": 0},
        {"overlap": 1},
        {"threshold": 0},
        {"linkage": "ward"},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MapperConfig(**kwargs)


def test_summary_and_components_on_a_hand_made_graph():
    graph = MapperGraph(
        nodes={
            "a": frozenset({"1", "2"}),
            "b": frozenset({"2", "3"}),
            "c": frozenset({"4"}),
        },
        edges=frozenset({frozenset({"a", "b"})}),
    )
    assert connected_components(graph) == [{"a", "b"}, {"c"}]
    summary = summarize(graph, n_chunks=5)
    assert summary["nodes"] == 3
    assert summary["edges"] == 1
    assert summary["components"] == 2
    assert summary["singleton_nodes"] == 1
    assert summary["chunks_in_some_node"] == pytest.approx(4 / 5)


def test_node_labels_pick_the_distinctive_words_of_each_node():
    graph = MapperGraph(
        nodes={"a": frozenset({"1", "2"}), "b": frozenset({"3", "4"})}, edges=frozenset()
    )
    texts = {
        "1": "insulin glucose diabetes patients",
        "2": "insulin resistance diabetes study",
        "3": "tumour cancer cells study",
        "4": "tumour metastasis cancer patients",
    }
    labels = node_labels(graph, texts, n_terms=2)
    assert set(labels["a"].split(", ")) == {"insulin", "diabetes"}
    assert set(labels["b"].split(", ")) == {"tumour", "cancer"}


def test_html_view_keeps_plain_node_ids_and_carries_the_labels(tmp_path):
    """KeplerMapper's script builds CSS selectors from node names ("#node-" + name), so a
    name with spaces or commas makes every hover fail. Names must stay plain ids, and the
    labels travel in the separate map read by the added script."""
    import json
    import re

    x = blobs(bridge=True, n_blob=10, n_bridge=10)
    params = fit_mapper(x, CONFIG)
    graph = build_mapper_graph(x, ids(len(x)), params)
    labels = {n: f"dna, chromatin, {n}" for n in graph.nodes}
    path = tmp_path / "graph.html"
    save_html(x, ids(len(x)), params, labels, ids(len(x)), path)
    html = path.read_text(encoding="utf-8")

    data = json.loads(re.search(r"const graph = (\{.*?\});\n", html, re.DOTALL).group(1))
    names = {node["name"] for node in data["nodes"]}
    assert names == set(graph.nodes)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)
    assert all(node["tooltip"]["node_id"] in names for node in data["nodes"])

    embedded = json.loads(re.search(r"const LABELS = (\{.*?\});", html).group(1))
    assert embedded == labels


FAR_POINTS = {
    "inside the range": lambda: _unit(-np.eye(DIM)[0] - np.eye(DIM)[1] + 0.1),
    "beyond the range": beyond_the_range,
}


@pytest.mark.parametrize("where", FAR_POINTS)
@pytest.mark.parametrize("lens", ["pca", "centroid"])
def test_adding_a_far_point_with_the_same_params_leaves_other_nodes_unchanged(lens, where):
    """The graph diff (#5) needs this: only the region of the new chunk changes."""
    x = blobs(bridge=True)
    params = fit_mapper(x, MapperConfig(lens=lens, n_intervals=5, overlap=0.3, threshold=0.05))
    before = build_mapper_graph(x, ids(len(x)), params)

    far = FAR_POINTS[where]()
    after = build_mapper_graph(np.vstack([x, far]), ids(len(x)) + ["new"], params)

    holding_new = {n for n, c in after.nodes.items() if "new" in c}
    assert holding_new
    unchanged = {n: c for n, c in after.nodes.items() if n not in holding_new}
    assert unchanged == before.nodes
    assert {e for e in after.edges if not e & holding_new} == before.edges


def test_a_lone_point_in_an_interval_still_gets_a_node():
    """KeplerMapper drops intervals with fewer points than the clusterer's minimum."""
    x = blobs(bridge=False)
    params = fit_mapper(x, MapperConfig(n_intervals=10, overlap=0.2, threshold=0.05))
    lens = params.lens(x)
    mid = x[np.argmin(lens)] * 0.5 + x[np.argmax(lens)] * 0.5  # alone in the empty middle
    after = build_mapper_graph(np.vstack([x, _unit(mid)]), ids(len(x)) + ["new"], params)
    assert "new" in set().union(*after.nodes.values())


@pytest.mark.parametrize("linkage", ["average", "complete", "single"])
def test_every_linkage_separates_the_blobs(linkage):
    graph = graph_of(
        blobs(bridge=False),
        MapperConfig(n_intervals=5, overlap=0.3, linkage=linkage, threshold=0.05),
    )
    assert len(connected_components(graph)) == 2


def test_ids_sort_by_interval_with_a_hundred_intervals_or_more():
    x = blobs(bridge=True)
    graph = graph_of(x, MapperConfig(n_intervals=120, overlap=0.5, threshold=0.05))
    nodes = sorted(graph.nodes)
    assert all(len(n.split("r")[0]) == 4 for n in nodes), "i + three digits"
    intervals = [node_interval(n) for n in nodes]
    assert intervals == sorted(intervals) and max(intervals) >= 100


def test_params_compare_by_identity_without_raising():
    x = blobs(bridge=True)
    params = fit_mapper(x, CONFIG)
    assert params in [params] and len({params, params}) == 1  # == and hash, no ValueError
    assert params != fit_mapper(x, CONFIG)
