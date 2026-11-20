"""Worked toy example: Mapper on a noisy circle, step by step (issue #9).

Builds the Mapper graph of points sampled from a noisy circle, using the
height (y coordinate) as the lens, and saves one figure per step. It also
builds two "bad cover" cases in which the loop is lost, to show when the
nerve lemma does not apply.

Usage (from the repository root):

    python experiments/toy_circle.py

All figures (PDF for the report, PNG for the slides) and a captions file are
written to report/figures/toy/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "report" / "figures" / "toy"

# --------------------------------------------------------------------------
# Shared style, so every figure looks the same in the report and the slides
# --------------------------------------------------------------------------
STYLE = {
    "figure.figsize": (4.2, 4.2),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
CMAP = "viridis"
POINT_GREY = "#b0b0b0"
EDGE_COLOR = "#333333"
SEED = 0


# --------------------------------------------------------------------------
# Mapper, written out by hand so each step is visible
# --------------------------------------------------------------------------
def noisy_circle(n: int = 300, noise: float = 0.06, seed: int = SEED) -> np.ndarray:
    """Step 1: n points on the unit circle plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return points + rng.normal(scale=noise, size=points.shape)


def height_lens(points: np.ndarray) -> np.ndarray:
    """Step 2: the lens (filter function) maps each point to its height."""
    return points[:, 1]


def interval_cover(
    values: np.ndarray, n_intervals: int, overlap: float
) -> list[tuple[float, float]]:
    """Step 3: cover the range of the lens with n equal intervals.

    `overlap` is the fraction of each interval shared with its neighbour
    (0 = no overlap, 0.3 = 30 % overlap).
    """
    lo, hi = values.min(), values.max()
    # length L such that n intervals with overlap `overlap` exactly span [lo, hi]
    length = (hi - lo) / (n_intervals - (n_intervals - 1) * overlap)
    step = length * (1 - overlap)
    eps = 1e-9  # so the extreme points are included
    return [(lo + i * step - eps, lo + i * step + length + eps) for i in range(n_intervals)]


@dataclass
class Node:
    interval: int  # which interval of the cover it comes from
    members: np.ndarray  # indices of the points in this cluster


def cluster_preimages(points, values, cover, link_distance: float) -> list[Node]:
    """Step 4: cluster the points whose lens value falls in each interval.

    Single-linkage clustering: two points end up in the same cluster if they
    are joined by a chain of points closer than `link_distance`.
    """
    nodes = []
    for i, (a, b) in enumerate(cover):
        idx = np.where((values >= a) & (values <= b))[0]
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            nodes.append(Node(i, idx))
            continue
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=link_distance, linkage="single"
        ).fit_predict(points[idx])
        for lab in np.unique(labels):
            nodes.append(Node(i, idx[labels == lab]))
    return nodes


def nerve_edges(nodes: list[Node]) -> list[tuple[int, int]]:
    """Step 5: the nerve. Join two clusters if they share at least one point."""
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if np.intersect1d(nodes[i].members, nodes[j].members).size > 0:
                edges.append((i, j))
    return edges


def n_components(n_nodes: int, edges) -> int:
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(x) for x in range(n_nodes)})


def n_loops(n_nodes: int, edges) -> int:
    """Number of independent loops of a graph: E - V + (connected components)."""
    return len(edges) - n_nodes + n_components(n_nodes, edges)


def mapper(points, n_intervals, overlap, link_distance):
    values = height_lens(points)
    cover = interval_cover(values, n_intervals, overlap)
    nodes = cluster_preimages(points, values, cover, link_distance)
    edges = nerve_edges(nodes)
    return values, cover, nodes, edges


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _square(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def _interval_colors(n):
    cmap = plt.get_cmap(CMAP)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _draw_bands(ax, cover, colors, alpha=0.18):
    for (a, b), c in zip(cover, colors, strict=True):
        ax.add_patch(Rectangle((-1.45, a), 2.9, b - a, color=c, alpha=alpha, lw=0))


def _draw_graph(ax, points, nodes, edges, colors, show_points=True):
    if show_points:
        ax.scatter(points[:, 0], points[:, 1], s=6, c=POINT_GREY, alpha=0.5, lw=0)
    centres = np.array([points[n.members].mean(axis=0) for n in nodes])
    for i, j in edges:
        ax.plot(*centres[[i, j]].T, color=EDGE_COLOR, lw=2, zorder=2)
    sizes = np.array([len(n.members) for n in nodes])
    ax.scatter(
        centres[:, 0],
        centres[:, 1],
        s=40 + 3 * sizes,
        c=[colors[n.interval] for n in nodes],
        edgecolors=EDGE_COLOR,
        lw=1,
        zorder=3,
    )


def _count(n: int, word: str) -> str:
    """'1 loop', '0 loops', '3 edges'..."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _save(fig, out: Path, name: str, captions: dict, caption: str):
    # No creation date in the PDF, so regenerating unchanged figures gives identical files
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    captions[name] = caption


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(out: Path, n_intervals=6, overlap=0.3, link_distance=0.25):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    captions: dict[str, str] = {}

    points = noisy_circle()
    values, cover, nodes, edges = mapper(points, n_intervals, overlap, link_distance)
    colors = _interval_colors(len(cover))

    # 1. Data
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=8, c="#444444", lw=0)
    _square(ax, "1. Data: a noisy circle")
    _save(
        fig,
        out,
        "01_data",
        captions,
        f"{len(points)} points sampled from a circle with Gaussian noise.",
    )

    # 2. Lens
    fig, ax = plt.subplots()
    sc = ax.scatter(points[:, 0], points[:, 1], s=8, c=values, cmap=CMAP, lw=0)
    fig.colorbar(sc, ax=ax, shrink=0.75, label="height $f(x, y) = y$")
    _square(ax, "2. Lens: height")
    _save(
        fig,
        out,
        "02_lens",
        captions,
        "The lens assigns each point its height; colour shows the lens value.",
    )

    # 3. Cover
    fig, ax = plt.subplots()
    _draw_bands(ax, cover, colors)
    ax.scatter(points[:, 0], points[:, 1], s=8, c="#444444", lw=0)
    _square(ax, f"3. Cover: {n_intervals} intervals, {int(overlap * 100)}% overlap")
    _save(
        fig,
        out,
        "03_cover",
        captions,
        f"The range of heights is covered by {n_intervals} overlapping intervals (bands).",
    )

    # 4. Clusters
    fig, ax = plt.subplots()
    _draw_bands(ax, cover, colors, alpha=0.10)
    ax.scatter(points[:, 0], points[:, 1], s=6, c=POINT_GREY, lw=0)
    for k, node in enumerate(nodes):
        p = points[node.members]
        ax.scatter(p[:, 0], p[:, 1], s=10, color=colors[node.interval], lw=0)
        cx, cy = p.mean(axis=0)
        ax.annotate(
            str(k + 1),
            (cx, cy),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.2", "fc": EDGE_COLOR, "lw": 0},
        )
    _square(ax, f"4. Clusters per interval ({len(nodes)} in total)")
    per_band = np.bincount([n.interval for n in nodes], minlength=len(cover))
    n_split = int((per_band > 1).sum())
    _save(
        fig,
        out,
        "04_clusters",
        captions,
        f"Points in each band are clustered separately: {n_split} of the {n_intervals} bands "
        f"split into more than one cluster, giving {len(nodes)} clusters in total.",
    )

    # 5. Nerve
    fig, ax = plt.subplots()
    _draw_bands(ax, cover, colors, alpha=0.10)
    _draw_graph(ax, points, nodes, edges, colors)
    _square(ax, "5. Nerve: link clusters that share points")
    _save(
        fig,
        out,
        "05_nerve",
        captions,
        "Each cluster becomes a node; clusters that share points (in the overlaps) are joined.",
    )

    # 6. Mapper graph alone
    loops = n_loops(len(nodes), edges)
    fig, ax = plt.subplots()
    _draw_graph(ax, points, nodes, edges, colors, show_points=False)
    _square(ax, f"6. Mapper graph: {_count(loops, 'loop')}")
    if loops == 1:
        graph_caption = "The resulting Mapper graph has 1 loop, recovering the hole of the circle."
    else:
        graph_caption = (
            f"The resulting Mapper graph has {_count(loops, 'loop')}, "
            "so it does not recover the single hole of the circle."
        )
    _save(fig, out, "06_graph", captions, graph_caption)

    summary = {"good": (len(nodes), len(edges), loops)}

    # 7-8. Bad covers: the loop is lost.
    # Captions are templates filled in with the actual edges, pieces and loops of each graph.
    bad_cases = [
        (
            "07_bad_coarse",
            2,
            0.3,
            "Too coarse: 2 intervals",
            (
                "With only two intervals each band holds one arc; their overlap has two pieces, "
                "but the nerve records only {edges}, so the graph has {loops}."
            ),
        ),
        (
            "08_bad_no_overlap",
            n_intervals,
            0.0,
            "No overlap",
            (
                "Without overlap no cluster shares points with another, so the nerve has {edges} "
                "and the graph falls apart into {pieces} with {loops}."
            ),
        ),
    ]
    for name, k, ov, title, caption in bad_cases:
        _, b_cover, b_nodes, b_edges = mapper(points, k, ov, link_distance)
        b_loops = n_loops(len(b_nodes), b_edges)
        b_colors = _interval_colors(len(b_cover))
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
        _draw_bands(axes[0], b_cover, b_colors)
        axes[0].scatter(points[:, 0], points[:, 1], s=8, c="#444444", lw=0)
        _square(axes[0], f"Cover: {k} intervals, {int(ov * 100)}% overlap")
        _draw_graph(axes[1], points, b_nodes, b_edges, b_colors, show_points=False)
        _square(axes[1], f"Mapper graph: {_count(b_loops, 'loop')}")
        fig.suptitle(title, fontsize=12)
        caption = caption.format(
            edges=_count(len(b_edges), "edge"),
            pieces=_count(n_components(len(b_nodes), b_edges), "piece"),
            loops=_count(b_loops, "loop"),
        )
        _save(fig, out, name, captions, caption)
        summary[name] = (len(b_nodes), len(b_edges), b_loops)

    with open(out / "captions.md", "w", encoding="utf-8") as f:
        f.write("# Captions for the toy example figures\n\n")
        for name, caption in captions.items():
            f.write(f"- `{name}`: {caption}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    args = parser.parse_args()

    summary = make_figures(args.out)
    for case, (v, e, loops) in summary.items():
        print(f"{case:>18}: {v} nodes, {e} edges, {loops} loop(s)")
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
