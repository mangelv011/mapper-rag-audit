"""Baseline selectors that Mapper is compared with (#6).

All three implement `selection.Selector` and follow the same conventions as `MapperSelector`
(#5), so the comparison in #7 is fair:

- The candidates are the ids in `questions` that have correct chunks in `before.qrels`.
- New chunks are the chunk ids of `after` that are not in `before`, or whose text changed.
  An ingestion that removes chunks raises ValueError.
- Only Random fills the whole budget. The other two return only the questions their rule marks,
  so a selector that marks fewer questions than the budget returns a shorter list.
- Ties are broken by question id, so every ranking is deterministic.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans

from mapper_rag_audit.data import Corpus


def new_chunk_rows(before: Corpus, after: Corpus) -> list[int]:
    """Rows of `after` holding chunks that are new or whose text changed in the ingestion."""
    texts_before = {c.chunk_id: c.text for c in before.chunks}
    ids_after = {c.chunk_id for c in after.chunks}
    removed = texts_before.keys() - ids_after
    if removed:
        raise ValueError(
            f"chunks removed by the ingestion: {len(removed)}; removals are not supported"
        )
    return [i for i, c in enumerate(after.chunks) if texts_before.get(c.chunk_id) != c.text]


def candidates(before: Corpus, questions: Sequence[str]) -> list[str]:
    """Ids in `questions` that have correct chunks, without duplicates, in the given order."""
    return [q for q in dict.fromkeys(questions) if before.qrels.get(q)]


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class RandomSelector:
    """Uniform random order of the candidates, seeded.

    It ignores the ingestion completely, so it is the floor every other selector must beat.
    The whole candidate list is shuffled once and then cut, so the selection for a smaller
    budget is a prefix of the one for a larger budget, as the interface requires.
    """

    def __init__(self, seed: int = 0, name: str = "random"):
        self.seed = seed
        self.name = name

    def select(
        self, before: Corpus, after: Corpus, questions: Sequence[str], budget: int
    ) -> list[str]:
        new_chunk_rows(before, after)  # same contract as the others: no removals
        ranked = sorted(candidates(before, questions))
        random.Random(self.seed).shuffle(ranked)
        return ranked[: max(budget, 0)]


class KMeansSelector:
    """Clusters that received new chunks, with no notion of neighbours or edges.

    Parameters:
    - `n_clusters`: k for k-means. Rule: k equals the number of nodes of the Mapper graph of the
      corpus before the ingestion, so both methods split the corpus into the same number of
      regions. The experiment (#7) passes that number in.
    - `seed`, `n_init`: k-means is seeded and restarted `n_init` times, keeping the best fit.

    `select` fits k-means on the chunk embeddings of `before` (unit length, so Euclidean
    distance ranks like cosine), then assigns every chunk of `after` to its nearest centroid.
    A cluster is marked if at least one new chunk lands in it. A candidate question is
    selected if one of its correct chunks is in a marked cluster. Questions are ranked by the
    largest number of new chunks among the marked clusters holding their correct chunks, then
    by question id. This mirrors how Mapper ranks questions in affected nodes, without the
    graph structure.
    """

    def __init__(self, n_clusters: int, seed: int = 0, n_init: int = 10, name: str = "kmeans"):
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
        self.n_clusters = n_clusters
        self.seed = seed
        self.n_init = n_init
        self.name = name

    def select(
        self, before: Corpus, after: Corpus, questions: Sequence[str], budget: int
    ) -> list[str]:
        new_rows = new_chunk_rows(before, after)
        pool = candidates(before, questions)
        if not new_rows or not pool or budget <= 0:
            return []

        n_clusters = min(self.n_clusters, len(before.chunks))
        kmeans = KMeans(n_clusters=n_clusters, n_init=self.n_init, random_state=self.seed)
        kmeans.fit(_unit_rows(before.embeddings))
        labels = kmeans.predict(_unit_rows(after.embeddings))

        new_per_cluster = np.bincount(labels[new_rows], minlength=n_clusters)
        cluster_of = {c.chunk_id: int(labels[i]) for i, c in enumerate(after.chunks)}

        keys: dict[str, tuple[int, str]] = {}
        for q in pool:
            counts = [new_per_cluster[cluster_of[c]] for c in before.qrels[q] if c in cluster_of]
            most_new = max(counts, default=0)
            if most_new > 0:
                keys[q] = (-int(most_new), q)
        return sorted(keys, key=keys.__getitem__)[:budget]


class NearestNeighboursSelector:
    """Questions whose correct chunks are among the nearest old chunks of a new chunk.

    Parameters:
    - `m`: how many nearest old chunks are taken for each new chunk. Default 10, twice the
      retrieval k of 5. A new chunk breaks a question by entering its top-5, which only
      happens if the new chunk is close to the question and so, usually, to its correct
      chunk. The question-to-chunk and chunk-to-chunk neighbourhoods are not the same, so
      m keeps a margin above k. Because questions are ranked by similarity, a larger m only
      adds candidates at the end of the ranking. It changes the result only for large budgets.

    Old chunks are the chunks of `after` that are not new. Similarity is cosine. A candidate
    question is selected if one of its correct chunks is among the m nearest old chunks of
    any new chunk. Questions are ranked by the highest such similarity, closest first, then
    by question id.
    """

    def __init__(self, m: int = 10, name: str = "nearest_neighbours"):
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        self.m = m
        self.name = name

    def select(
        self, before: Corpus, after: Corpus, questions: Sequence[str], budget: int
    ) -> list[str]:
        new_rows = new_chunk_rows(before, after)
        pool = candidates(before, questions)
        new_set = set(new_rows)
        old_rows = [i for i in range(len(after.chunks)) if i not in new_set]
        if not new_rows or not old_rows or not pool or budget <= 0:
            return []

        unit = _unit_rows(after.embeddings)
        sims = unit[new_rows] @ unit[old_rows].T  # one row per new chunk
        m = min(self.m, len(old_rows))
        # Stable sort: among equally similar old chunks, the earlier one in the corpus wins.
        nearest = np.argsort(-sims, axis=1, kind="stable")[:, :m]

        best: dict[str, float] = {}  # old chunk id -> highest similarity to a new chunk
        for r, cols in enumerate(nearest):
            for col in cols:
                chunk_id = after.chunks[old_rows[col]].chunk_id
                best[chunk_id] = max(best.get(chunk_id, -np.inf), float(sims[r, col]))

        keys: dict[str, tuple[float, str]] = {}
        for q in pool:
            hits = [best[c] for c in before.qrels[q] if c in best]
            if hits:
                keys[q] = (-max(hits), q)
        return sorted(keys, key=keys.__getitem__)[:budget]
