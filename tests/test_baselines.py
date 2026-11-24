"""Baseline selectors on tiny 2-D corpora with known expected rankings."""

import math

import numpy as np
import pytest

from mapper_rag_audit.baselines import (
    KMeansSelector,
    NearestNeighboursSelector,
    RandomSelector,
    new_chunk_rows,
)
from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.selection import Selector, run_selector


def _vec(degrees):
    rad = math.radians(degrees)
    return (math.cos(rad), math.sin(rad))


def corpus(chunks: dict[str, float], qrels: dict[str, list[str]], texts=None) -> Corpus:
    """chunks: chunk id -> angle in degrees. Questions are the qrels keys (embedding unused)."""
    texts = texts or {}
    return Corpus(
        chunks=[Chunk(c, c.split("#")[0], texts.get(c, c)) for c in chunks],
        embeddings=np.array([_vec(a) for a in chunks.values()], dtype=np.float32),
        questions=[Question(q, q) for q in qrels],
        question_embeddings=np.ones((len(qrels), 2), dtype=np.float32),
        qrels=qrels,
    )


# --- shared conventions -------------------------------------------------------------------

SELECTORS = [
    RandomSelector(seed=0),
    KMeansSelector(n_clusters=2),
    NearestNeighboursSelector(m=2),
]

SIMPLE = {"a#0": 0.0, "b#0": 90.0, "c#0": 180.0}
SIMPLE_QRELS = {"qa": ["a#0"], "qb": ["b#0"], "qc": ["c#0"]}


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s.name)
def test_every_baseline_implements_the_interface_and_respects_the_budget(selector):
    before = corpus(SIMPLE, SIMPLE_QRELS)
    after = corpus({**SIMPLE, "new#0": 5.0}, SIMPLE_QRELS)
    assert isinstance(selector, Selector)
    for budget in (0, 1, 2, 10):
        selected = run_selector(selector, before, after, list(SIMPLE_QRELS), budget)
        assert len(selected) <= budget


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s.name)
def test_an_ingestion_that_removes_chunks_is_rejected(selector):
    before = corpus(SIMPLE, SIMPLE_QRELS)
    after = corpus({"a#0": 0.0, "b#0": 90.0}, SIMPLE_QRELS)
    with pytest.raises(ValueError, match="removed"):
        selector.select(before, after, list(SIMPLE_QRELS), 3)


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s.name)
def test_questions_without_correct_chunks_are_never_selected(selector):
    qrels = {**SIMPLE_QRELS, "q_none": []}
    before = corpus(SIMPLE, qrels)
    after = corpus({**SIMPLE, "new#0": 5.0}, qrels)
    assert "q_none" not in selector.select(before, after, list(qrels), 10)


def test_new_chunks_are_added_or_changed_chunks():
    before = corpus(SIMPLE, SIMPLE_QRELS)
    after = corpus({**SIMPLE, "new#0": 5.0}, SIMPLE_QRELS, texts={"b#0": "edited"})
    assert new_chunk_rows(before, after) == [1, 3]


# --- random -------------------------------------------------------------------------------


def test_random_is_a_seeded_permutation_and_smaller_budgets_are_prefixes():
    qrels = {f"q{i:02d}": ["a#0"] for i in range(20)}
    before = corpus(SIMPLE, qrels)
    questions = list(qrels)
    full = RandomSelector(seed=7).select(before, before, questions, 20)
    assert sorted(full) == questions
    assert full != questions  # actually shuffled
    assert RandomSelector(seed=7).select(before, before, questions, 20) == full
    assert RandomSelector(seed=8).select(before, before, questions, 20) != full
    for budget in (0, 1, 5, 19):
        assert RandomSelector(seed=7).select(before, before, questions, budget) == full[:budget]
    # The order the candidates are passed in does not change the result.
    assert RandomSelector(seed=7).select(before, before, questions[::-1], 20) == full


# --- k-means ------------------------------------------------------------------------------

# Three well-separated groups of old chunks: A near 0, B near 90 and C near 180 degrees.
GROUPS = {"a1#0": 0, "a2#0": 6, "b1#0": 88, "b2#0": 94, "c1#0": 178, "c2#0": 184}
GROUP_QRELS = {
    "qa": ["a1#0"],
    "qb": ["b2#0"],
    "qc": ["c1#0"],
    "qcb": ["c2#0", "b1#0"],  # one correct chunk in C, one in B
}


def test_kmeans_ranks_questions_by_new_chunks_in_their_cluster():
    # Two new chunks land in A and one in B; C receives none.
    before = corpus(GROUPS, GROUP_QRELS)
    after = corpus({**GROUPS, "n1#0": 2, "n2#0": 4, "n3#0": 91}, GROUP_QRELS)
    selector = KMeansSelector(n_clusters=3, seed=0)
    # qa: cluster with 2 new. qb and qcb: cluster B with 1 new, tie broken by id. qc: unmarked.
    assert selector.select(before, after, list(GROUP_QRELS), 10) == ["qa", "qb", "qcb"]
    assert selector.select(before, after, list(GROUP_QRELS), 2) == ["qa", "qb"]


def test_kmeans_counts_a_changed_chunk_as_new():
    before = corpus(GROUPS, GROUP_QRELS)
    after = corpus(GROUPS, GROUP_QRELS, texts={"c2#0": "edited"})
    assert KMeansSelector(n_clusters=3).select(before, after, list(GROUP_QRELS), 10) == [
        "qc",
        "qcb",
    ]


def test_kmeans_selects_nothing_when_nothing_is_new():
    before = corpus(GROUPS, GROUP_QRELS)
    assert KMeansSelector(n_clusters=3).select(before, before, list(GROUP_QRELS), 10) == []


def test_kmeans_with_more_clusters_than_chunks_still_runs():
    before = corpus(SIMPLE, SIMPLE_QRELS)
    after = corpus({**SIMPLE, "new#0": 1.0}, SIMPLE_QRELS)
    assert KMeansSelector(n_clusters=50).select(before, after, list(SIMPLE_QRELS), 10) == ["qa"]


def test_kmeans_rejects_a_bad_number_of_clusters():
    with pytest.raises(ValueError):
        KMeansSelector(n_clusters=0)


# --- nearest neighbours of new chunks -----------------------------------------------------

OLD = {"o0#0": 0, "o10#0": 10, "o20#0": 20, "o90#0": 90, "o180#0": 180}
OLD_QRELS = {
    "q0": ["o0#0"],
    "q10": ["o10#0"],
    "q20": ["o20#0"],
    "q180": ["o180#0"],
    "qboth": ["o10#0", "o90#0"],
}


def test_nearest_neighbours_ranks_by_similarity_to_the_closest_new_chunk():
    # n4 (4 deg): 2 nearest old chunks are o0 (cos 4) and o10 (cos 6).
    # n92 (92 deg): 2 nearest old chunks are o90 (cos 2) and o20 (cos 72).
    before = corpus(OLD, OLD_QRELS)
    after = corpus({**OLD, "n4#0": 4, "n92#0": 92}, OLD_QRELS)
    selector = NearestNeighboursSelector(m=2)
    # qboth via o90 (cos 2), q0 via o0 (cos 4), q10 via o10 (cos 6), q20 via o20 (cos 72).
    # q180's chunk is not among the 2 nearest of any new chunk.
    assert selector.select(before, after, list(OLD_QRELS), 10) == ["qboth", "q0", "q10", "q20"]
    assert selector.select(before, after, list(OLD_QRELS), 2) == ["qboth", "q0"]


def test_nearest_neighbours_m_limits_how_far_from_a_new_chunk_it_looks():
    before = corpus(OLD, OLD_QRELS)
    after = corpus({**OLD, "n4#0": 4}, OLD_QRELS)
    assert NearestNeighboursSelector(m=1).select(before, after, list(OLD_QRELS), 10) == ["q0"]
    assert NearestNeighboursSelector(m=3).select(before, after, list(OLD_QRELS), 10) == [
        "q0",
        "q10",
        "qboth",
        "q20",
    ]


def test_nearest_neighbours_never_counts_a_new_chunk_as_a_neighbour():
    # n5 is the nearest chunk to n4, but it is new, so o0 is still the only neighbour with m=1.
    before = corpus(OLD, OLD_QRELS)
    after = corpus({**OLD, "n4#0": 4, "n5#0": 5}, OLD_QRELS)
    assert NearestNeighboursSelector(m=1).select(before, after, list(OLD_QRELS), 10) == ["q0"]


def test_nearest_neighbours_rejects_a_bad_m():
    with pytest.raises(ValueError):
        NearestNeighboursSelector(m=0)


# --- end to end on a synthetic ingestion ---------------------------------------------------


def test_all_baselines_run_end_to_end_on_a_synthetic_trap_ingestion():
    """Traps placed right on top of some questions break them; every baseline runs and keeps
    the budget, and the two that look at the ingestion rank a broken question high."""
    from mapper_rag_audit.retrieval import broken_by_ingestion

    rng = np.random.default_rng(0)
    dim, n_chunks, n_questions = 16, 120, 40
    chunk_vecs = rng.normal(size=(n_chunks, dim))
    chunk_ids = [f"d{i}#0" for i in range(n_chunks)]
    # Each question sits close to its own correct chunk.
    q_vecs = chunk_vecs[:n_questions] + 0.05 * rng.normal(size=(n_questions, dim))
    qrels = {f"q{i}": [chunk_ids[i]] for i in range(n_questions)}
    questions = [Question(q, q) for q in qrels]

    def make(ids, vecs):
        return Corpus(
            chunks=[Chunk(c, c.split("#")[0], c) for c in ids],
            embeddings=vecs.astype(np.float32),
            questions=questions,
            question_embeddings=q_vecs.astype(np.float32),
            qrels=qrels,
        )

    before = make(chunk_ids, chunk_vecs)
    targets = [3, 11, 25]
    traps = np.vstack([q_vecs[t] for t in targets for _ in range(5)])  # 5 copies per target
    trap_ids = [f"trap{t}_{j}#0" for t in targets for j in range(5)]
    after = make(chunk_ids + trap_ids, np.vstack([chunk_vecs, traps]))

    broken = broken_by_ingestion(before, after, list(qrels), k=5)
    assert broken == {f"q{t}" for t in targets}

    budget = 5
    for selector in (
        RandomSelector(seed=0),
        KMeansSelector(n_clusters=8, seed=0),
        NearestNeighboursSelector(m=10),
    ):
        selected = run_selector(selector, before, after, list(qrels), budget)
        assert len(selected) <= budget
        if selector.name != "random":
            assert broken & set(selected[:3]), f"{selector.name} missed every broken question"


def test_a_larger_m_can_change_the_selection_for_the_same_budget():
    # Example from the review of #24: a new neighbour of one new chunk can outrank the
    # neighbour of another new chunk, so m changes the selection, not only its tail.
    qrels = {"q10": ["o10#0"], "q20": ["o20#0"], "q180": ["o180#0"]}
    before = corpus({"o10#0": 10, "o20#0": 20, "o180#0": 180}, qrels)
    after = corpus({"o10#0": 10, "o20#0": 20, "o180#0": 180, "n0#0": 0, "n155#0": 155}, qrels)
    assert NearestNeighboursSelector(m=1).select(before, after, list(qrels), 2) == ["q10", "q180"]
    assert NearestNeighboursSelector(m=2).select(before, after, list(qrels), 2) == ["q10", "q20"]
