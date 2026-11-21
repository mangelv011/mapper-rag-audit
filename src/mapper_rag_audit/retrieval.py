"""Top-k retrieval and the shared definition of a failing question.

A question fails when none of its correct chunks is among the top-k chunks retrieved for it.
Every selector (Mapper and the baselines) is scored against this one definition.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from mapper_rag_audit.data import Corpus

DEFAULT_K = 5


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def top_k(corpus: Corpus, question_ids: Iterable[str], k: int = DEFAULT_K) -> dict[str, list[str]]:
    """Ids of the k chunks most similar (cosine) to each question, most similar first.

    Ties are broken by chunk order in the corpus, so results are deterministic.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    row = {q.question_id: i for i, q in enumerate(corpus.questions)}
    ids = list(question_ids)
    missing = [q for q in ids if q not in row]
    if missing:
        raise KeyError(f"questions not in the corpus: {missing[:5]}")
    if not ids:
        return {}

    queries = _unit_rows(corpus.question_embeddings[[row[q] for q in ids]])
    scores = queries @ _unit_rows(corpus.embeddings).T
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    chunk_ids = [c.chunk_id for c in corpus.chunks]
    return {q: [chunk_ids[j] for j in order[i]] for i, q in enumerate(ids)}


def failing_questions(corpus: Corpus, questions: Iterable[str], k: int = DEFAULT_K) -> set[str]:
    """Questions none of whose correct chunks is among their top-k retrieved chunks."""
    retrieved = top_k(corpus, questions, k)
    return {q for q, hits in retrieved.items() if not set(hits) & set(corpus.qrels.get(q, []))}


def broken_by_ingestion(
    before: Corpus, after: Corpus, questions: Iterable[str], k: int = DEFAULT_K
) -> set[str]:
    """Questions that pass on `before` and fail on `after`.

    Questions that already fail before the ingestion are excluded: an ingestion cannot break them.
    """
    ids = list(questions)
    return failing_questions(after, ids, k) - failing_questions(before, ids, k)
