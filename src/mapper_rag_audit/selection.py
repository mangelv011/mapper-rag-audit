"""Shared interface for the methods that choose which questions to re-test after an ingestion.

Mapper (#5) and the baselines (#6) all implement `Selector`, so the experiment (#7) can run
them the same way and score them against `retrieval.broken_by_ingestion`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mapper_rag_audit.data import Corpus


@runtime_checkable
class Selector(Protocol):
    """Chooses up to `budget` questions to re-test after an ingestion.

    `name` is a short label used in results tables and figures (e.g. "mapper", "random").

    `select` gets:
    - `before`: the corpus before the ingestion.
    - `after`: the corpus after it. The ingestion may add, change or remove chunks; the
      questions and their question embeddings are the same as in `before`.
    - `questions`: the candidate question ids, the pool to choose from.
    - `budget`: the maximum number of questions that may be re-tested.

    It returns question ids ranked by priority, most likely to be broken first. The list has
    no duplicates, only ids from `questions`, and at most `budget` items. Because it is ranked,
    a run with budget b can be scored on any prefix, so one call serves several smaller budgets.

    A selector must not look at which questions actually fail: it gets embeddings, chunks and
    qrels, but never runs retrieval against the correct chunks to decide.
    """

    name: str

    def select(
        self, before: Corpus, after: Corpus, questions: Sequence[str], budget: int
    ) -> list[str]: ...


def check_selection(selected: Sequence[str], questions: Sequence[str], budget: int) -> None:
    """Raise ValueError if `selected` breaks the `Selector.select` contract."""
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if len(selected) > budget:
        raise ValueError(f"selected {len(selected)} questions, budget is {budget}")
    if len(set(selected)) != len(selected):
        raise ValueError("selection has duplicate question ids")
    unknown = set(selected) - set(questions)
    if unknown:
        raise ValueError(f"selection has ids outside the candidates: {sorted(unknown)[:5]}")


def run_selector(
    selector: Selector, before: Corpus, after: Corpus, questions: Sequence[str], budget: int
) -> list[str]:
    """Call `selector.select` and check that its output keeps the contract."""
    selected = list(selector.select(before, after, questions, budget))
    check_selection(selected, questions, budget)
    return selected
