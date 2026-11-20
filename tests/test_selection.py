"""The Selector interface, exercised by a trivial selector."""

import numpy as np
import pytest

from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.selection import Selector, check_selection, run_selector


class FirstQuestions:
    """Trivial selector: the first `budget` candidates, in the order given."""

    name = "first"

    def select(self, before, after, questions, budget):
        return list(questions[:budget])


class Overspends:
    name = "overspends"

    def select(self, before, after, questions, budget):
        return list(questions)


@pytest.fixture
def corpus():
    return Corpus(
        chunks=[Chunk("d#0", "d", "text")],
        embeddings=np.ones((1, 2), dtype=np.float32),
        questions=[Question(q, q) for q in ("q1", "q2", "q3")],
        question_embeddings=np.ones((3, 2), dtype=np.float32),
        qrels={"q1": ["d#0"], "q2": ["d#0"], "q3": ["d#0"]},
    )


def test_trivial_selector_implements_the_interface(corpus):
    selector = FirstQuestions()
    assert isinstance(selector, Selector)
    questions = ["q1", "q2", "q3"]
    assert run_selector(selector, corpus, corpus, questions, budget=2) == ["q1", "q2"]
    assert run_selector(selector, corpus, corpus, questions, budget=0) == []
    assert run_selector(selector, corpus, corpus, questions, budget=10) == questions


def test_run_selector_rejects_a_selection_over_budget(corpus):
    with pytest.raises(ValueError, match="budget"):
        run_selector(Overspends(), corpus, corpus, ["q1", "q2", "q3"], budget=1)


@pytest.mark.parametrize(
    ("selected", "match"),
    [(["q1", "q1"], "duplicate"), (["q9"], "outside"), (["q1", "q2", "q3"], "budget")],
)
def test_check_selection_rejects_contract_violations(selected, match):
    with pytest.raises(ValueError, match=match):
        check_selection(selected, ["q1", "q2", "q3"], budget=2)
