"""failing_questions on a hand-made corpus of 2-D vectors where the answer is known."""

import numpy as np
import pytest

from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.retrieval import broken_by_ingestion, failing_questions, top_k


def _corpus(chunks: dict[str, tuple[float, float]], questions, qrels) -> Corpus:
    """chunks: chunk id -> 2-D vector. questions: question id -> 2-D vector."""
    return Corpus(
        chunks=[Chunk(cid, cid.split("#")[0], cid) for cid in chunks],
        embeddings=np.array(list(chunks.values()), dtype=np.float32),
        questions=[Question(q, q) for q in questions],
        question_embeddings=np.array(list(questions.values()), dtype=np.float32),
        qrels=qrels,
    )


# Chunks at 0, 30, 70 and 90 degrees. Vectors are deliberately not unit length:
# retrieval must use cosine, not the raw dot product.
CHUNKS = {
    "east#0": (5.0, 0.0),
    "ne30#0": (0.866, 0.5),
    "n70#0": (0.342, 0.940),
    "north#0": (0.0, 3.0),
}
# q_east is at about 3 degrees, the other two point exactly at a chunk direction.
QUESTIONS = {"q_east": (1.0, 0.05), "q_north": (0.0, 1.0), "q_ne30": (0.866, 0.5)}
QRELS = {
    "q_east": ["east#0"],  # its correct chunk is its nearest one: never fails
    "q_north": ["east#0"],  # its correct chunk is the farthest one: fails for k < 4
    "q_ne30": ["n70#0", "north#0"],  # correct chunks are 3rd and 4th nearest
}


@pytest.fixture
def corpus():
    return _corpus(CHUNKS, QUESTIONS, QRELS)


def test_top_k_ranks_by_cosine_most_similar_first(corpus):
    assert top_k(corpus, ["q_north"], k=4)["q_north"] == ["north#0", "n70#0", "ne30#0", "east#0"]
    assert top_k(corpus, ["q_east"], k=2)["q_east"] == ["east#0", "ne30#0"]


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, {"q_north", "q_ne30"}),
        (2, {"q_north", "q_ne30"}),  # q_ne30's top 2 are ne30 and east
        (3, {"q_north"}),  # n70 is 3rd for q_ne30
        (4, set()),  # k covers the whole corpus
        (10, set()),  # k larger than the corpus is clipped
    ],
)
def test_failing_questions_on_known_answers(corpus, k, expected):
    assert failing_questions(corpus, list(QUESTIONS), k) == expected


def test_failing_questions_only_considers_the_given_questions(corpus):
    assert failing_questions(corpus, ["q_east", "q_ne30"], k=1) == {"q_ne30"}
    assert failing_questions(corpus, [], k=1) == set()


def test_a_question_without_correct_chunks_always_fails():
    corpus = _corpus(CHUNKS, {"q": (1.0, 0.0)}, {})
    assert failing_questions(corpus, ["q"], k=4) == {"q"}


def test_ties_are_broken_by_corpus_order():
    corpus = _corpus({"a#0": (1.0, 0.0), "b#0": (2.0, 0.0)}, {"q": (1.0, 0.0)}, {"q": ["b#0"]})
    assert top_k(corpus, ["q"], k=1)["q"] == ["a#0"]
    assert failing_questions(corpus, ["q"], k=1) == {"q"}


def test_rejects_bad_k_and_unknown_questions(corpus):
    with pytest.raises(ValueError):
        failing_questions(corpus, ["q_east"], k=0)
    with pytest.raises(KeyError):
        failing_questions(corpus, ["nope"], k=1)


def test_broken_by_ingestion_excludes_questions_that_already_fail(corpus):
    # A trap chunk in exactly q_east's direction pushes east#0 out of its top-1.
    # q_north and q_ne30 already fail at k=1 before the ingestion, so they are not "broken".
    after = _corpus({**CHUNKS, "trap#0": (1.0, 0.05)}, QUESTIONS, QRELS)
    assert failing_questions(after, list(QUESTIONS), k=1) == {"q_east", "q_north", "q_ne30"}
    assert broken_by_ingestion(corpus, after, list(QUESTIONS), k=1) == {"q_east"}
    assert broken_by_ingestion(corpus, after, list(QUESTIONS), k=2) == set()
