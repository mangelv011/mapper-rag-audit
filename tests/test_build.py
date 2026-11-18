import numpy as np

from conftest import FakeTokenizer
from mapper_rag_audit.data import RawBeir, build_corpus, read_beir


def test_chunk_is_title_plus_text_and_maps_to_its_document(raw_dir, fake_embedder, fake_tokenizer):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder, fake_tokenizer)
    by_id = {c.chunk_id: c for c in corpus.chunks}
    assert by_id["d1#0"].doc_id == "d1"
    assert by_id["d1#0"].text == "Alpha\nAbout alpha."
    assert by_id["d3#0"].text == "No title here."
    assert [c.chunk_id for c in corpus.chunks] == ["d1#0", "d2#0", "d3#0"]


def test_questions_are_those_with_test_qrels_and_map_to_correct_chunks(
    raw_dir, fake_embedder, fake_tokenizer
):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder, fake_tokenizer)
    assert [q.question_id for q in corpus.questions] == ["q1", "q2"]
    assert corpus.questions[0].text == "what is alpha?"
    # q2 also lists a missing document (dX) and a score-0 pair (d1): neither is correct.
    assert corpus.qrels == {"q1": ["d1#0", "d3#0"], "q2": ["d2#0"]}


def test_question_without_any_present_correct_chunk_is_dropped(
    raw_dir, fake_embedder, fake_tokenizer
):
    raw = read_beir(raw_dir)
    raw = RawBeir(raw.documents, raw.queries, [("q1", "d1", 1), ("q2", "dX", 1)])
    corpus = build_corpus(raw, fake_embedder, fake_tokenizer)
    assert [q.question_id for q in corpus.questions] == ["q1"]
    assert list(corpus.qrels) == ["q1"]
    assert corpus.question_embeddings.shape[0] == 1


def test_embeddings_have_one_row_per_chunk_and_question_using_query_mode(
    raw_dir, fake_embedder, fake_tokenizer
):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder, fake_tokenizer)
    assert corpus.embeddings.shape == (3, 3)
    assert corpus.question_embeddings.shape == (2, 3)
    expected = fake_embedder.embed_queries(["what is alpha?", "what is beta?"])
    assert np.array_equal(corpus.question_embeddings, expected)


def test_question_maps_to_every_chunk_of_its_relevant_document(raw_dir, fake_embedder):
    raw = read_beir(raw_dir)
    long_text = " ".join(f"Sent{i} has four words." for i in range(6))  # 30 words
    raw.documents[0]["text"] = long_text
    corpus = build_corpus(raw, fake_embedder, FakeTokenizer(max_tokens=14))
    d1_chunks = [c.chunk_id for c in corpus.chunks if c.doc_id == "d1"]
    assert len(d1_chunks) > 1
    assert corpus.qrels["q1"] == [*d1_chunks, "d3#0"]
    assert corpus.embeddings.shape[0] == len(corpus.chunks)
