import numpy as np

from mapper_rag_audit.data import RawBeir, build_corpus, read_beir


def test_chunk_is_title_plus_abstract_and_maps_to_its_document(raw_dir, fake_embedder):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder)
    by_id = {c.chunk_id: c for c in corpus.chunks}
    assert by_id["d1"].doc_id == "d1"
    assert by_id["d1"].text == "Alpha\nAbout alpha."
    assert by_id["d3"].text == "No title here."
    assert [c.chunk_id for c in corpus.chunks] == ["d1", "d2", "d3"]


def test_questions_are_those_with_test_qrels_and_map_to_correct_chunks(raw_dir, fake_embedder):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder)
    assert [q.question_id for q in corpus.questions] == ["q1", "q2"]
    assert corpus.questions[0].text == "what is alpha?"
    # q2 also lists a missing document (dX) and a score-0 pair (d1): neither is correct.
    assert corpus.qrels == {"q1": ["d1", "d3"], "q2": ["d2"]}


def test_question_without_any_present_correct_chunk_is_dropped(raw_dir, fake_embedder):
    raw = read_beir(raw_dir)
    raw = RawBeir(raw.documents, raw.queries, [("q1", "d1", 1), ("q2", "dX", 1)])
    corpus = build_corpus(raw, fake_embedder)
    assert [q.question_id for q in corpus.questions] == ["q1"]
    assert list(corpus.qrels) == ["q1"]
    assert corpus.question_embeddings.shape[0] == 1


def test_embeddings_have_one_row_per_chunk_and_question_using_query_mode(raw_dir, fake_embedder):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder)
    assert corpus.embeddings.shape == (3, 3)
    assert corpus.question_embeddings.shape == (2, 3)
    expected = fake_embedder.embed_queries(["what is alpha?", "what is beta?"])
    assert np.array_equal(corpus.question_embeddings, expected)
