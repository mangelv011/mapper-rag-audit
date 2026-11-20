from pathlib import Path

import pytest

import helpers
from mapper_rag_audit.data import build_corpus, load_corpus, read_beir

REAL_DATA = Path(__file__).resolve().parent.parent / "data"


def test_every_question_has_a_correct_chunk_in_the_fixture_corpus(
    raw_dir, fake_embedder, fake_tokenizer
):
    raw = read_beir(raw_dir)
    corpus = build_corpus(raw, fake_embedder, fake_tokenizer)
    helpers.assert_every_question_has_a_correct_chunk_in_the_corpus(corpus)
    helpers.assert_chunks_fit_and_belong_to_documents(
        corpus, {d["_id"] for d in raw.documents}, fake_tokenizer
    )
    helpers.assert_chunks_are_verbatim_slices_of_their_documents(corpus, raw.documents)


@pytest.mark.skipif(not (REAL_DATA / "qrels.json").exists(), reason="run prepare_data.py first")
def test_every_question_has_a_correct_chunk_in_the_real_corpus():
    helpers.assert_every_question_has_a_correct_chunk_in_the_corpus(load_corpus(REAL_DATA))


@pytest.mark.skipif(not (REAL_DATA / "qrels.json").exists(), reason="run prepare_data.py first")
def test_no_chunk_of_the_real_corpus_exceeds_the_model_limit():
    from mapper_rag_audit.embedding import SentenceTransformerEmbedder

    tokenizer = SentenceTransformerEmbedder.load("cpu").tokenizer()
    doc_ids = {d["_id"] for d in read_beir(REAL_DATA / "raw" / "scifact").documents}
    helpers.assert_chunks_fit_and_belong_to_documents(load_corpus(REAL_DATA), doc_ids, tokenizer)


@pytest.mark.skipif(not (REAL_DATA / "qrels.json").exists(), reason="run prepare_data.py first")
def test_every_chunk_of_the_real_corpus_is_a_verbatim_slice_of_its_abstract():
    documents = read_beir(REAL_DATA / "raw" / "scifact").documents
    helpers.assert_chunks_are_verbatim_slices_of_their_documents(load_corpus(REAL_DATA), documents)


@pytest.mark.skipif(not (REAL_DATA / "qrels.json").exists(), reason="run prepare_data.py first")
def test_real_tokenizer_splits_keep_the_original_text():
    from mapper_rag_audit.embedding import SentenceTransformerEmbedder

    tokenizer = SentenceTransformerEmbedder.load("cpu").tokenizer()
    text = "Naïve T-cells (CD4+) hydrolase " * 40
    spans = tokenizer.split(text, 30)
    assert len(spans) > 1
    for a, b in spans:
        assert text[a:b] == text[a:b].strip() and text[a:b] in text
    assert "".join(text[a:b] for a, b in spans).replace(" ", "") == text.replace(" ", "")
