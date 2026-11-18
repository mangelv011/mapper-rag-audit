import numpy as np

from mapper_rag_audit.data import build_corpus, load_corpus, read_beir, save_corpus


def test_save_then_load_returns_the_same_corpus(raw_dir, fake_embedder, fake_tokenizer, tmp_path):
    corpus = build_corpus(read_beir(raw_dir), fake_embedder, fake_tokenizer)
    save_corpus(corpus, tmp_path / "data")
    loaded = load_corpus(tmp_path / "data")
    assert loaded.chunks == corpus.chunks
    assert loaded.questions == corpus.questions
    assert loaded.qrels == corpus.qrels
    assert loaded.embeddings.dtype == np.float32
    assert np.array_equal(loaded.embeddings, corpus.embeddings)
    assert np.array_equal(loaded.question_embeddings, corpus.question_embeddings)


def test_non_ascii_text_survives_the_round_trip(raw_dir, fake_embedder, fake_tokenizer, tmp_path):
    raw = read_beir(raw_dir)
    raw.documents[0]["text"] = "β-amyloid → naïve 肿瘤"
    corpus = build_corpus(raw, fake_embedder, fake_tokenizer)
    save_corpus(corpus, tmp_path)
    assert load_corpus(tmp_path).chunks[0].text == "Alpha\nβ-amyloid → naïve 肿瘤"
