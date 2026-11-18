import numpy as np

from mapper_rag_audit.embedding import (
    MODEL_REVISION,
    QUERY_PREFIX,
    ModelTokenizer,
    SentenceTransformerEmbedder,
    choose_device,
)


def test_device_preference_is_cuda_then_mps_then_cpu():
    assert choose_device(cuda=True, mps=True) == "cuda"
    assert choose_device(cuda=False, mps=True) == "mps"
    assert choose_device(cuda=False, mps=False) == "cpu"


class RecordingModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.append((list(texts), kwargs))
        return np.zeros((len(texts), 2), dtype=np.float32)


def test_queries_get_the_bge_instruction_prefix_and_passages_do_not():
    model = RecordingModel()
    embedder = SentenceTransformerEmbedder(model)
    embedder.embed_queries(["q"])
    embedder.embed_passages(["p"])
    (q_texts, q_kwargs), (p_texts, p_kwargs) = model.seen
    assert q_texts == [QUERY_PREFIX + "q"]
    assert p_texts == ["p"]
    assert q_kwargs["normalize_embeddings"] and p_kwargs["normalize_embeddings"]


def test_vectors_are_returned_as_float32():
    class Float64Model:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 2), dtype=np.float64)

    vectors = SentenceTransformerEmbedder(Float64Model()).embed_passages(["p"])
    assert vectors.dtype == np.float32


class FakeHfTokenizer:
    """Word-level tokenizer with the slice of the Hugging Face API the wrapper uses."""

    def __call__(self, text, add_special_tokens=True, truncation=False, verbose=True):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return [-1, *ids, -2] if add_special_tokens else ids

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def test_model_tokenizer_counts_special_tokens_and_does_not_truncate():
    tok = ModelTokenizer(FakeHfTokenizer(), max_tokens=512)
    assert tok.max_tokens == 512
    assert tok.count("") == 2
    assert tok.count(" ".join(["x"] * 1000)) == 1002


def test_model_tokenizer_splits_into_balanced_pieces_within_the_limit():
    tok = ModelTokenizer(FakeHfTokenizer(), max_tokens=512)
    pieces = tok.split(" ".join(["x"] * 25), 9)
    assert [len(p.split()) for p in pieces] == [9, 9, 7]
    assert tok.split("a b c", 9) == ["w0 w1 w2"]


def test_load_pins_the_model_revision_and_device(monkeypatch):
    import sentence_transformers

    calls = []

    class Recorder:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Recorder)
    SentenceTransformerEmbedder.load("cpu")
    assert calls == [("BAAI/bge-small-en-v1.5", {"device": "cpu", "revision": MODEL_REVISION})]
    assert len(MODEL_REVISION) == 40


def test_tokenizer_uses_the_models_own_tokenizer_and_limit():
    class Model:
        tokenizer = FakeHfTokenizer()
        max_seq_length = 77

    tok = SentenceTransformerEmbedder(Model()).tokenizer()
    assert tok.max_tokens == 77
    assert tok.count("a b") == 4
