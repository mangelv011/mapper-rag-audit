import numpy as np

from mapper_rag_audit.embedding import QUERY_PREFIX, SentenceTransformerEmbedder, choose_device


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
