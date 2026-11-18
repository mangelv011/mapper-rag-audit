"""Embedding model wrapper. The model library is imported only when a model is built."""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Instruction recommended by the model card for short queries in retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def choose_device(cuda: bool, mps: bool) -> str:
    return "cuda" if cuda else "mps" if mps else "cpu"


def detect_device() -> str:
    import torch

    return choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())


class SentenceTransformerEmbedder:
    def __init__(self, model, batch_size: int = 64):
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def load(cls, device: str) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer

        return cls(SentenceTransformer(MODEL_NAME, device=device))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + t for t in texts])
