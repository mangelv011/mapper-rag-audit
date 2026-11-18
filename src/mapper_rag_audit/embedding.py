"""Embedding model wrapper. The model library is imported only when a model is built."""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pinned so every machine embeds with the same weights and tokenizer files.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# Instruction recommended by the model card for short queries in retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def choose_device(cuda: bool, mps: bool) -> str:
    return "cuda" if cuda else "mps" if mps else "cpu"


def detect_device() -> str:
    import torch

    return choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())


class ModelTokenizer:
    """The embedding model's own tokenizer, as the chunker's token counter."""

    def __init__(self, hf_tokenizer, max_tokens: int):
        self._tok = hf_tokenizer
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        encoded = self._tok(text, add_special_tokens=True, truncation=False, verbose=False)
        return len(encoded["input_ids"])

    def split(self, text: str, max_body_tokens: int) -> list[str]:
        ids = self._tok.encode(text, add_special_tokens=False)
        pieces = -(-len(ids) // max_body_tokens)
        size = -(-len(ids) // pieces)
        return [self._tok.decode(ids[i : i + size]) for i in range(0, len(ids), size)]


class SentenceTransformerEmbedder:
    def __init__(self, model, batch_size: int = 64):
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def load(cls, device: str) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer

        return cls(SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION))

    def tokenizer(self) -> ModelTokenizer:
        return ModelTokenizer(self._model.tokenizer, self._model.max_seq_length)

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
