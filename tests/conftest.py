import json
from pathlib import Path

import numpy as np
import pytest

CORPUS = [
    {"_id": "d1", "title": "Alpha", "text": "About alpha.", "metadata": {}},
    {"_id": "d2", "title": "Beta", "text": "About beta.", "metadata": {}},
    {"_id": "d3", "title": "", "text": "No title here.", "metadata": {}},
]
QUERIES = [
    {"_id": "q1", "text": "what is alpha?", "metadata": {}},
    {"_id": "q2", "text": "what is beta?", "metadata": {}},
    {"_id": "q3", "text": "train-only question", "metadata": {}},
]
QRELS_TEST = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t1\nq2\td2\t1\nq2\tdX\t1\nq2\td1\t0\n"


class FakeEmbedder:
    """Deterministic stand-in for the real model: passages and queries differ."""

    def _embed(self, texts, offset):
        rows = [[len(t) + offset, sum(map(ord, t)) % 7 + 1, 1.0] for t in texts]
        arr = np.array(rows, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def embed_passages(self, texts):
        return self._embed(texts, 0)

    def embed_queries(self, texts):
        return self._embed(texts, 100)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "scifact"
    (raw / "qrels").mkdir(parents=True)
    for name, rows in (("corpus.jsonl", CORPUS), ("queries.jsonl", QUERIES)):
        (raw / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (raw / "qrels" / "test.tsv").write_text(QRELS_TEST, encoding="utf-8")
    return raw
