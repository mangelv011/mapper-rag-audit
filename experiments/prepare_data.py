"""Download SciFact, chunk it, embed it and cache everything under data/."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from mapper_rag_audit.data import build_corpus, read_beir, save_corpus
from mapper_rag_audit.download import ensure_dataset
from mapper_rag_audit.embedding import MODEL_NAME, SentenceTransformerEmbedder, detect_device


class TimedEmbedder:
    """Adds up the time spent inside the embedder, leaving chunking out."""

    def __init__(self, embedder):
        self._embedder = embedder
        self.seconds = 0.0

    def _timed(self, embed, texts):
        start = time.perf_counter()
        try:
            return embed(texts)
        finally:
            self.seconds += time.perf_counter() - start

    def embed_passages(self, texts):
        return self._timed(self._embedder.embed_passages, texts)

    def embed_queries(self, texts):
        return self._timed(self._embedder.embed_queries, texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw_dir = ensure_dataset(args.data_dir)
    raw = read_beir(raw_dir)

    device = detect_device()
    embedder = SentenceTransformerEmbedder.load(device)
    tokenizer = embedder.tokenizer()
    timed = TimedEmbedder(embedder)
    corpus = build_corpus(raw, timed, tokenizer)
    save_corpus(corpus, args.data_dir)

    asked = {q for q, _, _ in raw.qrels}
    chunks_per_doc = Counter(c.doc_id for c in corpus.chunks)
    split_docs = sum(n > 1 for n in chunks_per_doc.values())
    print(
        f"documents={len(raw.documents)} chunks={len(corpus.chunks)} documents_split={split_docs}"
    )
    print(
        f"max chunk tokens={max(tokenizer.count(c.text) for c in corpus.chunks)} limit={tokenizer.max_tokens}"
    )
    print(f"test questions={len(asked)} kept={len(corpus.questions)}")
    pairs = sum(len(v) for v in corpus.qrels.values())
    print(f"correct chunks={pairs} from {len(raw.qrels)} question-document pairs")
    print(f"model={MODEL_NAME} device={device} embedding_seconds={timed.seconds:.1f}")
    print(f"saved to {args.data_dir}")


if __name__ == "__main__":
    main()
