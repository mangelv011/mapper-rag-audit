"""Download SciFact, chunk it, embed it and cache everything under data/."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mapper_rag_audit.data import build_corpus, read_beir, save_corpus
from mapper_rag_audit.download import ensure_dataset
from mapper_rag_audit.embedding import MODEL_NAME, SentenceTransformerEmbedder, detect_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw_dir = ensure_dataset(args.data_dir)
    raw = read_beir(raw_dir)

    device = detect_device()
    embedder = SentenceTransformerEmbedder.load(device)
    start = time.perf_counter()
    corpus = build_corpus(raw, embedder)
    elapsed = time.perf_counter() - start
    save_corpus(corpus, args.data_dir)

    asked = {q for q, _, _ in raw.qrels}
    print(f"documents={len(raw.documents)} chunks={len(corpus.chunks)}")
    print(f"test questions={len(asked)} kept={len(corpus.questions)}")
    print(f"qrels pairs kept={sum(len(v) for v in corpus.qrels.values())} of {len(raw.qrels)}")
    print(f"model={MODEL_NAME} device={device} embedding_seconds={elapsed:.1f}")
    print(f"saved to {args.data_dir}")


if __name__ == "__main__":
    main()
