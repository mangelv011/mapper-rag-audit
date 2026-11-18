"""Build, save and load the processed SciFact corpus."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str


@dataclass(frozen=True)
class RawBeir:
    documents: list[dict]
    queries: list[dict]
    qrels: list[tuple[str, str, int]]


@dataclass(frozen=True)
class Corpus:
    chunks: list[Chunk]
    embeddings: np.ndarray
    questions: list[Question]
    question_embeddings: np.ndarray
    qrels: dict[str, list[str]]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_beir(raw_dir: Path) -> RawBeir:
    with (raw_dir / "qrels" / "test.tsv").open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        qrels = [(r["query-id"], r["corpus-id"], int(r["score"])) for r in rows]
    return RawBeir(
        documents=_read_jsonl(raw_dir / "corpus.jsonl"),
        queries=_read_jsonl(raw_dir / "queries.jsonl"),
        qrels=qrels,
    )


def build_corpus(raw: RawBeir, embedder: Embedder) -> Corpus:
    chunks = [
        Chunk(d["_id"], d["_id"], f"{d['title']}\n{d['text']}" if d["title"] else d["text"])
        for d in raw.documents
    ]
    embeddings = embedder.embed_passages([c.text for c in chunks])

    present = {c.chunk_id for c in chunks}
    qrels: dict[str, list[str]] = {}
    for question_id, chunk_id, score in raw.qrels:
        if score > 0 and chunk_id in present:
            qrels.setdefault(question_id, []).append(chunk_id)
    questions = [Question(q["_id"], q["text"]) for q in raw.queries if q["_id"] in qrels]
    question_embeddings = embedder.embed_queries([q.text for q in questions])
    return Corpus(chunks, embeddings, questions, question_embeddings, qrels)


def _write_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_corpus(corpus: Corpus, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "embeddings.npy", corpus.embeddings)
    np.save(data_dir / "question_embeddings.npy", corpus.question_embeddings)
    _write_json(data_dir / "chunks.json", [vars(c) for c in corpus.chunks])
    _write_json(data_dir / "questions.json", [vars(q) for q in corpus.questions])
    _write_json(data_dir / "qrels.json", corpus.qrels)


def load_corpus(data_dir: Path) -> Corpus:
    return Corpus(
        chunks=[Chunk(**c) for c in _read_json(data_dir / "chunks.json")],
        embeddings=np.load(data_dir / "embeddings.npy"),
        questions=[Question(**q) for q in _read_json(data_dir / "questions.json")],
        question_embeddings=np.load(data_dir / "question_embeddings.npy"),
        qrels=_read_json(data_dir / "qrels.json"),
    )
