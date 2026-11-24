"""Tests for trap-document generator and ground truth of broken questions."""

import random

import numpy as np
import pytest

from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.traps import (
    TRAP_TYPES,
    apply_ingestion,
    build_ingestion,
    generate_traps,
    load_ingestion,
    mutate_contradiction,
    mutate_mix,
    mutate_near_copy,
    save_ingestion,
)


@pytest.fixture
def synthetic_corpus():
    """Corpus with several chunks and passing questions with known qrels."""
    chunks = [
        Chunk(
            "doc1#0",
            "doc1",
            "Study of Alpha\nAlpha expression increased by 45% in human patients, showing that alpha is required for survival.",
        ),
        Chunk(
            "doc2#0",
            "doc2",
            "Analysis of Beta\nBeta levels were elevated in 2018. Results demonstrate that beta protects against cellular apoptosis.",
        ),
        Chunk(
            "doc3#0",
            "doc3",
            "Gamma and Delta\nGamma activation promotes inflammation. In vivo trials showed 3-fold higher response.",
        ),
        Chunk(
            "doc4#0",
            "doc4",
            "Unrelated Topic Zeta\nPlanetary astronomy studies show distant orbital trajectories around stellar clusters.",
        ),
    ]
    # Simple embeddings (4 chunks, 3 dimensions)
    embeddings = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.1, 1.0, 0.0],
            [0.9, 0.2, 0.1],
            [0.0, 0.0, 1.0],  # doc4 is distant in 3rd dimension
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    questions = [
        Question("q1", "What does alpha expression show?"),
        Question("q2", "Does beta protect against apoptosis?"),
        Question("q3", "How does gamma affect inflammation?"),
    ]
    q_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    q_embeddings /= np.linalg.norm(q_embeddings, axis=1, keepdims=True)

    qrels = {
        "q1": ["doc1#0"],
        "q2": ["doc2#0"],
        "q3": ["doc3#0"],
    }
    return Corpus(
        chunks=chunks,
        embeddings=embeddings,
        questions=questions,
        question_embeddings=q_embeddings,
        qrels=qrels,
    )


def test_mutate_near_copy_changes_facts():
    rng = random.Random(42)
    original = "Title\nExpression increased by 50% in patients, showing 3-fold higher activity."
    mutated, meta = mutate_near_copy(original, rng)
    assert mutated != original
    assert "Title" in mutated
    assert "decreased" in mutated or "lower" in mutated or "percent" in meta["changes"]


def test_mutate_contradiction_inverts_claims():
    rng = random.Random(42)
    original = "Title\nOur findings demonstrate that protein X is required for survival and protects against apoptosis."
    mutated, meta = mutate_contradiction(original, rng)
    assert mutated != original
    assert "Title" in mutated
    assert (
        "dispensable" in mutated
        or "fails to demonstrate" in mutated
        or "exacerbates" in mutated
        or "inversion" in meta["changes"]
        or "refut" in mutated
    )


def test_mutate_mix_joins_two_topics(fake_tokenizer):
    text_a = "Biology of Cells\nCell division requires kinase activation. Spindle poles assemble symmetrically."
    text_b = "Astronomy of Comets\nComet tails contain water ice. Sublimation occurs near the perihelion."
    mixed, _meta = mutate_mix(text_a, text_b, max_tokens=100, tokenizer=fake_tokenizer)
    assert "Biology of Cells and Astronomy of Comets" in mixed
    assert "Cell division" in mixed
    assert "tails contain" in mixed or "Sublimation" in mixed
    assert fake_tokenizer.count(mixed) <= 100


@pytest.mark.parametrize("trap_type", TRAP_TYPES)
def test_running_twice_with_same_seed_gives_identical_output(
    synthetic_corpus, fake_embedder, fake_tokenizer, tmp_path, trap_type
):
    first, second = tmp_path / "first", tmp_path / "second"
    for out_dir in (first, second):
        ingestion = build_ingestion(
            before=synthetic_corpus,
            trap_type=trap_type,
            n=3,
            seed=123,
            embedder=fake_embedder,
            tokenizer=fake_tokenizer,
        )
        assert len(ingestion.trap_chunks) == 3
        save_ingestion(ingestion, out_dir)

    for name in ("trap_chunks.json", "trap_embeddings.npy", "broken_questions.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_different_seeds_give_different_sampling(synthetic_corpus, fake_tokenizer):
    traps_a = generate_traps(synthetic_corpus, trap_type="near_copy", n=2, seed=1, tokenizer=fake_tokenizer)
    traps_b = generate_traps(synthetic_corpus, trap_type="near_copy", n=2, seed=999, tokenizer=fake_tokenizer)
    # The chunk ids contain the seed, so they must differ
    assert [t.chunk_id for t in traps_a] != [t.chunk_id for t in traps_b]


def test_apply_ingestion_adds_chunks_and_keeps_qrels(synthetic_corpus, fake_embedder):
    traps = generate_traps(synthetic_corpus, trap_type="near_copy", n=2, seed=42)
    trap_vecs = fake_embedder.embed_passages([t.text for t in traps])
    after = apply_ingestion(synthetic_corpus, traps, trap_vecs)

    assert len(after.chunks) == len(synthetic_corpus.chunks) + len(traps)
    assert len(after.embeddings) == len(synthetic_corpus.embeddings) + len(traps)
    assert after.qrels == synthetic_corpus.qrels
    assert after.questions == synthetic_corpus.questions


def test_build_ingestion_and_roundtrip_save_load(synthetic_corpus, fake_embedder, fake_tokenizer, tmp_path):
    out_dir = tmp_path / "ingestion_test"
    ingestion = build_ingestion(
        before=synthetic_corpus,
        trap_type="near_copy",
        n=2,
        seed=42,
        embedder=fake_embedder,
        tokenizer=fake_tokenizer,
        k=1,
    )
    save_ingestion(ingestion, out_dir)

    assert (out_dir / "trap_chunks.json").exists()
    assert (out_dir / "trap_embeddings.npy").exists()
    assert (out_dir / "broken_questions.json").exists()
    assert (out_dir / "manifest.json").exists()

    loaded = load_ingestion(out_dir, before=synthetic_corpus)
    assert loaded.trap_type == ingestion.trap_type
    assert loaded.n == ingestion.n
    assert loaded.seed == ingestion.seed
    assert loaded.broken_questions == ingestion.broken_questions
    assert len(loaded.trap_chunks) == len(ingestion.trap_chunks)
    np.testing.assert_allclose(loaded.trap_embeddings, ingestion.trap_embeddings)
    assert len(loaded.after.chunks) == len(ingestion.after.chunks)


def test_invalid_parameters_raise_errors(synthetic_corpus):
    with pytest.raises(ValueError, match="trap_type"):
        generate_traps(synthetic_corpus, trap_type="invalid", n=1, seed=42)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least 1"):
        generate_traps(synthetic_corpus, trap_type="near_copy", n=0, seed=42)
