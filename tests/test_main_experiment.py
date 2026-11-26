"""Unit tests for the main experiment module (#7)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from mapper_rag_audit.baselines import (
    KMeansSelector,
    NearestNeighboursSelector,
    RandomSelector,
)
from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.experiment import (
    ExperimentRun,
    _mean_std,
    export_runs_csv,
    export_summary_csv,
    format_latex_table,
    format_markdown_table,
    rank_selector_with_scores,
    score_ranking,
    summarize_runs,
)
from mapper_rag_audit.graph_diff import MapperSelector, Rules
from mapper_rag_audit.mapper import MapperConfig, build_corpus_graph, fit_corpus


def test_mean_std_calculation():
    # Empty
    assert _mean_std([]) == (0.0, 0.0)
    # Single element
    assert _mean_std([5.0]) == (5.0, 0.0)
    # Multiple elements
    vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    mean, std = _mean_std(vals)
    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(2.138, abs=0.01)


def test_score_ranking_with_ties_and_regressions():
    ranking = ["q1", "q2", "q3", "q4", "q5"]
    scores = [(0, -2), (0, -1), (0, -1), (1, -1), (1, 0)]
    broken = ["q2", "q4"]
    targets = ["q1", "q2"]
    budgets = [2, 3, 5, 6]

    runs = score_ranking(
        selector_name="test_sel",
        trap_type="near_copy",
        seed=42,
        ranking=ranking,
        scores=scores,
        broken_questions=broken,
        target_questions=targets,
        budgets=budgets,
    )

    assert len(runs) == 4

    # Budget 2: selected ["q1", "q2"] -> detected q2 (1 of 2 = 0.5)
    r2 = runs[0]
    assert r2.budget == 2
    assert r2.selected == 2
    assert r2.detected == 1
    assert r2.detection_rate == pytest.approx(0.5)
    assert r2.targets_hit == 2  # q1, q2
    assert r2.target_rate == pytest.approx(1.0)
    # The 2nd element is q2 with score (0, -1). q3 also has (0, -1). So tie block is 2.
    assert r2.tie_block_size == 2

    # Budget 3: selected ["q1", "q2", "q3"] -> detected q2 (1 of 2 = 0.5)
    r3 = runs[1]
    assert r3.budget == 3
    assert r3.detected == 1
    assert r3.tie_block_size == 2

    # Budgets 5 and 6 include the entire ranking, so neither has a cut tie.
    r5 = runs[2]
    assert r5.budget == 5
    assert r5.detected == 2
    assert r5.detection_rate == pytest.approx(1.0)
    assert r5.tie_block_size == 0
    assert runs[3].budget == 6
    assert runs[3].tie_block_size == 0


def test_score_ranking_when_no_broken_questions():
    ranking = ["q1", "q2"]
    scores = [1, 2]
    runs = score_ranking(
        selector_name="test_sel",
        trap_type="mix",
        seed=0,
        ranking=ranking,
        scores=scores,
        broken_questions=[],
        target_questions=["q1"],
        budgets=[1],
    )
    assert len(runs) == 1
    assert math.isnan(runs[0].detection_rate)
    assert runs[0].broken == 0
    assert runs[0].detected == 0


def test_summarize_runs_groups_and_pools():
    runs = [
        # Seed 0: 2 broken, 1 detected (DR = 0.5)
        ExperimentRun("mapper", "near_copy", 0, 15, 20, 15, 2, 1, 0.5, 2, 2, 1.0, 10),
        # Seed 1: 2 broken, 2 detected (DR = 1.0)
        ExperimentRun("mapper", "near_copy", 1, 15, 20, 15, 2, 2, 1.0, 2, 1, 0.5, 12),
        # Seed 2: 0 broken, 0 detected (DR = nan)
        ExperimentRun(
            "mapper", "near_copy", 2, 15, 20, 15, 0, 0, float("nan"), 2, 1, 0.5, 8
        ),
        ExperimentRun(
            "mapper", "near_copy", 3, 15, 10, 10, 0, 0, float("nan"), 2, 1, 0.5, 0
        ),
    ]

    summary = summarize_runs(runs)
    assert len(summary) == 1
    s = summary[0]
    assert s.selector == "mapper"
    assert s.trap_type == "near_copy"
    assert s.budget == 15
    assert s.ingestions == 4
    assert s.ingestions_with_broken == 2
    assert s.total_broken == 4
    assert s.total_detected == 3
    # Pooled DR = 3 / 4 = 0.75
    assert s.pooled_detection_rate == pytest.approx(0.75)
    # Mean DR over seeds with broken questions: (0.5 + 1.0) / 2 = 0.75
    assert s.detection_rate_mean == pytest.approx(0.75)
    assert s.detection_rate_std == pytest.approx(0.3535, abs=0.01)
    assert s.tie_block_observations == 3
    # No-cut runs are excluded: (10 + 12 + 8) / 3 = 10.0
    assert s.tie_block_size_mean == pytest.approx(10.0)


def test_csv_and_table_export_roundtrip(tmp_path: Path):
    runs = [
        ExperimentRun("mapper", "near_copy", 0, 15, 20, 15, 2, 1, 0.5, 2, 2, 1.0, 10),
        ExperimentRun("random", "near_copy", 0, 15, 20, 15, 2, 0, 0.0, 2, 1, 0.5, 1),
    ]
    runs_file = tmp_path / "runs.csv"
    export_runs_csv(runs, runs_file)
    assert runs_file.exists()

    with runs_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["selector"] == "mapper"
        assert reader[0]["detection_rate"] == "0.5000"

    summary = summarize_runs(runs)
    summary_file = tmp_path / "summary.csv"
    export_summary_csv(summary, summary_file)
    assert summary_file.exists()

    md_table = format_markdown_table(summary)
    assert "Mapper (all rules)" in md_table
    assert "Random" in md_table
    assert "Cut Tie Block (mean ± std; n)" in md_table
    assert "not evidence that selectors are equivalent" in md_table

    tex_table = format_latex_table(summary)
    assert "\\begin{table}" in tex_table
    assert "Mapper" in tex_table
    assert "tie-block size" in tex_table
    assert "\\end{table}" in tex_table


def test_nearest_neighbour_ranking_preserves_close_cosine_order():
    chunks = [Chunk("c0#0", "c0", "old 0"), Chunk("c1#0", "c1", "old 1")]
    vectors = np.array([[0.5, np.sqrt(0.75)], [0.5000001, np.sqrt(1 - 0.5000001**2)]])
    questions = [Question("q0", "Question 0"), Question("q1", "Question 1")]
    before = Corpus(chunks, vectors, questions, vectors.copy(), {"q0": ["c0#0"], "q1": ["c1#0"]})
    after = Corpus(
        chunks + [Chunk("trap#0", "trap", "new")],
        np.vstack([vectors, [1.0, 0.0]]),
        questions,
        vectors.copy(),
        before.qrels,
    )
    selector = NearestNeighboursSelector(m=2)

    ranked, scores = rank_selector_with_scores(selector, before, after, ["q0", "q1"])

    assert ranked == selector.select(before, after, ["q0", "q1"], budget=2)
    assert ranked == ["q1", "q0"]
    assert scores[0] < scores[1]


def test_all_selectors_run_on_synthetic_corpus():
    """Verify that rank_selector_with_scores works across all 5 selectors on a synthetic corpus."""
    dim = 8
    rng = np.random.default_rng(123)

    # 12 chunks, 4 questions
    c_vecs = rng.normal(size=(12, dim)).astype(np.float32)
    c_vecs /= np.linalg.norm(c_vecs, axis=1, keepdims=True)
    chunks = [Chunk(f"c{i}#0", f"c{i}", f"Title {i}\nBody content {i}") for i in range(12)]

    q_vecs = c_vecs[:4] + 0.01 * rng.normal(size=(4, dim)).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    questions = [Question(f"q{i}", f"Question {i}") for i in range(4)]
    qrels = {f"q{i}": [f"c{i}#0"] for i in range(4)}

    before = Corpus(
        chunks=chunks,
        embeddings=c_vecs,
        questions=questions,
        question_embeddings=q_vecs,
        qrels=qrels,
    )

    # Ingestion: add 2 trap chunks close to c0
    trap_vecs = c_vecs[[0, 0]] + 0.02 * rng.normal(size=(2, dim)).astype(np.float32)
    trap_vecs /= np.linalg.norm(trap_vecs, axis=1, keepdims=True)
    trap_chunks = [
        Chunk("trap0#0", "trap0", "Trap title\nTrap body near c0"),
        Chunk("trap1#0", "trap1", "Trap title 2\nTrap body 2 near c0"),
    ]

    after = Corpus(
        chunks=chunks + trap_chunks,
        embeddings=np.vstack([c_vecs, trap_vecs]),
        questions=questions,
        question_embeddings=q_vecs,
        qrels=qrels,
    )

    candidate_ids = [q.question_id for q in questions]

    # Pre-fit mapper
    params = fit_corpus(before, MapperConfig(n_intervals=3, overlap=0.3))
    gb = build_corpus_graph(before, params)
    ga = build_corpus_graph(after, params)

    selectors = [
        RandomSelector(seed=0),
        KMeansSelector(n_clusters=len(gb.nodes), seed=0, n_init=1),
        NearestNeighboursSelector(m=3),
        MapperSelector(
            lambda c: fit_corpus(c, MapperConfig(n_intervals=3, overlap=0.3)),
            build_corpus_graph,
            rules=Rules(),
            name="mapper",
        ),
        MapperSelector(
            lambda c: fit_corpus(c, MapperConfig(n_intervals=3, overlap=0.3)),
            build_corpus_graph,
            rules=Rules(new_edge=False, neighbours=False),
            name="mapper_no_edges",
        ),
    ]

    for sel in selectors:
        ranked, scores = rank_selector_with_scores(
            sel,
            before,
            after,
            candidate_ids,
            graph_before=gb,
            graph_after=ga,
        )
        assert len(ranked) == len(scores)
        assert len(ranked) <= len(candidate_ids)
        assert set(ranked).issubset(set(candidate_ids))
        assert ranked == sel.select(before, after, candidate_ids, budget=len(ranked))
        if sel.name != "random":
            # q0 is target of the traps, so informed selectors should rank it first or second
            assert "q0" in ranked[:2]
