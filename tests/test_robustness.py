"""Tests for the 100-seed robustness analysis (#31)."""

from __future__ import annotations

import json
import math

import pytest

from mapper_rag_audit.experiment import ExperimentRun, export_runs_csv, load_runs_csv
from mapper_rag_audit.robustness import (
    difference_interval,
    format_robustness_table,
    read_break_events,
)


def _run(selector: str, seed: int, broken: int, detected: int, budget: int = 15, trap="mix"):
    return ExperimentRun(
        selector=selector,
        trap_type=trap,
        seed=seed,
        budget=budget,
        candidates=0,
        selected=0,
        broken=broken,
        detected=detected,
        detection_rate=detected / broken if broken else float("nan"),
        targets=20,
        targets_hit=0,
        target_rate=0.0,
        tie_block_size=0,
    )


def test_difference_interval_resamples_whole_seeds():
    # Per seed, a - b is 0 of 1 and -1 of 1: two draws give -100, -50 or 0 (1/4, 1/2, 1/4).
    runs = [_run("a", 0, 1, 1), _run("b", 0, 1, 1), _run("a", 1, 1, 0), _run("b", 1, 1, 1)]

    assert difference_interval(runs, "a", "b", 15) == pytest.approx((-50.0, -100.0, 0.0))


def test_difference_interval_pools_the_trap_types_of_a_seed():
    runs = [_run("a", 0, 1, 1, trap="mix"), _run("a", 0, 1, 0, trap="near_copy")]
    runs += [_run("b", 0, 1, 0, trap="mix"), _run("b", 0, 1, 0, trap="near_copy")]

    assert difference_interval(runs, "a", "b", 15) == pytest.approx((50.0, 50.0, 50.0))


def test_difference_interval_draws_as_many_seeds_as_there_are():
    # One of four seeds differs: the mean of four draws reaches -75 often enough, -100 too rarely.
    runs = [_run("b", seed, 1, 1) for seed in range(4)]
    runs += [_run("a", seed, 1, int(seed > 0)) for seed in range(4)]

    assert difference_interval(runs, "a", "b", 15) == pytest.approx((-25.0, -75.0, 0.0))


def test_difference_interval_skips_draws_without_break_events():
    runs = [_run("a", 0, 0, 0), _run("b", 0, 0, 0), _run("a", 1, 1, 0), _run("b", 1, 1, 1)]

    assert difference_interval(runs, "a", "b", 15) == pytest.approx((-100.0, -100.0, -100.0))


def test_latex_table_reports_counts_intervals_and_question_concentration():
    selectors = ("mapper", "mapper_no_edges", "nearest_neighbours", "kmeans", "random")
    # Broken questions found out of 2, in seed 0 and seed 1.
    found = {
        "mapper": (2, 0),
        "mapper_no_edges": (2, 0),
        "nearest_neighbours": (2, 2),
        "kmeans": (2, 2),
        "random": (0, 0),
    }
    runs = [
        _run(s, seed, 2, found[s][seed], budget=b)
        for s in selectors
        for seed in (0, 1)
        for b in (15, 30)
    ]
    events = [("mix", 0, "7"), ("mix", 0, "8"), ("near_copy", 0, "7"), ("mix", 1, "7")]

    table = format_robustness_table(runs, events)

    assert r"Mapper (all rules) & 2/4 (50.0) & 2/4 (50.0) \\" in table
    assert r"Nearest neighbours & 4/4 (100.0) & 4/4 (100.0) \\" in table
    assert r"Nearest neighbours & $-50.0$ & $-50.0$ \\" in table
    assert r"& [$-100.0$, $+0.0$] & [$-100.0$, $+0.0$] \\" in table
    assert r"Random & $+50.0$ & $+50.0$ \\" in table
    assert r"& [$+0.0$, $+100.0$] & [$+0.0$, $+100.0$] \\" in table
    assert r"No-edges variant (post hoc) & $+0.0$ & $+0.0$ \\" in table
    assert "2 seeds, 4 break events, 3 distinct (seed, question) cases and 2 distinct questions" in table
    assert "the three most frequent questions account for 4 of the events" in table
    assert r"\label{tab:robustness}" in table


def test_latex_table_rejects_events_that_do_not_match_the_runs():
    selectors = ("mapper", "mapper_no_edges", "nearest_neighbours", "kmeans", "random")
    runs = [_run(s, 0, 2, 1) for s in selectors]

    with pytest.raises(ValueError, match="1 break events listed, 2 in the runs"):
        format_robustness_table(runs, [("mix", 0, "7")])


def test_runs_csv_loads_back_with_nan_for_ingestions_without_breaks(tmp_path):
    path = tmp_path / "runs.csv"
    export_runs_csv([_run("a", 3, 2, 1), _run("a", 4, 0, 0)], path)

    first, second = load_runs_csv(path)

    assert first == _run("a", 3, 2, 1)
    assert (second.seed, second.broken) == (4, 0)
    assert math.isnan(second.detection_rate)


def test_break_events_come_from_the_ingestions_named_in_the_runs(tmp_path):
    for name, broken in (("mix_n20_s3", ["9", "5"]), ("mix_n20_s4", []), ("mix_n20_s5", ["1"])):
        folder = tmp_path / "ingestions" / name
        folder.mkdir(parents=True)
        (folder / "broken_questions.json").write_text(json.dumps(broken), encoding="utf-8")
    runs = [_run("a", 3, 2, 1), _run("b", 3, 2, 1), _run("a", 4, 0, 0)]

    assert read_break_events(tmp_path, runs) == [("mix", 3, "9"), ("mix", 3, "5")]
