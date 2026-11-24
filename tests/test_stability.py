import math

import pytest

from mapper_rag_audit.stability import Run, compare_to_default, jaccard, summarize


def test_jaccard():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard(["a"], ["a"]) == 1.0
    assert jaccard(["a"], []) == 0.0
    assert jaccard([], []) == 1.0, "two empty selections agree"


def run(setting, ingestion, ranking, broken=(), targets=()):
    return Run(setting, ingestion, ranking, frozenset(broken), frozenset(targets))


def test_rows_score_each_budget_prefix_against_the_default_on_the_same_ingestion():
    runs = [
        run("default", 1, ["a", "b", "c", "d"], broken=["c"], targets=["a", "c", "x"]),
        run("other", 1, ["a", "c", "e", "f"], broken=["c"], targets=["a", "c", "x"]),
        run("default", 2, ["p", "q"], broken=[], targets=["q"]),
        run("other", 2, ["q", "p"], broken=[], targets=["q"]),
    ]
    rows = {
        (r["setting"], r["ingestion"], r["budget"]): r
        for r in compare_to_default(runs, "default", [1, 2])
    }
    assert len(rows) == 8
    assert rows[("default", 1, 2)]["jaccard"] == 1.0
    assert rows[("other", 1, 1)]["jaccard"] == 1.0  # {a} vs {a}
    assert rows[("other", 1, 2)]["jaccard"] == pytest.approx(1 / 3)  # {a,c} vs {a,b}
    assert rows[("other", 1, 2)]["detected"] == 1 and rows[("default", 1, 2)]["detected"] == 0
    assert rows[("other", 1, 2)]["targets_hit"] == 2 and rows[("other", 1, 2)]["targets"] == 3
    assert rows[("other", 2, 1)]["jaccard"] == 0.0  # {q} vs {p}
    assert rows[("other", 2, 2)]["jaccard"] == 1.0  # same set, other order
    assert rows[("other", 1, 1)]["region_jaccard"] == pytest.approx(2 / 6)  # {a,c} of {a..f}
    assert rows[("other", 1, 1)]["candidates"] == 4


def test_a_short_ranking_selects_fewer_than_the_budget():
    rows = compare_to_default([run("default", 1, ["a"])], "default", [5])
    assert rows[0]["selected"] == 1


def test_missing_default_run_raises():
    with pytest.raises(ValueError, match="default"):
        compare_to_default([run("default", 1, ["a"]), run("other", 2, ["a"])], "default", [1])


def test_summary_pools_the_rates_and_averages_the_jaccard():
    rows = [
        {
            "setting": "s",
            "budget": 5,
            "broken": 2,
            "detected": 1,
            "targets": 20,
            "targets_hit": 4,
            "jaccard": 0.5,
            "region_jaccard": 0.2,
            "candidates": 10,
        },
        {
            "setting": "s",
            "budget": 5,
            "broken": 0,
            "detected": 0,
            "targets": 20,
            "targets_hit": 6,
            "jaccard": 1.0,
            "region_jaccard": 0.4,
            "candidates": 20,
        },
        {
            "setting": "s",
            "budget": 9,
            "broken": 0,
            "detected": 0,
            "targets": 20,
            "targets_hit": 0,
            "jaccard": 1.0,
            "region_jaccard": 1.0,
            "candidates": 0,
        },
    ]
    by_budget = {s["budget"]: s for s in summarize(rows)}
    s5 = by_budget[5]
    assert s5["ingestions"] == 2
    assert s5["detection_rate"] == 0.5, (
        "1 of 2 broken, pooled; the ingestion with 0 broken adds nothing"
    )
    assert s5["target_rate"] == pytest.approx(10 / 40)
    assert s5["jaccard_mean"] == 0.75
    assert s5["jaccard_std"] == pytest.approx(math.sqrt(0.125))
    assert s5["region_jaccard_mean"] == pytest.approx(0.3)
    assert s5["candidates_mean"] == 15
    assert math.isnan(by_budget[9]["detection_rate"]), "no broken question: the rate is undefined"
    assert by_budget[9]["jaccard_std"] == 0.0
