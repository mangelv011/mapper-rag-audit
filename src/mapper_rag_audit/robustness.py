"""Pooled detection rates over many seeds, with bootstrap intervals (#31)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mapper_rag_audit.experiment import DISPLAY_NAMES, ExperimentRun
from mapper_rag_audit.traps import default_ingestion_dir

DRAWS = 10_000
SELECTORS = ("mapper", "mapper_no_edges", "nearest_neighbours", "kmeans", "random")
COMPARISONS = {
    "nearest_neighbours": "Nearest neighbours",
    "kmeans": "k-means",
    "random": "Random",
    "mapper_no_edges": "No-edges variant (post hoc)",
}


def _per_seed(runs: Sequence[ExperimentRun], selector: str, budget: int, seeds: list[int]):
    """Detected and broken counts of each seed, summed over its trap types."""
    detected, broken = np.zeros(len(seeds)), np.zeros(len(seeds))
    index = {seed: i for i, seed in enumerate(seeds)}
    for r in runs:
        if r.selector == selector and r.budget == budget:
            detected[index[r.seed]] += r.detected
            broken[index[r.seed]] += r.broken
    return detected, broken


def difference_interval(
    runs: Sequence[ExperimentRun], selector: str, other: str, budget: int
) -> tuple[float, float, float]:
    """Pooled detection rate of ``selector`` minus ``other``, in percentage points.

    Returns the difference and its 95 % percentile bootstrap interval. Whole seeds are
    resampled because the trap types of a seed share their target questions.
    """
    seeds = sorted({r.seed for r in runs})
    detected, broken = _per_seed(runs, selector, budget, seeds)
    difference = detected - _per_seed(runs, other, budget, seeds)[0]
    draws = np.random.default_rng(0).integers(0, len(seeds), size=(DRAWS, len(seeds)))
    totals = broken[draws].sum(axis=1)
    sampled = 100 * difference[draws].sum(axis=1)[totals > 0] / totals[totals > 0]
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(100 * difference.sum() / broken.sum()), float(low), float(high)


def read_break_events(
    data_dir: Path, runs: Sequence[ExperimentRun], n_traps: int = 20
) -> list[tuple[str, int, str]]:
    """Broken questions of every ingestion in ``runs`` as (trap type, seed, question id)."""
    events = []
    for trap_type, seed in sorted({(r.trap_type, r.seed) for r in runs}):
        path = default_ingestion_dir(data_dir, trap_type, n_traps, seed) / "broken_questions.json"
        questions = json.loads(path.read_text(encoding="utf-8"))
        events += [(trap_type, seed, str(question)) for question in questions]
    return events


def format_robustness_table(
    runs: Sequence[ExperimentRun], events: Sequence[tuple[str, int, str]]
) -> str:
    """LaTeX table of pooled detection and of Mapper minus each other selector.

    ``events`` lists every broken question as (trap type, seed, question id).
    """
    budgets = sorted({r.budget for r in runs})
    seeds = sorted({r.seed for r in runs})
    in_runs = sum(r.broken for r in runs if r.selector == "mapper" and r.budget == budgets[0])
    if len(events) != in_runs:
        raise ValueError(f"{len(events)} break events listed, {in_runs} in the runs")

    def pooled(selector: str, budget: int) -> str:
        detected, broken = _per_seed(runs, selector, budget, seeds)
        found, total = int(detected.sum()), int(broken.sum())
        return f"{found}/{total} ({100 * found / total:.1f})"

    def interval(other: str, budget: int) -> tuple[str, str]:
        point, low, high = difference_interval(runs, "mapper", other, budget)
        return f"${point:+.1f}$", f"[${low:+.1f}$, ${high:+.1f}$]"

    def rows(names: dict[str, str], cell) -> list[str]:
        return [
            f"    {name} & " + " & ".join(cell(key, b) for b in budgets) + r" \\"
            for key, name in names.items()
        ]

    def interval_rows() -> list[str]:
        # The interval goes on its own line so the table fits the page width.
        lines = []
        for other, name in COMPARISONS.items():
            points, bounds = zip(*(interval(other, b) for b in budgets))
            lines.append(f"    {name} & " + " & ".join(points) + r" \\")
            lines.append("    & " + " & ".join(bounds) + r" \\")
        return lines

    def group(title: str) -> list[str]:
        columns = len(budgets) + 1
        return [r"    \hline", rf"    \multicolumn{{{columns}}}{{l}}{{{title}}} \\"]

    questions = Counter(question for _, _, question in events)
    top = sum(count for _, count in questions.most_common(3))
    cases = {(seed, question) for _, seed, question in events}
    caption = (
        f"{len(seeds)} seeds, {len(events)} break events, {len(cases)} distinct (seed, question) "
        f"cases and {len(questions)} distinct questions, the same for every selector and budget; "
        f"the three most frequent questions account for {top} of the events. Intervals are "
        f"percentile bootstrap over seeds ({DRAWS:,} draws)."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l" + "r" * len(budgets) + "}",
        r"    \hline",
        "    Selector & " + " & ".join(f"Budget {b}" for b in budgets) + r" \\",
        *group(r"Broken questions found out of all break events (\%)"),
        *rows({s: DISPLAY_NAMES[s] for s in SELECTORS}, pooled),
        *group(r"Mapper (all rules) minus, in points [95\% interval]"),
        *interval_rows(),
        r"    \hline",
        r"  \end{tabular}",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:robustness}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
