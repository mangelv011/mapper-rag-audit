"""Measures for the stability study (#8): how much the selection of Mapper depends on its settings.

One run is one Mapper setting applied to one ingestion: the ranked list of questions the selector
returns, plus the ground truth of that ingestion. `compare_to_default` scores every run at each
budget against the run of the default setting on the same ingestion, and `summarize` pools the
scores per setting and budget.

Per run and budget:
- detected: broken questions inside the selection (#3 ground truth, #7 metric).
- targets_hit: target questions inside the selection. Each trap chunk is written against one
  target question, so a target is a question the ingestion was aimed at, broken or not. An
  ingestion of 20 chunks breaks 0 to 2 questions on SciFact, too few to compare settings with
  alone; the 20 targets give a denser signal of whether the selection lands where the ingestion
  did.
- jaccard: Jaccard index between this selection and the selection of the default setting at
  the same budget on the same ingestion. Two empty selections count as equal (1.0).
- region_jaccard: the same index on the whole candidate sets (every question with a correct
  chunk in a marked node), so it does not depend on the budget. The ranking breaks the many
  ties among candidates by question id, so a small change in the candidates can swap which
  tied questions fit in the budget; this index separates that from a change of the marked
  region itself.

The rankings must therefore be complete (every candidate), not cut at a budget.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class Run:
    setting: Hashable  # e.g. a MapperConfig
    ingestion: Hashable  # e.g. (trap_type, seed)
    ranking: Sequence[str]  # every candidate question id, best first
    broken: frozenset[str]
    targets: frozenset[str]


def compare_to_default(
    runs: Sequence[Run], default: Hashable, budgets: Sequence[int]
) -> list[dict]:
    """One row per run and budget, scored against the default setting on the same ingestion."""
    base = {r.ingestion: r.ranking for r in runs if r.setting == default}
    missing = {r.ingestion for r in runs} - base.keys()
    if missing:
        raise ValueError(
            f"no run of the default setting for ingestions {sorted(map(str, missing))}"
        )
    rows = []
    for run in runs:
        for budget in budgets:
            selected = set(run.ranking[:budget])
            rows.append(
                {
                    "setting": run.setting,
                    "ingestion": run.ingestion,
                    "budget": budget,
                    "candidates": len(run.ranking),
                    "selected": len(selected),
                    "broken": len(run.broken),
                    "detected": len(selected & run.broken),
                    "targets": len(run.targets),
                    "targets_hit": len(selected & run.targets),
                    "jaccard": jaccard(selected, base[run.ingestion][:budget]),
                    "region_jaccard": jaccard(run.ranking, base[run.ingestion]),
                }
            )
    return rows


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def summarize(rows: Sequence[dict]) -> list[dict]:
    """Pool the rows per (setting, budget).

    Detection and target rates are pooled (hits summed over the ingestions, divided by the
    summed totals), because most ingestions break 0 to 2 questions and a per-ingestion rate is
    undefined at 0 and jumps between 0, 0.5 and 1. The Jaccard indices and the number of
    candidates are averaged over the ingestions, with the sample standard deviation.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["setting"], row["budget"]), []).append(row)
    out = []
    for (setting, budget), group in groups.items():
        broken = sum(r["broken"] for r in group)
        detected = sum(r["detected"] for r in group)
        targets = sum(r["targets"] for r in group)
        hit = sum(r["targets_hit"] for r in group)
        mean, std = _mean_std([r["jaccard"] for r in group])
        region_mean, region_std = _mean_std([r["region_jaccard"] for r in group])
        out.append(
            {
                "setting": setting,
                "budget": budget,
                "ingestions": len(group),
                "broken": broken,
                "detected": detected,
                "detection_rate": detected / broken if broken else math.nan,
                "targets": targets,
                "targets_hit": hit,
                "target_rate": hit / targets if targets else math.nan,
                "jaccard_mean": mean,
                "jaccard_std": std,
                "region_jaccard_mean": region_mean,
                "region_jaccard_std": region_std,
                "candidates_mean": sum(r["candidates"] for r in group) / len(group),
            }
        )
    return out
