"""Robustness check of the main experiment over many seeds (#31).

Reads the committed runs and broken questions and writes the report table.

Usage:
    python experiments/robustness.py                      # from results/robustness/
    python experiments/robustness.py --runs R --from-data data   # after a new full run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mapper_rag_audit.experiment import load_runs_csv
from mapper_rag_audit.robustness import format_robustness_table, read_break_events

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "robustness"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=RESULTS / "runs.csv")
    parser.add_argument("--events", type=Path, default=RESULTS / "broken_questions.csv")
    parser.add_argument(
        "--from-data",
        type=Path,
        help="Data directory whose ingestions rewrite the broken questions file",
    )
    parser.add_argument(
        "--table", type=Path, default=REPO_ROOT / "report" / "figures" / "robustness_table.tex"
    )
    args = parser.parse_args()

    runs = load_runs_csv(args.runs)
    if args.from_data:
        with args.events.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["trap_type", "seed", "question_id"])
            writer.writerows(read_break_events(args.from_data, runs))
    with args.events.open(encoding="utf-8") as f:
        events = [
            (row["trap_type"], int(row["seed"]), row["question_id"]) for row in csv.DictReader(f)
        ]

    table = format_robustness_table(runs, events)
    args.table.write_text(table, encoding="utf-8", newline="\n")
    print(table)


if __name__ == "__main__":
    main()
