"""Count the questions that already fail on the original corpus, for several values of k.

Run after prepare_data.py:  python experiments/baseline_failures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mapper_rag_audit.data import load_corpus
from mapper_rag_audit.retrieval import DEFAULT_K, failing_questions

KS = (1, 3, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    questions = [q.question_id for q in corpus.questions]
    n = len(questions)
    print(f"questions={n} chunks={len(corpus.chunks)}")
    print("k\tfailing\tpercent\tcan_break")
    for k in sorted({*KS, DEFAULT_K}):
        failing = len(failing_questions(corpus, questions, k))
        mark = "  <- default" if k == DEFAULT_K else ""
        print(f"{k}\t{failing}\t{100 * failing / n:.1f}%\t{n - failing}{mark}")


if __name__ == "__main__":
    main()
