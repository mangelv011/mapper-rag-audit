"""Generate trap documents and compute ground truth of broken questions.

Usage:
    python experiments/make_ingestions.py --type near_copy --n 20 --seed 42
    python experiments/make_ingestions.py --type contradiction --n 20 --seed 42
    python experiments/make_ingestions.py --type mix --n 20 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mapper_rag_audit.data import load_corpus
from mapper_rag_audit.embedding import SentenceTransformerEmbedder, detect_device
from mapper_rag_audit.retrieval import DEFAULT_K, failing_questions
from mapper_rag_audit.traps import (
    TRAP_TYPES,
    build_ingestion,
    default_ingestion_dir,
    save_ingestion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type",
        dest="trap_type",
        type=str,
        required=True,
        choices=TRAP_TYPES,
        help="Type of trap document to generate (near_copy, contradiction, mix)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of trap chunks to generate and inject (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and mutations (default: 42)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k threshold for retrieval failure (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to directory containing processed corpus (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory to save trap files (default: data/ingestions/{type}_n{n}_s{seed})",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    device = detect_device()
    embedder = SentenceTransformerEmbedder.load(device)
    tokenizer = embedder.tokenizer()

    ingestion = build_ingestion(
        before=corpus,
        trap_type=args.trap_type,  # type: ignore[arg-type]
        n=args.n,
        seed=args.seed,
        embedder=embedder,
        tokenizer=tokenizer,
        k=args.k,
    )

    out_dir = args.output_dir or default_ingestion_dir(
        args.data_dir, args.trap_type, args.n, args.seed
    )
    save_ingestion(ingestion, out_dir)

    all_qids = [q.question_id for q in corpus.questions]
    failing_before = failing_questions(corpus, all_qids, k=args.k)
    passing_before = len(all_qids) - len(failing_before)

    n_broken = len(ingestion.broken_questions)
    n_targets = len(ingestion.target_questions)
    broken_pct = (n_broken / passing_before * 100) if passing_before > 0 else 0.0

    print(
        f"Ingestion generated: type={args.trap_type} n={args.n} seed={args.seed} k={args.k}"
    )
    print(f"Targeted questions: {n_targets}")
    print(
        f"Broken questions: {n_broken} broken of {n_targets} targeted questions "
        f"({broken_pct:.2f}% of the {passing_before} that initially passed)"
    )
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
