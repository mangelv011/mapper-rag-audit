"""Main experiment: four selectors x three trap types x three budgets (#7).

Runs Random, k-means, Nearest neighbours, Mapper (all rules), and Mapper (no edges)
on 30 ingestions (3 trap types x 10 seeds, 20 chunks each) at budgets 5 %, 10 %, 20 %
(15, 30, 60 questions).

Produces:
- results/main_experiment/runs.csv: raw scores for each run.
- results/main_experiment/summary.csv: aggregated mean ± std and pooled rates across seeds.
- results/main_experiment/table.md: markdown results table.
- report/figures/main_results_table.tex: LaTeX results table.
- report/figures/main_experiment.pdf and results/main_experiment/figure.png: 3-panel figure
  showing detection rate vs budget for each trap type.

Trap types use paired target-question samples within each seed.

Usage:
    python experiments/main_experiment.py              # run full experiment, table and plots
    python experiments/main_experiment.py --plot-only  # redraw table and plots from existing CSV
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mapper_rag_audit.baselines import (
    KMeansSelector,
    NearestNeighboursSelector,
    RandomSelector,
    _unit_rows,
)
from mapper_rag_audit.data import Corpus, load_corpus
from mapper_rag_audit.embedding import SentenceTransformerEmbedder, detect_device
from mapper_rag_audit.experiment import (
    ExperimentRun,
    SummaryRow,
    export_runs_csv,
    export_summary_csv,
    format_latex_table,
    format_markdown_table,
    rank_selector_with_scores,
    score_ranking,
    summarize_runs,
)
from mapper_rag_audit.graph_diff import MapperSelector, Rules
from mapper_rag_audit.mapper import (
    MapperConfig,
    build_corpus_graph,
    fit_corpus,
)
from mapper_rag_audit.traps import (
    TRAP_TYPES,
    Ingestion,
    TrapType,
    build_ingestion,
    default_ingestion_dir,
    load_ingestion,
    save_ingestion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = list(range(10))
DEFAULT_BUDGETS = [15, 30, 60]
DEFAULT_N_TRAPS = 20


def ensure_ingestions(
    data_dir: Path,
    trap_types: list[TrapType],
    seeds: list[int],
    n_traps: int = DEFAULT_N_TRAPS,
    device: str = "auto",
) -> dict[tuple[str, int], Ingestion]:
    """Ensure all required ingestions exist on disk and load them."""
    before: Corpus | None = None
    embedder: SentenceTransformerEmbedder | None = None
    tokenizer = None

    ingestions: dict[tuple[str, int], Ingestion] = {}
    for trap_type in trap_types:
        for seed in seeds:
            out_dir = default_ingestion_dir(data_dir, trap_type, n_traps, seed)
            manifest = out_dir / "manifest.json"
            if manifest.exists():
                if before is None:
                    before = load_corpus(data_dir)
                ingestion = load_ingestion(out_dir, before=before)
            else:
                if before is None:
                    before = load_corpus(data_dir)
                if embedder is None:
                    dev = detect_device() if device == "auto" else device
                    embedder = SentenceTransformerEmbedder.load(dev)
                    tokenizer = embedder.tokenizer()
                ingestion = build_ingestion(
                    before,
                    trap_type,
                    n_traps,
                    seed,
                    embedder,
                    tokenizer=tokenizer,
                    k=5,
                )
                save_ingestion(ingestion, out_dir)
                print(
                    f"Generated {out_dir.name}: {len(ingestion.broken_questions)} broken",
                    flush=True,
                )
            ingestions[(trap_type, seed)] = ingestion
    return ingestions


def plot_detection_rate_vs_budget(
    summary: list[SummaryRow],
    trap_types: list[str],
    budgets: list[int],
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Generate 3-panel figure: detection rate vs budget, one panel per trap type."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "mapper": {
            "label": "Mapper (all rules)",
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
        },
        "mapper_no_edges": {
            "label": "Mapper (no edges or neighbours)",
            "color": "#17becf",
            "marker": "s",
            "linestyle": "--",
            "linewidth": 1.75,
        },
        "nearest_neighbours": {
            "label": "Nearest neighbours",
            "color": "#ff7f0e",
            "marker": "^",
            "linestyle": "-",
            "linewidth": 1.75,
        },
        "kmeans": {
            "label": "k-means",
            "color": "#2ca02c",
            "marker": "D",
            "linestyle": "-",
            "linewidth": 1.75,
        },
        "random": {
            "label": "Random",
            "color": "#7f7f7f",
            "marker": "x",
            "linestyle": ":",
            "linewidth": 1.5,
        },
    }

    trap_titles = {
        "near_copy": "Near-copy traps",
        "contradiction": "Contradiction traps",
        "mix": "Mix traps",
    }

    fig, axes = plt.subplots(1, len(trap_types), figsize=(7.5, 3.2), sharey=True)
    if len(trap_types) == 1:
        axes = [axes]

    # Index summary by (trap_type, selector, budget)
    summary_map = {(r.trap_type, r.selector, r.budget): r for r in summary}

    for ax, trap_type in zip(axes, trap_types):
        ax.set_title(trap_titles.get(trap_type, trap_type), fontsize=10, fontweight="bold")
        ax.set_xlabel("Budget (questions)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(budgets)
        ax.set_xticklabels(budgets, fontsize=9)
        ax.set_ylim(-0.05, 1.05)

        for selector, style in styles.items():
            means = []
            stds = []
            for b in budgets:
                key = (trap_type, selector, b)
                if key in summary_map:
                    row = summary_map[key]
                    means.append(row.detection_rate_mean)
                    stds.append(row.detection_rate_std)
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))

            ax.errorbar(
                budgets,
                means,
                yerr=stds,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                capsize=3.5,
                capthick=1,
                markersize=6,
            )

    axes[0].set_ylabel("Detection rate (mean ± std)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_pdf} and {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "main_experiment"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=REPO_ROOT / "report" / "figures"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Seeds for ingestions"
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS, help="Budget counts"
    )
    parser.add_argument(
        "--trap-types",
        nargs="+",
        default=list(TRAP_TYPES),
        choices=list(TRAP_TYPES),
        help="Trap types to evaluate",
    )
    parser.add_argument(
        "--n-traps", type=int, default=DEFAULT_N_TRAPS, help="Number of traps per ingestion"
    )
    parser.add_argument(
        "--kmeans-n-init", type=int, default=10, help="n_init for KMeansSelector"
    )
    parser.add_argument(
        "--device", default="auto", help="device for embedding new ingestions (auto, cpu, etc.)"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures and tables from existing runs.csv without re-running",
    )
    args = parser.parse_args()

    runs_csv = args.results_dir / "runs.csv"
    summary_csv = args.results_dir / "summary.csv"
    table_md = args.results_dir / "table.md"
    table_tex = args.figures_dir / "main_results_table.tex"
    fig_pdf = args.figures_dir / "main_experiment.pdf"
    fig_png = args.results_dir / "figure.png"

    if args.plot_only:
        if not runs_csv.exists():
            raise FileNotFoundError(f"Cannot --plot-only: {runs_csv} not found")
        print(f"Reading existing runs from {runs_csv}...")
        runs = []
        with runs_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                det = float(row["detection_rate"]) if row["detection_rate"] else float("nan")
                runs.append(
                    ExperimentRun(
                        selector=row["selector"],
                        trap_type=row["trap_type"],
                        seed=int(row["seed"]),
                        budget=int(row["budget"]),
                        candidates=int(row["candidates"]),
                        selected=int(row["selected"]),
                        broken=int(row["broken"]),
                        detected=int(row["detected"]),
                        detection_rate=det,
                        targets=int(row["targets"]),
                        targets_hit=int(row["targets_hit"]),
                        target_rate=float(row["target_rate"]),
                        tie_block_size=int(row["tie_block_size"]),
                    )
                )
        summary = summarize_runs(runs)
        export_summary_csv(summary, summary_csv)
        table_md.write_text(format_markdown_table(summary), encoding="utf-8")
        table_tex.parent.mkdir(parents=True, exist_ok=True)
        table_tex.write_text(format_latex_table(summary), encoding="utf-8")
        plot_detection_rate_vs_budget(
            summary,
            list(dict.fromkeys(r.trap_type for r in runs)),
            sorted({r.budget for r in runs}),
            fig_pdf,
            fig_png,
        )
        print("Done regenerating plots and tables.")
        return

    start_time = time.perf_counter()
    print("=== Main Experiment: Four Selectors x Three Trap Types x Three Budgets ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")
    print(f"Budgets: {args.budgets}")
    print(f"Trap types: {args.trap_types}")
    print(f"Traps per ingestion: {args.n_traps}")
    print(f"KMeans n_init: {args.kmeans_n_init}")

    # 1. Ensure ingestions
    print("\n[1/4] Ensuring ingestions...")
    t0 = time.perf_counter()
    ingestions = ensure_ingestions(
        args.data_dir,
        args.trap_types,
        args.seeds,
        n_traps=args.n_traps,
        device=args.device,
    )
    print(f"All {len(ingestions)} ingestions ready in {time.perf_counter() - t0:.1f}s.")

    # 2. Prepare before-corpus and fitted Mapper graph
    print("\n[2/4] Building Mapper graph on baseline corpus...")
    t0 = time.perf_counter()
    corpus = load_corpus(args.data_dir)
    mapper_params = fit_corpus(corpus, MapperConfig())
    graph_before = build_corpus_graph(corpus, mapper_params)
    n_nodes = len(graph_before.nodes)
    print(
        f"Mapper baseline graph built in {time.perf_counter() - t0:.1f}s "
        f"({n_nodes} nodes, {len(graph_before.edges)} edges)."
    )

    # 3. Fit KMeans on baseline corpus once per seed
    print(f"\n[3/4] Fitting k-means on baseline corpus (k={n_nodes}, n_init={args.kmeans_n_init})...")
    kmeans_cache: dict[int, KMeans] = {}
    unit_before = _unit_rows(corpus.embeddings)
    for seed in args.seeds:
        t_km = time.perf_counter()
        km = KMeans(
            n_clusters=min(n_nodes, len(corpus.chunks)),
            n_init=args.kmeans_n_init,
            random_state=seed,
        )
        km.fit(unit_before)
        kmeans_cache[seed] = km
        print(f"  k-means seed={seed} fitted in {time.perf_counter() - t_km:.1f}s.")

    # 4. Evaluate selectors across all ingestions
    print("\n[4/4] Evaluating selectors across ingestions...")
    candidate_questions = [q.question_id for q in corpus.questions]

    mapper_full = MapperSelector(fit_corpus, build_corpus_graph, rules=Rules(), name="mapper")
    mapper_no_edges = MapperSelector(
        fit_corpus,
        build_corpus_graph,
        rules=Rules(new_edge=False, neighbours=False),
        name="mapper_no_edges",
    )
    nn_selector = NearestNeighboursSelector(m=10, name="nearest_neighbours")

    runs: list[ExperimentRun] = []
    total_ingestions = len(ingestions)

    for idx, ((trap_type, seed), ingestion) in enumerate(sorted(ingestions.items()), 1):
        t_ing = time.perf_counter()
        after = ingestion.after
        broken = ingestion.broken_questions
        targets = ingestion.target_questions

        # Build graph_after once for this ingestion (shared by mapper and mapper_no_edges)
        graph_after = build_corpus_graph(after, mapper_params)

        # Evaluate Mapper (full)
        ranked_m, scores_m = rank_selector_with_scores(
            mapper_full,
            corpus,
            after,
            candidate_questions,
            graph_before=graph_before,
            graph_after=graph_after,
        )
        runs.extend(
            score_ranking(
                "mapper", trap_type, seed, ranked_m, scores_m, broken, targets, args.budgets
            )
        )

        # Evaluate Mapper (no edges)
        ranked_mne, scores_mne = rank_selector_with_scores(
            mapper_no_edges,
            corpus,
            after,
            candidate_questions,
            graph_before=graph_before,
            graph_after=graph_after,
        )
        runs.extend(
            score_ranking(
                "mapper_no_edges",
                trap_type,
                seed,
                ranked_mne,
                scores_mne,
                broken,
                targets,
                args.budgets,
            )
        )

        # Evaluate Nearest neighbours
        ranked_nn, scores_nn = rank_selector_with_scores(
            nn_selector, corpus, after, candidate_questions
        )
        runs.extend(
            score_ranking(
                "nearest_neighbours",
                trap_type,
                seed,
                ranked_nn,
                scores_nn,
                broken,
                targets,
                args.budgets,
            )
        )

        # Evaluate k-means (using cached model for this seed)
        km_sel = KMeansSelector(
            n_clusters=n_nodes, seed=seed, n_init=args.kmeans_n_init, name="kmeans"
        )
        ranked_km, scores_km = rank_selector_with_scores(
            km_sel,
            corpus,
            after,
            candidate_questions,
            fitted_kmeans=kmeans_cache[seed],
        )
        runs.extend(
            score_ranking(
                "kmeans", trap_type, seed, ranked_km, scores_km, broken, targets, args.budgets
            )
        )

        # Evaluate Random
        rand_sel = RandomSelector(seed=seed, name="random")
        ranked_rnd, scores_rnd = rank_selector_with_scores(
            rand_sel, corpus, after, candidate_questions
        )
        runs.extend(
            score_ranking(
                "random",
                trap_type,
                seed,
                ranked_rnd,
                scores_rnd,
                broken,
                targets,
                args.budgets,
            )
        )

        print(
            f"  [{idx}/{total_ingestions}] {trap_type} s={seed}: "
            f"{len(broken)} broken, finished in {time.perf_counter() - t_ing:.2f}s."
        )

    # 5. Summarize and export
    print("\nExporting results...")
    summary = summarize_runs(runs)
    export_runs_csv(runs, runs_csv)
    export_summary_csv(summary, summary_csv)
    print(f"Saved {len(runs)} run records to {runs_csv}")
    print(f"Saved {len(summary)} summary records to {summary_csv}")

    md_content = format_markdown_table(summary)
    table_md.write_text(md_content, encoding="utf-8")
    print(f"Saved markdown table to {table_md}")

    tex_content = format_latex_table(summary)
    table_tex.parent.mkdir(parents=True, exist_ok=True)
    table_tex.write_text(tex_content, encoding="utf-8")
    print(f"Saved LaTeX table to {table_tex}")

    plot_detection_rate_vs_budget(summary, args.trap_types, args.budgets, fig_pdf, fig_png)

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal experiment runtime: {elapsed:.1f}s ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
