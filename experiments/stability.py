"""Stability of the Mapper selection under changes of lens, intervals and overlap (#8).

Usage (from the repository root, after experiments/prepare_data.py):

    python experiments/stability.py               # ingestions, runs, tables and heatmap
    python experiments/stability.py --plot-only   # redraw tables and heatmap from the CSV

For each trap type and seed it loads the ingestion from data/ingestions/<type>_n20_s<seed>/, or
generates and saves it there first (as experiments/make_ingestions.py does), so #7 can reuse the
same ingestions. For each Mapper setting (2 lenses x 3 interval counts x 3 overlaps, average
linkage at the default threshold) it runs the Mapper selector (#5) on every ingestion, with the
cover fitted on the corpus before. Each run is scored at budgets of 5, 10 and 20 % of the
questions against the ground truth and against the default setting; see
mapper_rag_audit.stability for the measures.

Writes:
- results/stability/runs.csv: one row per setting, ingestion and budget.
- results/stability/summary.csv: pooled per setting and budget.
- results/stability/table.md and report/figures/stability_table.tex: the summary as tables.
- report/figures/stability_heatmap.pdf (and results/stability/heatmap.png): intervals x overlap
  per lens, at the 10 % budget, for detection rate, target rate and Jaccard with the default,
  and the Jaccard of the whole candidate sets (the marked region, independent of the budget).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from mapper_rag_audit.data import load_corpus
from mapper_rag_audit.graph_diff import MapperSelector
from mapper_rag_audit.mapper import MapperConfig, build_corpus_graph, fit_corpus
from mapper_rag_audit.stability import Run, compare_to_default, summarize
from mapper_rag_audit.traps import (
    TRAP_TYPES,
    build_ingestion,
    default_ingestion_dir,
    load_ingestion,
    save_ingestion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = MapperConfig()
LENSES = ("pca", "centroid")
INTERVALS = (10, 15, 20)
OVERLAPS = (0.3, 0.5, 0.7)
GRID = [
    replace(DEFAULT, lens=lens, n_intervals=n, overlap=o)
    for lens in LENSES
    for n in INTERVALS
    for o in OVERLAPS
]
BUDGET_FRACTIONS = (0.05, 0.10, 0.20)
HEATMAP_FRACTION = 0.10
N_TRAPS = 20
SETTING_FIELDS = ("lens", "n_intervals", "overlap")
# Sequential blue ramp, light (near 0) to dark (near 1).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def ensure_ingestions(data_dir: Path, seeds: list[int], device: str) -> list[Path]:
    before = None
    embedder = tokenizer = None
    dirs = []
    for trap_type in TRAP_TYPES:
        for seed in seeds:
            out = default_ingestion_dir(data_dir, trap_type, N_TRAPS, seed)
            if not (out / "manifest.json").exists():
                if embedder is None:
                    from mapper_rag_audit.embedding import (
                        SentenceTransformerEmbedder,
                        detect_device,
                    )

                    before = load_corpus(data_dir)
                    embedder = SentenceTransformerEmbedder.load(
                        detect_device() if device == "auto" else device
                    )
                    tokenizer = embedder.tokenizer()
                ingestion = build_ingestion(
                    before, trap_type, N_TRAPS, seed, embedder, tokenizer=tokenizer
                )
                save_ingestion(ingestion, out)
                print(f"generated {out.name}: {len(ingestion.broken_questions)} broken", flush=True)
            dirs.append(out)
    return dirs


_BEFORE = None
_INGESTIONS = None


def _init_worker(data_dir: Path, ingestion_dirs: list[Path]) -> None:
    global _BEFORE, _INGESTIONS
    from threadpoolctl import threadpool_limits

    # One thread per worker: the workers already run in parallel, and nested BLAS/OpenMP
    # threads in every worker made each setting several times slower.
    threadpool_limits(1)
    _BEFORE = load_corpus(data_dir)
    _INGESTIONS = [load_ingestion(d, _BEFORE) for d in ingestion_dirs]


def _run_setting(config: MapperConfig) -> list[Run]:
    """The full ranking of the Mapper selector with `config` on every ingestion."""
    start = time.perf_counter()
    before = _BEFORE
    params = fit_corpus(before, config)
    graph_before = build_corpus_graph(before, params)
    selector = MapperSelector(
        lambda corpus: params,
        lambda corpus, p: graph_before if corpus is before else build_corpus_graph(corpus, p),
    )
    questions = [q.question_id for q in before.questions]
    budget = len(questions)  # every candidate: budgets are prefixes, the set is the region
    runs = [
        Run(
            setting=config,
            ingestion=(ing.trap_type, ing.seed),
            ranking=selector.select(before, ing.after, questions, budget),
            broken=frozenset(ing.broken_questions),
            targets=frozenset(ing.target_questions),
        )
        for ing in _INGESTIONS
    ]
    print(
        f"{config.lens:8s} {config.n_intervals:2d} intervals {config.overlap:.1f} overlap: "
        f"{time.perf_counter() - start:5.1f} s",
        flush=True,
    )
    return runs


def _flatten(row: dict) -> dict:
    out = {f: getattr(row["setting"], f) for f in SETTING_FIELDS}
    if "ingestion" in row:
        out["trap_type"], out["seed"] = row["ingestion"]
    out.update({k: v for k, v in row.items() if k not in ("setting", "ingestion")})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def read_summary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key == "lens":
                continue
            row[key] = float(value) if "." in value or value == "nan" else int(value)
    return rows


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.0%}"


def write_tables(summary: list[dict], md_path: Path, tex_path: Path) -> None:
    budgets = sorted({r["budget"] for r in summary})
    rows = {}
    for r in summary:
        rows.setdefault((r["lens"], r["n_intervals"], r["overlap"]), {})[r["budget"]] = r
    keys = sorted(rows, key=lambda k: (LENSES.index(k[0]), k[1], k[2]))
    default_key = (DEFAULT.lens, DEFAULT.n_intervals, DEFAULT.overlap)
    any_row = next(iter(rows.values()))[budgets[0]]

    n_questions = round(budgets[0] / BUDGET_FRACTIONS[0])
    header = (
        f"Pooled over {any_row['ingestions']} ingestions ({any_row['broken']} broken questions, "
        f"{any_row['targets']} target questions). Candidates: mean size of the candidate set. "
        f"Region: Jaccard of the candidate set with the default's. Budget cells: detection / "
        f"target rate / Jaccard of the selection with the default's (mean ± std). Budgets in "
        f"questions out of {n_questions}."
    )
    md = [
        header,
        "",
        "| lens | intervals | overlap | candidates | region | "
        + " | ".join(f"budget {b}" for b in budgets)
        + " |",
        "|---|---:|---:|---:|---|" + "---|" * len(budgets),
    ]
    for key in keys:
        cells = [
            f"{_pct(c['detection_rate'])} / {_pct(c['target_rate'])} / "
            f"{c['jaccard_mean']:.2f} ± {c['jaccard_std']:.2f}"
            for c in (rows[key][b] for b in budgets)
        ]
        mark = " (default)" if key == default_key else ""
        c0 = rows[key][budgets[0]]
        region = f"{c0['region_jaccard_mean']:.2f} ± {c0['region_jaccard_std']:.2f}"
        md.append(
            f"| {key[0]}{mark} | {key[1]} | {key[2]} | {c0['candidates_mean']:.0f} | {region} | "
            + " | ".join(cells)
            + " |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        "% Generated by experiments/stability.py; do not edit by hand.",
        "\\begin{tabular}{lrrrr" + "rrr" * len(budgets) + "}",
        "\\hline",
        " & & & & & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{budget {b}}}" for b in budgets)
        + " \\\\",
        "Lens & Int. & Ov. & Cand. & Region & "
        + " & ".join(["Det. & Targ. & Jacc."] * len(budgets))
        + " \\\\",
        "\\hline",
    ]
    for key in keys:
        cells = []
        for b in budgets:
            c = rows[key][b]
            cells += [
                _pct(c["detection_rate"]).replace("%", "\\%"),
                _pct(c["target_rate"]).replace("%", "\\%"),
                f"{c['jaccard_mean']:.2f}",
            ]
        lens = f"\\textbf{{{key[0]}}}" if key == default_key else key[0]
        c0 = rows[key][budgets[0]]
        head = f"{lens} & {key[1]} & {key[2]} & {c0['candidates_mean']:.0f} & "
        tex.append(head + f"{c0['region_jaccard_mean']:.2f} & " + " & ".join(cells) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def plot_heatmap(summary: list[dict], budget: int, paths: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    metrics = [
        ("detection_rate", "Detection rate"),
        ("target_rate", "Target questions selected"),
        ("jaccard_mean", "Jaccard with the default"),
        ("region_jaccard_mean", "Region Jaccard (all candidates)"),
    ]
    by_key = {
        (r["lens"], r["n_intervals"], r["overlap"]): r for r in summary if r["budget"] == budget
    }
    fig, axes = plt.subplots(
        len(LENSES), len(metrics), figsize=(12.5, 5.6), constrained_layout=True
    )
    for i, lens in enumerate(LENSES):
        for j, (metric, title) in enumerate(metrics):
            ax = axes[i, j]
            grid = np.array(
                [[by_key[(lens, n, o)][metric] for o in OVERLAPS] for n in INTERVALS], dtype=float
            )
            image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
            for a in range(len(INTERVALS)):
                for b in range(len(OVERLAPS)):
                    value = grid[a, b]
                    ax.text(
                        b,
                        a,
                        "n/a" if math.isnan(value) else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if value > 0.55 else "#1f1f1f",
                    )
            if lens == DEFAULT.lens:
                a = INTERVALS.index(DEFAULT.n_intervals)
                b = OVERLAPS.index(DEFAULT.overlap)
                ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, lw=2, ec="#1f1f1f"))
            ax.set_xticks(range(len(OVERLAPS)), [f"{o:.1f}" for o in OVERLAPS])
            ax.set_yticks(range(len(INTERVALS)), [str(n) for n in INTERVALS])
            ax.tick_params(length=0, labelsize=8, colors="#555555")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if i == len(LENSES) - 1:
                ax.set_xlabel("overlap", fontsize=9, color="#555555")
            if j == 0:
                ax.set_ylabel(f"{lens} lens\nintervals", fontsize=9, color="#555555")
    fig.colorbar(image, ax=axes, shrink=0.8, aspect=30).ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Mapper selection at a budget of {budget} questions (outlined: default setting)",
        fontsize=10,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "stability")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--jobs", type=int, default=min(9, os.cpu_count() or 1))
    parser.add_argument(
        "--device", default="auto", help="device to embed new ingestions on (auto, cpu, cuda, mps)"
    )
    parser.add_argument("--plot-only", action="store_true", help="redraw from summary.csv")
    args = parser.parse_args()
    out = args.out_dir
    figures = REPO_ROOT / "report" / "figures"

    if not args.plot_only:
        start = time.perf_counter()
        dirs = ensure_ingestions(args.data_dir, args.seeds, args.device)
        with ProcessPoolExecutor(
            args.jobs, initializer=_init_worker, initargs=(args.data_dir, dirs)
        ) as pool:
            runs = [r for runs in pool.map(_run_setting, GRID) for r in runs]
        n_questions = len(load_corpus(args.data_dir).questions)
        budgets = [round(f * n_questions) for f in BUDGET_FRACTIONS]
        rows = compare_to_default(runs, DEFAULT, budgets)
        write_csv(out / "runs.csv", [_flatten(r) for r in rows])
        write_csv(out / "summary.csv", [_flatten(r) for r in summarize(rows)])
        print(f"runtime: {time.perf_counter() - start:.0f} s for {len(runs)} runs")

    summary = read_summary(out / "summary.csv")
    budgets = sorted({r["budget"] for r in summary})
    heatmap_budget = budgets[BUDGET_FRACTIONS.index(HEATMAP_FRACTION)]
    write_tables(summary, out / "table.md", figures / "stability_table.tex")
    plot_heatmap(summary, heatmap_budget, [figures / "stability_heatmap.pdf", out / "heatmap.png"])
    print((out / "table.md").read_text(encoding="utf-8"))
    print(f"wrote {out} and {figures / 'stability_heatmap.pdf'}")


if __name__ == "__main__":
    main()
