# mapper-rag-audit

Auditing and maintaining RAG knowledge bases with Mapper.

## Idea

A tool that analyzes how the knowledge in a RAG system is organized and helps detect problems when documents are added or modified.

Mapper is used to build a map of the documents: which groups of information exist and how they relate to each other. After a new ingestion, the map shows which areas of the RAG changed and which questions should be tested again to check that retrieval still works.

## Research question

Does Mapper detect regressions with fewer tests than traditional methods such as clustering or random sampling?

## Goals

- Study Mapper as a technique that applies topology to data analysis.
- Understand its mathematical foundations in depth, and how they improve on earlier approaches.
- Apply it to a concrete use case, with real data where possible, run experiments and analyze the results.
- Identify its advantages and limitations.
- Explain the ideas clearly, including the mathematics behind them.

## Outputs

- **Report**: the underlying mathematical theory, the technique, the use case and its results, conclusions, and references (both information sources and data sources).
- **Code** used to run the experiments, plus the datasets, or links to their sources when they are large.
- **Short talk** (about 7 minutes): mathematical foundations, main ideas of the algorithm, experimental results and conclusions.

## Team

- [@mangelv011](https://github.com/mangelv011)
- [@santimartinez1717](https://github.com/santimartinez1717)
- [@ndelval](https://github.com/ndelval)
- [@gvidalquadras](https://github.com/gvidalquadras)
- [@Victoriaguilllen](https://github.com/Victoriaguilllen)

## Data

The experiments use [SciFact](https://github.com/allenai/scifact) in the [BEIR](https://github.com/beir-cellar/beir) format, test split. `experiments/prepare_data.py` downloads it, so never download it by hand.

| Item | Count |
|---|---|
| Documents (abstracts) in the corpus | 5,183 |
| Chunks | 5,658 |
| Documents split into more than one chunk | 455 |
| Largest chunk, in tokens (limit 512) | 512 |
| Questions in `queries.jsonl` (train and test) | 1,109 |
| Test questions (with test qrels) | 300 |
| Question to document pairs (test qrels) | 339 |
| Question to correct chunk pairs | 397 |
| Test questions without a correct chunk in the corpus | 0 |

Counts were measured on the downloaded files. 232 test questions have one correct chunk, the other 68 have two to nine, because a relevant document can be split.

- **Download**: `scifact.zip` from the BEIR mirror (2,816,079 bytes). The script checks its SHA-256 against a pinned value and refuses archive members that would leave `data/raw/`.
- **Licence**: stated in SciFact's [LICENSE.md](https://github.com/allenai/scifact/blob/master/LICENSE.md). The abstracts in `corpus.jsonl` are ODC-By 1.0 (from S2ORC). The claims and evidence annotations are CC BY 4.0. BEIR redistributes the data and grants no licence of its own.
- **Chunking rule**: a chunk is the title, a line break and a run of whole abstract sentences. Its token count, special tokens included, never exceeds the model's `max_seq_length` of 512, so no chunk is truncated. An abstract that fits stays one chunk. A longer one is cut into the smallest number of chunks that fit, with sentences packed so the chunks are roughly equal in size. A single sentence that is too long is cut on token boundaries. The body of every chunk is an exact slice of the original abstract (same case, accents and spacing), and the bodies of one document follow it in order. Sentences end at `.`, `!` or `?` followed by a space and a capital letter, except after `e.g.`, `i.e.`, `vs.` and `et al.`. Chunk ids are `<document id>#<n>` and the chunk to document map is stored in `chunks.json`. A question's correct chunks are all chunks of its relevant documents.
- **Embedding model**: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) through sentence-transformers, pinned to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions, L2-normalised float32. Questions get the model's query instruction ("Represent this sentence for searching relevant passages: "), chunks do not. The script uses CUDA if available, then Apple MPS, then CPU, and prints the device and the embedding time. On a laptop CPU the embedding calls took about 9 minutes (552 s); chunking is not included in that figure.

### Run it

Use Python 3.11 or newer. macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/prepare_data.py
pytest
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python experiments\prepare_data.py
pytest
```

The pinned PyTorch publishes no wheel for Intel Macs, so macOS needs Apple silicon.

### Files in `data/`

| File | Content |
|---|---|
| `raw/` | the unpacked download |
| `chunks.json` | chunk id, document id and text, in `embeddings.npy` row order |
| `embeddings.npy` | one row per chunk |
| `questions.json` | question id and text, in `question_embeddings.npy` row order |
| `question_embeddings.npy` | one row per question |
| `qrels.json` | question id to its list of correct chunk ids |

Load everything with `mapper_rag_audit.data.load_corpus(Path("data"))`; it takes under a second. A question whose correct documents are all missing from the corpus is dropped, and a pair that points to a missing document is ignored. SciFact has neither case.

## Retrieval and failing questions

Retrieval ranks every chunk by cosine similarity between its embedding and the question's embedding (`mapper_rag_audit.retrieval.top_k`). Ties go to the chunk that comes first in the corpus.

A question **fails** when none of its correct chunks is in its top-k (`failing_questions`). A question is **broken by an ingestion** when it passes before the ingestion and fails after it (`broken_by_ingestion`). Questions that already fail on the original corpus are left out, because an ingestion cannot break them. Every selector is scored against these definitions.

**k = 5.** Reasons:

- A RAG system usually passes a handful of chunks to the language model, and 5 is a common default.
- With k = 1, 39% of the questions already fail, which leaves few questions an ingestion can break.
- With k = 10, a trap document has to push the correct chunk below ten others. Each trap then breaks fewer questions, and the experiment says less.
- With k = 5, 234 of the 300 questions pass and can be broken. Most of them have a single correct chunk, so one well-placed trap chunk can still push it out.

Questions that already fail on the original corpus (300 test questions):

| k | Failing | Percent | Can be broken |
|---|---|---|---|
| 1 | 117 | 39.0% | 183 |
| **5** | **66** | **22.0%** | **234** |
| 10 | 47 | 15.7% | 253 |

These figures were measured on the corpus built in #1. They will change slightly if the chunking changes. To recompute them for k = 1, 3, 5, 10 and 20, run:

```bash
python experiments/baseline_failures.py
```

## SciFact trap ingestions

Each ingestion starts from the original SciFact corpus and adds 20 synthetic chunks.
All three runs target the same 20 questions that originally passed at k = 5.
`near_copy` alters details in a relevant chunk, `contradiction` applies rules intended
to reverse or negate a finding, and `mix` combines text from a relevant chunk and a
chunk from a different document with low cosine similarity to it.

Run these commands after `prepare_data.py`:

```bash
python experiments/make_ingestions.py --type near_copy --n 20 --seed 42 --k 5
python experiments/make_ingestions.py --type contradiction --n 20 --seed 42 --k 5
python experiments/make_ingestions.py --type mix --n 20 --seed 42 --k 5
```

Each run saves the synthetic chunks, their embeddings, the regression question IDs,
and the run manifest under `data/ingestions/<type>_n20_s42/`.

At k = 5, 66 of 300 questions failed before ingestion and 234 passed. We rechecked
all 300 questions after each ingestion.

| Trap type | Newly failing among the 20 targets | Newly failing among the 234 initial passes | Total failing after ingestion |
|---|---:|---:|---:|
| `near_copy` | 2/20 (10%) | 2/234 (0.85%) | 68/300 (22.67%) |
| `contradiction` | 1/20 (5%) | 1/234 (0.43%) | 67/300 (22.33%) |
| `mix` | 1/20 (5%) | 1/234 (0.43%) | 67/300 (22.33%) |

All regressions involved targeted questions. Question `1049` failed after all three
ingestions, and `1041` failed only after `near_copy`. These results describe one seed
and one batch size. They measure top-k retrieval coverage.
The [trap examples](report/trap_examples.md) summarize five generated chunks per type.

The runs reused the prepared SciFact corpus and embedded the new chunks with
the pinned BGE model revision. The reported regression numbers were measured
on an Intel Mac CPU and reproduce identically under the pinned versions in
`requirements.txt` (Python 3.11, PyTorch 2.15.0, sentence-transformers 6.3.0,
transformers 5.23.0, tokenizers 0.23.3, and NumPy 2.4.6).

## Mapper graph

The Mapper graph is the map of the corpus (`mapper_rag_audit.mapper`). The lens sends each chunk embedding to one number; the cover cuts the range of the lens into overlapping intervals; inside each interval the chunks are clustered, and each cluster is a node; two nodes share an edge when they share a chunk.

```python
params = fit_mapper(embeddings_before, MapperConfig())        # lens and cover range
graph = build_mapper_graph(embeddings, chunk_ids, params)     # MapperGraph(nodes, edges)
```

Fitting and building are separate so that the graph after an ingestion reuses the lens and the cover fitted on the corpus before it. Refitting would shift every interval and reshuffle the whole graph. A new chunk whose lens value falls outside the fitted range goes to the end interval. `fit_corpus` and `build_corpus_graph` wrap the two steps for a `Corpus`, with the signatures `MapperSelector` (#5) takes. A node id is `i<interval>r<row>`: the interval and the first embedding row in the node, so a node that keeps its chunks keeps its id. The id names a set of chunks, not a region: a node that holds no new chunk can still change its chunks after an ingestion (see "The cost" below), and then its id can change too. `node_interval` reads the interval back from an id. Building is deterministic.

**Library: [KeplerMapper](https://github.com/scikit-tda/kepler-mapper) 2.1.0.** Its cover takes fixed limits, which is what reusing the cover needs; it accepts a precomputed lens and any scikit-learn clusterer; and it writes the interactive HTML view. tda-mapper 0.12 requires `numpy<2`, which conflicts with the pinned numpy 2.4.6.

**Defaults** (`MapperConfig()`):

| Setting | Value | Why |
|---|---|---|
| Lens | first principal component | Linear, so it applies unchanged to new chunks. On SciFact it orders the corpus from public health and epidemiology to molecular biology. It explains 8.5% of the variance. |
| Intervals | 15 | At a 0.27 cut, the fewest side effects in the simulated ingestion: 20, against 28 with 10 intervals and 41 with 20. |
| Overlap | 50% | Against 30%, it roughly halves the number of connected components. |
| Clustering | average linkage, cosine distance, cut at 0.27 | See "Why average linkage and not DBSCAN" below. |

### Why average linkage and not DBSCAN

The first version clustered with DBSCAN, as Mapper is often run, with `min_samples=1` so that no chunk is dropped as noise. That setting is exactly single linkage: two chunks are in the same cluster when a chain of chunks joins them, each step shorter than `eps`. It works when groups are separated by empty space, as in the toy circle (#9). Text embeddings have no such gaps. On SciFact the median cosine distance from a chunk to its nearest neighbour is 0.14, and between two random chunks 0.37, so there is no `eps` that separates topics. At 0.14 more than half of the chunks have no neighbour that close and stay alone. A little above, chains of neighbours of neighbours join unrelated topics into giant clusters.

Average linkage merges two groups only when the mean distance between all their chunks is below the threshold, so a single bridge between two topics does not merge them. It still puts every chunk in a node. It is the same scikit-learn `AgglomerativeClustering` as the toy example, with another linkage.

Measured with 15 intervals and 50% overlap ("alone" is the share of chunks found only in one-chunk nodes):

| Linkage | Cut | Nodes | Components | Alone | Largest node | Touched | Side effects |
|---|---|---|---|---|---|---|---|
| single | 0.14 | 8,381 | 3,766 | 52.5% | 68 | 40 | 0 |
| single | 0.16 | 6,759 | 2,778 | 37.1% | 301 | 40 | 0 |
| single | 0.18 | 4,564 | 1,683 | 23.4% | 688 | 35 | 0 |
| single | 0.20 | 2,481 | 816 | 11.5% | 1,144 | 26 | 0 |
| average | 0.25 | 2,601 | 393 | 3.9% | 89 | 39 | 19 |
| **average** | **0.27** | **1,761** | **216** | **2.4%** | **228** | **39** | **20** |
| average | 0.30 | 908 | 88 | 1.1% | 580 | 39 | 25 |
| average | 0.35 | 292 | 29 | 0.4% | 1,292 | 22 | 17 |

No single-linkage row is usable for the graph diff (#5). A one-chunk node is not a region: a trap chunk next to it lands in a new node with no neighbours, and the diff cannot say which part of the corpus it affects. A node with 1,144 chunks (20% of the corpus) marks a fifth of the corpus as affected after any ingestion near it, which is what the method is meant to avoid. With average linkage at 0.27, 2.4% of the chunks are alone and the largest node holds 228 chunks (4% of the corpus). A lower cut leaves more chunks alone; a higher one makes giant nodes again.

**The cost.** The last two columns come from a simulated ingestion: 20 new chunks, each a slightly perturbed copy of a random existing chunk, with the graph rebuilt from the same fitted parameters. "Touched" counts the nodes holding a new chunk. "Side effects" counts the nodes that hold no new chunk but whose chunks differ from every node of the graph before. Single linkage never has side effects: a new chunk can only join the clusters it links. Average linkage has some, because a new chunk changes the mean distances, and so which nearby groups merge. At 0.27, 20 nodes change on top of the 39 touched. These changes are local, and the graph diff (#5) sees them as splits or merges. The stability study (#8) measures how much the selected questions depend on this choice. Single linkage stays available as `MapperConfig(linkage="single", threshold=...)`.

To recompute the distances and the table, with all settings tried (about 2 minutes on a laptop CPU), run:

```bash
python experiments/mapper_settings.py
```

**The graph of SciFact** with the defaults:

| Item | Count |
|---|---|
| Nodes | 1,761 |
| Edges | 2,225 |
| Connected components | 216 |
| Nodes in the largest component | 1,255 |
| One-chunk nodes | 500 |
| Median node size, in chunks | 3 |
| Largest node, in chunks | 228 |
| Chunks in at least one node | 5,658 (100%) |

Each node is labelled with the three terms of highest mean TF-IDF over its chunks (English stop words removed), for example `dna, chromatin, histone` or `risk, mortality, men`. To build the graph and the view, run:

```bash
python experiments/build_mapper_graph.py
```

It writes `data/mapper/graph.json` (settings, counts, and each node with its label and chunk ids) and `data/mapper/graph.html`. Open the HTML in a browser with internet access (it loads d3 from a CDN). Hover over a node: the browser tooltip shows its id and label, and the panel on the left (open it with `[+]` if it is closed) adds the chunk titles. The colour is the mean lens value. We checked the tooltip and the panel in Chrome; in Safari the tooltip may not show, so use Chrome or another Chromium browser. Other settings can be passed as options, for example `--threshold 0.3 --n-intervals 10`.

## Selectors

A selector chooses which questions to re-test after an ingestion. Mapper and every baseline implement `mapper_rag_audit.selection.Selector`:

```python
class Selector(Protocol):
    name: str
    def select(self, before: Corpus, after: Corpus, questions: Sequence[str], budget: int) -> list[str]: ...
```

`select` returns at most `budget` distinct question ids from `questions`, ranked by priority (most likely broken first), so any prefix is the selection for a smaller budget. A selector never runs the failure check itself. `run_selector` calls a selector and checks that its output keeps this contract.

## Graph diff selector

`mapper_rag_audit.graph_diff` compares the Mapper graph before an ingestion with the graph after it (same Mapper parameters) and ranks the questions to re-run. `select_questions(graph_before, graph_after, new_chunk_ids, qrels, budget, rules)` returns question ids, at most `budget`. `diff_graphs` returns the matching, the affected and marked nodes and the new-chunk count per node. The module docstring states the matching and ranking rules. Each rule (`new_chunks`, `new_edge`, `split`, `merge`, `neighbours`) has a flag in `Rules`, all on by default.

`MapperSelector(fit, build, rules, name)` adapts this to the shared selector interface (`name` and `select(before, after, questions, budget)`). `fit(before)` learns the Mapper parameters and `build(corpus, params)` makes a graph; the selector calls `build` on both corpora with the same parameters. A chunk is new if its id is missing from the corpus before or its text changed. The questions come from `questions` and their correct chunks from the qrels of the corpus before. The after corpus must start with the original chunk ids in order. The selector raises `ValueError` if that prefix is missing.

Use `MapperSelector(fit_corpus, build_corpus_graph)` with the corpus adapters from `mapper_rag_audit.mapper`.

## Baselines

Mapper is compared with three simpler selectors in `mapper_rag_audit.baselines`. They follow the same conventions as the Mapper selector:
- The candidates are the questions with correct chunks.
- A new chunk is a chunk id that is added or whose text changes.
- An ingestion that removes chunks is rejected.
- Ties are broken by question id.

| Selector | Rule | Parameters |
|---|---|---|
| `RandomSelector` | A seeded uniform shuffle of the candidates. It ignores the ingestion and is the floor every method must beat. | `seed` |
| `KMeansSelector` | Fits k-means on the chunks before the ingestion and assigns the chunks after it to the nearest centroid. It marks the clusters that receive a new chunk and ranks questions by the number of new chunks in their correct chunk's cluster. It does not use neighbours or edges. | `n_clusters` = the number of nodes in the Mapper graph of the corpus before the ingestion, so both split the corpus into the same number of regions. `seed`, `n_init` = 10. |
| `NearestNeighboursSelector` | Takes the `m` nearest old chunks of each new chunk (cosine) and ranks questions whose correct chunk is among them, closest first. | `m` = 10, twice the retrieval k, as a margin. Because the ranking is by similarity, a larger `m` only adds candidates at the end. |

Only Random always fills the budget. The other two return only the questions their rule marks, as Mapper does. A shorter list means the method found nothing else worth re-testing.

## License

[MIT](LICENSE)
