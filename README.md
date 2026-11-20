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

## Graph diff selector

`mapper_rag_audit.graph_diff` compares the Mapper graph before an ingestion with the graph after it (same Mapper parameters) and ranks the questions to re-run. `select_questions(graph_before, graph_after, new_chunk_ids, qrels, budget, rules)` returns question ids, at most `budget`. `diff_graphs` returns the matching, the affected and marked nodes and the new-chunk count per node. The module docstring states the matching and ranking rules. Each rule (`new_chunks`, `new_edge`, `split`, `merge`, `neighbours`) has a flag in `Rules`, all on by default.

`MapperSelector(fit, build, rules, name)` adapts this to the shared selector interface (`name` and `select(before, after, questions, budget)`). `fit(before)` learns the Mapper parameters and `build(corpus, params)` makes a graph; the selector calls `build` on both corpora with the same parameters. A chunk is new if its id is missing from the corpus before or its text changed. The questions come from `questions` and their correct chunks from the qrels of the corpus before. An ingestion that removes chunks raises `ValueError`, because our ingestions only add chunks.

The graph format (`MapperGraph`: node id to chunk ids, plus edges) is provisional until the Mapper builder exists, so `fit` and `build` are injected.

## License

[MIT](LICENSE)
