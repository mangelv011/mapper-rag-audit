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
| Chunks | 5,183 |
| Questions in `queries.jsonl` (train and test) | 1,109 |
| Test questions (with test qrels) | 300 |
| Question to correct chunk pairs (test qrels) | 339 |
| Test questions without a correct chunk in the corpus | 0 |

Counts were measured on the downloaded files. 277 test questions have one correct chunk, the other 23 have two to five.

- **Download**: `scifact.zip` from the BEIR mirror (2,816,079 bytes). The script checks its SHA-256 against a pinned value and refuses archive members that would leave `data/raw/`.
- **Licence**: stated in SciFact's [LICENSE.md](https://github.com/allenai/scifact/blob/master/LICENSE.md). The abstracts in `corpus.jsonl` are ODC-By 1.0 (from S2ORC). The claims and evidence annotations are CC BY 4.0. BEIR redistributes the data and grants no licence of its own.
- **Chunking rule**: one abstract is one chunk: the title, a line break and the abstract text. A document without a title uses the abstract alone. Chunk ids equal document ids, and the chunk to document map is stored anyway.
- **Embedding model**: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) through sentence-transformers, 384 dimensions, L2-normalised float32. Questions get the model's query instruction ("Represent this sentence for searching relevant passages: "), chunks do not. The script uses CUDA if available, then Apple MPS, then CPU, and prints the device and the embedding time. On a laptop CPU it took 494 s.

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

## License

[MIT](LICENSE)
