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

## License

[MIT](LICENSE)
