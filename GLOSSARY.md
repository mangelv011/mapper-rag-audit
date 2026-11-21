# Glossary

- **chunk**: a piece of a document that is embedded and retrieved: the title plus a run of whole sentences of the abstract, never longer than the model's token limit. A short document is one chunk, a long one is several.
- **corpus**: the set of chunks the RAG system retrieves from, with their embeddings.
- **question**: a test query with known correct chunks. SciFact calls it a claim or query.
- **correct chunk**: a chunk that the dataset marks as relevant to a question (score above 0).
- **ingestion**: adding or changing documents in the corpus.
- **embedding**: the L2-normalised float32 vector a model assigns to a chunk or question.
- **top-k**: the k chunks whose embeddings have the highest cosine similarity to a question's embedding. The project uses k = 5 (see the README).
- **failing question**: a question none of whose correct chunks is in its top-k. Defined once, in `mapper_rag_audit.retrieval.failing_questions`.
- **broken question**: a question that passes before an ingestion and fails after it. Questions that already fail before are never counted as broken.
- **selector**: a method that, given the corpus before and after an ingestion, ranks the questions to re-test (Mapper or a baseline). Interface: `mapper_rag_audit.selection.Selector`.
- **budget**: the maximum number of questions a selector may pick for re-testing.
- **Mapper graph**: the graph Mapper builds over the chunk embeddings. Each node holds a set of chunks; an edge joins two nodes that share at least one chunk.
- **node**: a vertex of a Mapper graph, identified by an id and holding chunk ids.
- **affected node**: a node of the graph after an ingestion that the diff rules flag as changed: it received new chunks, gained an edge, or took part in a split or merge.
- **marked node**: an affected node, or a direct neighbour of one when neighbour expansion is on. Questions whose correct chunk lies in a marked node are candidates.
- **matching**: the map from each node of the graph after an ingestion to the node of the graph before it that shares most old chunks with it (Jaccard).
