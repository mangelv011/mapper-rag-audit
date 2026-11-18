# Glossary

- **chunk**: a piece of a document that is embedded and retrieved: the title plus a run of whole sentences of the abstract, never longer than the model's token limit. A short document is one chunk, a long one is several.
- **corpus**: the set of chunks the RAG system retrieves from, with their embeddings.
- **question**: a test query with known correct chunks. SciFact calls it a claim or query.
- **correct chunk**: a chunk that the dataset marks as relevant to a question (score above 0).
- **ingestion**: adding or changing documents in the corpus.
- **embedding**: the L2-normalised float32 vector a model assigns to a chunk or question.
