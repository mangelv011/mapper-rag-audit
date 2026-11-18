# Glossary

- **chunk**: a piece of a document that is embedded and retrieved. In this project one abstract (title and abstract text) is one chunk.
- **corpus**: the set of chunks the RAG system retrieves from, with their embeddings.
- **question**: a test query with known correct chunks. SciFact calls it a claim or query.
- **correct chunk**: a chunk that the dataset marks as relevant to a question (score above 0).
- **ingestion**: adding or changing documents in the corpus.
- **embedding**: the L2-normalised float32 vector a model assigns to a chunk or question.
