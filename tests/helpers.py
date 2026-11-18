from mapper_rag_audit.data import Corpus


def assert_every_question_has_a_correct_chunk_in_the_corpus(corpus: Corpus) -> None:
    chunk_ids = {c.chunk_id for c in corpus.chunks}
    assert corpus.questions, "no questions"
    for q in corpus.questions:
        correct = corpus.qrels.get(q.question_id, [])
        assert correct, f"{q.question_id} has no correct chunk"
        assert set(correct) <= chunk_ids, f"{q.question_id} points outside the corpus"


def assert_chunks_fit_and_belong_to_documents(corpus: Corpus, doc_ids: set[str], tokenizer) -> None:
    assert corpus.chunks, "no chunks"
    assert len({c.chunk_id for c in corpus.chunks}) == len(corpus.chunks), "duplicate chunk ids"
    assert corpus.embeddings.shape[0] == len(corpus.chunks)
    for c in corpus.chunks:
        assert c.doc_id in doc_ids, f"{c.chunk_id} belongs to an unknown document"
        assert c.chunk_id.startswith(f"{c.doc_id}#")
        assert tokenizer.count(c.text) <= tokenizer.max_tokens, f"{c.chunk_id} is truncated"
