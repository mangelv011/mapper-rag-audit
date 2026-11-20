from mapper_rag_audit.data import Corpus


def assert_every_question_has_a_correct_chunk_in_the_corpus(corpus: Corpus) -> None:
    chunk_ids = {c.chunk_id for c in corpus.chunks}
    assert corpus.questions, "no questions"
    assert set(corpus.qrels) == {q.question_id for q in corpus.questions}, (
        "qrels and questions differ"
    )
    for q in corpus.questions:
        correct = corpus.qrels.get(q.question_id, [])
        assert correct, f"{q.question_id} has no correct chunk"
        assert len(set(correct)) == len(correct), f"{q.question_id} lists a chunk twice"
        assert set(correct) <= chunk_ids, f"{q.question_id} points outside the corpus"


def assert_chunks_fit_and_belong_to_documents(corpus: Corpus, doc_ids: set[str], tokenizer) -> None:
    assert corpus.chunks, "no chunks"
    assert len({c.chunk_id for c in corpus.chunks}) == len(corpus.chunks), "duplicate chunk ids"
    assert corpus.embeddings.shape[0] == len(corpus.chunks)
    for c in corpus.chunks:
        assert c.doc_id in doc_ids, f"{c.chunk_id} belongs to an unknown document"
        assert c.chunk_id.startswith(f"{c.doc_id}#")
        assert tokenizer.count(c.text) <= tokenizer.max_tokens, f"{c.chunk_id} is truncated"


def assert_chunks_are_verbatim_slices_of_their_documents(
    corpus: Corpus, documents: list[dict]
) -> None:
    """Each chunk is its title, a line break and an exact slice; the slices follow the abstract."""
    by_doc: dict[str, list] = {}
    for c in corpus.chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    assert set(by_doc) == {d["_id"] for d in documents}, "a document has no chunk"
    for d in documents:
        prefix = f"{d['title']}\n" if d["title"] else ""
        text, pos = d["text"], 0
        for c in by_doc[d["_id"]]:
            assert c.text.startswith(prefix), f"{c.chunk_id} lost its title"
            body = c.text[len(prefix) :]
            start = text.find(body, pos)
            assert start >= 0, f"{c.chunk_id} is not an in-order slice of its abstract"
            assert text[pos:start].strip() == "", f"{c.chunk_id} skips text before it"
            pos = start + len(body)
        assert text[pos:].strip() == "", f"{d['_id']} loses its ending"
