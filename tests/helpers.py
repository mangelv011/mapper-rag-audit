from mapper_rag_audit.data import Corpus


def assert_every_question_has_a_correct_chunk_in_the_corpus(corpus: Corpus) -> None:
    chunk_ids = {c.chunk_id for c in corpus.chunks}
    assert corpus.questions, "no questions"
    for q in corpus.questions:
        correct = corpus.qrels.get(q.question_id, [])
        assert correct, f"{q.question_id} has no correct chunk"
        assert set(correct) <= chunk_ids, f"{q.question_id} points outside the corpus"
