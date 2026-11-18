from pathlib import Path

import pytest

from helpers import assert_every_question_has_a_correct_chunk_in_the_corpus as check
from mapper_rag_audit.data import build_corpus, load_corpus, read_beir

REAL_DATA = Path(__file__).resolve().parent.parent / "data"


def test_every_question_has_a_correct_chunk_in_the_fixture_corpus(raw_dir, fake_embedder):
    check(build_corpus(read_beir(raw_dir), fake_embedder))


@pytest.mark.skipif(not (REAL_DATA / "qrels.json").exists(), reason="run prepare_data.py first")
def test_every_question_has_a_correct_chunk_in_the_real_corpus():
    check(load_corpus(REAL_DATA))
