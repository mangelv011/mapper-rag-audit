import pytest

from conftest import FakeTokenizer
from mapper_rag_audit.chunking import Chunk, chunk_document, split_sentences

TOK = FakeTokenizer(max_tokens=12)  # budget: 12 - 2 special = 10 words, title included


def sentences(n, words=4, prefix="S"):
    return " ".join(f"{prefix}{i} " + "w " * (words - 2) + "end." for i in range(n))


def test_document_that_fits_stays_one_chunk():
    assert chunk_document("d1", "Alpha", "One two three.", TOK) == [
        Chunk("d1#0", "d1", "Alpha\nOne two three.")
    ]


def test_document_without_title_has_no_leading_newline():
    assert chunk_document("d1", "", "One two.", TOK)[0].text == "One two."


def test_long_document_is_split_into_balanced_chunks_of_whole_sentences():
    text = sentences(6)  # 6 sentences of 4 words; room for 9 body words after the title
    chunks = chunk_document("d1", "T", text, TOK)
    assert [c.chunk_id for c in chunks] == ["d1#0", "d1#1", "d1#2"]
    assert {c.doc_id for c in chunks} == {"d1"}
    parts = split_sentences(text)
    assert [c.text for c in chunks] == [
        "T\n" + " ".join(parts[0:2]),
        "T\n" + " ".join(parts[2:4]),
        "T\n" + " ".join(parts[4:6]),
    ]


def test_uses_the_smallest_number_of_chunks():
    chunks = chunk_document("d1", "T", sentences(4), TOK)  # 16 words, 9 per chunk -> 2
    assert len(chunks) == 2


def test_every_chunk_fits_the_token_budget():
    for n in range(1, 15):
        for chunk in chunk_document("d1", "Some title", sentences(n, words=3), TOK):
            assert TOK.count(chunk.text) <= TOK.max_tokens


def test_sentence_longer_than_the_budget_is_split_on_token_boundaries():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_document("d1", "T", " ".join(words), TOK)
    assert len(chunks) == 3
    assert all(c.text.startswith("T\n") and TOK.count(c.text) <= 12 for c in chunks)
    assert " ".join(c.text[2:] for c in chunks) == " ".join(words)


def test_title_that_leaves_no_room_for_text_is_an_error():
    with pytest.raises(ValueError, match="title"):
        chunk_document("d1", " ".join(["t"] * 10), "Text.", TOK)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One. Two three! Four?", ["One.", "Two three!", "Four?"]),
        ("Use e.g. this one. Next", ["Use e.g. this one.", "Next"]),
        ("Value was 1.5 mg. Done.", ["Value was 1.5 mg.", "Done."]),
        ("Use e.g. TP53 here. Next", ["Use e.g. TP53 here.", "Next"]),
        ("Cost 10 vs. Dollars 5. Next", ["Cost 10 vs. Dollars 5.", "Next"]),
        ("As Verhaak et al. Found it. Next", ["As Verhaak et al. Found it.", "Next"]),
        ("Is i.e. Foo ok? Yes", ["Is i.e. Foo ok?", "Yes"]),
        ("See vol. 5 now. Next", ["See vol. 5 now.", "Next"]),
        ("  ", []),
        ("No end", ["No end"]),
    ],
)
def test_sentence_splitting(text, expected):
    assert split_sentences(text) == expected


class SuperlinearTokenizer(FakeTokenizer):
    """Counting a joined text costs more than the sum of its parts, like re-tokenised pieces."""

    def count(self, text):
        return super().count(text) + len(text.split()) // 4


def test_chunks_still_fit_when_token_counts_are_not_additive():
    tok = SuperlinearTokenizer(max_tokens=20)
    chunks = chunk_document("d1", "T", sentences(30, words=3), tok)
    assert len(chunks) > 1
    assert all(tok.count(c.text) <= 20 for c in chunks)


def test_document_exactly_at_the_limit_stays_one_chunk():
    text = "w " * 9  # title 1 + 9 words + 2 special = 12; a text that fits is kept verbatim
    assert [c.text for c in chunk_document("d1", "T", text, TOK)] == ["T\n" + text]
    assert len(chunk_document("d1", "T", text + " x", TOK)) == 2


def test_sentences_that_fill_the_budget_exactly_share_a_chunk():
    text = "A a a a a a. B b b. C c c c c c. D d d."  # 6, 3, 6, 3 words; 9 words fit per chunk
    chunks = chunk_document("d1", "T", text, TOK)
    expected = ["T\nA a a a a a. B b b.", "T\nC c c c c c. D d d."]
    assert [c.text for c in chunks] == expected
