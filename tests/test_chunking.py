import re

import pytest

from conftest import FakeTokenizer
from mapper_rag_audit.chunking import Chunk, chunk_document, sentence_spans

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
    parts = [text[a:b] for a, b in sentence_spans(text)]
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
    assert [text[a:b] for a, b in sentence_spans(text)] == expected


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


def bodies(chunks, title="T"):
    return [c.text[len(title) + 1 :] for c in chunks]


def assert_bodies_are_verbatim_and_cover_the_text(chunks, text, title="T"):
    pos = 0
    for body in bodies(chunks, title):
        start = text.find(body, pos)
        assert start >= 0, f"{body!r} is not an in-order slice of the text"
        assert text[pos:start].strip() == "", "text skipped between chunks"
        pos = start + len(body)
    assert text[pos:].strip() == ""


def test_chunk_bodies_keep_the_original_spacing_and_line_breaks():
    text = "A a a a a a.  B b b.\nC c c c c c.\t\tD d d."
    chunks = chunk_document("d1", "T", text, TOK)
    assert len(chunks) == 2
    assert bodies(chunks) == ["A a a a a a.  B b b.", "C c c c c c.\t\tD d d."]
    assert_bodies_are_verbatim_and_cover_the_text(chunks, text)


def test_pieces_of_a_long_sentence_are_slices_of_the_original_text():
    text = (
        "  Naive  w1 w2 w3 w4\nw5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22  "
    )
    chunks = chunk_document("d1", "T", text, TOK)
    assert len(chunks) == 3
    assert all(TOK.count(c.text) <= 12 for c in chunks)
    assert_bodies_are_verbatim_and_cover_the_text(chunks, text)


class SubwordTokenizer:
    """First 2 characters of a word are a token, then 3 each, so cutting a word shifts its tokens."""

    max_tokens = 12

    @staticmethod
    def _tokens(text):
        spans = []
        for word in re.finditer(r"\S+", text):
            start, size = word.start(), 2
            while start < word.end():
                spans.append((start, min(start + size, word.end())))
                start, size = spans[-1][1], 3
        return spans

    def count(self, text):
        return len(self._tokens(text)) + 2

    def split(self, text, max_body_tokens):
        tokens = self._tokens(text)
        return [
            (tokens[i][0], tokens[min(i + max_body_tokens, len(tokens)) - 1][1])
            for i in range(0, len(tokens), max_body_tokens)
        ]


def test_pieces_that_grow_when_retokenised_are_cut_smaller_until_they_fit():
    tok = SubwordTokenizer()
    text = "x" * 60  # 21 tokens; a 9-token slice from mid-word re-counts as 10
    chunks = chunk_document("d1", "T", text, tok)
    assert len(chunks) == 3  # cut at 8 tokens, not 9
    assert all(tok.count(c.text) <= tok.max_tokens for c in chunks)
    assert "".join(bodies(chunks)) == text


class UncuttableTokenizer(FakeTokenizer):
    def split(self, text, max_body_tokens):
        return [(0, len(text))]


def test_text_that_cannot_be_cut_to_fit_raises_an_error_naming_the_document():
    with pytest.raises(ValueError, match="doc-42"):
        chunk_document("doc-42", "T", " ".join(["w"] * 30), UncuttableTokenizer(max_tokens=12))


def test_sentence_spans_exclude_the_whitespace_around_the_text_and_between_sentences():
    assert sentence_spans("  One.  Two  ") == [(2, 6), (8, 11)]
