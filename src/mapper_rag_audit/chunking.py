"""Split documents into chunks that fit the embedding model's token limit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

Span = tuple[int, int]


class Tokenizer(Protocol):
    max_tokens: int

    def count(self, text: str) -> int:
        """Tokens in text, special tokens included."""

    def split(self, text: str, max_body_tokens: int) -> list[Span]:
        """Character spans of text, cut on token boundaries, of at most max_body_tokens each."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


# A sentence ends at . ! or ? followed by a space and a capital letter, unless the full stop
# belongs to a common abbreviation ("e.g. TP53", "vs. Dollars", "et al. found"). Numbers such
# as "1.5" have no space after the dot. A wrong cut only moves a chunk boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!et al\.)\s+(?=[A-Z])")


def sentence_spans(text: str) -> list[Span]:
    """Character spans of the sentences of text, without surrounding whitespace."""
    bounds = [0, *(i for m in _SENTENCE_END.finditer(text) for i in m.span()), len(text)]
    spans = []
    for start, end in zip(bounds[::2], bounds[1::2], strict=True):
        piece = text[start:end]
        if piece.strip():
            spans.append(
                (start + len(piece) - len(piece.lstrip()), end - len(piece) + len(piece.rstrip()))
            )
    return spans


def _groups_needed(sizes: list[int], cap: int) -> list[list[int]]:
    groups: list[list[int]] = [[]]
    load = 0
    for i, size in enumerate(sizes):
        if groups[-1] and load + size > cap:
            groups.append([])
            load = 0
        groups[-1].append(i)
        load += size
    return groups


def _balanced_groups(sizes: list[int], budget: int) -> list[list[int]]:
    """Fewest contiguous groups with load <= budget, then the smallest largest load."""
    total = sum(sizes)
    for n in range(-(-total // budget), len(sizes) + 1):
        lo, hi = max(sizes), budget
        while lo < hi:
            mid = (lo + hi) // 2
            if len(_groups_needed(sizes, mid)) <= n:
                hi = mid
            else:
                lo = mid + 1
        groups = _groups_needed(sizes, lo)
        if len(groups) <= n and max(sum(sizes[i] for i in g) for g in groups) <= budget:
            return groups
    raise AssertionError("unreachable: every unit fits the budget")


def _cut_to_fit(sentence: str, offset: int, budget: int, tok: Tokenizer) -> list[Span] | None:
    """Spans of an over-long sentence that still fit the budget once re-tokenised on their own."""
    overhead = tok.count("")
    for cut in range(budget, 0, -1):
        spans = [(offset + a, offset + b) for a, b in tok.split(sentence, cut)]
        if all(tok.count(sentence[a - offset : b - offset]) - overhead <= budget for a, b in spans):
            return spans
    return None


def chunk_document(doc_id: str, title: str, text: str, tok: Tokenizer) -> list[Chunk]:
    """Chunks whose bodies are exact slices of text, in order."""
    prefix = f"{title}\n" if title else ""
    if tok.count(prefix + text) <= tok.max_tokens:
        return [Chunk(f"{doc_id}#0", doc_id, prefix + text)]

    overhead = tok.count("")
    budget = tok.max_tokens - tok.count(prefix)
    while budget >= 1:
        units: list[Span] = []
        for start, end in sentence_spans(text):
            if tok.count(text[start:end]) - overhead <= budget:
                units.append((start, end))
                continue
            pieces = _cut_to_fit(text[start:end], start, budget, tok)
            if pieces is None:
                raise ValueError(f"{doc_id} cannot be cut into pieces of at most {budget} tokens")
            units.extend(pieces)
        sizes = [tok.count(text[a:b]) - overhead for a, b in units]
        groups = _balanced_groups(sizes, budget)
        bodies = [text[units[g[0]][0] : units[g[-1]][1]] for g in groups]
        if all(tok.count(prefix + b) <= tok.max_tokens for b in bodies):
            return [Chunk(f"{doc_id}#{i}", doc_id, prefix + b) for i, b in enumerate(bodies)]
        budget -= 1
    raise ValueError(f"title of {doc_id} leaves no room for text in a chunk")
