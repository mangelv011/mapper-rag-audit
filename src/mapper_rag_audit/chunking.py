"""Split documents into chunks that fit the embedding model's token limit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    max_tokens: int

    def count(self, text: str) -> int:
        """Tokens in text, special tokens included."""

    def split(self, text: str, max_body_tokens: int) -> list[str]:
        """Cut text on token boundaries into balanced pieces of at most max_body_tokens."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


# A sentence ends at . ! or ? followed by a space and a capital letter, unless the full stop
# belongs to a common abbreviation ("e.g. TP53", "vs. Dollars", "et al. found"). Numbers such
# as "1.5" have no space after the dot. A wrong cut only moves a chunk boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!et al\.)\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    return _SENTENCE_END.split(text.strip()) if text.strip() else []


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


def chunk_document(doc_id: str, title: str, text: str, tok: Tokenizer) -> list[Chunk]:
    prefix = f"{title}\n" if title else ""
    if tok.count(prefix + text) <= tok.max_tokens:
        return [Chunk(f"{doc_id}#0", doc_id, prefix + text)]

    overhead = tok.count("")
    budget = tok.max_tokens - tok.count(prefix)
    while True:
        if budget < 1:
            raise ValueError(f"title of {doc_id} leaves no room for text in a chunk")
        units: list[str] = []
        for sentence in split_sentences(text):
            if tok.count(sentence) - overhead > budget:
                units.extend(tok.split(sentence, budget))
            else:
                units.append(sentence)
        sizes = [tok.count(u) - overhead for u in units]
        texts = [prefix + " ".join(units[i] for i in g) for g in _balanced_groups(sizes, budget)]
        if all(tok.count(t) <= tok.max_tokens for t in texts):
            return [Chunk(f"{doc_id}#{i}", doc_id, t) for i, t in enumerate(texts)]
        budget -= 1
