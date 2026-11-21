"""Trap-document generator and ground truth of broken questions.

An ingestion adds trap chunks designed to distract retrieval and push correct chunks out
of top-k for target questions. Three trap types:
- `near_copy`: correct chunk with altered facts (numbers, entities, direction).
- `contradiction`: same topic, but asserting the opposite finding (rule-based inversion).
- `mix`: text joining two unrelated topics, bridging topological regions in Mapper.

Ground truth broken questions are computed using `retrieval.broken_by_ingestion`.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mapper_rag_audit.chunking import Chunk, Tokenizer, sentence_spans
from mapper_rag_audit.data import Corpus, Embedder
from mapper_rag_audit.retrieval import DEFAULT_K, broken_by_ingestion, failing_questions

TrapType = Literal["near_copy", "contradiction", "mix"]
TRAP_TYPES: tuple[TrapType, ...] = ("near_copy", "contradiction", "mix")

_DIRECTION_PAIRS: list[tuple[str, str]] = [
    ("increased", "decreased"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increasing", "decreasing"),
    ("higher", "lower"),
    ("highest", "lowest"),
    ("upregulated", "downregulated"),
    ("upregulation", "downregulation"),
    ("positive", "negative"),
    ("positively", "negatively"),
    ("improved", "worsened"),
    ("improvement", "deterioration"),
    ("elevated", "reduced"),
    ("elevation", "reduction"),
    ("enhances", "impairs"),
    ("enhanced", "impaired"),
    ("promotes", "suppresses"),
    ("promoted", "suppressed"),
    ("activates", "inhibits"),
    ("activated", "inhibited"),
    ("activation", "inhibition"),
    ("accelerates", "decelerates"),
    ("accelerated", "decelerated"),
    ("stimulates", "blocks"),
    ("greater", "smaller"),
    ("more", "less"),
]

# Map word -> replacement and vice versa
_DIRECTION_MAP: dict[str, str] = {}
for _w1, _w2 in _DIRECTION_PAIRS:
    _DIRECTION_MAP[_w1] = _w2
    _DIRECTION_MAP[_w2] = _w1
    _DIRECTION_MAP[_w1.capitalize()] = _w2.capitalize()
    _DIRECTION_MAP[_w2.capitalize()] = _w1.capitalize()

_CONTRADICTION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bis required for\b", re.IGNORECASE), "is dispensable for"),
    (re.compile(r"\bare required for\b", re.IGNORECASE), "are dispensable for"),
    (re.compile(r"\bis essential for\b", re.IGNORECASE), "is not essential for"),
    (re.compile(r"\bare essential for\b", re.IGNORECASE), "are non-essential for"),
    (re.compile(r"\bis associated with\b", re.IGNORECASE), "is not associated with"),
    (re.compile(r"\bare associated with\b", re.IGNORECASE), "are not associated with"),
    (re.compile(r"\bplays a critical role in\b", re.IGNORECASE), "plays no role in"),
    (re.compile(r"\bplays an important role in\b", re.IGNORECASE), "plays no detectable role in"),
    (re.compile(r"\bplays a role in\b", re.IGNORECASE), "plays no role in"),
    (re.compile(r"\bdemonstrates that\b", re.IGNORECASE), "fails to demonstrate that"),
    (re.compile(r"\bdemonstrate that\b", re.IGNORECASE), "do not demonstrate that"),
    (re.compile(r"\bshows that\b", re.IGNORECASE), "fails to show that"),
    (re.compile(r"\bshow that\b", re.IGNORECASE), "do not show that"),
    (re.compile(r"\bconfirms that\b", re.IGNORECASE), "refutes that"),
    (re.compile(r"\bsuggests that\b", re.IGNORECASE), "provides no evidence that"),
    (re.compile(r"\bsuggest that\b", re.IGNORECASE), "contradict the claim that"),
    (re.compile(r"\bindicates that\b", re.IGNORECASE), "does not indicate that"),
    (re.compile(r"\bindicate that\b", re.IGNORECASE), "do not indicate that"),
    (re.compile(r"\bcorrelated with\b", re.IGNORECASE), "uncorrelated with"),
    (re.compile(r"\bcorrelates with\b", re.IGNORECASE), "does not correlate with"),
    (re.compile(r"\bleads to\b", re.IGNORECASE), "fails to lead to"),
    (re.compile(r"\bprotects against\b", re.IGNORECASE), "exacerbates"),
    (re.compile(r"\bprotects from\b", re.IGNORECASE), "sensitizes to"),
    (re.compile(r"\bwas effective\b", re.IGNORECASE), "was ineffective"),
    (re.compile(r"\bis effective\b", re.IGNORECASE), "is ineffective"),
    (re.compile(r"\bcaused\b", re.IGNORECASE), "did not cause"),
    (re.compile(r"\bcauses\b", re.IGNORECASE), "fails to cause"),
    (re.compile(r"\binduces\b", re.IGNORECASE), "fails to induce"),
    (re.compile(r"\binduce\b", re.IGNORECASE), "do not induce"),
]

_ENTITY_SWAPS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin humans\b", re.IGNORECASE), "in murine models"),
    (re.compile(r"\bin vivo\b", re.IGNORECASE), "in vitro"),
    (re.compile(r"\bpatients\b", re.IGNORECASE), "healthy controls"),
    (re.compile(r"\bserum\b", re.IGNORECASE), "plasma"),
]


@dataclass(frozen=True)
class TrapChunk:
    chunk_id: str
    doc_id: str
    text: str
    trap_type: TrapType
    target_question_id: str
    source_chunk_ids: list[str]
    metadata: dict[str, str]

    def to_chunk(self) -> Chunk:
        return Chunk(chunk_id=self.chunk_id, doc_id=self.doc_id, text=self.text)


@dataclass(frozen=True)
class Ingestion:
    trap_type: TrapType
    n: int
    seed: int
    k: int
    target_questions: list[str]
    trap_chunks: list[TrapChunk]
    trap_embeddings: np.ndarray
    broken_questions: list[str]
    after: Corpus


def mutate_near_copy(text: str, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Mutate numbers, direction words, or entities in a chunk text to alter factual claims."""
    lines = text.split("\n", 1)
    title = lines[0]
    body = lines[1] if len(lines) > 1 else ""

    changes: list[str] = []

    # 1. Mutate directional terms
    def _swap_direction(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = _DIRECTION_MAP.get(word, word)
        if replacement != word and len(changes) < 3:
            changes.append(f"direction:{word}->{replacement}")
            return replacement
        return word

    dir_pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _DIRECTION_MAP) + r")\b"
    )
    new_body = dir_pattern.sub(_swap_direction, body)

    # 2. Mutate percentages or numbers
    def _mutate_number(match: re.Match[str]) -> str:
        if len(changes) >= 3:
            return match.group(0)
        raw_val = match.group(1)
        suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
        try:
            if "%" in match.group(0):
                val = float(raw_val)
                new_val = max(5, min(95, round(100.0 - val, 1)))
                if new_val == val:
                    new_val = 15.0 if val > 50 else 85.0
                changes.append(f"percent:{val}%->{new_val}%")
                return f"{new_val:g}%"
            elif re.search(r"\b(19\d\d|20\d\d)\b", raw_val):
                year = int(raw_val)
                new_year = year - 20 if year > 2000 else year + 15
                changes.append(f"year:{year}->{new_year}")
                return str(new_year)
            else:
                num = int(raw_val)
                new_num = num * 10 if num in (1, 2, 3) else max(1, num // 2)
                changes.append(f"number:{num}->{new_num}")
                return f"{new_num}{suffix}"
        except ValueError:
            return match.group(0)

    num_pattern = re.compile(r"\b(\d+(?:\.\d+)?)(%|\s*fold|\s*mg|\s*h)?\b")
    new_body = num_pattern.sub(_mutate_number, new_body)

    # 3. Mutate entities if few changes
    if len(changes) < 2:
        for ent_pat, rep in _ENTITY_SWAPS:
            if ent_pat.search(new_body):
                new_body = ent_pat.sub(rep, new_body, count=1)
                changes.append(f"entity:{rep}")
                break

    # 4. Fallback guarantee: if unchanged, negate explicitly
    if not changes or new_body == body:
        new_body = body + " Note: Subsequent re-evaluations failed to replicate these measurements."
        changes.append("fallback:appended_negative_note")

    mutated_text = f"{title}\n{new_body}" if body else title
    return mutated_text, {"changes": "; ".join(changes)}


def mutate_contradiction(text: str, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Produce an opposite claim on the same topic using scientific negation rules."""
    lines = text.split("\n", 1)
    title = lines[0]
    body = lines[1] if len(lines) > 1 else ""

    changes: list[str] = []
    new_body = body

    for pattern, replacement in _CONTRADICTION_REPLACEMENTS:
        if pattern.search(new_body):
            new_body = pattern.sub(replacement, new_body, count=1)
            changes.append(f"inversion:{replacement}")
            if len(changes) >= 2:
                break

    # Also invert direction terms to reinforce the opposite finding
    dir_pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _DIRECTION_MAP) + r")\b"
    )

    def _swap_dir(m: re.Match[str]) -> str:
        w = m.group(0)
        rep = _DIRECTION_MAP.get(w, w)
        if rep != w and len(changes) < 4:
            changes.append(f"direction:{w}->{rep}")
            return rep
        return w

    new_body = dir_pattern.sub(_swap_dir, new_body)

    # If no rule triggered, or to strengthen the contradiction, wrap concluding claim
    if not changes or new_body == body:
        spans = sentence_spans(body)
        if spans:
            last_span = spans[-1]
            last_sentence = body[last_span[0] : last_span[1]]
            contradiction_prefix = rng.choice(
                [
                    "In contrast to prevailing models, extensive experiments demonstrate that ",
                    "Contradicting previous observations, our direct tests show that ",
                    "Here we provide evidence refuting that ",
                ]
            )
            inverted_sentence = (
                contradiction_prefix + last_sentence[0].lower() + last_sentence[1:]
            )
            new_body = body[: last_span[0]] + inverted_sentence
            changes.append("concluding_sentence_inversion")
        else:
            new_body = (
                body
                + " Crucially, subsequent controlled trials conclusively refuted this hypothesis."
            )
            changes.append("fallback_refutation")

    mutated_text = f"{title}\n{new_body}" if body else title
    return mutated_text, {"changes": "; ".join(changes)}


def mutate_mix(
    text_a: str,
    text_b: str,
    max_tokens: int = 512,
    tokenizer: Tokenizer | None = None,
) -> tuple[str, dict[str, str]]:
    """Join text from two unrelated topics so new links between topological regions appear."""
    lines_a = text_a.split("\n", 1)
    title_a = lines_a[0].strip()
    body_a = lines_a[1].strip() if len(lines_a) > 1 else lines_a[0].strip()

    lines_b = text_b.split("\n", 1)
    title_b = lines_b[0].strip()
    body_b = lines_b[1].strip() if len(lines_b) > 1 else lines_b[0].strip()

    combined_title = f"{title_a} and {title_b}" if title_a != title_b else title_a

    spans_a = sentence_spans(body_a)
    spans_b = sentence_spans(body_b)

    sentences_a = (
        [body_a[start:end] for start, end in spans_a] if spans_a else [body_a]
    )
    sentences_b = (
        [body_b[start:end] for start, end in spans_b] if spans_b else [body_b]
    )

    take_a = max(1, len(sentences_a) // 2)
    take_b = max(1, len(sentences_b) // 2)

    half_a = " ".join(sentences_a[:take_a])
    half_b = " ".join(sentences_b[-take_b:])

    bridge = " Concurrently, cross-domain observations indicate that "
    combined_body = f"{half_a}{bridge}{half_b}"
    combined_text = f"{combined_title}\n{combined_body}"

    # Verify token length if tokenizer is provided
    if tokenizer is not None:
        while tokenizer.count(combined_text) > max_tokens and take_b > 1:
            take_b -= 1
            half_b = " ".join(sentences_b[-take_b:])
            combined_body = f"{half_a}{bridge}{half_b}"
            combined_text = f"{combined_title}\n{combined_body}"

        # If still over, cut body to fit
        if tokenizer.count(combined_text) > max_tokens:
            spans = tokenizer.split(combined_body, max_tokens - tokenizer.count(combined_title) - 4)
            if spans:
                combined_body = combined_body[spans[0][0] : spans[0][1]]
                combined_text = f"{combined_title}\n{combined_body}"

    meta = {
        "source_a": title_a[:50],
        "source_b": title_b[:50],
        "sentences_a": str(take_a),
        "sentences_b": str(take_b),
    }
    return combined_text, meta


def _pick_unrelated_chunk(
    target_chunk: Chunk,
    corpus: Corpus,
    rng: random.Random,
    exclude_chunk_ids: set[str],
) -> Chunk:
    """Select a chunk from an unrelated topic/region."""
    chunk_ids = [c.chunk_id for c in corpus.chunks]
    valid_indices = [
        i
        for i, c in enumerate(corpus.chunks)
        if c.chunk_id not in exclude_chunk_ids and c.doc_id != target_chunk.doc_id
    ]
    if not valid_indices:
        return target_chunk

    # If embeddings are present, pick from lowest cosine similarity
    if len(corpus.embeddings) == len(corpus.chunks) and target_chunk.chunk_id in chunk_ids:
        t_idx = chunk_ids.index(target_chunk.chunk_id)
        t_vec = corpus.embeddings[t_idx]
        norm = np.linalg.norm(t_vec)
        if norm > 0:
            sims = (corpus.embeddings[valid_indices] @ t_vec) / (
                np.linalg.norm(corpus.embeddings[valid_indices], axis=1) * norm + 1e-9
            )
            # Pick from the lowest quartile of similarity
            cutoff = np.percentile(sims, 25)
            low_sim_indices = [
                valid_indices[j] for j, s in enumerate(sims) if s <= cutoff
            ]
            if low_sim_indices:
                chosen_idx = rng.choice(low_sim_indices)
                return corpus.chunks[chosen_idx]

    # Fallback: choose from distant index in corpus
    chosen_idx = rng.choice(valid_indices)
    return corpus.chunks[chosen_idx]


def generate_traps(
    corpus: Corpus,
    trap_type: TrapType,
    n: int,
    seed: int,
    tokenizer: Tokenizer | None = None,
    k: int = DEFAULT_K,
) -> list[TrapChunk]:
    """Generate n trap chunks tied to target questions of the given trap type."""
    if trap_type not in TRAP_TYPES:
        raise ValueError(f"trap_type must be one of {TRAP_TYPES}, got {trap_type!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    rng = random.Random(seed)
    chunk_by_id = {c.chunk_id: c for c in corpus.chunks}

    # Find questions that currently pass at top-k and have correct chunks in the corpus
    all_qids = [q.question_id for q in corpus.questions]
    failing_initial = failing_questions(corpus, all_qids, k=k)
    passing_qids = sorted(
        [
            q
            for q in all_qids
            if q not in failing_initial
            and q in corpus.qrels
            and any(cid in chunk_by_id for cid in corpus.qrels[q])
        ]
    )

    if not passing_qids:
        # Fallback to any questions with correct chunks
        passing_qids = sorted(
            [q for q in all_qids if corpus.qrels.get(q)]
        )

    if not passing_qids:
        raise ValueError("Corpus has no questions with valid qrels")

    # Sample n target questions deterministically
    if n <= len(passing_qids):
        target_questions = rng.sample(passing_qids, n)
    else:
        target_questions = [rng.choice(passing_qids) for _ in range(n)]

    traps: list[TrapChunk] = []
    for i, target_qid in enumerate(target_questions):
        correct_cids = sorted(
            [cid for cid in corpus.qrels[target_qid] if cid in chunk_by_id]
        )
        target_cid = rng.choice(correct_cids)
        chunk_a = chunk_by_id[target_cid]
        doc_id = f"trap_{trap_type}_s{seed}_{i}"
        chunk_id = f"{doc_id}#0"

        if trap_type == "near_copy":
            text, meta = mutate_near_copy(chunk_a.text, rng)
            source_ids = [target_cid]
        elif trap_type == "contradiction":
            text, meta = mutate_contradiction(chunk_a.text, rng)
            source_ids = [target_cid]
        elif trap_type == "mix":
            chunk_b = _pick_unrelated_chunk(
                chunk_a,
                corpus,
                rng,
                exclude_chunk_ids=set(corpus.qrels.get(target_qid, [])),
            )
            text, meta = mutate_mix(chunk_a.text, chunk_b.text, tokenizer=tokenizer)
            source_ids = [target_cid, chunk_b.chunk_id]
        else:
            raise ValueError(f"Unknown trap type: {trap_type}")

        traps.append(
            TrapChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                trap_type=trap_type,
                target_question_id=target_qid,
                source_chunk_ids=source_ids,
                metadata=meta,
            )
        )

    return traps


def apply_ingestion(
    before: Corpus,
    trap_chunks: Sequence[Chunk | TrapChunk],
    trap_embeddings: np.ndarray,
) -> Corpus:
    """Return a new Corpus with trap chunks and embeddings added to the existing ones."""
    chunks_to_add = [
        c.to_chunk() if isinstance(c, TrapChunk) else c for c in trap_chunks
    ]
    all_chunks = list(before.chunks) + chunks_to_add
    all_embeddings = np.vstack([before.embeddings, trap_embeddings])
    return Corpus(
        chunks=all_chunks,
        embeddings=all_embeddings,
        questions=before.questions,
        question_embeddings=before.question_embeddings,
        qrels=before.qrels,
    )


def build_ingestion(
    before: Corpus,
    trap_type: TrapType,
    n: int,
    seed: int,
    embedder: Embedder,
    tokenizer: Tokenizer | None = None,
    k: int = DEFAULT_K,
) -> Ingestion:
    """Generate traps, embed them, apply ingestion, and compute broken questions."""
    traps = generate_traps(before, trap_type, n, seed, tokenizer=tokenizer, k=k)
    trap_embeddings = embedder.embed_passages([t.text for t in traps])
    after = apply_ingestion(before, traps, trap_embeddings)

    all_qids = [q.question_id for q in before.questions]
    broken = broken_by_ingestion(before, after, all_qids, k=k)

    return Ingestion(
        trap_type=trap_type,
        n=n,
        seed=seed,
        k=k,
        target_questions=[t.target_question_id for t in traps],
        trap_chunks=traps,
        trap_embeddings=trap_embeddings,
        broken_questions=sorted(broken),
        after=after,
    )


def save_ingestion(ingestion: Ingestion, out_dir: Path) -> Path:
    """Write trap chunks, embeddings and broken questions to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trap_embeddings.npy", ingestion.trap_embeddings)

    chunks_data = [
        {
            "chunk_id": t.chunk_id,
            "doc_id": t.doc_id,
            "text": t.text,
            "trap_type": t.trap_type,
            "target_question_id": t.target_question_id,
            "source_chunk_ids": t.source_chunk_ids,
            "metadata": t.metadata,
        }
        for t in ingestion.trap_chunks
    ]
    with (out_dir / "trap_chunks.json").open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    with (out_dir / "broken_questions.json").open("w", encoding="utf-8") as f:
        json.dump(ingestion.broken_questions, f, indent=2)

    manifest = {
        "trap_type": ingestion.trap_type,
        "n": ingestion.n,
        "seed": ingestion.seed,
        "k": ingestion.k,
        "broken_count": len(ingestion.broken_questions),
        "target_questions": ingestion.target_questions,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def load_ingestion(
    out_dir: Path, before: Corpus | None = None
) -> Ingestion:
    """Load an ingestion from out_dir. If before is given, reconstructs the after corpus."""
    with (out_dir / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)

    with (out_dir / "broken_questions.json").open(encoding="utf-8") as f:
        broken_questions = json.load(f)

    with (out_dir / "trap_chunks.json").open(encoding="utf-8") as f:
        chunks_data = json.load(f)

    trap_embeddings = np.load(out_dir / "trap_embeddings.npy")
    trap_chunks = [
        TrapChunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            text=c["text"],
            trap_type=c["trap_type"],
            target_question_id=c["target_question_id"],
            source_chunk_ids=c["source_chunk_ids"],
            metadata=c["metadata"],
        )
        for c in chunks_data
    ]

    after = (
        apply_ingestion(before, trap_chunks, trap_embeddings)
        if before is not None
        else None  # type: ignore[assignment]
    )

    return Ingestion(
        trap_type=manifest["trap_type"],
        n=manifest["n"],
        seed=manifest["seed"],
        k=manifest.get("k", DEFAULT_K),
        target_questions=manifest["target_questions"],
        trap_chunks=trap_chunks,
        trap_embeddings=trap_embeddings,
        broken_questions=broken_questions,
        after=after,  # type: ignore[arg-type]
    )


def default_ingestion_dir(data_dir: Path, trap_type: str, n: int, seed: int) -> Path:
    return data_dir / "ingestions" / f"{trap_type}_n{n}_s{seed}"
