"""Pick the questions to re-run after an ingestion by comparing two Mapper graphs.

The graph format is provisional; the Mapper builder will produce it. An edge means the two
nodes share at least one chunk, but edges are taken as given and never recomputed.

Both graphs come from the same Mapper parameters: `graph_before` on the corpus before the
ingestion, `graph_after` on the corpus after it.

Matching. For an after-node v, its old chunks are v minus the new chunks. v is matched to the
before-node u that maximises Jaccard(old chunks of v, u). Ties go to the smaller node id. A node
with no old chunks, or no overlap with any before-node, has no match. The reverse matching maps
each before-node u to the after-node v maximising Jaccard(u, old chunks of v), same tie rule.

Rules that mark an after-node as affected; each has a flag in `Rules`, all on by default:
- new_chunks: v contains at least one new chunk.
- new_edge: an edge {v, w} of graph_after whose endpoints are both matched, to different
  before-nodes m(v) != m(w), where {m(v), m(w)} is not an edge of graph_before. Both v and w
  are affected. An unmatched endpoint is left to new_chunks.
- split: two or more after-nodes are matched to the same before-node. All of them are affected.
- merge: two or more before-nodes have the same after-node as reverse match. That after-node is
  affected.

The neighbours flag widens the set: marked = affected plus the graph_after neighbours of
affected nodes. With it off, marked = affected.

Ranking. A question is a candidate if one of its correct chunks lies in a marked node. Sort key:
affected tier first (0 if one of those chunks lies in an affected node, else 1), then the largest
number of new chunks among the marked nodes holding one of its correct chunks (more first), then
question id. The first `budget` ids are returned; a budget above the number of candidates returns
all of them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MapperGraph:
    nodes: dict[str, frozenset[str]]  # node id -> chunk ids
    edges: frozenset[frozenset[str]]  # each edge is a set of two node ids


@dataclass(frozen=True)
class Rules:
    new_chunks: bool = True
    new_edge: bool = True
    split: bool = True
    merge: bool = True
    neighbours: bool = True


@dataclass(frozen=True)
class GraphDiff:
    matching: dict[str, str]  # after-node id -> before-node id; unmatched nodes are absent
    affected: frozenset[str]
    marked: frozenset[str]
    new_chunk_counts: dict[str, int]  # after-node id -> number of new chunks in it


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


def _match(chunks: frozenset[str], candidates: dict[str, frozenset[str]]) -> str | None:
    best, best_score = None, 0.0
    for node_id in sorted(candidates):
        score = _jaccard(chunks, candidates[node_id])
        if score > best_score:
            best, best_score = node_id, score
    return best


def _match_all(
    sources: dict[str, frozenset[str]], targets: dict[str, frozenset[str]]
) -> dict[str, str]:
    matched = {source: _match(chunks, targets) for source, chunks in sources.items()}
    return {source: target for source, target in matched.items() if target is not None}


def _sources_by_target(matching: dict[str, str]) -> list[tuple[str, set[str]]]:
    """Each target hit by two or more sources, with those sources."""
    grouped: dict[str, set[str]] = {}
    for source, target in matching.items():
        grouped.setdefault(target, set()).add(source)
    return [(target, sources) for target, sources in grouped.items() if len(sources) > 1]


def diff_graphs(
    graph_before: MapperGraph,
    graph_after: MapperGraph,
    new_chunk_ids: set[str],
    rules: Rules = Rules(),
) -> GraphDiff:
    old_after = {v: chunks - new_chunk_ids for v, chunks in graph_after.nodes.items()}
    matching = _match_all(old_after, graph_before.nodes)
    counts = {v: len(chunks & new_chunk_ids) for v, chunks in graph_after.nodes.items()}

    affected: set[str] = set()
    if rules.new_chunks:
        affected |= {v for v, n in counts.items() if n}
    if rules.new_edge:
        for edge in graph_after.edges:
            v, w = sorted(edge)
            if v not in matching or w not in matching or matching[v] == matching[w]:
                continue
            if frozenset((matching[v], matching[w])) not in graph_before.edges:
                affected |= edge
    if rules.split:
        for _, sources in _sources_by_target(matching):
            affected |= sources
    if rules.merge:
        reverse = _match_all(graph_before.nodes, old_after)
        affected |= {target for target, _ in _sources_by_target(reverse)}

    marked = set(affected)
    if rules.neighbours:
        for edge in graph_after.edges:
            if edge & affected:
                marked |= edge
    return GraphDiff(matching, frozenset(affected), frozenset(marked), counts)


def select_questions(
    graph_before: MapperGraph,
    graph_after: MapperGraph,
    new_chunk_ids: set[str],
    qrels: dict[str, list[str]],
    budget: int,
    rules: Rules = Rules(),
) -> list[str]:
    diff = diff_graphs(graph_before, graph_after, new_chunk_ids, rules)
    keys: dict[str, tuple[int, int, str]] = {}
    for question_id, chunk_ids in qrels.items():
        correct = set(chunk_ids)
        holders = [v for v in diff.marked if graph_after.nodes[v] & correct]
        if holders:
            tier = 0 if any(v in diff.affected for v in holders) else 1
            most_new = max(diff.new_chunk_counts[v] for v in holders)
            keys[question_id] = (tier, -most_new, question_id)
    return sorted(keys, key=keys.__getitem__)[: max(budget, 0)]
