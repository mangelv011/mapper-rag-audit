import numpy as np
import pytest

from mapper_rag_audit.chunking import Chunk
from mapper_rag_audit.data import Corpus, Question
from mapper_rag_audit.graph_diff import (
    MapperGraph,
    MapperSelector,
    Rules,
    diff_graphs,
    select_questions,
)

NONE = Rules(new_chunks=False, new_edge=False, split=False, merge=False, neighbours=False)


def graph(nodes: dict[str, set[str]], *edges: str) -> MapperGraph:
    """Edges are written "a-b"."""
    return MapperGraph(
        nodes={k: frozenset(v) for k, v in nodes.items()},
        edges=frozenset(frozenset(e.split("-")) for e in edges),
    )


def only(**flags: bool) -> Rules:
    return Rules(**{**NONE.__dict__, **flags})


# Rule new_chunks
#   before:  A{c1,c2} - B{c3,c4}
#   after:   A{c1,c2,n1} - B{c3,c4}      n1 is new: only A is affected
def test_node_with_a_new_chunk_is_affected():
    before = graph({"A": {"c1", "c2"}, "B": {"c3", "c4"}}, "A-B")
    after = graph({"A": {"c1", "c2", "n1"}, "B": {"c3", "c4"}}, "A-B")
    diff = diff_graphs(before, after, {"n1"}, only(new_chunks=True))
    assert diff.affected == {"A"}
    assert diff.marked == {"A"}
    assert diff.new_chunk_counts == {"A": 1, "B": 0}


def test_new_chunks_flag_off_marks_nothing():
    before = graph({"A": {"c1", "c2"}})
    after = graph({"A": {"c1", "c2", "n1"}})
    diff = diff_graphs(before, after, {"n1"}, NONE)
    assert diff.affected == set()
    assert diff.marked == set()


# Matching
#   before:  A{c1,c2}   B{c3,c4}
#   after:   X{c1,c2,n1}  Y{n2}      X old chunks {c1,c2} equal A; Y has no old chunk
def test_after_node_matches_the_before_node_with_most_overlap_and_new_chunks_do_not_count():
    before = graph({"A": {"c1", "c2"}, "B": {"c3", "c4"}})
    after = graph({"X": {"c1", "c2", "n1"}, "Y": {"n2"}})
    diff = diff_graphs(before, after, {"n1", "n2"}, NONE)
    assert diff.matching == {"X": "A"}


# Tie: old chunks {c1}; P{c1,c2} and Q{c1,c3} both give Jaccard 1/2; smaller id wins
def test_matching_tie_goes_to_the_smaller_node_id():
    before = graph({"Q": {"c1", "c3"}, "P": {"c1", "c2"}})
    after = graph({"X": {"c1"}})
    assert diff_graphs(before, after, set(), NONE).matching == {"X": "P"}


# A cluster inside the overlap of two intervals is a node in both of them, with the same
# chunks: before and after hold the pair i00r0 and i01r0. Both after-nodes have identical old
# chunks, so Jaccard ties, and without the id tie rule both would match i00r0 and the duplicate
# would be counted as a split.
def test_duplicate_nodes_of_overlapping_intervals_are_not_counted_as_a_split():
    before = graph({"i00r0": {"c1", "c2"}, "i01r0": {"c1", "c2"}})
    after = graph({"i00r0": {"c1", "c2", "n1"}, "i01r0": {"c1", "c2", "n1"}})
    diff = diff_graphs(before, after, {"n1"}, only(new_chunks=True, split=True, merge=True))
    assert diff.matching == {"i00r0": "i00r0", "i01r0": "i01r0"}
    assert diff.affected == {"i00r0", "i01r0"}


# Jaccard, not raw overlap: old {c1,c2}; P{c1,c2,c3,c4,c5} shares 2 of 5; Q{c1} shares 1 of 2
def test_matching_uses_jaccard_not_overlap_size():
    before = graph({"P": {"c1", "c2", "c3", "c4", "c5"}, "Q": {"c1"}})
    after = graph({"X": {"c1", "c2"}})
    assert diff_graphs(before, after, set(), NONE).matching == {"X": "Q"}


# Rule new_edge
#   before:  A - B - C            (no A-C)
#   after:   A - B - C, A - C     same nodes, no new chunks: A and C are affected, B is not
def test_edge_between_nodes_that_were_not_neighbours_marks_both_ends():
    nodes = {"A": {"c1", "c2"}, "B": {"c3", "c4"}, "C": {"c5", "c6"}}
    before = graph(nodes, "A-B", "B-C")
    after = graph(nodes, "A-B", "B-C", "A-C")
    diff = diff_graphs(before, after, set(), only(new_edge=True))
    assert diff.affected == {"A", "C"}


def test_edge_that_already_existed_between_the_matched_nodes_marks_nothing():
    # renamed nodes: X matches A, Y matches B, and A-B was an edge before
    before = graph({"A": {"c1", "c2"}, "B": {"c3", "c4"}}, "A-B")
    after = graph({"X": {"c1", "c2"}, "Y": {"c3", "c4"}}, "X-Y")
    assert diff_graphs(before, after, set(), only(new_edge=True)).affected == set()


def test_new_edge_flag_off_ignores_new_edges():
    nodes = {"A": {"c1"}, "B": {"c2"}}
    diff = diff_graphs(graph(nodes), graph(nodes, "A-B"), set(), only(new_chunks=True))
    assert diff.affected == set()


# An unmatched endpoint (only new chunks) is left to new_chunks, not new_edge
#   before:  A{c1}
#   after:   A{c1} - N{n1}
def test_new_edge_ignores_an_unmatched_endpoint():
    before = graph({"A": {"c1"}})
    after = graph({"A": {"c1"}, "N": {"n1"}}, "A-N")
    assert diff_graphs(before, after, {"n1"}, only(new_edge=True)).affected == set()


# Rule split
#   before:  A{c1,c2,c3,c4}   B{c5}
#   after:   X{c1,c2} - Y{c3,c4}   Z{c5}     X and Y both match A: both affected, Z is not
SPLIT_BEFORE = graph({"A": {"c1", "c2", "c3", "c4"}, "B": {"c5"}})
SPLIT_AFTER = graph({"X": {"c1", "c2"}, "Y": {"c3", "c4"}, "Z": {"c5"}}, "X-Y")


def test_nodes_matched_to_the_same_before_node_are_all_affected():
    diff = diff_graphs(SPLIT_BEFORE, SPLIT_AFTER, set(), only(split=True))
    assert diff.affected == {"X", "Y"}


def test_split_flag_off_ignores_splits():
    assert diff_graphs(SPLIT_BEFORE, SPLIT_AFTER, set(), only(new_chunks=True)).affected == set()


def test_edge_between_two_halves_of_a_split_is_not_a_new_edge():
    assert diff_graphs(SPLIT_BEFORE, SPLIT_AFTER, set(), only(new_edge=True)).affected == set()


# Rule merge
#   before:  A{c1,c2}   B{c3,c4}   C{c5}
#   after:   X{c1,c2,c3,c4}   Z{c5}     A and B both map to X: X is affected, Z is not
MERGE_BEFORE = graph({"A": {"c1", "c2"}, "B": {"c3", "c4"}, "C": {"c5"}})
MERGE_AFTER = graph({"X": {"c1", "c2", "c3", "c4"}, "Z": {"c5"}})


def test_after_node_that_two_before_nodes_map_to_is_affected():
    diff = diff_graphs(MERGE_BEFORE, MERGE_AFTER, set(), only(merge=True))
    assert diff.affected == {"X"}


def test_merge_flag_off_ignores_merges():
    assert diff_graphs(MERGE_BEFORE, MERGE_AFTER, set(), only(split=True)).affected == set()


# Neighbour expansion
#   after:   A - B - C - D      only B holds a new chunk
#   marked:  A, B, C (direct neighbours of B), not D (two steps away)
CHAIN = {"A": {"c1"}, "B": {"c2", "n1"}, "C": {"c3"}, "D": {"c4"}}
CHAIN_BEFORE = graph({"A": {"c1"}, "B": {"c2"}, "C": {"c3"}, "D": {"c4"}}, "A-B", "B-C", "C-D")
CHAIN_AFTER = graph(CHAIN, "A-B", "B-C", "C-D")


def test_neighbours_of_affected_nodes_are_marked_but_not_affected():
    diff = diff_graphs(CHAIN_BEFORE, CHAIN_AFTER, {"n1"}, only(new_chunks=True, neighbours=True))
    assert diff.affected == {"B"}
    assert diff.marked == {"A", "B", "C"}


def test_neighbours_flag_off_marks_only_affected_nodes():
    diff = diff_graphs(CHAIN_BEFORE, CHAIN_AFTER, {"n1"}, only(new_chunks=True))
    assert diff.marked == {"B"}


# Ranking and budget
#   before:  A{c1}  B{c2} - C{c3}  D{c4}
#   after:   A{c1,n1,n2}  B{c2,n3} - C{c3}  D{c4}
#   affected A (2 new), B (1 new); marked also C (neighbour of B); D is not marked
RANK_BEFORE = graph({"A": {"c1"}, "B": {"c2"}, "C": {"c3"}, "D": {"c4"}}, "B-C")
RANK_AFTER = graph({"A": {"c1", "n1", "n2"}, "B": {"c2", "n3"}, "C": {"c3"}, "D": {"c4"}}, "B-C")
NEW = {"n1", "n2", "n3"}
QRELS = {
    "q5": ["c2", "c1"],  # best node is A (2 new), same key as q1, so id order decides
    "q4": ["c4"],  # D is not marked
    "q3": ["c3"],  # neighbour only
    "q2": ["c2"],  # affected, 1 new
    "q1": ["c1"],  # affected, 2 new
    "q0": ["missing"],  # chunk is in no node
}


def select(budget: int) -> list[str]:
    return select_questions(RANK_BEFORE, RANK_AFTER, NEW, QRELS, budget)


def test_questions_are_ranked_by_tier_then_new_chunks_then_id():
    assert select(10) == ["q1", "q5", "q2", "q3"]


def test_budget_cuts_the_ranking():
    assert select(2) == ["q1", "q5"]


def test_budget_zero_selects_nothing():
    assert select(0) == []


def test_budget_larger_than_the_candidates_does_not_pad():
    assert len(select(100)) == 4


def test_neighbour_question_ranks_after_affected_even_with_more_new_chunks():
    # split marks X and Y with no new chunks; N is their neighbour and holds one new chunk
    #   before:  A{c1,c2}   after:  X{c1} - Y{c2} - N{c3,n1}
    before = graph({"A": {"c1", "c2"}, "B": {"c3"}})
    after = graph({"X": {"c1"}, "Y": {"c2"}, "N": {"c3", "n1"}}, "Y-N")
    rules = Rules(new_chunks=False, new_edge=False, split=True, merge=False, neighbours=True)
    qrels = {"a": ["c3"], "b": ["c1"]}
    assert select_questions(before, after, {"n1"}, qrels, 10, rules) == ["b", "a"]


def test_question_with_a_chunk_in_an_affected_and_a_neighbour_node_counts_as_affected():
    before = graph({"A": {"c1"}, "B": {"c2"}}, "A-B")
    after = graph({"A": {"c1", "n1"}, "B": {"c2", "c1"}}, "A-B")
    rules = only(new_chunks=True, neighbours=True)
    # c1 sits in A (affected) and in B (neighbour of A); c2 only in B
    assert select_questions(before, after, {"n1"}, {"p": ["c2"], "q": ["c1"]}, 10, rules) == [
        "q",
        "p",
    ]


# Jaccard divides by the union: old {c1..c4}; P{c1} shares 1 of 4, Q{c1..c5} shares 4 of 5
def test_matching_prefers_the_larger_jaccard_over_the_smaller_candidate():
    before = graph({"P": {"c1"}, "Q": {"c1", "c2", "c3", "c4", "c5"}})
    after = graph({"X": {"c1", "c2", "c3", "c4"}})
    assert diff_graphs(before, after, set(), NONE).matching == {"X": "Q"}


# New chunks are removed before matching. X has old chunks {c1,c2} and three new ones.
# Old chunks only: P scores 1/3, Q scores 2/7, so P wins. With the new chunks counted:
# P scores 1/6, Q scores 2/10, and Q would win.
def test_matching_ignores_new_chunks_in_the_after_node():
    before = graph({"P": {"c1", "c3"}, "Q": {"c1", "c2", "c4", "c5", "c6", "c7", "c8"}})
    after = graph({"X": {"c1", "c2", "n1", "n2", "n3"}})
    assert diff_graphs(before, after, {"n1", "n2", "n3"}, NONE).matching == {"X": "P"}


# Reverse matching also ignores new chunks. On old chunks A{c1} maps to X (1 vs 1/3); counting the
# new chunks it would map to W (1/4 vs 1/3), and then A and B would both map to W: a false merge.
#   before:  A{c1}   B{c2,c3}
#   after:   X{c1,n1,n2,n3}   W{c1,c2,c3}
def test_reverse_matching_ignores_new_chunks_of_the_after_node():
    before = graph({"A": {"c1"}, "B": {"c2", "c3"}})
    after = graph({"X": {"c1", "n1", "n2", "n3"}, "W": {"c1", "c2", "c3"}})
    assert diff_graphs(before, after, {"n1", "n2", "n3"}, only(merge=True)).affected == set()


# A question whose chunk lies in an affected and a neighbour node is in the affected tier.
#   before:  A{c1,c2}   B{c3}
#   after:   X{c1} - Y{c2} - N{c3,c1,n1}    X, Y split from A (affected, 0 new); N is Y's neighbour
#   q holds c1 (in X and N), p holds c3 (only N, same new-chunk count): q wins on tier, not on id
def test_one_affected_holder_puts_the_question_in_the_affected_tier():
    before = graph({"A": {"c1", "c2"}, "B": {"c3"}})
    after = graph({"X": {"c1"}, "Y": {"c2"}, "N": {"c3", "c1", "n1"}}, "X-Y", "Y-N")
    rules = Rules(new_chunks=False, new_edge=False, split=True, merge=False, neighbours=True)
    assert select_questions(before, after, {"n1"}, {"q": ["c1"], "p": ["c3"]}, 10, rules) == [
        "q",
        "p",
    ]


# No ingestion: the same overlapping graph before and after, renamed nodes.
#   before:  A{c1,c2,c3} - B{c3,c4,c5} - C{c5,c6}
#   after:   X{c1,c2,c3} - Y{c3,c4,c5} - Z{c5,c6}
# All rules on: nothing may be affected, so nothing is selected.
def test_unchanged_graph_with_overlapping_nodes_selects_nothing():
    before = graph(
        {"A": {"c1", "c2", "c3"}, "B": {"c3", "c4", "c5"}, "C": {"c5", "c6"}}, "A-B", "B-C"
    )
    after = graph(
        {"X": {"c1", "c2", "c3"}, "Y": {"c3", "c4", "c5"}, "Z": {"c5", "c6"}}, "X-Y", "Y-Z"
    )
    diff = diff_graphs(before, after, set())
    assert diff.matching == {"X": "A", "Y": "B", "Z": "C"}
    assert diff.affected == set()
    assert select_questions(before, after, set(), {"q1": ["c1"], "q2": ["c4"]}, budget=5) == []


# MapperSelector: fake fit/build, tiny corpora. Embeddings are zeros; the fakes ignore them.
def corpus(texts: dict[str, str], qrels: dict[str, list[str]]) -> Corpus:
    return Corpus(
        chunks=[Chunk(chunk_id=c, doc_id=c.split("#")[0], text=t) for c, t in texts.items()],
        embeddings=np.zeros((len(texts), 2), dtype=np.float32),
        questions=[Question(q, q) for q in qrels],
        question_embeddings=np.zeros((len(qrels), 2), dtype=np.float32),
        qrels=qrels,
    )


class Fakes:
    """fit and build that record their calls; build returns the graph registered for a corpus."""

    def __init__(self, graphs: dict[int, MapperGraph]):
        self.graphs = graphs
        self.fit_calls: list[Corpus] = []
        self.build_calls: list[tuple[Corpus, object]] = []

    def fit(self, corpus: Corpus) -> object:
        self.fit_calls.append(corpus)
        return object()

    def build(self, corpus: Corpus, params: object) -> MapperGraph:
        self.build_calls.append((corpus, params))
        return self.graphs[id(corpus)]


# before:  A{a#0,a#1} - B{b#0}        after adds n#0 to A: q_a (correct chunk a#0) is selected
def setup_one_added_chunk(**selector_args):
    before = corpus({"a#0": "x", "a#1": "y", "b#0": "z"}, {"q_a": ["a#0"], "q_b": ["b#0"]})
    after = corpus(
        {"a#0": "x", "a#1": "y", "b#0": "z", "n#0": "new"}, {"q_a": ["a#0"], "q_b": ["b#0"]}
    )
    graph_before = graph({"A": {"a#0", "a#1"}, "B": {"b#0"}})
    graph_after = graph({"A": {"a#0", "a#1", "n#0"}, "B": {"b#0"}})
    fakes = Fakes({id(before): graph_before, id(after): graph_after})
    selector = MapperSelector(fakes.fit, fakes.build, **selector_args)
    return selector, fakes, before, after


def test_an_added_chunk_selects_the_questions_of_its_node():
    selector, _, before, after = setup_one_added_chunk(rules=only(new_chunks=True))
    assert selector.select(before, after, ["q_a", "q_b"], 5) == ["q_a"]


def test_name_defaults_to_mapper_and_can_be_set():
    selector, *_ = setup_one_added_chunk()
    assert selector.name == "mapper"
    assert setup_one_added_chunk(name="m2")[0].name == "m2"


def test_fit_runs_once_on_before_and_both_graphs_get_the_same_params():
    selector, fakes, before, after = setup_one_added_chunk()
    selector.select(before, after, ["q_a", "q_b"], 5)
    assert fakes.fit_calls == [before]
    assert [c for c, _ in fakes.build_calls] == [before, after]
    params = {id(p) for _, p in fakes.build_calls}
    assert len(params) == 1


def test_a_chunk_with_changed_text_counts_as_new():
    before = corpus({"a#0": "old", "b#0": "z"}, {"q_a": ["a#0"], "q_b": ["b#0"]})
    after = corpus({"a#0": "edited", "b#0": "z"}, {"q_a": ["a#0"], "q_b": ["b#0"]})
    graphs = {
        id(before): graph({"A": {"a#0"}, "B": {"b#0"}}),
        id(after): graph({"A": {"a#0"}, "B": {"b#0"}}),
    }
    fakes = Fakes(graphs)
    selector = MapperSelector(fakes.fit, fakes.build, only(new_chunks=True))
    assert selector.select(before, after, ["q_a", "q_b"], 5) == ["q_a"]


def test_an_unchanged_chunk_is_not_new():
    before = corpus({"a#0": "x"}, {"q_a": ["a#0"]})
    same = corpus({"a#0": "x"}, {"q_a": ["a#0"]})
    fakes = Fakes({id(before): graph({"A": {"a#0"}}), id(same): graph({"A": {"a#0"}})})
    selector = MapperSelector(fakes.fit, fakes.build, only(new_chunks=True))
    assert selector.select(before, same, ["q_a"], 5) == []


def test_questions_outside_the_pool_are_never_returned():
    selector, _, before, after = setup_one_added_chunk()
    assert selector.select(before, after, ["q_b"], 5) == []


def test_pool_ids_without_qrels_are_ignored():
    selector, _, before, after = setup_one_added_chunk()
    assert selector.select(before, after, ["q_a", "unknown"], 5) == ["q_a"]


def test_budget_caps_the_selection():
    before = corpus({"a#0": "x", "b#0": "z"}, {"q1": ["a#0"], "q2": ["b#0"]})
    after = corpus({"a#0": "x", "b#0": "z", "n#0": "n", "n#1": "m"}, before.qrels)
    graphs = {
        id(before): graph({"A": {"a#0"}, "B": {"b#0"}}),
        id(after): graph({"A": {"a#0", "n#0"}, "B": {"b#0", "n#1"}}),
    }
    fakes = Fakes(graphs)
    selector = MapperSelector(fakes.fit, fakes.build)
    assert selector.select(before, after, ["q1", "q2"], 1) == ["q1"]
    assert selector.select(before, after, ["q1", "q2"], 2) == ["q1", "q2"]


def test_removed_chunks_raise_and_say_how_many():
    before = corpus({"a#0": "x", "a#1": "y", "b#0": "z"}, {"q": ["a#0"]})
    after = corpus({"a#0": "x"}, {"q": ["a#0"]})
    fakes = Fakes({})
    selector = MapperSelector(fakes.fit, fakes.build)
    with pytest.raises(ValueError, match="chunks removed by the ingestion: 2"):
        selector.select(before, after, ["q"], 5)
    assert fakes.fit_calls == []


def test_rules_are_passed_to_the_diff():
    # With neighbours on, q_b (in B, next to affected A) is selected as well.
    before = corpus({"a#0": "x", "b#0": "z"}, {"q_a": ["a#0"], "q_b": ["b#0"]})
    after = corpus({"a#0": "x", "b#0": "z", "n#0": "n"}, before.qrels)
    graphs = {
        id(before): graph({"A": {"a#0"}, "B": {"b#0"}}, "A-B"),
        id(after): graph({"A": {"a#0", "n#0"}, "B": {"b#0"}}, "A-B"),
    }
    fakes = Fakes(graphs)
    on = MapperSelector(fakes.fit, fakes.build, only(new_chunks=True, neighbours=True))
    off = MapperSelector(fakes.fit, fakes.build, only(new_chunks=True, neighbours=False))
    assert on.select(before, after, ["q_a", "q_b"], 5) == ["q_a", "q_b"]
    assert off.select(before, after, ["q_a", "q_b"], 5) == ["q_a"]


def test_qrels_come_from_the_corpus_before_the_ingestion():
    before = corpus({"a#0": "x", "b#0": "z"}, {"q": ["a#0"]})
    after = corpus({"a#0": "x", "b#0": "z", "n#0": "n"}, {"q": ["b#0"]})
    fakes = Fakes(
        {
            id(before): graph({"A": {"a#0"}, "B": {"b#0"}}),
            id(after): graph({"A": {"a#0", "n#0"}, "B": {"b#0"}}),
        }
    )
    selector = MapperSelector(fakes.fit, fakes.build, only(new_chunks=True))
    assert selector.select(before, after, ["q"], 5) == ["q"]
