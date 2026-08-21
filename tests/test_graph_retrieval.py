import networkx as nx
from langchain_core.documents import Document

from graph_retrieval import _match_entities, graph_retrieve

CHUNKS = [
    Document(page_content="Chunk 0: about remote work eligibility"),
    Document(page_content="Chunk 1: about the equipment stipend"),
    Document(page_content="Chunk 2: about expense approval chain"),
    Document(page_content="Chunk 3: unrelated chunk about parking"),
]


def _sample_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("remote employees", chunk_indices={0})
    g.add_node("90 days tenure", chunk_indices={0})
    g.add_node("equipment stipend", chunk_indices={1})
    g.add_node("expense report", chunk_indices={2})
    g.add_node("finance department", chunk_indices={2})
    g.add_edge("remote employees", "90 days tenure", relation="requires", chunk_index=0)
    g.add_edge("remote employees", "equipment stipend", relation="receives", chunk_index=1)
    g.add_edge("expense report", "finance department", relation="requires approval from", chunk_index=2)
    return g


def test_one_hop_traversal_reaches_connected_chunks():
    docs = graph_retrieve("What do remote employees get?", _sample_graph(), CHUNKS, max_hops=1, top_n=5)
    assert {d.page_content for d in docs} == {CHUNKS[0].page_content, CHUNKS[1].page_content}


def test_no_entity_match_returns_empty_for_caller_fallback():
    docs = graph_retrieve("completely unrelated question about the weather", _sample_graph(), CHUNKS)
    assert docs == []


def test_traversal_is_undirected():
    # 'finance department' is only the *target* of an edge from 'expense
    # report' -- matching on it should still reach 'expense report' via
    # undirected traversal.
    docs = graph_retrieve("What does the finance department require?", _sample_graph(), CHUNKS, max_hops=1)
    assert CHUNKS[2].page_content in [d.page_content for d in docs]


def test_top_n_truncates_results():
    docs = graph_retrieve("What do remote employees get?", _sample_graph(), CHUNKS, max_hops=1, top_n=1)
    assert len(docs) == 1


def test_match_entities_is_case_insensitive_substring():
    g = _sample_graph()
    assert "remote employees" in _match_entities("What about REMOTE EMPLOYEES policy?", g)
    assert _match_entities("nothing relevant here", g) == []
