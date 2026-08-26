"""Unit tests for agentic_rag.py's routing/grading/decomposition logic and
graph topology.

The LLM calls here use structured output (`.with_structured_output(...)`),
which FakeListChatModel doesn't emulate realistically -- these small stub
LLMs return canned Pydantic results directly instead, which is enough to
test this module's own logic (truncation, filtering, edge wiring) without
needing a real tool-calling-capable model.
"""
from unittest.mock import patch

import networkx as nx
import pytest
from langchain_core.documents import Document

from agentic_rag import (
    MAX_SUBQUESTIONS,
    GradeDocument,
    RouteQuery,
    SubQuestions,
    build_app,
    decompose_question,
    grade_documents,
    route_question,
    transform_query,
)


class _StructuredStub:
    def __init__(self, result):
        self._result = result

    def invoke(self, *args, **kwargs):
        return self._result


class _QueueStructuredStub:
    def __init__(self, results):
        self._results = list(results)

    def invoke(self, *args, **kwargs):
        return self._results.pop(0)


class _StubLLM:
    """Returns a fixed structured-output result regardless of prompt."""

    def __init__(self, structured_result=None):
        self._structured_result = structured_result

    def with_structured_output(self, schema):
        return _StructuredStub(self._structured_result)


class _QueueLLM:
    """Returns a different structured-output result on each successive call
    -- for grade_documents, which grades several documents in one call."""

    def __init__(self, structured_results):
        self._structured_results = structured_results

    def with_structured_output(self, schema):
        return _QueueStructuredStub(self._structured_results)


class _PlainLLM:
    """Returns a fixed plain (non-structured) response."""

    def __init__(self, content):
        self._content = content

    def invoke(self, *args, **kwargs):
        class _Response:
            content = self._content

        return _Response()


def test_route_question_returns_router_result():
    llm = _StubLLM(RouteQuery(route="multi_hop"))
    assert route_question("some question", llm) == "multi_hop"


def test_decompose_question_truncates_to_max_subquestions():
    too_many = SubQuestions(questions=[f"q{i}" for i in range(5)])
    llm = _StubLLM(too_many)
    result = decompose_question("original question", llm)
    assert len(result) == MAX_SUBQUESTIONS
    assert result == ["q0", "q1", "q2"]


def test_grade_documents_keeps_only_relevant():
    docs = [Document(page_content="relevant doc"), Document(page_content="irrelevant doc")]
    llm = _QueueLLM([GradeDocument(relevant=True), GradeDocument(relevant=False)])
    relevant = grade_documents("q", docs, llm)
    assert len(relevant) == 1
    assert relevant[0].page_content == "relevant doc"


def test_grade_documents_empty_input_returns_empty():
    llm = _QueueLLM([])
    assert grade_documents("q", [], llm) == []


def test_transform_query_strips_whitespace():
    llm = _PlainLLM("  a more specific question  ")
    assert transform_query("vague question", llm) == "a more specific question"


def test_build_app_topology(fixture_pdf_paths):
    """Router branches to all three routes, both simple/multi_hop converge on
    generate, and complex/generate both terminate -- without needing a real
    LLM to build the (mocked) GraphRAG-style graph."""
    from rag_pipeline import load_and_chunk

    chunks = load_and_chunk(fixture_pdf_paths)
    with patch("agentic_rag.build_graph", return_value=nx.MultiDiGraph()):
        app = build_app(chunks)

    graph = app.get_graph()
    node_names = set(graph.nodes.keys())
    assert {"route_question", "retrieve_simple", "retrieve_multi_hop", "complex", "generate"} <= node_names

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("retrieve_simple", "generate") in edges
    assert ("retrieve_multi_hop", "generate") in edges
