"""GraphRAG-style multi-hop retrieval, built on LangChain + NetworkX.

Not Microsoft's `graphrag` package -- that library ships its own indexing
pipeline (Leiden community detection, hierarchical community summarization,
its own config/CLI, parquet artifacts) built for large corpora with real
community structure. Integrating it here wouldn't be meaningfully
exercisable against a small single-repo-scale document set, and it doesn't
compose with the hybrid retriever/reranker/contextual-chunking stack already
built in rag_pipeline.py. This module implements the same conceptual
technique -- LLM-extracted entities/relationships, graph traversal for
multi-hop retrieval -- directly on top of this repo's existing LangChain
plumbing and NetworkX, at a scope that's actually exercisable and
verifiable here. When describing this feature (README, UI copy, etc.), call
it "GraphRAG-style multi-hop retrieval" -- never imply Microsoft's graphrag
library is what's running.

Entity linking at query time is a plain case-insensitive substring match
against known graph node names, not an embedding-based entity linker --
a deliberate scope simplification, noted here so it isn't mistaken for a
production entity-resolution system.
"""
from typing import List, Optional

import networkx as nx
from langchain_core.documents import Document
from pydantic import BaseModel, Field

EXTRACTION_PROMPT = """Extract every distinct factual relationship stated in the following text.
Each relationship is a (source entity, relation, target entity) triple, e.g.
("remote employees", "must be reachable between", "10:00 AM and 3:00 PM").
Keep entity names short and concrete (a policy section, a role, an amount, a
time window, a department, etc.). Only extract relationships actually stated
in the text -- do not infer anything not written.

Text:
{text}"""


class Relationship(BaseModel):
    source: str = Field(description="Source entity, short and concrete.")
    relation: str = Field(description="Short verb phrase describing the relationship.")
    target: str = Field(description="Target entity, short and concrete.")


class ChunkGraph(BaseModel):
    relationships: List[Relationship] = Field(
        description="Every distinct entity relationship stated in this text chunk."
    )


def extract_relationships(chunk_text: str, llm) -> List[Relationship]:
    """One LLM call (structured output) per chunk -- pulls out its (source,
    relation, target) triples. Requires a tool-calling-capable model."""
    structured_llm = llm.with_structured_output(ChunkGraph)
    result = structured_llm.invoke(EXTRACTION_PROMPT.format(text=chunk_text))
    return result.relationships


def build_graph(chunks: List[Document], llm) -> nx.MultiDiGraph:
    """Extract relationships from every chunk and assemble a directed graph.

    Each node is an entity name; node["chunk_indices"] holds every chunk
    index that mentioned it. Each edge carries the relation label plus the
    chunk index it came from, so a traversal result can be mapped back to
    source text.
    """
    graph = nx.MultiDiGraph()
    for idx, chunk in enumerate(chunks):
        for rel in extract_relationships(chunk.page_content, llm):
            for entity in (rel.source, rel.target):
                if graph.has_node(entity):
                    graph.nodes[entity]["chunk_indices"].add(idx)
                else:
                    graph.add_node(entity, chunk_indices={idx})
            graph.add_edge(rel.source, rel.target, relation=rel.relation, chunk_index=idx)
    return graph


def _match_entities(query: str, graph: nx.MultiDiGraph) -> List[str]:
    """Plain substring match between the query and known entity names --
    see module docstring for why this isn't an embedding-based linker."""
    query_lower = query.lower()
    return [node for node in graph.nodes if node.lower() in query_lower or query_lower in node.lower()]


def graph_retrieve(
    query: str,
    graph: nx.MultiDiGraph,
    chunks: List[Document],
    max_hops: int = 2,
    top_n: int = 5,
) -> List[Document]:
    """Match entities mentioned in `query`, walk up to `max_hops` from each in
    the undirected view of the graph, and return the chunks associated with
    every entity reached -- the multi-hop retrieval step. Falls back to an
    empty list (caller should fall back to hybrid retrieval) if no entities
    in the query match the graph at all.
    """
    seed_entities = _match_entities(query, graph)
    if not seed_entities:
        return []

    undirected = graph.to_undirected(as_view=True)
    reached: set[str] = set()
    for entity in seed_entities:
        reached.update(nx.single_source_shortest_path_length(undirected, entity, cutoff=max_hops).keys())

    chunk_indices: set[int] = set()
    for entity in reached:
        chunk_indices.update(graph.nodes[entity].get("chunk_indices", set()))

    return [chunks[i] for i in sorted(chunk_indices)[:top_n]]
