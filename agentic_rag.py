"""Agentic router: classify each question and send it down the cheapest
retrieval path that can actually answer it, adapting LangGraph's published
Adaptive-RAG / Corrective-RAG (CRAG) reference pattern
(langchain-ai/langgraph, examples/rag/langgraph_adaptive_rag.ipynb and
langgraph_crag.ipynb) -- same shape: a structured-output router picks a
route, retrieval is followed by an LLM relevance grade per document, and a
conditional edge decides generate vs. rewrite-and-retry.

Routes:
  - simple    -> the existing hybrid retriever (rag_pipeline.py), straight
                 to generation. No grading overhead -- this mirrors the
                 Default prompting path exactly.
  - multi_hop -> graph_retrieval.py's GraphRAG-style multi-hop retrieval
                 (falls back to hybrid retrieval if no graph entities match
                 the question), straight to generation.
  - complex   -> query decomposition (2-3 sub-questions) + a bounded
                 CRAG-style grade/rewrite/retry loop per sub-question, then
                 a synthesis step combining the sub-answers. This is the
                 "agent loop with query decomposition + CRAG-style
                 self-correction" branch.

The per-sub-question grade/rewrite/retry loop is implemented as ordinary
bounded control flow inside one node function rather than as its own nested
LangGraph nodes -- a node function can contain normal Python logic; it
doesn't need a sub-graph just to run a bounded loop. Keeps the top-level
graph the same shape as the reference tutorials it's adapted from. Every
LLM call inside that loop still gets the run's `config` forwarded to it
(see the node functions in build_app below), so each grade/rewrite/answer
call still shows up as its own nested span in Langfuse -- observability
isn't lost by not being a separate graph node.
"""
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from graph_retrieval import build_graph, graph_retrieve
from observability import trace_config
from query_translation import chatgpt
from rag_pipeline import QA_SYSTEM_PROMPT, build_hybrid_retriever, build_vector_store, load_and_chunk

MAX_SUBQUESTIONS = 3
MAX_RETRIES_PER_SUBQUESTION = 1


class RouteQuery(BaseModel):
    route: Literal["simple", "multi_hop", "complex"] = Field(
        description=(
            "'simple' for a single-fact lookup answerable from one document/chunk; "
            "'multi_hop' for a question that requires connecting facts across "
            "multiple entities or relationships (relational, multi-step); "
            "'complex' for an ambiguous, multi-part, or high-stakes question that "
            "needs decomposition and careful self-checking before answering."
        )
    )


class GradeDocument(BaseModel):
    relevant: bool = Field(description="True if the document is relevant to answering the question.")


class SubQuestions(BaseModel):
    questions: List[str] = Field(description="2 to 3 sub-questions that together answer the original question.")


class AgentState(TypedDict, total=False):
    question: str
    route: str
    documents: List[Document]
    generation: str


def route_question(question: str, llm, config=None) -> str:
    router = llm.with_structured_output(RouteQuery)
    return router.invoke(
        f"Classify this question for retrieval routing.\n\nQuestion: {question}",
        config=config,
    ).route


def grade_documents(question: str, documents: List[Document], llm, config=None) -> List[Document]:
    """CRAG-style relevance grading: one structured-output LLM call per
    document, keep only the ones graded relevant."""
    grader = llm.with_structured_output(GradeDocument)
    relevant = []
    for doc in documents:
        grade = grader.invoke(
            f"Question: {question}\n\nDocument:\n{doc.page_content}\n\n"
            "Is this document relevant to answering the question?",
            config=config,
        )
        if grade.relevant:
            relevant.append(doc)
    return relevant


def transform_query(question: str, llm, config=None) -> str:
    result = llm.invoke(
        "Rewrite this question to be more specific and better suited for "
        f"retrieval from a document search index. Answer with only the "
        f"rewritten question.\n\nOriginal question: {question}",
        config=config,
    )
    return result.content.strip()


def decompose_question(question: str, llm, config=None) -> List[str]:
    decomposer = llm.with_structured_output(SubQuestions)
    result = decomposer.invoke(
        "Break this question into 2 to 3 sub-questions that, answered "
        f"together, fully answer it. If it's already a single simple "
        f"question, return it unchanged as the only sub-question.\n\n"
        f"Question: {question}",
        config=config,
    )
    return result.questions[:MAX_SUBQUESTIONS]


def build_app(chunks: List[Document], llm=None):
    """Construct the routed StateGraph over already-parsed-and-chunked
    documents. Returns a compiled LangGraph app with
    .invoke({"question": ...}).

    Takes `chunks` rather than PDF paths so callers that already have chunks
    in hand (e.g. app.py, which parses uploaded files into a Streamlit temp
    directory that may not outlive the request) don't need to re-parse from
    disk. Use `build_app_from_pdfs` below for the file-path convenience path.
    """
    llm = llm or chatgpt
    vector_store = build_vector_store(chunks)
    hybrid_retriever = build_hybrid_retriever(vector_store, chunks)
    graph = build_graph(chunks, llm)

    # Every node below takes a `config` second argument -- LangGraph injects
    # the run's config (including Langfuse callbacks, if configured; see
    # observability.py) into any node function whose signature accepts it.
    # Forwarding that same config into every nested .invoke() call is what
    # nests the router/retrieve/grade/rewrite/generate/decompose LLM calls
    # under one trace per question in Langfuse instead of each showing up as
    # an unrelated, disconnected span.

    def node_route(state: AgentState, config) -> AgentState:
        return {"route": route_question(state["question"], llm, config=config)}

    def node_retrieve_simple(state: AgentState, config) -> AgentState:
        docs = hybrid_retriever.invoke(state["question"], config=config)
        return {"documents": docs}

    def node_retrieve_multi_hop(state: AgentState, config) -> AgentState:
        docs = graph_retrieve(state["question"], graph, chunks)
        if not docs:
            docs = hybrid_retriever.invoke(state["question"], config=config)
        return {"documents": docs}

    def node_generate(state: AgentState, config) -> AgentState:
        context = "\n\n".join(d.page_content for d in state.get("documents", []))
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("system", "Context: {context}"),
                ("human", "{input}"),
            ]
        )
        chain = answer_prompt | llm
        response = chain.invoke({"context": context, "input": state["question"]}, config=config)
        return {"generation": response.content}

    def node_complex(state: AgentState, config) -> AgentState:
        """Decompose -> per-sub-question retrieve/grade/rewrite-retry/answer
        -> synthesize. The bounded CRAG loop lives here (see module
        docstring for why this isn't its own set of graph nodes)."""
        sub_questions = decompose_question(state["question"], llm, config=config)
        qa_pairs = []
        all_relevant_docs: List[Document] = []
        for sub_q in sub_questions:
            docs = hybrid_retriever.invoke(sub_q, config=config)
            relevant = grade_documents(sub_q, docs, llm, config=config)
            retries = 0
            current_q = sub_q
            while not relevant and retries < MAX_RETRIES_PER_SUBQUESTION:
                current_q = transform_query(current_q, llm, config=config)
                docs = hybrid_retriever.invoke(current_q, config=config)
                relevant = grade_documents(sub_q, docs, llm, config=config)
                retries += 1

            all_relevant_docs.extend(relevant)
            context = "\n\n".join(d.page_content for d in relevant)
            sub_answer = llm.invoke(
                f"Context:\n{context}\n\nQuestion: {sub_q}\n\n"
                "Answer concisely using only the context above. If the "
                "context doesn't contain the answer, say so.",
                config=config,
            ).content
            qa_pairs.append(f"Sub-question: {sub_q}\nAnswer: {sub_answer}")

        synthesis = llm.invoke(
            "Combine these sub-question answers into one clear, complete "
            f"answer to the original question.\n\nOriginal question: {state['question']}\n\n"
            + "\n\n".join(qa_pairs),
            config=config,
        ).content
        return {"generation": synthesis, "documents": all_relevant_docs}

    def route_edge(state: AgentState) -> str:
        return state["route"]

    workflow = StateGraph(AgentState)
    workflow.add_node("route_question", node_route)
    workflow.add_node("retrieve_simple", node_retrieve_simple)
    workflow.add_node("retrieve_multi_hop", node_retrieve_multi_hop)
    workflow.add_node("generate", node_generate)
    workflow.add_node("complex", node_complex)

    workflow.set_entry_point("route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_edge,
        {"simple": "retrieve_simple", "multi_hop": "retrieve_multi_hop", "complex": "complex"},
    )
    workflow.add_edge("retrieve_simple", "generate")
    workflow.add_edge("retrieve_multi_hop", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("complex", END)

    return workflow.compile()


def build_app_from_pdfs(pdf_paths: List[str], llm=None, **chunk_kwargs):
    """Convenience path for callers (eval scripts, CLI) that start from PDF
    files on disk rather than already-parsed chunks."""
    chunks = load_and_chunk(pdf_paths, **chunk_kwargs)
    return build_app(chunks, llm=llm)
