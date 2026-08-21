"""Reusable RAG pipeline: parse -> chunk -> index -> hybrid retrieve -> generate.

This mirrors the "Default (Based on User Query)" path in app.py, but as
plain importable functions with no Streamlit dependency, so it can be
driven by non-interactive callers -- eval scripts, CI, tests -- without a
live Streamlit session. app.py's interactive chat keeps its own
history-aware retrieval chain (multi-turn chat needs chat_history the
single-turn eval path here does not); this module is the single-turn core
those two use cases share.
"""
from typing import Iterable, Optional

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from chunking_strategies import CHUNKING_STRATEGY
from contextual_retrieval import contextualize_chunks
from parser import PARSING_PDF
from query_translation import chatgpt, embeddings

QA_SYSTEM_PROMPT = """You are an intelligent and detail-oriented QA Chatbot designed to generate comprehensive and accurate answers.
Your primary task is to provide a clear, detailed, and context-aware response to the user's question.

Instructions:
- Carefully analyze the provided context and use it to craft a complete and precise answer.
- If the question is ambiguous or lacks sufficient detail, take help from user .
- Ensure the answer is structured, easy to understand, and includes examples or explanations where necessary.
- use provided context whenever it is necessary ,otherwise dont use it .
"""


def load_and_chunk(
    pdf_paths: Iterable[str],
    parsing_strategy: str = "PyMuPDFLoader",
    chunking_strategy: str = "RecursiveCharacterTextSplitter",
    use_contextual: bool = False,
    contextual_llm=None,
):
    """Parse every PDF in `pdf_paths` and split the combined docs into chunks.

    If `use_contextual` is True, each chunk gets an LLM-generated blurb
    situating it within its source document prepended before it's returned
    (see contextual_retrieval.py) -- one extra LLM call per chunk, so this is
    opt-in rather than the default.
    """
    docs = []
    for path in pdf_paths:
        docs.extend(PARSING_PDF(parsing_strategy, path))
    splitter = CHUNKING_STRATEGY(chunking_strategy)
    chunks = splitter.split_documents(docs)

    if use_contextual:
        chunks = contextualize_chunks(chunks, contextual_llm or chatgpt)

    return chunks


def build_vector_store(chunks, persist_directory: Optional[str] = None) -> Chroma:
    kwargs = {"persist_directory": persist_directory} if persist_directory else {}
    return Chroma.from_documents(chunks, embeddings, **kwargs)


def build_hybrid_retriever(
    vector_store: Chroma,
    chunks,
    k: int = 3,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> EnsembleRetriever:
    """Dense (Chroma) + sparse (BM25) retrieval, combined as in app.py."""
    dense = vector_store.as_retriever(search_kwargs={"k": k})
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_weight, sparse_weight])


def build_reranking_retriever(
    base_retriever,
    top_n: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> ContextualCompressionRetriever:
    """Wrap `base_retriever` with cross-encoder reranking, via LangChain's own
    CrossEncoderReranker/HuggingFaceCrossEncoder -- no custom scoring logic.

    `base_retriever` should return a wider candidate pool than `top_n` (e.g.
    build_hybrid_retriever(..., k=10)) so there's something for the reranker
    to actually rerank.
    """
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


def build_qa_chain(retriever, llm=None):
    llm = llm or chatgpt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("system", "Context: {context}"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def answer(question: str, chain) -> dict:
    """Run one single-turn question through a chain built by build_qa_chain.

    Returns {"answer": str, "contexts": list[str]} -- the shape RAGAS expects
    for its `answer` and `contexts` fields.
    """
    result = chain.invoke({"input": question})
    contexts = [doc.page_content for doc in result.get("context", [])]
    return {"answer": result["answer"], "contexts": contexts}
