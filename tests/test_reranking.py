from rag_pipeline import build_hybrid_retriever, build_reranking_retriever, build_vector_store, load_and_chunk


def test_reranker_truncates_to_top_n(fixture_pdf_paths):
    chunks = load_and_chunk(fixture_pdf_paths)
    vector_store = build_vector_store(chunks)
    base_retriever = build_hybrid_retriever(vector_store, chunks, k=10)
    reranking_retriever = build_reranking_retriever(base_retriever, top_n=2)

    docs = reranking_retriever.invoke("What is the equipment stipend amount?")

    assert len(docs) <= 2
    assert len(docs) > 0


def test_reranked_results_come_from_base_retriever(fixture_pdf_paths):
    chunks = load_and_chunk(fixture_pdf_paths)
    vector_store = build_vector_store(chunks)
    base_retriever = build_hybrid_retriever(vector_store, chunks, k=10)
    base_docs = {d.page_content for d in base_retriever.invoke("expense report approval")}

    reranking_retriever = build_reranking_retriever(base_retriever, top_n=2)
    reranked_docs = reranking_retriever.invoke("expense report approval")

    assert all(d.page_content in base_docs for d in reranked_docs)
