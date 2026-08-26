from rag_pipeline import answer, build_hybrid_retriever, build_qa_chain, build_vector_store, load_and_chunk


def test_load_and_chunk_produces_chunks_from_fixtures(fixture_pdf_paths):
    chunks = load_and_chunk(fixture_pdf_paths)
    assert len(chunks) > 0
    assert any("90 days" in c.page_content for c in chunks)


def test_hybrid_retriever_surfaces_relevant_chunk(fixture_pdf_paths):
    chunks = load_and_chunk(fixture_pdf_paths)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(vector_store, chunks)

    docs = retriever.invoke("What is the equipment stipend amount?")

    assert any("$750" in d.page_content for d in docs)


def test_answer_returns_expected_shape(fixture_pdf_paths, fake_llm):
    chunks = load_and_chunk(fixture_pdf_paths)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(vector_store, chunks)
    chain = build_qa_chain(retriever, llm=fake_llm)

    result = answer("What is the equipment stipend amount?", chain)

    assert set(result.keys()) == {"answer", "contexts"}
    assert result["answer"] == "fake response"
    assert isinstance(result["contexts"], list)
    assert len(result["contexts"]) > 0
