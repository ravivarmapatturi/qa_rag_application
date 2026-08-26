"""Contextual retrieval: prepend an LLM-generated, document-situated summary
to each chunk before it's embedded and BM25-indexed.

Follows Anthropic's published contextual-retrieval technique --
https://www.anthropic.com/news/contextual-retrieval, reference implementation
at github.com/anthropics/anthropic-cookbook (capabilities/contextual-embeddings)
-- same document/chunk prompt template and the same "prepend context, then
embed the combined text" pattern. Their reference calls the Anthropic Messages
API directly with `cache_control` prompt caching; this repo doesn't use the
Anthropic API anywhere else, so this adapts the technique to the LLM already
wired up in query_translation.py (chatgpt, currently Groq) instead of adding a
third API key dependency for one step. That means we lose their explicit
prompt-caching cost optimization (a provider-specific mechanic, orthogonal to
the retrieval-quality technique itself) -- each chunk is still one full LLM
call with the whole document in context, so this is meaningfully more
expensive and slower than plain chunking. It's opt-in for that reason (see
`use_contextual` in rag_pipeline.load_and_chunk).
"""
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from observability import trace_config

DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>"""

CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


def situate_context(llm, doc_content: str, chunk_content: str) -> str:
    """One LLM call: ask for a short (~50-100 token) blurb situating `chunk_content`
    within `doc_content`. Mirrors the cookbook's `situate_context` exactly, minus
    the Anthropic-specific cache_control block (see module docstring)."""
    message = HumanMessage(
        content=(
            DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_content)
            + "\n\n"
            + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content)
        )
    )
    response = llm.invoke([message], config=trace_config(run_name="contextual_retrieval.situate_context"))
    return response.content.strip()


def contextualize_chunks(chunks: list[Document], llm) -> list[Document]:
    """Group chunks by source document, generate a situating context for each,
    and return new Documents whose page_content is `context + "\\n\\n" + chunk`
    -- the same text gets embedded and BM25-indexed (one field, matching the
    cookbook's `text_to_embed` pattern), while the original chunk text is kept
    in metadata["original_content"] so the UI can still show clean source
    excerpts instead of the prepended context.
    """
    by_source: dict[str, list[Document]] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        by_source.setdefault(source, []).append(chunk)

    contextualized: list[Document] = []
    for source, source_chunks in by_source.items():
        full_doc = "\n\n".join(c.page_content for c in source_chunks)
        for chunk in source_chunks:
            context = situate_context(llm, full_doc, chunk.page_content)
            contextualized.append(
                Document(
                    page_content=f"{context}\n\n{chunk.page_content}",
                    metadata={**chunk.metadata, "original_content": chunk.page_content},
                )
            )
    return contextualized
